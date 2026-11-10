from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "panel_ine_cno_ai_exposure.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
FORECAST_HORIZON = 8
TEST_PERIODS = 8
VALIDATION_PERIODS = 4


def empirical_contraction_probability(predicted_growth: float, residuals: np.ndarray) -> float:
    """Estimate P(actual growth < 0) from out-of-sample backtest residuals."""
    if residuals.size == 0:
        return float("nan")
    # Add-one smoothing avoids exact probabilities of zero or one in a small sample.
    contractions = int(np.sum(residuals < -predicted_growth))
    return float((contractions + 1) / (residuals.size + 2))


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"No se puede escribir el archivo '{path}'. "
            "Cierralo en Excel, Power BI, el explorador de Windows o cualquier visor que lo tenga abierto y vuelve a ejecutar el script."
        ) from exc


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(DATA_PATH, parse_dates=["period_date"])
    panel = panel.sort_values(["cno_major", "period_date"]).reset_index(drop=True)
    panel["cno_major"] = panel["cno_major"].astype(str).str.replace(".0", "", regex=False)
    panel["quarter"] = panel["quarter"].astype(int)
    panel["year"] = panel["year"].astype(int)
    panel["ai_exposure_category"] = panel["ai_exposure_category"].fillna("sin_categoria")
    panel["ai_exposure_score"] = panel["language_modeling_percentile_cno_proxy"].fillna(0) / 100.0
    panel["anthropic_exposure_score"] = panel["anthropic_percentile_cno_proxy"].fillna(0) / 100.0
    panel["employment_log"] = np.log(panel["employment_thousand_people"].clip(lower=1))
    panel["time_index"] = (
        (panel["year"] - panel["year"].min()) * 4 + (panel["quarter"] - 1)
    ).astype(float)
    panel["target_growth_next_quarter_pct"] = panel["employment_growth_next_quarter_pct"]
    return panel


FEATURE_COLUMNS = [
    "employment_log",
    "employment_qoq_pct",
    "employment_yoy_pct",
    "ai_exposure_score",
    "anthropic_exposure_score",
    "time_index",
    "quarter_sin",
    "quarter_cos",
    "interaction_qoq_ai",
    "interaction_yoy_ai",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    angle = 2 * np.pi * (frame["quarter"] - 1) / 4
    frame["quarter_sin"] = np.sin(angle)
    frame["quarter_cos"] = np.cos(angle)
    frame["interaction_qoq_ai"] = frame["employment_qoq_pct"].fillna(0) * frame["ai_exposure_score"]
    frame["interaction_yoy_ai"] = frame["employment_yoy_pct"].fillna(0) * frame["ai_exposure_score"]
    return frame


def build_design_matrix(df: pd.DataFrame, categories: List[str]) -> Tuple[np.ndarray, List[str]]:
    numeric = df[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=float)
    cno_dummies = pd.get_dummies(df["cno_major"].astype(str), prefix="cno")
    cno_dummies = cno_dummies.reindex(columns=categories, fill_value=0)
    X = np.column_stack([numeric, cno_dummies.to_numpy(dtype=float)])
    columns = FEATURE_COLUMNS + categories
    return X, columns


def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1.0
    return (X_train - means) / stds, (X_test - means) / stds, means, stds


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    intercept = np.ones((X.shape[0], 1))
    X_ext = np.hstack([intercept, X])
    penalty = np.eye(X_ext.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(X_ext.T @ X_ext + alpha * penalty, X_ext.T @ y)


def predict_ridge(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    intercept = np.ones((X.shape[0], 1))
    X_ext = np.hstack([intercept, X])
    return X_ext @ beta


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    baseline = np.mean(y_true)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - baseline) ** 2))
    return 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot


def choose_alpha(
    train_df: pd.DataFrame, validation_df: pd.DataFrame, categories: List[str]
) -> float:
    X_train, _ = build_design_matrix(train_df, categories)
    y_train = train_df["target_growth_next_quarter_pct"].to_numpy(dtype=float)
    X_val, _ = build_design_matrix(validation_df, categories)
    y_val = validation_df["target_growth_next_quarter_pct"].to_numpy(dtype=float)
    X_train_std, X_val_std, _, _ = standardize_train_test(X_train, X_val)

    best_alpha = ALPHAS[0]
    best_rmse = float("inf")
    for alpha in ALPHAS:
        beta = fit_ridge(X_train_std, y_train, alpha)
        pred = predict_ridge(X_val_std, beta)
        score = rmse(y_val, pred)
        if score < best_rmse:
            best_rmse = score
            best_alpha = alpha
    return best_alpha


def backtest(panel: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    model_df = add_features(
        panel[panel["target_growth_next_quarter_pct"].notna()].copy()
    ).reset_index(drop=True)
    categories = sorted(pd.get_dummies(model_df["cno_major"].astype(str), prefix="cno").columns)
    periods = sorted(model_df["periodo"].unique())
    test_periods = periods[-TEST_PERIODS:]
    validation_periods = periods[-(TEST_PERIODS + VALIDATION_PERIODS) : -TEST_PERIODS]

    train_core = model_df[~model_df["periodo"].isin(test_periods + validation_periods)]
    validation_df = model_df[model_df["periodo"].isin(validation_periods)]
    alpha = choose_alpha(train_core, validation_df, categories)

    predictions = []
    residuals = []
    naive_residuals = []
    for period in test_periods:
        train_slice = model_df[model_df["period_date"] < model_df.loc[model_df["periodo"] == period, "period_date"].min()]
        test_slice = model_df[model_df["periodo"] == period].copy()
        X_train, _ = build_design_matrix(train_slice, categories)
        y_train = train_slice["target_growth_next_quarter_pct"].to_numpy(dtype=float)
        X_test, _ = build_design_matrix(test_slice, categories)
        y_test = test_slice["target_growth_next_quarter_pct"].to_numpy(dtype=float)
        X_train_std, X_test_std, _, _ = standardize_train_test(X_train, X_test)
        beta = fit_ridge(X_train_std, y_train, alpha)
        pred = predict_ridge(X_test_std, beta)
        test_slice["prediction_next_quarter_pct"] = pred
        test_slice["naive_prediction_next_quarter_pct"] = test_slice["employment_qoq_pct"].fillna(0.0)
        test_slice["zero_growth_prediction_pct"] = 0.0
        predictions.append(test_slice)
        residuals.extend((y_test - pred).tolist())
        naive_residuals.extend(
            (y_test - test_slice["naive_prediction_next_quarter_pct"].to_numpy(dtype=float)).tolist()
        )

    evaluation = pd.concat(predictions, ignore_index=True)
    metrics = {
        "alpha": alpha,
        "observations_train": int(len(train_core) + len(validation_df)),
        "observations_test": int(len(evaluation)),
        "rmse_model": rmse(
            evaluation["target_growth_next_quarter_pct"].to_numpy(dtype=float),
            evaluation["prediction_next_quarter_pct"].to_numpy(dtype=float),
        ),
        "mae_model": mae(
            evaluation["target_growth_next_quarter_pct"].to_numpy(dtype=float),
            evaluation["prediction_next_quarter_pct"].to_numpy(dtype=float),
        ),
        "r2_model": r2(
            evaluation["target_growth_next_quarter_pct"].to_numpy(dtype=float),
            evaluation["prediction_next_quarter_pct"].to_numpy(dtype=float),
        ),
        "rmse_naive": rmse(
            evaluation["target_growth_next_quarter_pct"].to_numpy(dtype=float),
            evaluation["naive_prediction_next_quarter_pct"].to_numpy(dtype=float),
        ),
        "mae_naive": mae(
            evaluation["target_growth_next_quarter_pct"].to_numpy(dtype=float),
            evaluation["naive_prediction_next_quarter_pct"].to_numpy(dtype=float),
        ),
        "rmse_zero": rmse(
            evaluation["target_growth_next_quarter_pct"].to_numpy(dtype=float),
            evaluation["zero_growth_prediction_pct"].to_numpy(dtype=float),
        ),
        "mae_zero": mae(
            evaluation["target_growth_next_quarter_pct"].to_numpy(dtype=float),
            evaluation["zero_growth_prediction_pct"].to_numpy(dtype=float),
        ),
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0,
        "residual_p05": float(np.quantile(residuals, 0.05)),
        "residual_p95": float(np.quantile(residuals, 0.95)),
        "backtest_residuals": residuals,
        "test_period_start": test_periods[0],
        "test_period_end": test_periods[-1],
    }
    return evaluation, metrics


def fit_final_model(panel: pd.DataFrame, alpha: float) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    model_df = add_features(panel[panel["target_growth_next_quarter_pct"].notna()].copy())
    categories = sorted(pd.get_dummies(model_df["cno_major"].astype(str), prefix="cno").columns)
    X, _ = build_design_matrix(model_df, categories)
    y = model_df["target_growth_next_quarter_pct"].to_numpy(dtype=float)
    X_std, _, means, stds = standardize_train_test(X, X)
    beta = fit_ridge(X_std, y, alpha)
    return beta, categories, means, stds


def quarter_from_timestamp(timestamp: pd.Timestamp) -> Tuple[int, int]:
    quarter = ((timestamp.month - 1) // 3) + 1
    return int(timestamp.year), quarter


def build_row_from_history(
    cno_row: pd.Series,
    history: List[Tuple[pd.Timestamp, float]],
    reference_start_year: int,
) -> dict:
    current_date, current_employment = history[-1]
    year, quarter = quarter_from_timestamp(current_date)
    prev_employment = history[-2][1] if len(history) >= 2 else np.nan
    last_year_employment = history[-5][1] if len(history) >= 5 else np.nan

    qoq = (
        ((current_employment - prev_employment) / prev_employment) * 100
        if pd.notna(prev_employment) and prev_employment not in [0, np.nan]
        else np.nan
    )
    yoy = (
        ((current_employment - last_year_employment) / last_year_employment) * 100
        if pd.notna(last_year_employment) and last_year_employment not in [0, np.nan]
        else np.nan
    )

    return {
        "cno_major": cno_row["cno_major"],
        "cno_title": cno_row["cno_title"],
        "period_date": current_date,
        "periodo": f"{year}T{quarter}",
        "employment_thousand_people": current_employment,
        "employment_qoq_pct": qoq,
        "employment_yoy_pct": yoy,
        "quarter": quarter,
        "year": year,
        "time_index": float((year - reference_start_year) * 4 + (quarter - 1)),
        "ai_exposure_score": cno_row["ai_exposure_score"],
        "anthropic_exposure_score": cno_row["anthropic_exposure_score"],
        "employment_log": float(np.log(max(current_employment, 1))),
        "ai_exposure_category": cno_row["ai_exposure_category"],
    }


def add_months(date: pd.Timestamp, months: int) -> pd.Timestamp:
    return date + pd.offsets.MonthEnd(months)


def forecast_future(
    panel: pd.DataFrame,
    beta: np.ndarray,
    categories: List[str],
    means: np.ndarray,
    stds: np.ndarray,
    residuals: np.ndarray,
    residual_p05: float,
    residual_p95: float,
) -> pd.DataFrame:
    latest_rows = (
        panel.sort_values("period_date")
        .groupby("cno_major", as_index=False)
        .tail(1)
        .sort_values("cno_major")
    )
    history_map: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
    for cno_major, group in panel.groupby("cno_major"):
        history_map[str(cno_major)] = list(
            group.sort_values("period_date")[["period_date", "employment_thousand_people"]].itertuples(index=False, name=None)
        )

    forecasts = []
    base_year = int(panel["year"].min())
    for _, latest in latest_rows.iterrows():
        cno_key = str(latest["cno_major"])
        history = history_map[cno_key].copy()
        for horizon in range(1, FORECAST_HORIZON + 1):
            feature_row = build_row_from_history(latest, history, base_year)
            feature_df = add_features(pd.DataFrame([feature_row]))
            X, _ = build_design_matrix(feature_df, categories)
            X_std = (X - means) / stds
            predicted_growth = float(predict_ridge(X_std, beta)[0])
            next_date = add_months(history[-1][0], 3)
            next_employment = history[-1][1] * (1 + predicted_growth / 100.0)
            contraction_probability = empirical_contraction_probability(predicted_growth, residuals)
            exposure_weighted_risk = (
                next_employment * contraction_probability * latest["ai_exposure_score"]
            )
            forecasts.append(
                {
                    "forecast_origin_period": latest["periodo"],
                    "forecast_horizon_quarters": horizon,
                    "cno_major": latest["cno_major"],
                    "cno_title": latest["cno_title"],
                    "ai_exposure_category": latest["ai_exposure_category"],
                    "current_employment_thousand_people": history[-1][1],
                    "forecast_period": f"{quarter_from_timestamp(next_date)[0]}T{quarter_from_timestamp(next_date)[1]}",
                    "forecast_period_date": next_date,
                    "predicted_growth_pct": predicted_growth,
                    "prediction_lower_90_pct": predicted_growth + residual_p05,
                    "prediction_upper_90_pct": predicted_growth + residual_p95,
                    "predicted_employment_thousand_people": next_employment,
                    "contraction_probability": contraction_probability,
                    "exposure_weighted_risk_thousand": exposure_weighted_risk,
                    "risk_index": exposure_weighted_risk * (1 + latest["ai_exposure_score"]),
                }
            )
            history.append((next_date, next_employment))
    forecast_df = pd.DataFrame(forecasts)
    horizon_rank = forecast_df.groupby("forecast_horizon_quarters")["risk_index"].rank(pct=True)
    forecast_df["risk_category"] = pd.cut(
        horizon_rank,
        bins=[0, 0.33, 0.66, 1.0],
        labels=["bajo", "medio", "alto"],
        include_lowest=True,
    )
    return forecast_df


def build_powerbi_exports(panel: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    latest_period = panel["period_date"].max()

    dim_cno = (
        panel[["cno_major", "cno_title", "ai_exposure_category", "mapped_soc_majors", "mapped_soc_occupations"]]
        .drop_duplicates()
        .sort_values("cno_major")
    )
    history = panel[
        [
            "periodo",
            "period_date",
            "year",
            "quarter",
            "cno_major",
            "cno_title",
            "employment_thousand_people",
            "employment_qoq_pct",
            "employment_yoy_pct",
            "language_modeling_percentile_cno_proxy",
            "anthropic_percentile_cno_proxy",
            "ai_exposure_category",
        ]
    ].copy()
    risk_summary = (
        forecast_df[forecast_df["forecast_horizon_quarters"] == 1]
        .sort_values("risk_index", ascending=False)
    )

    safe_to_csv(dim_cno, OUTPUT_DIR / "powerbi_dim_cno.csv")
    safe_to_csv(history, OUTPUT_DIR / "powerbi_fact_employment_history.csv")
    safe_to_csv(forecast_df, OUTPUT_DIR / "powerbi_fact_forecast.csv")
    safe_to_csv(risk_summary, OUTPUT_DIR / "powerbi_fact_risk_summary.csv")


def write_report(metrics: dict, forecast_df: pd.DataFrame) -> None:
    top_risk = forecast_df[forecast_df["forecast_horizon_quarters"] == 1].copy()
    top_risk = top_risk.sort_values("risk_index", ascending=False).head(10)

    metrics_table = pd.DataFrame(
        [
            ("alpha ridge", metrics["alpha"]),
            ("observaciones train", metrics["observations_train"]),
            ("observaciones test", metrics["observations_test"]),
            ("rmse modelo", metrics["rmse_model"]),
            ("mae modelo", metrics["mae_model"]),
            ("r2 modelo", metrics["r2_model"]),
            ("rmse baseline ingenua", metrics["rmse_naive"]),
            ("mae baseline ingenua", metrics["mae_naive"]),
            ("rmse baseline crecimiento cero", metrics["rmse_zero"]),
            ("mae baseline crecimiento cero", metrics["mae_zero"]),
            ("desv. residual", metrics["residual_std"]),
            ("percentil 5 residual", metrics["residual_p05"]),
            ("percentil 95 residual", metrics["residual_p95"]),
            ("inicio test", metrics["test_period_start"]),
            ("fin test", metrics["test_period_end"]),
        ],
        columns=["metrica", "valor"],
    )

    def markdown_table(df: pd.DataFrame, decimals: int = 3) -> str:
        table = df.copy()
        for col in table.columns:
            if pd.api.types.is_numeric_dtype(table[col]):
                table[col] = table[col].map(
                    lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}"
                )
            else:
                table[col] = table[col].astype(str)
        header = "| " + " | ".join(table.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
        rows = [
            "| " + " | ".join(str(row[col]) for col in table.columns) + " |"
            for _, row in table.iterrows()
        ]
        return "\n".join([header, sep, *rows])

    report = f"""# Modelo predictivo y paquete Power BI

## Objetivo

Esta fase incorpora un primer modelo de prediccion trimestral del empleo por grandes grupos CNO-11 y un indice operativo de priorizacion ocupacional asociado a la exposicion a IA.

El modelo no estima destruccion causal de empleo. Lo que produce es:

- prediccion del crecimiento del empleo del siguiente trimestre;
- probabilidad empirica de contraccion del empleo a corto plazo;
- indicador sintetico de empleo ponderado por exposicion y riesgo, que no representa despidos esperados.

## Metodologia

- Unidad de analisis: gran grupo CNO-11 por trimestre.
- Variable objetivo: `employment_growth_next_quarter_pct`.
- Modelo: regresion ridge con efectos fijos por gran grupo ocupacional.
- Variables explicativas:
  - logaritmo del empleo actual;
  - variacion trimestral e interanual del empleo;
  - proxies de exposicion a IA y Anthropic;
  - estacionalidad trimestral;
  - tendencia temporal;
  - interacciones entre dinamica del empleo y exposicion a IA.
- Validacion: backtesting walk-forward sobre los ultimos {TEST_PERIODS} trimestres con ajuste de hiperparametro en una ventana previa de {VALIDATION_PERIODS} trimestres.
- Incertidumbre: distribucion empirica de los residuos fuera de muestra, sin asumir normalidad.

## Resultados de validacion

{markdown_table(metrics_table, decimals=4)}

Interpretacion rapida:

- Si `rmse modelo` es menor que `rmse baseline ingenua`, el modelo mejora una referencia simple.
- El baseline de crecimiento cero permite comprobar si la mejora se mantiene frente a una referencia conservadora adicional.
- `r2 modelo` debe leerse con cautela por el tamano reducido del panel y la volatilidad trimestral.
- La probabilidad de contraccion y el intervalo predictivo se estiman con los residuos del backtesting; no son probabilidades causales.

## Ocupaciones con mayor riesgo a un trimestre vista

{markdown_table(top_risk[[
    "cno_major",
    "cno_title",
    "ai_exposure_category",
    "predicted_growth_pct",
    "contraction_probability",
    "exposure_weighted_risk_thousand",
    "risk_category",
]], decimals=3)}

## Archivos generados

- `data/processed/panel_ine_cno_forecast_backtest.csv`
- `data/processed/cno_future_risk_summary.csv`
- `data/processed/powerbi_dim_cno.csv`
- `data/processed/powerbi_fact_employment_history.csv`
- `data/processed/powerbi_fact_forecast.csv`
- `data/processed/powerbi_fact_risk_summary.csv`

## Uso en Power BI

La recomendacion es construir un modelo sencillo tipo estrella:

- `powerbi_dim_cno.csv` como dimension ocupacional.
- `powerbi_fact_employment_history.csv` como tabla historica.
- `powerbi_fact_forecast.csv` como tabla de predicciones por horizonte.
- `powerbi_fact_risk_summary.csv` como tabla KPI para el ultimo corte disponible.

Indicadores sugeridos:

- empleo actual;
- crecimiento interanual;
- prediccion de crecimiento proximo trimestre;
- probabilidad de contraccion;
- indice de empleo ponderado por exposicion y riesgo;
- ranking por `risk_index`.

## Limitaciones

- El panel es agregado: 10 grandes grupos CNO, no ocupaciones individuales finas.
- La exposicion a IA es una proxy internacional agregada, no una medida observada de automatizacion en Espana.
- `exposure_weighted_risk_thousand` es un indice sintetico de priorizacion analitica, no una prevision de despidos ni un recuento de puestos que desapareceran.
- Los intervalos y probabilidades se basan en 80 errores fuera de muestra y deben interpretarse como aproximaciones exploratorias.
"""

    (DOCS_DIR / "modelo_predictivo_y_powerbi.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    backtest_df, metrics = backtest(panel)
    beta, categories, means, stds = fit_final_model(panel, metrics["alpha"])
    forecast_df = forecast_future(
        panel,
        beta,
        categories,
        means,
        stds,
        np.asarray(metrics["backtest_residuals"], dtype=float),
        metrics["residual_p05"],
        metrics["residual_p95"],
    )

    safe_to_csv(backtest_df, OUTPUT_DIR / "panel_ine_cno_forecast_backtest.csv")
    risk_summary = (
        forecast_df[forecast_df["forecast_horizon_quarters"] == 1]
        .sort_values("risk_index", ascending=False)
    )
    safe_to_csv(risk_summary, OUTPUT_DIR / "cno_future_risk_summary.csv")
    build_powerbi_exports(panel, forecast_df)
    write_report(metrics, forecast_df)

    print("Forecast outputs written to data/processed")
    print("Methodology note written to docs/modelo_predictivo_y_powerbi.md")


if __name__ == "__main__":
    main()
