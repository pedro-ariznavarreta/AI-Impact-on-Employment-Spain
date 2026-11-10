from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "panel_ine_cno_ai_exposure.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "model_comparison"
DOCS_DIR = PROJECT_ROOT / "docs"
RANDOM_SEED = 20260915
TEST_PERIODS = 8
VALIDATION_PERIODS = 4

NUMERIC_FEATURES = [
    "employment_log", "employment_qoq_pct", "employment_yoy_pct",
    "ai_exposure_score", "anthropic_exposure_score", "time_index",
    "quarter_sin", "quarter_cos", "interaction_qoq_ai", "interaction_yoy_ai",
]


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["period_date"])
    df = df.sort_values(["period_date", "cno_major"]).reset_index(drop=True)
    df["cno_major"] = df["cno_major"].astype(str).str.replace(".0", "", regex=False)
    df["quarter"] = df["quarter"].astype(int)
    df["year"] = df["year"].astype(int)
    df["ai_exposure_category"] = df["ai_exposure_category"].fillna("sin_categoria")
    df["ai_exposure_score"] = df["language_modeling_percentile_cno_proxy"].fillna(0) / 100
    df["anthropic_exposure_score"] = df["anthropic_percentile_cno_proxy"].fillna(0) / 100
    df["employment_log"] = np.log(df["employment_thousand_people"].clip(lower=1))
    df["time_index"] = (df["year"] - df["year"].min()) * 4 + df["quarter"] - 1
    df["target"] = df["employment_growth_next_quarter_pct"]
    angle = 2 * np.pi * (df["quarter"] - 1) / 4
    df["quarter_sin"], df["quarter_cos"] = np.sin(angle), np.cos(angle)
    df["interaction_qoq_ai"] = df["employment_qoq_pct"].fillna(0) * df["ai_exposure_score"]
    df["interaction_yoy_ai"] = df["employment_yoy_pct"].fillna(0) * df["ai_exposure_score"]
    return df


def design(df: pd.DataFrame, dummy_columns: list[str]) -> tuple[np.ndarray, list[str]]:
    numeric = df[NUMERIC_FEATURES].fillna(0).to_numpy(float)
    dummies = pd.get_dummies(df["cno_major"], prefix="cno", dtype=float).reindex(columns=dummy_columns, fill_value=0)
    return np.column_stack([numeric, dummies.to_numpy(float)]), NUMERIC_FEATURES + dummy_columns


class Standardizer:
    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class LinearModel:
    def __init__(self, alpha: float = 0.0, l1_ratio: float = 0.0, max_iter: int = 4000):
        self.alpha, self.l1_ratio, self.max_iter = alpha, l1_ratio, max_iter

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearModel":
        self.intercept_ = float(y.mean())
        yc = y - self.intercept_
        if self.l1_ratio == 0:
            self.coef_ = np.linalg.pinv(x.T @ x + self.alpha * np.eye(x.shape[1])) @ x.T @ yc
            return self
        beta = np.zeros(x.shape[1])
        l1, l2 = self.alpha * self.l1_ratio, self.alpha * (1 - self.l1_ratio)
        norms = np.sum(x * x, axis=0) + l2
        for _ in range(self.max_iter):
            old = beta.copy()
            for j in range(x.shape[1]):
                residual = yc - x @ beta + x[:, j] * beta[j]
                rho = float(x[:, j] @ residual)
                beta[j] = np.sign(rho) * max(abs(rho) - l1, 0) / norms[j]
            if np.max(np.abs(beta - old)) < 1e-7:
                break
        self.coef_ = beta
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept_ + x @ self.coef_


@dataclass
class Node:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


class RegressionTree:
    def __init__(self, max_depth=3, min_leaf=12, max_features=None, rng=None):
        self.max_depth, self.min_leaf, self.max_features = max_depth, min_leaf, max_features
        self.rng = rng or np.random.default_rng(RANDOM_SEED)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RegressionTree":
        self.root = self._grow(x, y, 0)
        return self

    def _grow(self, x: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(float(y.mean()))
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or np.var(y) < 1e-12:
            return node
        features = np.arange(x.shape[1])
        if self.max_features and self.max_features < len(features):
            features = self.rng.choice(features, self.max_features, replace=False)
        best = (np.inf, None, None, None)
        for j in features:
            values = np.unique(x[:, j])
            if len(values) < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2
            if len(thresholds) > 40:
                thresholds = np.quantile(thresholds, np.linspace(.05, .95, 40))
            for threshold in thresholds:
                left = x[:, j] <= threshold
                n_left = int(left.sum())
                if n_left < self.min_leaf or len(y) - n_left < self.min_leaf:
                    continue
                loss = np.var(y[left]) * n_left + np.var(y[~left]) * (len(y) - n_left)
                if loss < best[0]:
                    best = (loss, int(j), float(threshold), left)
        if best[1] is None:
            return node
        _, node.feature, node.threshold, left = best
        node.left = self._grow(x[left], y[left], depth + 1)
        node.right = self._grow(x[~left], y[~left], depth + 1)
        return node

    def predict(self, x: np.ndarray) -> np.ndarray:
        def one(row, node):
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            return node.value
        return np.array([one(row, self.root) for row in x])


class RandomForest:
    def __init__(self, n_trees=80, max_depth=4, min_leaf=10, max_features=5, seed=RANDOM_SEED):
        self.n_trees, self.max_depth, self.min_leaf, self.max_features, self.seed = n_trees, max_depth, min_leaf, max_features, seed

    def fit(self, x, y):
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, len(y), len(y))
            tree = RegressionTree(self.max_depth, self.min_leaf, self.max_features, rng).fit(x[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, x):
        return np.mean([tree.predict(x) for tree in self.trees], axis=0)


class GradientBoosting:
    def __init__(self, n_estimators=80, learning_rate=.05, max_depth=2, min_leaf=12):
        self.n_estimators, self.learning_rate, self.max_depth, self.min_leaf = n_estimators, learning_rate, max_depth, min_leaf

    def fit(self, x, y):
        self.base = float(y.mean())
        pred = np.full(len(y), self.base)
        self.trees = []
        for i in range(self.n_estimators):
            tree = RegressionTree(self.max_depth, self.min_leaf, None, np.random.default_rng(RANDOM_SEED + i)).fit(x, y - pred)
            pred += self.learning_rate * tree.predict(x)
            self.trees.append(tree)
        return self

    def predict(self, x):
        pred = np.full(len(x), self.base)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(x)
        return pred


def rmse(y, p): return float(np.sqrt(np.mean((y-p)**2)))
def mae(y, p): return float(np.mean(np.abs(y-p)))
def r2(y, p):
    den = np.sum((y-y.mean())**2)
    return float(1 - np.sum((y-p)**2)/den) if den else 0.0


GRIDS: dict[str, list[dict]] = {
    "Regresion lineal": [{}],
    "Ridge": [{"alpha": x} for x in [.01, .1, 1, 10, 100]],
    "Elastic Net": [{"alpha": a, "l1_ratio": r} for a in [.01, .1, 1, 10] for r in [.25, .5, .75]],
    "Random Forest": [{"n_trees": 80, "max_depth": d, "min_leaf": m, "max_features": f} for d in [3, 5] for m in [8, 15] for f in [5, 10]],
    "Gradient Boosting": [{"n_estimators": n, "learning_rate": lr, "max_depth": d, "min_leaf": 12} for n in [50, 100] for lr in [.03, .07] for d in [1, 2]],
}


def make_model(name: str, params: dict):
    if name == "Regresion lineal": return LinearModel()
    if name == "Ridge": return LinearModel(alpha=params["alpha"])
    if name == "Elastic Net": return LinearModel(alpha=params["alpha"], l1_ratio=params["l1_ratio"])
    if name == "Random Forest": return RandomForest(**params)
    if name == "Gradient Boosting": return GradientBoosting(**params)
    raise KeyError(name)


def fit_predict(name, params, train, test, dummies):
    x_train, features = design(train, dummies)
    x_test, _ = design(test, dummies)
    scaler = Standardizer().fit(x_train)
    model = make_model(name, params).fit(scaler.transform(x_train), train["target"].to_numpy(float))
    return model.predict(scaler.transform(x_test)), model, scaler, features


def select_hyperparameters(df, train_periods, validation_periods, dummies):
    train, val = df[df.periodo.isin(train_periods)], df[df.periodo.isin(validation_periods)]
    rows, best = [], {}
    for name, grid in GRIDS.items():
        best_score, best_params = np.inf, None
        for params in grid:
            pred, _, _, _ = fit_predict(name, params, train, val, dummies)
            score = rmse(val.target.to_numpy(float), pred)
            rows.append({"modelo": name, "parametros": json.dumps(params), "rmse_validacion": score})
            if score < best_score:
                best_score, best_params = score, params
        best[name] = best_params
    return best, pd.DataFrame(rows)


def run_backtest(df, test_periods, dummies, params):
    rows = []
    for period in test_periods:
        test = df[df.periodo == period].copy()
        train = df[df.period_date < test.period_date.min()]
        y = test.target.to_numpy(float)
        for name in GRIDS:
            pred, _, _, _ = fit_predict(name, params[name], train, test, dummies)
            for i, (_, source) in enumerate(test.iterrows()):
                rows.append({"periodo": period, "period_date": source.period_date, "cno_major": source.cno_major,
                             "cno_title": source.cno_title, "actual": y[i], "modelo": name, "prediccion": pred[i]})
        for name, pred in [("Crecimiento cero", np.zeros(len(test))), ("Persistencia trimestral", test.employment_qoq_pct.fillna(0).to_numpy(float))]:
            for i, (_, source) in enumerate(test.iterrows()):
                rows.append({"periodo": period, "period_date": source.period_date, "cno_major": source.cno_major,
                             "cno_title": source.cno_title, "actual": y[i], "modelo": name, "prediccion": pred[i]})
    return pd.DataFrame(rows)


def add_ensemble(predictions: pd.DataFrame, validation_predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    candidates = ["Ridge", "Elastic Net", "Random Forest", "Gradient Boosting"]
    scores = validation_predictions.groupby("modelo").apply(lambda g: rmse(g.actual.to_numpy(), g.prediccion.to_numpy()), include_groups=False)
    selected = list(scores.loc[candidates].sort_values().head(3).index)
    inverse = np.array([1 / scores[name] for name in selected])
    weights = inverse / inverse.sum()
    wide = predictions.pivot_table(index=["periodo", "period_date", "cno_major", "cno_title", "actual"], columns="modelo", values="prediccion").reset_index()
    wide["prediccion"] = sum(w * wide[name] for name, w in zip(selected, weights))
    ensemble = wide[["periodo", "period_date", "cno_major", "cno_title", "actual", "prediccion"]].copy()
    ensemble["modelo"] = "Ensemble ponderado"
    predictions = pd.concat([predictions, ensemble], ignore_index=True)
    return predictions, {name: float(w) for name, w in zip(selected, weights)}


def metric_table(predictions):
    rows = []
    zero = predictions[predictions.modelo == "Crecimiento cero"]
    zero_rmse = rmse(zero.actual.to_numpy(), zero.prediccion.to_numpy())
    for name, g in predictions.groupby("modelo"):
        score = rmse(g.actual.to_numpy(), g.prediccion.to_numpy())
        rows.append({"modelo": name, "rmse": score, "mae": mae(g.actual.to_numpy(), g.prediccion.to_numpy()),
                     "r2": r2(g.actual.to_numpy(), g.prediccion.to_numpy()),
                     "mejora_rmse_vs_cero_pct": 100 * (zero_rmse - score) / zero_rmse})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def bootstrap_differences(predictions, best_name, reps=2000):
    wide = predictions.pivot_table(index=["periodo", "cno_major", "actual"], columns="modelo", values="prediccion").reset_index()
    periods = wide.periodo.unique()
    rng = np.random.default_rng(RANDOM_SEED)
    diffs = []
    for _ in range(reps):
        sample = rng.choice(periods, len(periods), replace=True)
        parts = [wide[wide.periodo == p] for p in sample]
        boot = pd.concat(parts)
        diffs.append(rmse(boot.actual.to_numpy(), boot[best_name].to_numpy()) - rmse(boot.actual.to_numpy(), boot["Crecimiento cero"].to_numpy()))
    return {"diferencia_rmse_media": float(np.mean(diffs)), "ic95_inferior": float(np.quantile(diffs, .025)), "ic95_superior": float(np.quantile(diffs, .975))}


def permutation_importance(df, train_periods, test_periods, dummies, name, params, repeats=20):
    train, test = df[df.periodo.isin(train_periods)], df[df.periodo.isin(test_periods)]
    base, model, scaler, features = fit_predict(name, params, train, test, dummies)
    y = test.target.to_numpy(float)
    baseline = rmse(y, base)
    x, _ = design(test, dummies)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for j, feature in enumerate(features):
        increases = []
        for _ in range(repeats):
            shuffled = x.copy(); shuffled[:, j] = rng.permutation(shuffled[:, j])
            increases.append(rmse(y, model.predict(scaler.transform(shuffled))) - baseline)
        rows.append({"variable": feature, "aumento_rmse": float(np.mean(increases)), "desviacion": float(np.std(increases))})
    return pd.DataFrame(rows).sort_values("aumento_rmse", ascending=False)


def fit_final_ensemble(df, dummies, params, weights):
    fitted = {}
    for name, weight in weights.items():
        x, features = design(df, dummies)
        scaler = Standardizer().fit(x)
        model = make_model(name, params[name]).fit(scaler.transform(x), df.target.to_numpy(float))
        fitted[name] = (weight, model, scaler, features)
    return fitted


def next_quarter(date):
    return date + pd.offsets.QuarterEnd(startingMonth=3)


def recursive_ensemble_forecast(panel, fitted, dummies, residuals, horizons=8):
    latest = panel.sort_values("period_date").groupby("cno_major", as_index=False).tail(1)
    histories = {str(k): list(g.sort_values("period_date")[["period_date", "employment_thousand_people"]].itertuples(index=False, name=None)) for k, g in panel.groupby("cno_major")}
    rows = []
    p05, p95 = np.quantile(residuals, [.05, .95])
    for _, source in latest.iterrows():
        history = histories[str(source.cno_major)].copy()
        for horizon in range(1, horizons + 1):
            date, employment = history[-1]
            previous = history[-2][1]
            previous_year = history[-5][1]
            year, quarter = date.year, ((date.month - 1) // 3) + 1
            row = source.copy()
            row["period_date"], row["periodo"], row["year"], row["quarter"] = date, f"{year}T{quarter}", year, quarter
            row["employment_thousand_people"] = employment
            row["employment_qoq_pct"] = (employment / previous - 1) * 100
            row["employment_yoy_pct"] = (employment / previous_year - 1) * 100
            row["employment_log"] = np.log(max(employment, 1))
            row["time_index"] = (year - panel.year.min()) * 4 + quarter - 1
            angle = 2 * np.pi * (quarter - 1) / 4
            row["quarter_sin"], row["quarter_cos"] = np.sin(angle), np.cos(angle)
            row["interaction_qoq_ai"] = row["employment_qoq_pct"] * row["ai_exposure_score"]
            row["interaction_yoy_ai"] = row["employment_yoy_pct"] * row["ai_exposure_score"]
            x, _ = design(pd.DataFrame([row]), dummies)
            components = {name: float(model.predict(scaler.transform(x))[0]) for name, (_, model, scaler, _) in fitted.items()}
            growth = sum(fitted[name][0] * value for name, value in components.items())
            future_date = next_quarter(date)
            future_employment = employment * (1 + growth / 100)
            probability = float((np.sum(residuals < -growth) + 1) / (len(residuals) + 2))
            weighted_risk = future_employment * probability * source.ai_exposure_score
            fy, fq = future_date.year, ((future_date.month - 1) // 3) + 1
            rows.append({"forecast_origin_period": source.periodo, "forecast_horizon_quarters": horizon,
                         "cno_major": source.cno_major, "cno_title": source.cno_title,
                         "ai_exposure_category": source.ai_exposure_category,
                         "current_employment_thousand_people": employment,
                         "forecast_period": f"{fy}T{fq}", "forecast_period_date": future_date,
                         "predicted_growth_pct": growth, "prediction_lower_90_pct": growth + p05,
                         "prediction_upper_90_pct": growth + p95,
                         "predicted_employment_thousand_people": future_employment,
                         "contraction_probability": probability,
                         "exposure_weighted_risk_thousand": weighted_risk,
                         "risk_index": weighted_risk * (1 + source.ai_exposure_score),
                         **{f"prediction_{name.lower().replace(' ', '_')}_pct": value for name, value in components.items()}})
            history.append((future_date, future_employment))
    forecast = pd.DataFrame(rows)
    ranks = forecast.groupby("forecast_horizon_quarters")["risk_index"].rank(pct=True)
    forecast["risk_category"] = pd.cut(ranks, [0, .33, .66, 1], labels=["bajo", "medio", "alto"], include_lowest=True)
    return forecast


def make_charts(metrics, predictions, importance):
    from PIL import Image, ImageDraw, ImageFont
    chart_dir = OUTPUT_DIR / "figures"; chart_dir.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=18); small = ImageFont.load_default(size=14)
    # Model metric comparison
    data = metrics.sort_values("rmse", ascending=False)
    image = Image.new("RGB", (1400, 760), "white"); draw = ImageDraw.Draw(image)
    draw.text((50, 25), "Comparacion del RMSE fuera de muestra", fill="black", font=font)
    max_value = data.rmse.max()
    for i, row in enumerate(data.itertuples()):
        y = 90 + i * 78; width = int(900 * row.rmse / max_value)
        color = "#2F6B73" if row.modelo == data.iloc[-1].modelo else "#9CB7BC"
        draw.text((50, y + 10), row.modelo, fill="black", font=small)
        draw.rectangle((330, y, 330 + width, y + 42), fill=color)
        draw.text((345 + width, y + 10), f"{row.rmse:.3f}", fill="black", font=small)
    image.save(chart_dir / "comparacion_rmse_modelos.png")
    # Permutation importance
    data = importance.head(10).sort_values("aumento_rmse")
    image = Image.new("RGB", (1400, 760), "white"); draw = ImageDraw.Draw(image)
    draw.text((50, 25), "Importancia por permutacion del mejor modelo", fill="black", font=font)
    max_value = max(data.aumento_rmse.max(), .001)
    for i, row in enumerate(data.itertuples()):
        y = 90 + i * 62; width = int(850 * max(row.aumento_rmse, 0) / max_value)
        draw.text((50, y + 8), row.variable, fill="black", font=small)
        draw.rectangle((390, y, 390 + width, y + 34), fill="#2F6B73")
        draw.text((405 + width, y + 8), f"{row.aumento_rmse:.3f}", fill="black", font=small)
    image.save(chart_dir / "importancia_variables.png")


def write_markdown(metrics, by_group, params, weights, robustness, importance):
    def md(df, decimals=4):
        out = df.copy()
        for col in out.select_dtypes(include=[np.number]).columns:
            out[col] = out[col].map(lambda v: f"{v:.{decimals}f}")
        header = "| " + " | ".join(out.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(out.columns)) + " |"
        rows = ["| " + " | ".join(str(row[col]) for col in out.columns) + " |" for _, row in out.iterrows()]
        return "\n".join([header, separator, *rows])
    report = f"""# Comparacion ampliada de modelos predictivos

## Diseno experimental

Se compararon dos baselines, tres modelos lineales, dos modelos no lineales basados en arboles y un ensemble ponderado. Todos se evaluaron sobre los mismos ocho trimestres, 2024T1-2025T4, mediante backtesting walk-forward. En cada corte, el entrenamiento, la imputacion y la estandarizacion utilizaron exclusivamente observaciones anteriores al trimestre predicho.

La seleccion de hiperparametros se realizo en una ventana temporal independiente de cuatro trimestres inmediatamente anterior al test. El ensemble combina los tres mejores candidatos segun el RMSE de esa ventana; sus pesos son inversamente proporcionales al RMSE de validacion.

## Hiperparametros seleccionados

```json
{json.dumps(params, indent=2, ensure_ascii=False)}
```

Pesos del ensemble: `{json.dumps(weights, ensure_ascii=False)}`.

## Resultados fuera de muestra

{md(metrics)}

## Resultados por grupo ocupacional

{md(by_group)}

## Robustez frente al baseline de crecimiento cero

Se remuestrearon bloques trimestrales completos para conservar la dependencia transversal entre ocupaciones. Diferencia RMSE del mejor modelo menos el baseline: {robustness['diferencia_rmse_media']:.4f}; IC bootstrap del 95 % [{robustness['ic95_inferior']:.4f}, {robustness['ic95_superior']:.4f}]. Un intervalo que incluya cero indica que la mejora no puede considerarse estable con esta muestra.

## Importancia por permutacion

{md(importance.head(15))}

## Interpretacion

La seleccion final debe basarse conjuntamente en rendimiento fuera de muestra, estabilidad e interpretabilidad. Una diferencia pequena frente al crecimiento cero no justifica por si sola un modelo mas complejo. La exposicion a IA permanece constante dentro de cada grupo y, por ello, sus importancias describen diferencias estructurales entre grupos, no cambios temporales de adopcion ni efectos causales.
"""
    (DOCS_DIR / "comparacion_modelos_recomendada.md").write_text(report, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); DOCS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_panel(); model_df = df[df.target.notna()].copy()
    periods = sorted(model_df.periodo.unique())
    test_periods = periods[-TEST_PERIODS:]
    validation_periods = periods[-(TEST_PERIODS + VALIDATION_PERIODS):-TEST_PERIODS]
    train_periods = periods[:-(TEST_PERIODS + VALIDATION_PERIODS)]
    dummies = sorted(pd.get_dummies(model_df.cno_major, prefix="cno").columns)
    params, tuning = select_hyperparameters(model_df, train_periods, validation_periods, dummies)
    validation = run_backtest(model_df[model_df.periodo.isin(train_periods + validation_periods)], validation_periods, dummies, params)
    predictions = run_backtest(model_df, test_periods, dummies, params)
    predictions, weights = add_ensemble(predictions, validation)
    metrics = metric_table(predictions)
    best_name = metrics.iloc[0].modelo
    by_group = predictions.groupby(["modelo", "cno_major", "cno_title"]).apply(
        lambda g: pd.Series({"rmse": rmse(g.actual.to_numpy(), g.prediccion.to_numpy()), "mae": mae(g.actual.to_numpy(), g.prediccion.to_numpy())}), include_groups=False
    ).reset_index()
    robustness = bootstrap_differences(predictions, best_name)
    importance_model = best_name if best_name in GRIDS else min(weights, key=lambda k: -weights[k])
    importance = permutation_importance(model_df, periods[:-TEST_PERIODS], test_periods, dummies, importance_model, params[importance_model])
    residuals = predictions[predictions.modelo == "Ensemble ponderado"].eval("actual - prediccion").to_numpy(float)
    final_models = fit_final_ensemble(model_df, dummies, params, weights)
    forecast = recursive_ensemble_forecast(df, final_models, dummies, residuals)
    predictions.to_csv(OUTPUT_DIR / "predicciones_backtesting_todos_modelos.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metricas_comparacion_modelos.csv", index=False)
    by_group.to_csv(OUTPUT_DIR / "metricas_por_grupo_y_modelo.csv", index=False)
    tuning.to_csv(OUTPUT_DIR / "ajuste_hiperparametros.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "importancia_variables_permutacion.csv", index=False)
    forecast.to_csv(OUTPUT_DIR / "previsiones_ensemble_8_trimestres.csv", index=False)
    forecast.to_csv(PROJECT_ROOT / "data" / "processed" / "powerbi_fact_forecast_v2.csv", index=False)
    forecast[forecast.forecast_horizon_quarters == 1].sort_values("risk_index", ascending=False).to_csv(PROJECT_ROOT / "data" / "processed" / "powerbi_fact_risk_summary_v2.csv", index=False)
    (OUTPUT_DIR / "configuracion_modelos.json").write_text(json.dumps({"hiperparametros": params, "pesos_ensemble": weights, "robustez": robustness}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(metrics, by_group[by_group.modelo.isin(metrics.head(4).modelo)], params, weights, robustness, importance)
    make_charts(metrics, predictions, importance)
    print(metrics.to_string(index=False))
    print("BEST_MODEL", best_name)


if __name__ == "__main__":
    main()
