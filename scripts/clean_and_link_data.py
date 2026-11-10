from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"


CNO_SOC_MAJOR_MAP = {
    "1": ["11"],
    "2": ["15", "17", "19", "23", "25", "27", "29"],
    "3": ["13", "21", "27", "29"],
    "4": ["43"],
    "5": ["31", "33", "35", "37", "39", "41"],
    "6": ["45"],
    "7": ["47", "49", "51"],
    "8": ["51", "53"],
    "9": ["35", "37", "45", "53"],
    "0": ["55"],
}


def normalize_name(value: str) -> str:
    value = value.strip().lower()
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
        "ü": "u",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def parse_ine_number(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text == "..":
        return np.nan
    text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_period(value: str):
    match = re.match(r"(\d{4})T([1-4])", str(value))
    if not match:
        return pd.NaT
    year = int(match.group(1))
    quarter = int(match.group(2))
    month = quarter * 3
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"No se puede escribir el archivo '{path}'. "
            "Cierralo en Excel, Power BI, el explorador de Windows o cualquier visor que lo tenga abierto y vuelve a ejecutar el script."
        ) from exc


def extract_cno_code(value: str):
    if value == "Total":
        return "total"
    match = re.match(r"^(\d)\s+", str(value))
    return match.group(1) if match else None


def extract_cno_title(value: str):
    return re.sub(r"^\d\s+", "", str(value)).strip()


def col_to_idx(ref):
    letters = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - 64
    return idx - 1


def read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in item.findall(".//a:t", ns)))

        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        relationship_id = None
        for sheet in workbook.find("a:sheets", ns):
            if sheet.attrib["name"] == sheet_name:
                relationship_id = sheet.attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
                break
        if relationship_id is None:
            raise ValueError(f"Sheet not found: {sheet_name}")

        relationships = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relationship in relationships:
            if relationship.attrib["Id"] == relationship_id:
                target = relationship.attrib["Target"]
                break
        if target is None:
            raise ValueError(f"Sheet relationship not found: {sheet_name}")

        sheet_path = "xl/" + target.lstrip("/")
        root = ET.fromstring(z.read(sheet_path))
        rows = []
        for row in root.findall(".//a:row", ns):
            values = []
            for cell in row.findall("a:c", ns):
                idx = col_to_idx(cell.attrib.get("r", "A1"))
                while len(values) <= idx:
                    values.append("")
                value_element = cell.find("a:v", ns)
                value = "" if value_element is None else value_element.text
                if cell.attrib.get("t") == "s" and value != "":
                    value = shared[int(value)]
                values[idx] = value
            rows.append(values)
    header = rows[0]
    body = [row + [""] * (len(header) - len(row)) for row in rows[1:]]
    return pd.DataFrame(body, columns=header)


def clean_ine_situation():
    path = RAW_DIR / "ine_epa_ocupados_situacion_sexo_ocupacion_65136.csv"
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")
    df.columns = [normalize_name(col) for col in df.columns]
    df["employment_thousand_people"] = df["total"].map(parse_ine_number)
    df["period_date"] = df["periodo"].map(parse_period)
    df["year"] = df["period_date"].dt.year
    df["quarter"] = df["periodo"].str.extract(r"T([1-4])").astype(float)
    df["cno_major"] = df["ocupacion_cno_11"].map(extract_cno_code)
    df["cno_title"] = df["ocupacion_cno_11"].map(extract_cno_title)
    ordered = df.sort_values(["sexo", "situacion_profesional", "cno_major", "period_date"])
    grouped = ordered.groupby(["sexo", "situacion_profesional", "cno_major"], dropna=False)
    ordered["employment_qoq_pct"] = (
        grouped["employment_thousand_people"].pct_change(fill_method=None) * 100
    )
    ordered["employment_yoy_pct"] = (
        grouped["employment_thousand_people"].pct_change(4, fill_method=None) * 100
    )
    return ordered


def clean_ine_age():
    path = RAW_DIR / "ine_epa_ocupados_edad_sexo_ocupacion_65135.csv"
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")
    df.columns = [normalize_name(col) for col in df.columns]
    df["employment_thousand_people"] = df["total"].map(parse_ine_number)
    df["period_date"] = df["periodo"].map(parse_period)
    df["year"] = df["period_date"].dt.year
    df["quarter"] = df["periodo"].str.extract(r"T([1-4])").astype(float)
    df["cno_major"] = df["ocupacion_cno_11"].map(extract_cno_code)
    df["cno_title"] = df["ocupacion_cno_11"].map(extract_cno_title)
    return df


def clean_aioe():
    aioe = read_xlsx_sheet(RAW_DIR / "aioe_data_appendix.xlsx", "Appendix A")
    aioe.columns = ["soc_code", "soc_title", "aioe_score"]
    aioe["aioe_score"] = pd.to_numeric(aioe["aioe_score"], errors="coerce")

    lm = read_xlsx_sheet(RAW_DIR / "aioe_language_modeling.xlsx", "LM AIOE")
    lm.columns = ["soc_code", "soc_title_lm", "language_modeling_aioe_score"]
    lm["language_modeling_aioe_score"] = pd.to_numeric(
        lm["language_modeling_aioe_score"], errors="coerce"
    )

    merged = aioe.merge(lm[["soc_code", "language_modeling_aioe_score"]], on="soc_code", how="left")
    merged["soc_major"] = merged["soc_code"].str.slice(0, 2)
    return merged


def clean_anthropic():
    job = pd.read_csv(RAW_DIR / "anthropic_job_exposure.csv", dtype={"occ_code": str})
    job = job.rename(
        columns={
            "occ_code": "soc_code",
            "title": "soc_title",
            "observed_exposure": "anthropic_observed_exposure",
        }
    )
    job["soc_major"] = job["soc_code"].str.slice(0, 2)
    job = job.drop_duplicates(subset=["soc_code"], keep="first").reset_index(drop=True)

    task = pd.read_csv(RAW_DIR / "anthropic_task_penetration.csv")
    task["penetration"] = pd.to_numeric(task["penetration"], errors="coerce")
    task = task.drop_duplicates().reset_index(drop=True)
    return job, task


def build_soc_exposure(aioe, anthropic_job):
    soc = aioe.merge(
        anthropic_job[["soc_code", "anthropic_observed_exposure"]],
        on="soc_code",
        how="left",
    )
    soc["aioe_percentile"] = soc["aioe_score"].rank(pct=True) * 100
    soc["language_modeling_aioe_percentile"] = (
        soc["language_modeling_aioe_score"].rank(pct=True) * 100
    )
    soc["anthropic_observed_percentile"] = (
        soc["anthropic_observed_exposure"].rank(pct=True) * 100
    )
    return soc


def build_cno_exposure(soc):
    rows = []
    for cno_major, soc_majors in CNO_SOC_MAJOR_MAP.items():
        # Aggregate directly over unique SOC occupations. The previous version
        # averaged SOC-major averages, which gave a small major the same weight
        # as a major containing many occupations.
        subset = soc[soc["soc_major"].isin(soc_majors)].drop_duplicates("soc_code")
        rows.append(
            {
                "cno_major": cno_major,
                "mapped_soc_majors": ",".join(soc_majors),
                "mapped_soc_occupations": int(subset["soc_code"].nunique()),
                "aioe_score_cno_proxy": subset["aioe_score"].mean(),
                "language_modeling_aioe_score_cno_proxy": subset["language_modeling_aioe_score"].mean(),
                "anthropic_observed_exposure_cno_proxy": subset["anthropic_observed_exposure"].mean(),
            }
        )
    cno = pd.DataFrame(rows)
    cno["aioe_percentile_cno_proxy"] = cno["aioe_score_cno_proxy"].rank(pct=True) * 100
    cno["language_modeling_percentile_cno_proxy"] = (
        cno["language_modeling_aioe_score_cno_proxy"].rank(pct=True) * 100
    )
    cno["anthropic_percentile_cno_proxy"] = (
        cno["anthropic_observed_exposure_cno_proxy"].rank(pct=True) * 100
    )
    cno["ai_exposure_category"] = pd.cut(
        cno["language_modeling_percentile_cno_proxy"],
        bins=[0, 33.333, 66.666, 100],
        labels=["baja", "media", "alta"],
        include_lowest=True,
    )
    return cno


def build_main_panel(ine_situation, cno_exposure):
    base = ine_situation[
        (ine_situation["sexo"] == "Ambos sexos")
        & (ine_situation["situacion_profesional"] == "Total")
        & (ine_situation["cno_major"] != "total")
    ].copy()
    panel = base.merge(cno_exposure, on="cno_major", how="left")
    panel = panel.sort_values(["cno_major", "period_date"])
    panel["employment_next_quarter_thousand_people"] = panel.groupby("cno_major")[
        "employment_thousand_people"
    ].shift(-1)
    panel["employment_growth_next_quarter_pct"] = (
        (panel["employment_next_quarter_thousand_people"] - panel["employment_thousand_people"])
        / panel["employment_thousand_people"]
        * 100
    )
    return panel


def validate_outputs(panel, soc_exposure, cno_exposure, task):
    expected_cno = set(CNO_SOC_MAJOR_MAP)
    actual_cno = set(panel["cno_major"].astype(str).unique())
    if actual_cno != expected_cno:
        raise ValueError(f"Panel CNO incompleto: esperado={sorted(expected_cno)}, real={sorted(actual_cno)}")
    if panel.duplicated(["cno_major", "periodo"]).any():
        raise ValueError("El panel contiene claves cno_major-periodo duplicadas")
    if soc_exposure["soc_code"].duplicated().any():
        raise ValueError("La tabla SOC integrada contiene codigos duplicados")
    if cno_exposure["cno_major"].duplicated().any():
        raise ValueError("La tabla de exposicion CNO contiene grupos duplicados")
    if task.duplicated().any():
        raise ValueError("La tabla de tareas Anthropic conserva duplicados exactos")
    counts = panel.groupby("cno_major")["periodo"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"Serie temporal desequilibrada por CNO: {counts.to_dict()}")


def write_summary(panel, ine_age, soc_exposure, cno_exposure, task):
    def markdown_table(df):
        text_df = df.copy()
        for col in text_df.columns:
            if pd.api.types.is_float_dtype(text_df[col]):
                text_df[col] = text_df[col].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
            else:
                text_df[col] = text_df[col].map(lambda value: "" if pd.isna(value) else str(value))
        headers = list(text_df.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in text_df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    latest_period = panel["period_date"].max()
    latest = panel[panel["period_date"] == latest_period].copy()
    latest = latest.sort_values("employment_thousand_people", ascending=False)
    recent = panel[panel["year"] >= 2021].copy()

    exposure_period = (
        recent.groupby(["period_date", "periodo", "ai_exposure_category"], observed=True)
        .agg(
            employment_thousand_people=("employment_thousand_people", "sum"),
            occupations=("cno_major", "nunique"),
        )
        .reset_index()
        .sort_values(["ai_exposure_category", "period_date"])
    )
    exposure_period["employment_yoy_pct"] = (
        exposure_period.groupby("ai_exposure_category", observed=True)["employment_thousand_people"]
        .pct_change(4, fill_method=None)
        .mul(100)
    )
    latest_exposure_period = exposure_period["period_date"].max()
    by_exposure = (
        exposure_period.groupby("ai_exposure_category", observed=True)
        .agg(
            occupations=("occupations", "max"),
            avg_total_employment_thousand_people=("employment_thousand_people", "mean"),
            avg_aggregate_yoy_growth_pct=("employment_yoy_pct", "mean"),
        )
        .reset_index()
        .merge(
            exposure_period.loc[
                exposure_period["period_date"] == latest_exposure_period,
                ["ai_exposure_category", "employment_thousand_people"],
            ].rename(columns={"employment_thousand_people": "latest_total_employment_thousand_people"}),
            on="ai_exposure_category",
            how="left",
        )
    )

    age_latest = ine_age[
        (ine_age["period_date"] == latest_period)
        & (ine_age["sexo"] == "Ambos sexos")
        & (ine_age["edad"] != "Total")
        & (ine_age["cno_major"] != "total")
    ].copy()

    task_summary = task["penetration"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])

    report = []
    report.append("# Analisis exploratorio inicial de datos laborales e IA\n")
    report.append(f"Fecha de generacion: {pd.Timestamp.today().date()}\n")
    report.append(f"Ultimo periodo INE disponible en los datos descargados: {latest_period.date()}\n")
    report.append("## Cobertura de datos procesados\n")
    report.append(f"- Panel principal INE-IA: {len(panel):,} filas y {panel['cno_major'].nunique()} grandes grupos CNO.\n")
    report.append(f"- Periodo cubierto por INE: {panel['periodo'].min()} a {panel['periodo'].max()}.\n")
    report.append(f"- Ocupaciones SOC unicas con exposicion AIOE procesadas: {soc_exposure['soc_code'].nunique():,}.\n")
    report.append(f"- Asignaciones CNO-SOC agregadas: {cno_exposure['mapped_soc_occupations'].sum():,}; pueden solaparse entre grupos CNO.\n")
    report.append(f"- Tareas Anthropic Economic Index unicas procesadas: {len(task):,}.\n")

    report.append("## Mercado laboral espanol actual por grandes grupos CNO\n")
    report.append(
        markdown_table(
            latest[
                [
                    "cno_major",
                    "cno_title",
                    "employment_thousand_people",
                    "employment_yoy_pct",
                    "ai_exposure_category",
                ]
            ]
        )
    )
    report.append("\n")

    report.append("## Proxy de exposicion a IA por grandes grupos CNO\n")
    report.append(
        markdown_table(
            cno_exposure.sort_values(
                "language_modeling_percentile_cno_proxy", ascending=False
            )
        )
    )
    report.append("\n")

    report.append("## Evolucion desde 2021 por categoria de exposicion a IA\n")
    report.append(markdown_table(by_exposure))
    report.append("\n")

    report.append("## Distribucion de penetracion de tareas en Anthropic Economic Index\n")
    report.append(markdown_table(task_summary.reset_index().rename(columns={"index": "metric", "penetration": "value"})))
    report.append("\n")

    report.append("## Notas metodologicas importantes\n")
    report.append("- El cruce CNO-SOC es una aproximacion agregada para analisis exploratorio. No debe presentarse como equivalencia exacta entre clasificaciones.\n")
    report.append("- AIOE y Anthropic miden exposicion o uso de IA, no destruccion directa de empleo.\n")
    report.append("- La variable de empleo del INE se interpreta en miles de personas.\n")
    report.append("- El periodo 2021-2026 se conserva para enfocar el auge reciente de IA generativa, aunque el dataset mantiene serie desde 2011.\n")

    (DOCS_DIR / "analisis_exploratorio_inicial.md").write_text("\n".join(report), encoding="utf-8")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ine_situation = clean_ine_situation()
    ine_age = clean_ine_age()
    aioe = clean_aioe()
    anthropic_job, anthropic_task = clean_anthropic()
    soc_exposure = build_soc_exposure(aioe, anthropic_job)
    cno_exposure = build_cno_exposure(soc_exposure)
    panel = build_main_panel(ine_situation, cno_exposure)
    validate_outputs(panel, soc_exposure, cno_exposure, anthropic_task)

    safe_to_csv(ine_situation, PROCESSED_DIR / "ine_employment_by_situation_clean.csv")
    safe_to_csv(ine_age, PROCESSED_DIR / "ine_employment_by_age_clean.csv")
    safe_to_csv(aioe, PROCESSED_DIR / "aioe_occupation_exposure_clean.csv")
    safe_to_csv(anthropic_job, PROCESSED_DIR / "anthropic_job_exposure_clean.csv")
    safe_to_csv(anthropic_task, PROCESSED_DIR / "anthropic_task_penetration_clean.csv")
    safe_to_csv(soc_exposure, PROCESSED_DIR / "soc_ai_exposure_integrated.csv")
    safe_to_csv(cno_exposure, PROCESSED_DIR / "cno_ai_exposure_proxy.csv")
    safe_to_csv(panel, PROCESSED_DIR / "panel_ine_cno_ai_exposure.csv")

    write_summary(panel, ine_age, soc_exposure, cno_exposure, anthropic_task)

    print("Processed files written to data/processed")
    print("Exploratory report written to docs/analisis_exploratorio_inicial.md")


if __name__ == "__main__":
    main()
