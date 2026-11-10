from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import csv
import datetime as dt


RAW_DIR = Path("data/raw")
LOG_PATH = Path("docs/download_log.csv")


SOURCES = [
    {
        "name": "ine_epa_ocupados_situacion_sexo_ocupacion",
        "url": "https://www.ine.es/jaxiT3/files/t/csv_bdsc/65136.csv",
        "filename": "ine_epa_ocupados_situacion_sexo_ocupacion_65136.csv",
        "source": "INE EPA",
        "purpose": "Empleo historico por situacion profesional, sexo y ocupacion.",
    },
    {
        "name": "ine_epa_ocupados_edad_sexo_ocupacion",
        "url": "https://www.ine.es/jaxiT3/files/t/csv_bdsc/65135.csv",
        "filename": "ine_epa_ocupados_edad_sexo_ocupacion_65135.csv",
        "source": "INE EPA",
        "purpose": "Empleo historico por grupo de edad, sexo y ocupacion.",
    },
    {
        "name": "aioe_data_appendix",
        "url": "https://github.com/AIOE-Data/AIOE/raw/main/AIOE_DataAppendix.xlsx",
        "filename": "aioe_data_appendix.xlsx",
        "source": "Felten, Raj y Seamans - AIOE",
        "purpose": "Puntuaciones academicas de exposicion ocupacional a IA.",
    },
    {
        "name": "aioe_language_modeling",
        "url": "https://github.com/AIOE-Data/AIOE/raw/main/Language%20Modeling%20AIOE%20and%20AIIE.xlsx",
        "filename": "aioe_language_modeling.xlsx",
        "source": "AIOE - Generative AI language modeling",
        "purpose": "Exposicion ocupacional especifica a modelos de lenguaje.",
    },
    {
        "name": "anthropic_job_exposure",
        "url": "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/labor_market_impacts/job_exposure.csv",
        "filename": "anthropic_job_exposure.csv",
        "source": "Anthropic Economic Index",
        "purpose": "Exposicion observada de trabajos a IA en datos recientes.",
    },
    {
        "name": "anthropic_task_penetration",
        "url": "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/labor_market_impacts/task_penetration.csv",
        "filename": "anthropic_task_penetration.csv",
        "source": "Anthropic Economic Index",
        "purpose": "Penetracion de tareas realizadas con IA.",
    },
    {
        "name": "bls_skills_detailed",
        "url": "https://www.bls.gov/emp/skills/skills.xlsx",
        "filename": "bls_skills.xlsx",
        "source": "BLS Skills Data",
        "purpose": "Habilidades por ocupacion y proyecciones de empleo.",
    },
]


def download(url: str, target: Path) -> int:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 TFM data research"})
    with urlopen(request, timeout=120) as response:
        content = response.read()
    target.write_bytes(content)
    return len(content)


def append_log(rows):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "name",
                "source",
                "url",
                "filename",
                "status",
                "bytes",
                "purpose",
                "error",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logs = []
    for source in SOURCES:
        target = RAW_DIR / source["filename"]
        status = "ok"
        size = 0
        error = ""
        try:
            size = download(source["url"], target)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            status = "error"
            error = repr(exc)
        logs.append(
            {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "name": source["name"],
                "source": source["source"],
                "url": source["url"],
                "filename": str(target),
                "status": status,
                "bytes": size,
                "purpose": source["purpose"],
                "error": error,
            }
        )
        print(f"{status.upper():5} {source['name']} {size} bytes")
    append_log(logs)


if __name__ == "__main__":
    main()
