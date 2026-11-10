from pathlib import Path
from subprocess import run
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent


def run_step(script_name: str) -> None:
    script_path = ROOT_DIR / "scripts" / script_name
    print(f"> Running {script_path.name}", flush=True)
    result = run([sys.executable, str(script_path)], cwd=str(ROOT_DIR))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    run_step("clean_and_link_data.py")
    run_step("train_forecast_model.py")
    run_step("enhanced_model_comparison.py")
    print("Full pipeline completed successfully.", flush=True)


if __name__ == "__main__":
    main()
