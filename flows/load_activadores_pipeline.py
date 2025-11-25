from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Optional

# Desactivar API Prefect remota para ejecución local/ephemeral
os.environ["PREFECT_API_URL"] = ""
os.environ.pop("PREFECT_API_KEY", None)
os.environ.pop("PREFECT_API_DATABASE_CONNECTION_URL", None)

try:
    from prefect import flow, task
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Prefect no está instalado en este entorno. Instala 'prefect>=2' para ejecutar el flujo de carga de activadores."
    ) from exc

from engine.audit import activadores_loader


def _run_main(module_main, args):
    code = module_main(args)
    if code not in (0, None):
        raise RuntimeError(f"El módulo {module_main.__module__} finalizó con código {code}.")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key in {"PREFECT_API_URL", "PREFECT_API_KEY", "PREFECT_API_DATABASE_CONNECTION_URL"}:
                continue
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # No es crítico; se seguirá con las variables ya presentes
        pass


@task(persist_result=True)
def load_activadores(run_date: str) -> str:
    _load_env_file(Path(__file__).resolve().parents[1] / ".env")
    args = []
    activ_dir = os.getenv("ACTIVADORES_DIR")
    db_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
    table = os.getenv("ACTIVADORES_TABLE") or "activadores_dinamicos_fase3"

    if activ_dir:
        parquet_path = os.path.join(activ_dir, "activadores_dinamicos_fase3_para_motor.parquet")
        csv_path = os.path.join(activ_dir, "activadores_dinamicos_fase3_para_motor.csv")
        args += ["--input", parquet_path if Path(parquet_path).exists() else csv_path]
    if db_url:
        args += ["--db-url", db_url]
    args += ["--table", table, "--run-date", run_date, "--if-exists", "replace"]

    _run_main(activadores_loader.main, args)
    return f"Activadores cargados en {table} para run_date={run_date}"


@flow(name="load_activadores_pipeline")
def load_activadores_pipeline(run_date: Optional[str] = None):
    # For ejecuciones locales/ephemerales, evitar apuntar a un servidor Prefect no disponible
    if os.getenv("PREFECT_API_URL"):
        os.environ.pop("PREFECT_API_URL", None)
        os.environ.pop("PREFECT_API_KEY", None)
    run_dt = run_date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    load_activadores(run_dt)


if __name__ == "__main__":
    # Ejecución local sin servidor Prefect: llama directamente al loader
    _load_env_file(Path(__file__).resolve().parents[1] / ".env")
    activ_dir = os.getenv("ACTIVADORES_DIR")
    input_path = (
        Path(activ_dir) / "activadores_dinamicos_fase3_para_motor.parquet"
        if activ_dir
        else Path("data/activadores/activadores_dinamicos_fase3_para_motor.parquet")
    )
    if not input_path.exists():
        alt_csv = input_path.with_suffix(".csv")
        input_path = alt_csv if alt_csv.exists() else input_path
    db_url = (
        activadores_loader._build_db_url_from_env()
        or os.getenv("DATABASE_URL")
        or os.getenv("DB_URL")
        or os.getenv("PREDICTIONS_DB_URL")
    )
    args = ["--input", str(input_path)]
    if db_url:
        args += ["--db-url", db_url]
    args += ["--table", os.getenv("ACTIVADORES_TABLE") or "activadores_dinamicos_fase3"]
    activadores_loader.main(args)
