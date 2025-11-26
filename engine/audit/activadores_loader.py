"""Carga activadores Fase 3 en base de datos (Postgres o fallback SQLite).

- Lee Parquet/CSV generado por Fase 3 (`activadores_dinamicos_fase3_para_motor`).
- Valida columnas requeridas y agrega metadatos (`run_date`, `ingestion_ts`).
- Inserta en la tabla `activadores_dinamicos_fase3` (configurable) con SQLAlchemy.
- Si falla la DB destino y se permite, usa SQLite local como respaldo.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from sqlalchemy import create_engine

LOGGER = logging.getLogger("audit.activadores_loader")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "activadores" / "activadores_dinamicos_fase3_para_motor.parquet"
DEFAULT_TABLE = "activadores_dinamicos_fase3"
REQUIRED_COLS = [
    "NumeroObjetivo",
    "PosOrigen",
    "PosDestino",
    "Lag",
    "NumeroCondicionante",
    "TipoRelacion",
    "Clasificacion_Fase2_5",
    "Peso_Bruto",
    "Peso_Normalizado",
    "Regla_Condicional",
    "Stability_Score",
    "Periodos_Fuertes",
]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception as exc:
        LOGGER.warning("No se pudo cargar .env (%s): %s", path, exc)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _build_db_url_from_env() -> str | None:
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    db = os.getenv("POSTGRES_DB") or os.getenv("PGDATABASE")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    if user and password and db:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return None


def _resolve_input(path: Path) -> Path:
    if path.exists():
        return path
    candidates = []
    if path.suffix:
        stem = path.with_suffix("")
        candidates = [stem.with_suffix(".parquet"), stem.with_suffix(".csv")]
    else:
        candidates = [path.with_suffix(".parquet"), path.with_suffix(".csv")]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No se encontró archivo de activadores. Probusado: {path} y {candidates}")


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en activadores: {missing}")


def _attach_metadata(df: pd.DataFrame, run_date: str) -> pd.DataFrame:
    out = df.copy()
    out["run_date"] = run_date
    out["ingestion_ts"] = dt.datetime.utcnow().isoformat()
    return out


def load_activadores_to_db(
    input_path: Path, db_url: str, table: str, run_date: str, if_exists: str, sqlite_fallback: bool = True
) -> int:
    resolved_input = _resolve_input(input_path)
    df = _read_table(resolved_input)
    _validate_columns(df)
    df = _attach_metadata(df, run_date)
    LOGGER.info("Escribiendo %s filas en tabla %s (if_exists=%s) desde %s", len(df), table, if_exists, resolved_input)
    try:
        engine = create_engine(db_url)
        df.to_sql(table, engine, if_exists=if_exists, index=False, method="multi")
        return len(df)
    except Exception as exc:
        LOGGER.error("Error escribiendo en %s: %s", db_url, exc)
        if not sqlite_fallback:
            raise
        fallback_path = REPO_ROOT / "data" / "activadores" / "activadores_fase3.sqlite"
        fallback_url = f"sqlite:///{fallback_path}"
        LOGGER.info("Usando fallback SQLite en %s", fallback_url)
        engine = create_engine(fallback_url)
        df.to_sql(table, engine, if_exists="replace", index=False)
        LOGGER.info("Escritura completada en fallback SQLite (%s filas)", len(df))
        return len(df)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carga activadores Fase 3 a Postgres.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Ruta al Parquet/CSV de activadores para motor.")
    parser.add_argument("--db-url", default=None, help="URL de conexión a la base de datos (SQLAlchemy/DSN).")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Nombre de la tabla destino.")
    parser.add_argument("--run-date", default=None, help="Fecha lógica de la corrida (YYYY-MM-DD).")
    parser.add_argument(
        "--if-exists",
        default="append",
        choices=["append", "replace"],
        help="Estrategia de escritura en la tabla destino (append por defecto).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    configure_logging()
    _load_env_file(REPO_ROOT / ".env")
    args = parse_args(argv)
    run_date = args.run_date or dt.datetime.utcnow().strftime("%Y-%m-%d")
    db_url = (
        args.db_url
        or _build_db_url_from_env()
        or os.getenv("DATABASE_URL")
        or os.getenv("DB_URL")
        or os.getenv("PREDICTIONS_DB_URL")
    )
    if not db_url:
        LOGGER.error(
            "No se encontró DB URL (ni --db-url ni env DATABASE_URL/DB_URL/PREDICTIONS_DB_URL ni credenciales PG* en .env)."
        )
        return 2
    try:
        load_activadores_to_db(
            input_path=Path(args.input),
            db_url=db_url,
            table=args.table,
            run_date=run_date,
            if_exists=args.if_exists,
            sqlite_fallback=True,
        )
    except Exception as exc:
        LOGGER.error("Carga de activadores falló: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
