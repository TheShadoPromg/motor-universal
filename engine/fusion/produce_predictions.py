from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from engine.derived_dynamic.helpers.storage import upload_artifact

LOGGER = logging.getLogger("produce_predictions")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DERIVED = REPO_ROOT / "data" / "derived"
LATEST_OUTPUT = DATA_DERIVED / "predictions_daily.parquet"
DEFAULT_BUCKET = os.getenv("PREDICTIONS_BUCKET", "motor-predictions")
DEFAULT_PREFIX = os.getenv("PREDICTIONS_PREFIX", "")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _parse_run_date(raw: Optional[str], fusion_df: pd.DataFrame) -> date:
    if raw:
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"run-date inválida '{raw}'.") from exc
    if not fusion_df.empty:
        return pd.to_datetime(fusion_df["fecha"]).max().date()
    return datetime.utcnow().date()


def _softmax(scores: pd.Series, temperature: float) -> pd.Series:
    if temperature <= 0:
        raise ValueError("temperature debe ser > 0.")
    scaled = scores / temperature
    scaled -= scaled.max()
    exp = np.exp(scaled)
    total = exp.sum()
    if not np.isfinite(total) or total == 0:
        LOGGER.warning("Softmax degenerada; se usará distribución uniforme.")
        return pd.Series(np.full(len(scores), 1 / len(scores)), index=scores.index)
    return exp / total


def _build_object_name(prefix: str, run_date: date, filename: str) -> str:
    clean = prefix.strip("/")
    parts = [clean] if clean else []
    parts.append(run_date.strftime("%Y/%m/%d"))
    parts.append(filename)
    return "/".join(parts)


def maybe_log_mlflow(
    mlflow_uri: Optional[str],
    params: Dict[str, str],
    metrics: Dict[str, float],
    artifact: Path,
) -> None:
    if not mlflow_uri:
        return
    try:
        import mlflow  # type: ignore
    except ImportError:
        LOGGER.warning("MLflow no disponible; omitiendo seguimiento.")
        return
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("predictions_daily")
        with mlflow.start_run(run_name="predictions_daily"):
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="outputs")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Error registrando en MLflow: %s", exc)


def _write_postgres(df: pd.DataFrame, db_url: str, table: str) -> None:
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        LOGGER.warning("sqlalchemy no está instalado; no se puede escribir en Postgres.")
        return

    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table} WHERE fecha = :fecha"), {"fecha": df["fecha"].iloc[0]})
        df.to_sql(table, conn, if_exists="append", index=False)
    LOGGER.info("Predicciones insertadas en %s (tabla=%s).", db_url, table)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibra el output de la fusión y genera predicciones diarias (parquet + Postgres opcional).",
    )
    parser.add_argument(
        "--fusion-path",
        default=str(DATA_DERIVED / "jugadas_fusionadas_3capas.parquet"),
        help="Parquet con el resultado de fusión.",
    )
    parser.add_argument("--run-date", default=None, help="Fecha objetivo (YYYY-MM-DD).")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperatura para recalibrar score_total antes de normalizar.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Cadena de conexión SQLAlchemy (ej. postgresql+psycopg2://user:pass@host/db).",
    )
    parser.add_argument(
        "--db-table",
        default="predictions_daily",
        help="Nombre de la tabla destino en Postgres.",
    )
    parser.add_argument("--mlflow-uri", default=None, help="URI de MLflow para registrar runs.")
    parser.add_argument("--s3-bucket", default=DEFAULT_BUCKET, help="Bucket S3/MinIO para snapshots.")
    parser.add_argument("--s3-prefix", default=DEFAULT_PREFIX, help="Prefijo dentro del bucket.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    fusion_path = Path(args.fusion_path).expanduser().resolve()
    if not fusion_path.exists():
        raise FileNotFoundError(f"No se encontró el parquet de fusión: {fusion_path}")
    fusion = pd.read_parquet(fusion_path)
    if fusion.empty:
        LOGGER.error("El dataset de fusión está vacío.")
        return 2

    fusion["fecha"] = pd.to_datetime(fusion["fecha"], errors="coerce")
    run_date = _parse_run_date(args.run_date, fusion["fecha"].dropna())
    target_ts = pd.Timestamp(run_date)
    mask = fusion["fecha"] == target_ts
    daily = fusion.loc[mask].copy()
    if daily.empty:
        last_ts = fusion["fecha"].dropna().max()
        if pd.isna(last_ts):
            raise ValueError(f"No se encontraron registros para la fecha {run_date} en {fusion_path}.")
        LOGGER.warning(
            "No se encontraron registros de fusión para %s; se reutiliza %s como base para pronosticar.",
            run_date,
            last_ts.date(),
        )
        daily = fusion.loc[fusion["fecha"] == last_ts].copy()
        daily["fecha"] = target_ts

    daily["score_total"] = daily["score_total"].astype(float)
    daily["prob_raw"] = daily["prob"].astype(float)
    daily["prob"] = _softmax(daily["score_total"], args.temperature)
    daily = daily.sort_values("prob", ascending=False).reset_index(drop=True)
    daily["rank"] = np.arange(1, len(daily) + 1)

    snapshot_path = DATA_DERIVED / f"predictions_{run_date}.parquet"
    DATA_DERIVED.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(snapshot_path, index=False)
    daily.to_parquet(LATEST_OUTPUT, index=False)

    if args.s3_bucket:
        object_name = _build_object_name(args.s3_prefix, run_date, "predictions.parquet")
        upload_artifact(snapshot_path, args.s3_bucket, object_name=object_name)
    else:
        LOGGER.info("Bucket S3 no configurado; se omite carga.")

    if args.db_url:
        _write_postgres(daily, args.db_url, args.db_table)

    entropy = float(-(daily["prob"] * np.log(daily["prob"] + 1e-12)).sum())
    LOGGER.info(
        "Predicciones %s -> top1=%s prob=%.4f | entropía=%.4f",
        run_date,
        daily.iloc[0]["numero"],
        float(daily.iloc[0]["prob"]),
        entropy,
    )

    maybe_log_mlflow(
        args.mlflow_uri,
        params={
            "run_date": run_date.isoformat(),
            "temperature": str(args.temperature),
            "fusion_path": str(fusion_path),
            "db_table": args.db_table if args.db_url else "",
        },
        metrics={
            "top1_prob": float(daily.iloc[0]["prob"]),
            "entropy": entropy,
        },
        artifact=snapshot_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
