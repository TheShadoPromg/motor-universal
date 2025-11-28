"""Agregado diario estructural basado en activadores core+periódico vigentes.

- Lee eventos normalizados, calcula distribuciones diarias con activadores Fase 3 (core+periódico) vía softmax.
- Exporta score_estructural por fecha/número en Parquet (snapshot diario + último).
- Opcional: validación GE, subida a bucket, tracking MLflow.

NOTA: Este reemplazo alinea struct_daily con el modelo estructural vigente (phase4 core+periódico),
      descartando el score legacy de recencia/frecuencia interna.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from engine.backtesting.phase4 import (
    _build_draws_index,
    _compute_prob_struct,
    _read_activadores,
    parse_activadores,
)
from engine.derived_dynamic.helpers.storage import upload_artifact
from engine.derived_dynamic.transform import load_or_generate_eventos

LOGGER = logging.getLogger("struct_daily")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DERIVED = REPO_ROOT / "data" / "derived"
DERIVED_LATEST = DATA_DERIVED / "struct_daily.parquet"
DEFAULT_BUCKET = os.getenv("STRUCT_DAILY_BUCKET", "motor-struct-daily")
DEFAULT_PREFIX = os.getenv("STRUCT_DAILY_PREFIX", "struct-daily")

DEFAULT_ACTIVADORES = REPO_ROOT / "data" / "activadores" / "activadores_dinamicos_fase3_para_motor.parquet"
DEFAULT_BETA = 1.0
DEFAULT_LAMBDA = 0.85  # recomendado según tuning actual


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _parse_run_date(raw: Optional[str], available_dates: List[date]) -> date:
    if raw:
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"run-date inválida '{raw}' (YYYY-MM-DD).") from exc
    return max(available_dates) if available_dates else datetime.utcnow().date()


def _panel_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte panel con columnas e_pos1/e_pos2/e_pos3 a formato long esperado por _build_draws_index."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
    melted_rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        for pos in (1, 2, 3):
            col = f"e_pos{pos}"
            if col not in df.columns:
                continue
            try:
                flag = int(row[col])
            except Exception:
                flag = 0
            if flag == 1:
                melted_rows.append(
                    {
                        "fecha": row["fecha"],
                        "posicion": pos,
                        "numero": int(str(row["numero"]).zfill(2)),
                    }
                )
    return pd.DataFrame(melted_rows)


def _compute_struct_scores(
    draw_index: Dict[date, List[tuple[int, int]]],
    activadores_df: pd.DataFrame,
    eval_dates: List[date],
    beta: float,
    mix_lambda: float,
) -> pd.DataFrame:
    activadores = parse_activadores(activadores_df)
    probs = _compute_prob_struct(activadores, draw_index, eval_dates, beta=beta, mix_lambda=mix_lambda)
    records: List[Dict[str, object]] = []
    for d, dist in probs.items():
        for num, p in enumerate(dist):
            records.append(
                {
                    "fecha": pd.to_datetime(d),
                    "numero": f"{num:02d}",
                    "score_estructural": float(p),
                    "detalle_estructural": "[]",
                }
            )
    df = pd.DataFrame(records)
    df = df.sort_values(["fecha", "numero"]).reset_index(drop=True)
    # garantía de grilla completa por fecha
    all_dates = df["fecha"].drop_duplicates().sort_values()
    base_nums = pd.DataFrame({"numero": [f"{i:02d}" for i in range(100)]})
    full_rows: List[pd.DataFrame] = []
    for ts in all_dates:
        merged = base_nums.merge(df[df["fecha"] == ts], on="numero", how="left")
        merged["fecha"] = ts
        merged["score_estructural"] = merged["score_estructural"].fillna(0.0)
        merged["detalle_estructural"] = merged["detalle_estructural"].fillna("[]")
        full_rows.append(merged)
    return pd.concat(full_rows, ignore_index=True).sort_values(["fecha", "numero"]).reset_index(drop=True)


def _append_or_replace(path_parquet: Path, path_csv: Path, df: pd.DataFrame) -> None:
    path_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path_parquet, index=False)
    df.to_csv(path_csv, index=False)


def maybe_run_gx(skip: bool, path: Path) -> str:
    if skip:
        LOGGER.info("Validación GE omitida.")
        return "skipped"
    try:
        import great_expectations as gx  # type: ignore

        ctx = gx.get_context()
        checkpoint = "struct_daily"
        result = ctx.run_checkpoint(checkpoint_name=checkpoint)
        status = "passed" if result.get("success") else "failed"
        LOGGER.info("Checkpoint '%s' finalizado con estado %s.", checkpoint, status)
        return status
    except FileNotFoundError as exc:
        LOGGER.warning("Checkpoint struct_daily no encontrado: %s", exc)
        return "missing"
    except ImportError:
        LOGGER.warning("Great Expectations no instalado; validación omitida.")
        return "missing"
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Validación GE falló: %s", exc)
        return "failed"


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
        LOGGER.warning("MLflow no disponible; omitiendo tracking.")
        return
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("struct_daily")
        with mlflow.start_run(run_name="struct_daily"):
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="outputs")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("MLflow falló: %s", exc)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcula score_estructural diario usando activadores core+periódico.")
    parser.add_argument("--run-date", default=None, help="Fecha a procesar (YYYY-MM-DD). Si se omite, usa la última disponible.")
    parser.add_argument("--all-dates", action="store_true", help="Si se pasa, calcula y exporta todas las fechas disponibles.")
    parser.add_argument("--activadores-path", default=str(DEFAULT_ACTIVADORES), help="Parquet/CSV de activadores estructurales.")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Beta del softmax estructural.")
    parser.add_argument("--lambda-mix", type=float, default=DEFAULT_LAMBDA, help="Mezcla con uniforme (1 = sin mezcla).")
    parser.add_argument("--s3-bucket", default=DEFAULT_BUCKET, help="Bucket S3/GCS opcional.")
    parser.add_argument("--s3-prefix", default=DEFAULT_PREFIX, help="Prefijo dentro del bucket.")
    parser.add_argument("--mlflow-uri", default=None, help="URI de MLflow.")
    parser.add_argument("--skip-validation", action="store_true", help="Omitir Great Expectations.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    events_panel, source_path, _fmt = load_or_generate_eventos()
    events_long = _panel_to_long(events_panel)
    draw_index = _build_draws_index(events_long)
    all_dates = sorted(draw_index.keys())
    if not all_dates:
        LOGGER.error("No hay fechas disponibles en eventos_numericos.")
        return 2
    run_date = _parse_run_date(args.run_date, all_dates)
    eval_dates = all_dates if args.all_dates else [d for d in all_dates if d <= run_date]
    if not eval_dates:
        LOGGER.error("No se encontraron fechas para procesar struct_daily.")
        return 2
    LOGGER.info(
        "Calculando struct_daily%s para %s fechas históricas (beta=%.2f, lambda=%.2f)...",
        " (batch)" if args.all_dates else "",
        len(eval_dates),
        args.beta,
        args.lambda_mix,
    )

    acts_df = _read_activadores(None, Path(args.activadores_path))
    df = _compute_struct_scores(draw_index, acts_df, eval_dates, beta=args.beta, mix_lambda=args.lambda_mix)
    df["fecha"] = df["fecha"].dt.strftime("%Y-%m-%d")

    snapshot_path = DATA_DERIVED / ("struct_daily_all.parquet" if args.all_dates else f"struct_daily_{run_date}.parquet")
    DATA_DERIVED.mkdir(parents=True, exist_ok=True)
    _append_or_replace(snapshot_path, snapshot_path.with_suffix(".csv"), df)
    df.to_parquet(DERIVED_LATEST, index=False)

    if args.s3_bucket:
        object_date = max(eval_dates) if args.all_dates else run_date
        filename = "struct_daily_all.parquet" if args.all_dates else "struct_daily.parquet"
        object_name = "/".join([args.s3_prefix.strip("/"), object_date.strftime("%Y/%m/%d"), filename]).strip("/")
        upload_artifact(snapshot_path, args.s3_bucket, object_name=object_name)
    else:
        LOGGER.info("Bucket no configurado; omitiendo carga.")

    gx_status = maybe_run_gx(args.skip_validation, snapshot_path)

    maybe_log_mlflow(
        args.mlflow_uri,
        params={
            "run_date": run_date.isoformat(),
            "beta": str(args.beta),
            "lambda_mix": str(args.lambda_mix),
            "events_path": str(source_path),
            "activadores_path": str(args.activadores_path),
        },
        metrics={
            "score_mean": float(df["score_estructural"].mean()),
            "score_std": float(df["score_estructural"].std()),
        },
        artifact=snapshot_path,
    )

    if gx_status == "failed":
        LOGGER.error("Great Expectations reportó fallas en struct_daily.")
        return 2
    LOGGER.info("struct_daily completado. Output: %s", snapshot_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
