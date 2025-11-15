from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from engine.derived_dynamic.helpers.storage import upload_artifact
from engine.derived_dynamic.transform import load_or_generate_eventos

LOGGER = logging.getLogger("struct_daily")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DERIVED = REPO_ROOT / "data" / "derived"
DERIVED_LATEST = DATA_DERIVED / "struct_daily.parquet"
DEFAULT_BUCKET = os.getenv("STRUCT_DAILY_BUCKET", "motor-struct-daily")
DEFAULT_PREFIX = os.getenv("STRUCT_DAILY_PREFIX", "struct-daily")

DEFAULT_WINDOW = 120
DEFAULT_RECENCY_WEIGHT = 0.6


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _parse_run_date(raw: Optional[str], available: pd.DatetimeIndex) -> date:
    if raw:
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"run-date inválida '{raw}'.") from exc
    if len(available) > 0:
        return available.max().date()
    return datetime.utcnow().date()


def _build_matrix(panel: pd.DataFrame) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    panel = panel.copy()
    panel["fecha"] = pd.to_datetime(panel["fecha"])
    panel["numero"] = panel["numero"].astype(str).str.zfill(2)
    panel["aparece"] = (
        panel[["e_pos1", "e_pos2", "e_pos3"]].sum(axis=1) > 0
    ).astype(int)
    pivot = panel.pivot(index="fecha", columns="numero", values="aparece").fillna(0).sort_index()
    pivot = pivot[[f"{i:02d}" for i in range(100)]]
    return pivot.index, pivot.to_numpy(dtype=np.int8)


def _find_run_index(dates: pd.DatetimeIndex, run_date: date) -> int:
    ts = pd.Timestamp(run_date)
    if ts not in dates:
        raise ValueError(f"No existe información para la fecha {run_date}.")
    return dates.get_loc(ts)


def _compute_structural_stats(
    dates: pd.DatetimeIndex,
    matrix: np.ndarray,
    run_idx: int,
    window_days: int,
    recency_weight: float,
) -> pd.DataFrame:
    run_ts = dates[run_idx]
    history = matrix[:run_idx]
    history_dates = dates[:run_idx]

    numbers = [f"{i:02d}" for i in range(matrix.shape[1])]
    window_start = run_ts - pd.Timedelta(days=window_days)
    window_mask = (history_dates >= window_start) & (history_dates < run_ts)
    window_length = int(window_mask.sum())
    window_data = history[window_mask]

    results: List[Dict[str, object]] = []

    for num_idx, num in enumerate(numbers):
        col_hist = history[:, num_idx]
        appearances = np.where(col_hist == 1)[0]
        if appearances.size > 0:
            last_seen_idx = appearances[-1]
            last_seen_date = history_dates[last_seen_idx]
            days_since_last = (run_ts - last_seen_date).days
        else:
            last_seen_date = None
            days_since_last = window_days + 1

        if window_length > 0:
            freq_window = float(window_data[:, num_idx].sum()) / window_length
        else:
            freq_window = 0.0

        recency_component = 1.0 - min(days_since_last, window_days) / window_days
        score = recency_weight * recency_component + (1 - recency_weight) * freq_window

        detail = {
            "ultima_fecha": last_seen_date.strftime("%Y-%m-%d") if last_seen_date is not None else None,
            "dias_desde_ultimo": days_since_last if last_seen_date is not None else None,
            "frecuencia_ventana": round(freq_window, 4),
            "ventana_dias": window_length,
            "paridad": "par" if int(num) % 2 == 0 else "impar",
            "alto_bajo": "alto" if int(num) >= 50 else "bajo",
            "decena": int(num) // 10,
            "unidad": int(num) % 10,
        }

        results.append(
            {
                "fecha": run_ts,
                "numero": num,
                "score_estructural": round(float(score), 6),
                "dias_desde_ultimo": int(days_since_last),
                "freq_ventana": round(freq_window, 6),
                "detalle_estructural": json.dumps(detail, ensure_ascii=False),
            }
        )
    return pd.DataFrame(results)


def _build_object_name(prefix: str, run_date: date, filename: str) -> str:
    clean = prefix.strip("/")
    parts = [clean] if clean else []
    parts.append(run_date.strftime("%Y/%m/%d"))
    parts.append(filename)
    return "/".join(parts)


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
    parser = argparse.ArgumentParser(
        description="Calcula score estructural por número usando recencia y frecuencia.",
    )
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW, help="Ventana histórica en días.")
    parser.add_argument(
        "--recency-weight",
        type=float,
        default=DEFAULT_RECENCY_WEIGHT,
        help="Peso de la componente de recencia (0-1).",
    )
    parser.add_argument("--run-date", default=None, help="Fecha a procesar (YYYY-MM-DD).")
    parser.add_argument("--s3-bucket", default=DEFAULT_BUCKET, help="Bucket S3 para snapshots.")
    parser.add_argument("--s3-prefix", default=DEFAULT_PREFIX, help="Prefijo dentro del bucket.")
    parser.add_argument("--mlflow-uri", default=None, help="URI de MLflow.")
    parser.add_argument("--skip-validation", action="store_true", help="Omitir Great Expectations.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    if not (0 <= args.recency_weight <= 1):
        raise ValueError("recency-weight debe estar entre 0 y 1.")
    if args.window_days <= 0:
        raise ValueError("window-days debe ser > 0.")

    panel, source, _fmt = load_or_generate_eventos()
    dates, matrix = _build_matrix(panel)
    if len(dates) == 0:
        LOGGER.error("No hay fechas disponibles en eventos_numericos.")
        return 2
    run_date = _parse_run_date(args.run_date, dates)
    run_idx = _find_run_index(dates, run_date)

    LOGGER.info(
        "Calculando struct_daily para %s (ventana=%s días, recency_weight=%.2f)...",
        run_date,
        args.window_days,
        args.recency_weight,
    )
    df = _compute_structural_stats(dates, matrix, run_idx, args.window_days, args.recency_weight)
    df["fecha"] = df["fecha"].dt.strftime("%Y-%m-%d")

    snapshot_path = DATA_DERIVED / f"struct_daily_{run_date}.parquet"
    DATA_DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(snapshot_path, index=False)
    df.to_parquet(DERIVED_LATEST, index=False)

    if args.s3_bucket:
        object_name = _build_object_name(args.s3_prefix, run_date, "struct_daily.parquet")
        upload_artifact(snapshot_path, args.s3_bucket, object_name=object_name)
    else:
        LOGGER.info("Bucket S3 no configurado; omitiendo carga.")

    gx_status = maybe_run_gx(args.skip_validation, snapshot_path)

    maybe_log_mlflow(
        args.mlflow_uri,
        params={
            "run_date": run_date.isoformat(),
            "window_days": str(args.window_days),
            "recency_weight": str(args.recency_weight),
            "events_path": str(source),
        },
        metrics={
            "score_promedio": float(df["score_estructural"].mean()),
            "dias_desde_ultimo_prom": float(df["dias_desde_ultimo"].mean()),
        },
        artifact=snapshot_path,
    )

    if gx_status == "failed":
        LOGGER.error("Great Expectations reportó fallas en struct_daily.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
