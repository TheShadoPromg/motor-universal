from __future__ import annotations

import argparse
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
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

DEFAULT_RECENCY_WEIGHT = 0.6
STRUCTURAL_WINDOW_CONFIG: "OrderedDict[str, Optional[int]]" = OrderedDict(
    [
        ("corto", 90),
        ("largo", 360),
    ]
)
# Ventana de referencia para score_estructural "legacy".
STRUCTURAL_MAIN_WINDOW = "largo"


@dataclass(frozen=True)
class WindowSlice:
    """Describe una ventana de tiempo sobre el histórico estructural."""

    label: str
    days: Optional[int]
    data: np.ndarray
    dates: pd.DatetimeIndex
    recency_span: int

    @property
    def length(self) -> int:
        return int(self.data.shape[0])


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


def _build_window_slices(
    history_dates: pd.DatetimeIndex,
    history_matrix: np.ndarray,
    run_ts: pd.Timestamp,
) -> Dict[str, WindowSlice]:
    """Prepara vistas del histórico para cada ventana definida en STRUCTURAL_WINDOW_CONFIG."""

    if len(history_dates) == 0:
        return {
            label: WindowSlice(
                label=label,
                days=days,
                data=history_matrix,
                dates=history_dates,
                recency_span=max(int(days or 1), 1),
            )
            for label, days in STRUCTURAL_WINDOW_CONFIG.items()
        }

    global_span = max((run_ts - history_dates.min()).days, 1)
    slices: Dict[str, WindowSlice] = {}
    for label, window_days in STRUCTURAL_WINDOW_CONFIG.items():
        if window_days is None:
            mask = np.ones(len(history_dates), dtype=bool)
        else:
            cutoff = run_ts - pd.Timedelta(days=int(window_days))
            mask = history_dates >= cutoff
        mask = np.asarray(mask)
        window_dates = history_dates[mask]
        window_data = history_matrix[mask]
        if len(window_dates) == 0:
            span = window_days if window_days is not None else global_span
        else:
            observed_span = max((run_ts - window_dates.min()).days, 1)
            span = observed_span if window_days is None else min(int(window_days), observed_span)
        slices[label] = WindowSlice(
            label=label,
            days=window_days,
            data=window_data,
            dates=window_dates,
            recency_span=max(int(span or 1), 1),
        )
    return slices


def _window_components(window: WindowSlice, num_idx: int, run_ts: pd.Timestamp) -> Tuple[float, float]:
    """Calcula frecuencia y recencia normalizada para un número dentro de una ventana."""

    if window.length == 0:
        return 0.0, 0.0
    column = window.data[:, num_idx]
    freq = float(column.sum()) / float(window.length)
    appearances = np.where(column == 1)[0]
    if appearances.size == 0:
        return freq, 0.0
    last_seen = window.dates[appearances[-1]]
    days_since_last = max((run_ts - last_seen).days, 0)
    span = max(window.recency_span, 1)
    recency = max(0.0, 1.0 - min(days_since_last, span) / span)
    return freq, recency


def _compute_structural_stats(
    dates: pd.DatetimeIndex,
    matrix: np.ndarray,
    run_idx: int,
    recency_weight: float,
) -> pd.DataFrame:
    """Combina recencia y frecuencia en cada ventana y expone scores corto/largo."""
    run_ts = dates[run_idx]
    history = matrix[:run_idx]
    history_dates = dates[:run_idx]

    numbers = [f"{i:02d}" for i in range(matrix.shape[1])]
    results: List[Dict[str, object]] = []
    history_span_days = max((run_ts - dates.min()).days, 1) if len(history_dates) > 0 else 1
    window_views = _build_window_slices(history_dates, history, run_ts)
    main_window = STRUCTURAL_MAIN_WINDOW if STRUCTURAL_MAIN_WINDOW in window_views else next(iter(window_views))

    for num_idx, num in enumerate(numbers):
        col_hist = history[:, num_idx] if len(history) > 0 else np.zeros(0, dtype=np.int8)
        appearances = np.where(col_hist == 1)[0]
        if appearances.size > 0:
            last_seen_idx = appearances[-1]
            last_seen_date = history_dates[last_seen_idx]
            days_since_last = (run_ts - last_seen_date).days
        else:
            last_seen_date = None
            days_since_last = history_span_days + 1

        window_metrics: Dict[str, Dict[str, float]] = {}
        for label, view in window_views.items():
            freq_component, recency_component = _window_components(view, num_idx, run_ts)
            score = recency_weight * recency_component + (1 - recency_weight) * freq_component
            window_metrics[label] = {
                "frecuencia": float(freq_component),
                "recencia": float(recency_component),
                "score": float(score),
                "observaciones": float(view.length),
                "ventana_dias": float(view.days or view.recency_span),
            }

        detail = {
            "ultima_fecha": last_seen_date.strftime("%Y-%m-%d") if last_seen_date is not None else None,
            "dias_desde_ultimo": days_since_last if last_seen_date is not None else None,
            "paridad": "par" if int(num) % 2 == 0 else "impar",
            "alto_bajo": "alto" if int(num) >= 50 else "bajo",
            "decena": int(num) // 10,
            "unidad": int(num) % 10,
            "ventanas": {
                label: {
                    "dias": int(window_views[label].days or window_views[label].recency_span),
                    "observaciones": int(metrics["observaciones"]),
                    "frecuencia": round(metrics["frecuencia"], 4),
                    "recencia": round(metrics["recencia"], 4),
                    "score": round(metrics["score"], 4),
                }
                for label, metrics in window_metrics.items()
            },
        }

        row: Dict[str, object] = {
            "fecha": run_ts,
            "numero": num,
            "dias_desde_ultimo": int(days_since_last),
            "detalle_estructural": json.dumps(detail, ensure_ascii=False),
        }
        for label, metrics in window_metrics.items():
            row[f"freq_ventana_{label}"] = round(metrics["frecuencia"], 6)
            row[f"recencia_{label}"] = round(metrics["recencia"], 6)
            row[f"score_estructural_{label}"] = round(metrics["score"], 6)

        row["score_estructural"] = row.get(f"score_estructural_{main_window}", 0.0)
        row["freq_ventana"] = row.get(f"freq_ventana_{main_window}", 0.0)
        results.append(row)

    df = pd.DataFrame(results)
    if f"score_estructural_{main_window}" in df.columns:
        # score_estructural se conserva como alias de la ventana principal (largo por defecto).
        df["score_estructural"] = df[f"score_estructural_{main_window}"]
    if f"freq_ventana_{main_window}" in df.columns:
        df["freq_ventana"] = df[f"freq_ventana_{main_window}"]
    return df


def _build_object_name(prefix: str, run_date: date, filename: str) -> str:
    clean = prefix.strip("/")
    parts = [clean] if clean else []
    parts.append(run_date.strftime("%Y/%m/%d"))
    parts.append(filename)
    return "/".join(parts)


def _shift_detail_payload(payload: str, delta_days: int) -> str:
    try:
        data = json.loads(payload)
    except Exception:
        return payload
    if isinstance(data, dict) and delta_days > 0:
        value = data.get("dias_desde_ultimo")
        if isinstance(value, int):
            data["dias_desde_ultimo"] = value + delta_days
    return json.dumps(data, ensure_ascii=False)


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
    panel, source, _fmt = load_or_generate_eventos()
    dates, matrix = _build_matrix(panel)
    if len(dates) == 0:
        LOGGER.error("No hay fechas disponibles en eventos_numericos.")
        return 2
    run_date = _parse_run_date(args.run_date, dates)
    target_ts = pd.Timestamp(run_date)
    if target_ts in dates:
        run_idx = _find_run_index(dates, run_date)
        base_ts = target_ts
        delta_days = 0
    else:
        base_ts = dates.max()
        run_idx = _find_run_index(dates, base_ts.date())
        delta_days = max((target_ts - base_ts).days, 0)
        LOGGER.warning(
            "No existe información estructural para %s; se reutiliza %s como base para pronosticar.",
            run_date,
            base_ts.date(),
        )

    history_length = int(run_idx)
    LOGGER.info(
        "Calculando struct_daily para %s usando %s fechas históricas (recency_weight=%.2f)...",
        run_date,
        history_length,
        args.recency_weight,
    )
    df = _compute_structural_stats(dates, matrix, run_idx, args.recency_weight)
    if delta_days > 0:
        df["fecha"] = pd.Timestamp(run_date)
        df["dias_desde_ultimo"] = (df["dias_desde_ultimo"] + delta_days).astype(int)
        df["detalle_estructural"] = df["detalle_estructural"].apply(
            lambda payload: _shift_detail_payload(payload, delta_days)
        )
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")

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
"""Capa estructural: combina frecuencia y recencia por ventanas para cada número.

- Calcula score_estructural (ventanas corto/largo) y detalle por número (última aparición, paridad, alto/bajo).
- Exporta parquet diario + último snapshot, sube a S3 opcional, valida con GE opcional y puede loguear a MLflow.
- Pensada para correr en el flujo diario del motor.
"""
