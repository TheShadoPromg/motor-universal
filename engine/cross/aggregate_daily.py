"""Capa cross: agrega activaciones origen→destino por lags y genera score_cruzado diario.

- Lee eventos normalizados y cuenta oportunidades/activaciones entre posiciones.
- Genera parquet diario + último snapshot, sube a S3 opcional, valida con GE opcional.
- Opcionalmente loguea a MLflow. Pensado para correr en pipeline diario (Prefect).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from engine.derived_dynamic.helpers.storage import upload_artifact
from engine.derived_dynamic.transform import load_or_generate_eventos

LOGGER = logging.getLogger("cross_daily")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DERIVED = REPO_ROOT / "data" / "derived"
DERIVED_LATEST = DATA_DERIVED / "cross_daily.parquet"
DEFAULT_BUCKET = os.getenv("CROSS_DAILY_BUCKET", "motor-cross-daily")
DEFAULT_PREFIX = os.getenv("CROSS_DAILY_PREFIX", "cross-daily")

DEFAULT_LAGS = [1, 2, 3, 7, 14, 30]
DEFAULT_DETAIL_TOP = 5


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _parse_run_date(raw: Optional[str], available_dates: Sequence[pd.Timestamp]) -> date:
    if raw:
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"run-date inválida '{raw}'. Formato esperado YYYY-MM-DD.") from exc
    if available_dates:
        return max(available_dates).date()
    return datetime.utcnow().date()


def _extract_positions(panel: pd.DataFrame) -> Tuple[List[pd.Timestamp], List[Dict[int, Optional[str]]]]:
    dates = pd.to_datetime(panel["fecha"]).sort_values().unique()
    panel = panel.set_index("fecha")
    per_date: List[Dict[int, Optional[str]]] = []
    for ts in dates:
        subset = panel.loc[ts]
        if isinstance(subset, pd.Series):
            subset = subset.to_frame().T
        data = {}
        for pos in (1, 2, 3):
            mask = subset[f"e_pos{pos}"] == 1
            if mask.any():
                num = subset.loc[mask, "numero"].iloc[0]
                data[pos] = num
            else:
                data[pos] = None
        per_date.append(data)
    return list(dates), per_date


def _build_detail(
    combos: Dict[Tuple[int, int, int], Tuple[int, int]],
    detail_top: int,
) -> str:
    if not combos:
        return "[]"
    rows = []
    for (lag, origin, dest), (op, act) in combos.items():
        ratio = (act / op) if op > 0 else 0.0
        rows.append(
            {
                "lag": lag,
                "origin": origin,
                "dest": dest,
                "op": op,
                "act": act,
                "ratio": round(ratio, 4),
            }
        )
    rows.sort(key=lambda item: (item["ratio"], item["act"], item["op"]), reverse=True)
    return json.dumps(rows[: detail_top], ensure_ascii=False)


def _calculate_cross_scores(
    dates: List[pd.Timestamp],
    per_date_positions: List[Dict[int, Optional[str]]],
    lags: Sequence[int],
    detail_top: int,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    numbers = [f"{i:02d}" for i in range(100)]

    for idx, current_date in enumerate(dates):
        combos_per_number: Dict[str, Dict[Tuple[int, int, int], Tuple[int, int]]] = {}
        totals: Dict[str, Tuple[int, int]] = {}

        for lag in lags:
            prev_idx = idx - lag
            if prev_idx < 0:
                continue
            prev_pos = per_date_positions[prev_idx]
            current_pos = per_date_positions[idx]
            for origin in (1, 2, 3):
                num = prev_pos.get(origin)
                if num is None:
                    continue
                combos = combos_per_number.setdefault(num, {})
                for dest in (1, 2, 3):
                    entry = combos.get((lag, origin, dest), (0, 0))
                    op, act = entry
                    op += 1
                    if current_pos.get(dest) == num:
                        act += 1
                    combos[(lag, origin, dest)] = (op, act)
                    tot = totals.get(num, (0, 0))
                    totals[num] = (tot[0] + 1, tot[1] + (1 if current_pos.get(dest) == num else 0))

        for num in numbers:
            combos = combos_per_number.get(num, {})
            op_total, act_total = totals.get(num, (0, 0))
            score = act_total / op_total if op_total > 0 else 0.0
            records.append(
                {
                    "fecha": current_date,
                    "numero": num,
                    "score_cruzado": float(score),
                    "oportunidades_total": int(op_total),
                    "activaciones_total": int(act_total),
                    "combinaciones_activas": len([c for c in combos.values() if c[0] > 0]),
                    "detalle_cruzado": _build_detail(combos, detail_top),
                }
            )
    return pd.DataFrame(records)


def _build_object_name(prefix: str, run_date: date, filename: str) -> str:
    clean = prefix.strip("/")
    parts = [clean] if clean else []
    parts.append(run_date.strftime("%Y/%m/%d"))
    parts.append(filename)
    return "/".join(parts)


def maybe_run_gx(skip: bool, path: Path) -> str:
    if skip:
        LOGGER.info("Validación GE omitida por bandera.")
        return "skipped"
    try:
        import great_expectations as gx  # type: ignore

        ctx = gx.get_context()
        checkpoint = "cross_daily"
        result = ctx.run_checkpoint(checkpoint_name=checkpoint)
        status = "passed" if result.get("success") else "failed"
        LOGGER.info("Checkpoint '%s' finalizado con estado %s.", checkpoint, status)
        return status
    except FileNotFoundError as exc:
        LOGGER.warning("Checkpoint cross_daily no encontrado: %s", exc)
        return "missing"
    except ImportError:
        LOGGER.warning("Great Expectations no está instalado; se omite validación.")
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
        mlflow.set_experiment("cross_daily")
        with mlflow.start_run(run_name="cross_daily"):
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
        description="Agrega activaciones cruzadas (origen→destino) y genera score_cruzado diario.",
    )
    parser.add_argument("--lags", default=",".join(map(str, DEFAULT_LAGS)), help="Lags a evaluar (ej. '1,2,3').")
    parser.add_argument("--run-date", default=None, help="Fecha a procesar (YYYY-MM-DD).")
    parser.add_argument("--all-dates", action="store_true", help="Si se pasa, exporta todas las fechas disponibles.")
    parser.add_argument(
        "--detail-top",
        type=int,
        default=DEFAULT_DETAIL_TOP,
        help="Cantidad máxima de combinaciones en detalle_cruzado.",
    )
    parser.add_argument("--s3-bucket", default=DEFAULT_BUCKET, help="Bucket S3 para snapshots.")
    parser.add_argument("--s3-prefix", default=DEFAULT_PREFIX, help="Prefijo dentro del bucket.")
    parser.add_argument("--mlflow-uri", default=None, help="URI de MLflow para tracking.")
    parser.add_argument("--skip-validation", action="store_true", help="Omitir Great Expectations.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    if not lags:
        raise ValueError("Debe proporcionar al menos un lag.")

    panel, source, _fmt = load_or_generate_eventos()
    panel["fecha"] = pd.to_datetime(panel["fecha"])
    dates, per_date_positions = _extract_positions(panel)
    if not dates:
        LOGGER.error("El dataset de eventos está vacío.")
        return 2
    latest_date = max(dates).date()
    if args.all_dates:
        run_date = latest_date
        trimmed_dates = dates
        trimmed_positions = per_date_positions
        forecast = False
        base_ts = pd.Timestamp(latest_date)
        LOGGER.info(
            "Calculando cross_daily para todas las fechas disponibles (%s fechas) con lags=%s ...",
            len(trimmed_dates),
            lags,
        )
    else:
        run_date = _parse_run_date(args.run_date, dates)
        target_ts = pd.Timestamp(run_date)
        if target_ts in dates:
            idx = dates.index(target_ts)
            base_ts = target_ts
            forecast = False
        else:
            base_ts = dates[-1]
            idx = len(dates) - 1
            forecast = True
            LOGGER.warning(
                "No existe información para la fecha %s en eventos_numericos; se reutiliza %s como base para pronosticar.",
                run_date,
                base_ts.date(),
            )

        trimmed_dates = dates[: idx + 1]
        trimmed_positions = per_date_positions[: idx + 1]
        LOGGER.info("Calculando cross_daily para %s con lags=%s ...", run_date, lags)

    df = _calculate_cross_scores(trimmed_dates, trimmed_positions, lags, max(args.detail_top, 1))
    df["fecha"] = df["fecha"].dt.strftime("%Y-%m-%d")

    snapshot_path = DATA_DERIVED / ("cross_daily_all.parquet" if args.all_dates else f"cross_daily_{run_date}.parquet")
    DATA_DERIVED.mkdir(parents=True, exist_ok=True)
    if args.all_dates:
        daily = df
    else:
        daily = df[df["fecha"] == base_ts.strftime("%Y-%m-%d")].copy()
        if forecast:
            daily["fecha"] = run_date.strftime("%Y-%m-%d")
    daily.to_parquet(snapshot_path, index=False)
    daily.to_parquet(DERIVED_LATEST, index=False)

    if args.s3_bucket:
        object_run_date = latest_date if args.all_dates else run_date
        object_name = _build_object_name(
            args.s3_prefix,
            object_run_date,
            "cross_daily_all.parquet" if args.all_dates else "cross_daily.parquet",
        )
        upload_artifact(snapshot_path, args.s3_bucket, object_name=object_name)
    else:
        LOGGER.info("Bucket S3 no configurado; se omite carga.")

    LOGGER.info(
        "Cross diario%s -> oportunidades=%s activaciones=%s score_promedio=%.4f",
        " (batch)" if args.all_dates else "",
        int(daily["oportunidades_total"].sum()),
        int(daily["activaciones_total"].sum()),
        float(daily["score_cruzado"].mean()),
    )

    gx_status = maybe_run_gx(args.skip_validation, snapshot_path)

    maybe_log_mlflow(
        args.mlflow_uri,
        params={
            "run_date": run_date.isoformat(),
            "lags": ",".join(map(str, lags)),
            "events_path": str(source),
            "mode": "all_dates" if args.all_dates else "single",
        },
        metrics={
            "oportunidades": float(daily["oportunidades_total"].sum()),
            "activaciones": float(daily["activaciones_total"].sum()),
            "score_promedio": float(daily["score_cruzado"].mean()),
        },
        artifact=snapshot_path,
    )

    if gx_status == "failed":
        LOGGER.error("Great Expectations reportó fallas en cross_daily.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
