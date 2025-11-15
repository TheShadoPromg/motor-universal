from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from engine.derived_dynamic.helpers.storage import upload_artifact

LOGGER = logging.getLogger("derived_dynamic.aggregate")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DERIVED = REPO_ROOT / "data" / "derived"
DERIVED_DYNAMIC_LATEST = DATA_DERIVED / "derived_dynamic.parquet"
DERIVED_DAILY_LATEST = DATA_DERIVED / "derived_daily.parquet"
DEFAULT_BUCKET = os.getenv("DERIVED_DAILY_BUCKET", "motor-derived-daily")
DEFAULT_PREFIX = os.getenv("DERIVED_DAILY_PREFIX", "")
DETAIL_TOP = 5


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _find_latest_snapshot() -> Optional[Path]:
    snapshots = sorted(DATA_DERIVED.glob("derived_dynamic_*.parquet"))
    if snapshots:
        return snapshots[-1]
    return None


def _resolve_input_path(manual: Optional[str]) -> Path:
    if manual:
        candidate = Path(manual).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"No se encontró el archivo de entrada: {candidate}")
        return candidate
    latest = _find_latest_snapshot()
    if latest:
        LOGGER.info("Usando snapshot más reciente: %s", latest)
        return latest
    if DERIVED_DYNAMIC_LATEST.exists():
        LOGGER.info("Usando dataset por defecto: %s", DERIVED_DYNAMIC_LATEST)
        return DERIVED_DYNAMIC_LATEST
    raise FileNotFoundError(
        "No se encontraron artefactos derived_dynamic (ni snapshots ni archivo canonical)."
    )


def _parse_target_date(raw: Optional[str], available: pd.Series) -> datetime:
    if raw:
        try:
            return datetime.strptime(raw, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"target-date inválida '{raw}' (YYYY-MM-DD).") from exc
    if not available.empty:
        return pd.to_datetime(available.max())
    return datetime.utcnow()


def _build_details(group: pd.DataFrame, top_n: int) -> str:
    ordered = group.sort_values(
        by=["activaciones", "consistencia", "oportunidades"],
        ascending=[False, False, False],
    ).head(top_n)
    payload: List[dict] = []
    for _, row in ordered.iterrows():
        payload.append(
            {
                "rel": row["tipo_relacion"],
                "lag": int(row["lag"]),
                "k": (int(row["k"]) if pd.notna(row["k"]) else None),
                "op": int(row["oportunidades"]),
                "act": int(row["activaciones"]),
                "consistencia": round(float(row["consistencia"]), 4),
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def _aggregate(df: pd.DataFrame, detail_top: int) -> pd.DataFrame:
    data = df.copy()
    data["fecha"] = pd.to_datetime(data["fecha"])
    data["numero"] = data["numero"].astype(str).str.zfill(2)
    data["weighted_consistencia"] = data["consistencia"] * data["oportunidades"]

    summary = (
        data.groupby(["fecha", "numero"], sort=True)
        .agg(
            oportunidades_total=("oportunidades", "sum"),
            activaciones_total=("activaciones", "sum"),
            weighted_consistencia=("weighted_consistencia", "sum"),
            combinaciones_totales=("tipo_relacion", "size"),
        )
        .reset_index()
    )
    summary["score_derivado"] = np.where(
        summary["oportunidades_total"] > 0,
        summary["weighted_consistencia"] / summary["oportunidades_total"],
        0.0,
    )

    relaciones = (
        data.loc[data["activaciones"] > 0]
        .groupby(["fecha", "numero"])["tipo_relacion"]
        .nunique()
        .rename("relaciones_activas")
        .reset_index()
    )
    lags = (
        data.loc[data["activaciones"] > 0]
        .groupby(["fecha", "numero"])["lag"]
        .nunique()
        .rename("lags_activos")
        .reset_index()
    )
    detalles = (
        data.groupby(["fecha", "numero"], group_keys=False)
        .apply(lambda g: _build_details(g, detail_top))
        .rename("detalle_derivado")
        .reset_index()
    )

    result = summary.merge(relaciones, how="left", on=["fecha", "numero"])
    result = result.merge(lags, how="left", on=["fecha", "numero"])
    result = result.merge(detalles, how="left", on=["fecha", "numero"])
    result["relaciones_activas"] = result["relaciones_activas"].fillna(0).astype(int)
    result["lags_activos"] = result["lags_activos"].fillna(0).astype(int)
    result["detalle_derivado"] = result["detalle_derivado"].fillna("[]")
    result = result.drop(columns=["weighted_consistencia"])
    return result[
        [
            "fecha",
            "numero",
            "score_derivado",
            "oportunidades_total",
            "activaciones_total",
            "relaciones_activas",
            "lags_activos",
            "combinaciones_totales",
            "detalle_derivado",
        ]
    ].sort_values(["fecha", "numero"]).reset_index(drop=True)


def _build_object_name(prefix: str, target_date: datetime, filename: str) -> str:
    clean = prefix.strip("/")
    parts = [clean] if clean else []
    parts.append(target_date.strftime("%Y/%m/%d"))
    parts.append(filename)
    return "/".join(parts)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agrega el derived_dynamic diario consolidando scores por número.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Ruta del derived_dynamic parquet a consumir (por defecto usa el último snapshot).",
    )
    parser.add_argument(
        "--target-date",
        default=None,
        help="Fecha (YYYY-MM-DD) cuyo grid 00-99 se exportará. Por defecto usa la fecha máxima disponible.",
    )
    parser.add_argument(
        "--detail-top",
        type=int,
        default=DETAIL_TOP,
        help="Cantidad de combinaciones destacadas que se incluyen en detalle_derivado (default: 5).",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="Bucket S3/MinIO opcional (override de DERIVED_DAILY_BUCKET).",
    )
    parser.add_argument(
        "--s3-prefix",
        default=None,
        help="Prefijo dentro del bucket antes de YYYY/MM/DD.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    input_path = _resolve_input_path(args.input)
    LOGGER.info("Leyendo dataset derivado desde %s ...", input_path)
    derived = pd.read_parquet(input_path)
    if derived.empty:
        LOGGER.error("El dataset derived_dynamic está vacío; no se puede agregar.")
        return 2

    aggregated = _aggregate(derived, max(args.detail_top, 1))
    target_date = _parse_target_date(args.target_date, aggregated["fecha"])
    mask = aggregated["fecha"] == target_date
    if not mask.any():
        LOGGER.error(
            "No se encontró información para la fecha objetivo %s.", target_date.strftime("%Y-%m-%d")
        )
        return 2
    daily = aggregated.loc[mask].copy()

    DATA_DERIVED.mkdir(parents=True, exist_ok=True)
    date_str = target_date.strftime("%Y-%m-%d")
    snapshot_path = DATA_DERIVED / f"derived_daily_{date_str}.parquet"
    daily.to_parquet(snapshot_path, index=False)
    LOGGER.info("Snapshot diario guardado en %s (%s filas).", snapshot_path, len(daily))
    daily.to_parquet(DERIVED_DAILY_LATEST, index=False)
    LOGGER.info("Última versión actualizada en %s.", DERIVED_DAILY_LATEST)

    bucket = args.s3_bucket or DEFAULT_BUCKET
    prefix = args.s3_prefix or DEFAULT_PREFIX
    if bucket:
        object_name = _build_object_name(prefix, target_date, "derived_daily.parquet")
        upload_artifact(snapshot_path, bucket, object_name=object_name)
    else:
        LOGGER.info("Bucket S3 no configurado; se omite la carga del derivado diario.")

    LOGGER.info(
        "Resumen -> score_promedio=%.4f, oportunidades=%s, activaciones=%s",
        float(daily["score_derivado"].mean()),
        int(daily["oportunidades_total"].sum()),
        int(daily["activaciones_total"].sum()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
