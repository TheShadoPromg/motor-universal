from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from engine.derived_dynamic.helpers.storage import upload_artifact

LOGGER = logging.getLogger("fusion_3capas")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DERIVED = REPO_ROOT / "data" / "derived"

DEFAULT_CROSS_PATH = DATA_DERIVED / "cross_daily.parquet"
DEFAULT_STRUCT_PATH = DATA_DERIVED / "struct_daily.parquet"
DEFAULT_DERIVED_PATH = DATA_DERIVED / "derived_daily.parquet"

LATEST_OUTPUT = DATA_DERIVED / "jugadas_fusionadas_3capas.parquet"
BUCKET_DEFAULT = os.getenv("FUSION_BUCKET", "motor-fusion")
PREFIX_DEFAULT = os.getenv("FUSION_PREFIX", "")

DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)
DEFAULT_THRESHOLD = 0.0
DEFAULT_TEMP = 1.0


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _parse_run_date(raw: Optional[str], available: pd.Series) -> date:
    if raw:
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"run-date inválida '{raw}' (YYYY-MM-DD).") from exc
    if not available.empty:
        return pd.to_datetime(available.max()).date()
    return datetime.utcnow().date()


def _parse_weights(raw: Optional[str]) -> tuple[float, float, float]:
    if raw:
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if len(tokens) != 3:
            raise ValueError("Las ponderaciones deben tener exactamente tres valores (C,E,D).")
        weights = tuple(float(t) for t in tokens)
    else:
        weights = DEFAULT_WEIGHTS
    if any(w < 0 for w in weights):
        raise ValueError("Las ponderaciones no pueden ser negativas.")
    if sum(weights) <= 0:
        raise ValueError("La suma de ponderaciones debe ser mayor a 0.")
    return weights  # type: ignore[return-value]


def _load_daily_frame(
    path: Path,
    run_date: date,
    dataset_name: str,
    score_column: str,
    detail_column: str,
    cached_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if cached_df is not None:
        df = cached_df.copy()
    else:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el dataset {dataset_name}: {path}")
        df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"El dataset {dataset_name} está vacío ({path}).")
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    mask = df["fecha"] == pd.Timestamp(run_date)
    daily = df.loc[mask].copy()
    if daily.empty:
        raise ValueError(f"{dataset_name} no contiene información para {run_date} (archivo: {path}).")

    daily["numero"] = daily["numero"].astype(str).str.zfill(2)
    columns = ["fecha", "numero", score_column]
    if detail_column in daily.columns:
        columns.append(detail_column)
    else:
        daily[detail_column] = "[]"
        columns.append(detail_column)

    daily = daily[columns]
    return _ensure_full_grid(daily, score_column, detail_column, dataset_name)


def _ensure_full_grid(
    df: pd.DataFrame,
    score_column: str,
    detail_column: str,
    dataset_name: str,
) -> pd.DataFrame:
    fecha = df["fecha"].iloc[0]
    numbers = pd.Index([f"{i:02d}" for i in range(100)], name="numero")
    base = pd.DataFrame({"numero": numbers})
    merged = base.merge(df.drop(columns="fecha"), on="numero", how="left")
    merged["fecha"] = fecha
    merged[score_column] = pd.to_numeric(merged[score_column], errors="coerce").fillna(0.0)
    merged[detail_column] = merged[detail_column].fillna("[]")
    missing = merged["numero"][merged[score_column].isna()].tolist()
    if missing:
        LOGGER.warning(
            "%s: se encontraron valores inválidos en %s; fueron reemplazados por 0.",
            dataset_name,
            missing,
        )
    merged[score_column] = merged[score_column].astype(float)
    merged["fecha"] = pd.to_datetime(merged["fecha"])
    return merged.sort_values("numero").reset_index(drop=True)


def _softmax(values: pd.Series, temperature: float) -> pd.Series:
    if temperature <= 0:
        raise ValueError("softmax-temp debe ser > 0.")
    scaled = values / temperature
    max_val = scaled.max()
    exp = np.exp(scaled - max_val)
    total = exp.sum()
    if total == 0 or not np.isfinite(total):
        LOGGER.warning("La suma de exp() es inválida; se asignará distribución uniforme.")
        return pd.Series(np.full(len(values), 1 / len(values)), index=values.index)
    return exp / total


def _build_tipo_convergencia(row: pd.Series, threshold: float) -> str:
    layers = []
    if row["score_cruzado"] > threshold:
        layers.append("C")
    if row["score_estructural"] > threshold:
        layers.append("E")
    if row["score_derivado"] > threshold:
        layers.append("D")
    return "+".join(layers) if layers else "Sin Activación"


def _parse_detail(payload: object) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            return []
        if payload.startswith("{") or payload.startswith("["):
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                LOGGER.warning("Detalle con formato JSON inválido: '%s'. Se devolverá literal.", payload)
                return [payload]  # type: ignore[list-item]
            return data if isinstance(data, list) else [data]
        return [payload]  # type: ignore[list-item]
    return []


def _combine_details(row: pd.Series) -> str:
    payload = {
        "cruzado": _parse_detail(row.get("detalle_cruzado")),
        "estructural": _parse_detail(row.get("detalle_estructural")),
        "derivado": _parse_detail(row.get("detalle_derivado")),
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_object_name(prefix: str, run_date: date, filename: str) -> str:
    clean_prefix = prefix.strip("/")
    segments = [seg for seg in [clean_prefix, run_date.strftime("%Y/%m/%d")] if seg]
    segments.append(filename)
    return "/".join(segments)


def maybe_run_great_expectations(skip_validation: bool, output_path: Path) -> str:
    if skip_validation:
        LOGGER.info("Validación con Great Expectations omitida por bandera.")
        return "skipped"
    try:
        import great_expectations as gx  # type: ignore

        ctx = gx.get_context()
        checkpoint_name = "fusion_daily"
        result = ctx.run_checkpoint(checkpoint_name=checkpoint_name)
        status = "passed" if result.get("success") else "failed"
        LOGGER.info("Checkpoint '%s' finalizado con estado %s.", checkpoint_name, status)
        return status
    except FileNotFoundError as exc:
        LOGGER.warning("Checkpoint de Great Expectations no encontrado: %s", exc)
        return "missing"
    except ImportError as exc:
        LOGGER.warning("Great Expectations no está instalado: %s", exc)
        return "missing"
    except Exception as exc:  # pragma: no cover - contexto externo
        LOGGER.warning("Validación de Great Expectations falló: %s", exc)
        return "failed"


def maybe_log_mlflow(
    mlflow_uri: Optional[str],
    params: Dict[str, str],
    metrics: Dict[str, float],
    artifact_path: Path,
) -> None:
    if not mlflow_uri:
        LOGGER.info("MLflow no configurado; omitiendo registro.")
        return
    try:
        import mlflow  # type: ignore
    except ImportError as exc:
        LOGGER.warning("MLflow no disponible: %s", exc)
        return

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("fusion_3capas")
        with mlflow.start_run(run_name="fusion_3capas"):
            for key, value in params.items():
                mlflow.log_param(key, value)
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value))
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path="outputs")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("No se pudieron registrar métricas en MLflow: %s", exc)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fusiona scores cruzado/estructural/derivado y produce distribución diaria 00-99.",
    )
    parser.add_argument("--cross-path", default=str(DEFAULT_CROSS_PATH), help="Parquet con score_cruzado diario.")
    parser.add_argument("--struct-path", default=str(DEFAULT_STRUCT_PATH), help="Parquet con score_estructural diario.")
    parser.add_argument("--derived-path", default=str(DEFAULT_DERIVED_PATH), help="Parquet con score_derivado diario.")
    parser.add_argument("--run-date", default=None, help="Fecha (YYYY-MM-DD) a procesar. Por defecto la más reciente.")
    parser.add_argument(
        "--weights",
        default=",".join(map(str, DEFAULT_WEIGHTS)),
        help="Ponderaciones C,E,D (por ejemplo '0.4,0.3,0.3').",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Umbral para considerar una capa como activa en tipo_convergencia.",
    )
    parser.add_argument(
        "--softmax-temp",
        type=float,
        default=DEFAULT_TEMP,
        help="Temperatura para la softmax que genera las probabilidades.",
    )
    parser.add_argument("--skip-validation", action="store_true", help="Omitir Great Expectations.")
    parser.add_argument("--mlflow-uri", default=None, help="URI de MLflow para registrar runs.")
    parser.add_argument("--s3-bucket", default=None, help="Bucket S3 donde subir el parquet resultante.")
    parser.add_argument("--s3-prefix", default=None, help="Prefijo opcional dentro del bucket.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    weights = _parse_weights(args.weights)
    cross_path = Path(args.cross_path).expanduser().resolve()
    struct_path = Path(args.struct_path).expanduser().resolve()
    derived_path = Path(args.derived_path).expanduser().resolve()

    derived_df = pd.read_parquet(derived_path)
    run_date = _parse_run_date(args.run_date, pd.to_datetime(derived_df["fecha"]))

    LOGGER.info("Procesando fecha %s con weights=%s ...", run_date, weights)
    cross = _load_daily_frame(cross_path, run_date, "cross_daily", "score_cruzado", "detalle_cruzado")
    structural = _load_daily_frame(struct_path, run_date, "struct_daily", "score_estructural", "detalle_estructural")
    derived = _load_daily_frame(
        derived_path, run_date, "derived_daily", "score_derivado", "detalle_derivado", cached_df=derived_df
    )

    merged = cross.merge(structural, on=["fecha", "numero"], how="inner")
    merged = merged.merge(derived, on=["fecha", "numero"], how="inner")

    merged["score_cruzado"] = merged["score_cruzado"].astype(float)
    merged["score_estructural"] = merged["score_estructural"].astype(float)
    merged["score_derivado"] = merged["score_derivado"].astype(float)

    merged["score_total"] = (
        merged["score_cruzado"] * weights[0]
        + merged["score_estructural"] * weights[1]
        + merged["score_derivado"] * weights[2]
    )
    merged["prob"] = _softmax(merged["score_total"], args.softmax_temp)
    merged["tipo_convergencia"] = merged.apply(
        lambda row: _build_tipo_convergencia(row, args.activation_threshold), axis=1
    )
    merged["detalles"] = merged.apply(_combine_details, axis=1)
    merged["fecha"] = merged["fecha"].dt.strftime("%Y-%m-%d")

    ordered = merged[
        [
            "fecha",
            "numero",
            "score_cruzado",
            "score_estructural",
            "score_derivado",
            "score_total",
            "prob",
            "tipo_convergencia",
            "detalles",
        ]
    ].sort_values(["fecha", "numero"])

    run_date_str = run_date.strftime("%Y-%m-%d")
    snapshot_path = DATA_DERIVED / f"jugadas_fusionadas_3capas_{run_date_str}.parquet"
    DATA_DERIVED.mkdir(parents=True, exist_ok=True)
    ordered.to_parquet(snapshot_path, index=False)
    shutil.copy2(snapshot_path, LATEST_OUTPUT)

    bucket = args.s3_bucket or BUCKET_DEFAULT
    prefix = args.s3_prefix or PREFIX_DEFAULT
    if bucket:
        object_name = _build_object_name(prefix, run_date, "jugadas_fusionadas_3capas.parquet")
        upload_artifact(snapshot_path, bucket, object_name=object_name)
    else:
        LOGGER.info("Bucket S3 no configurado; se omite carga del artefacto de fusión.")

    prob_sum = (
        ordered.groupby("fecha")["prob"].sum().agg(list)
        if not ordered.empty
        else []
    )
    LOGGER.info(
        "Fusión completada -> filas=%s, sum(prob)=%s, score_total∈[%.4f, %.4f]",
        len(ordered),
        prob_sum,
        float(ordered["score_total"].min()) if not ordered.empty else 0.0,
        float(ordered["score_total"].max()) if not ordered.empty else 0.0,
    )

    gx_status = maybe_run_great_expectations(args.skip_validation, snapshot_path)

    maybe_log_mlflow(
        args.mlflow_uri,
        params={
            "run_date": run_date_str,
            "weights": ",".join(map(str, weights)),
            "activation_threshold": str(args.activation_threshold),
            "softmax_temp": str(args.softmax_temp),
            "cross_path": str(cross_path),
            "struct_path": str(struct_path),
            "derived_path": str(derived_path),
        },
        metrics={
            "filas": float(len(ordered)),
            "prob_sum": float(ordered["prob"].sum()) if not ordered.empty else 0.0,
            "score_total_min": float(ordered["score_total"].min()) if not ordered.empty else 0.0,
            "score_total_max": float(ordered["score_total"].max()) if not ordered.empty else 0.0,
        },
        artifact_path=snapshot_path,
    )

    if gx_status == "failed":
        LOGGER.error("La validación de Great Expectations falló.")
        return 2

    LOGGER.info("Motor de fusión 3 capas finalizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
