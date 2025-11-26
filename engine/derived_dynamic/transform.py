from __future__ import annotations

import argparse
import logging
import unicodedata
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from engine._utils.schema import normalize_events_df
from engine.derived_dynamic.helpers.storage import upload_artifact

LOGGER = logging.getLogger("derived_dynamic")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_DERIVED = REPO_ROOT / "data" / "derived"
EVENTS_PARQUET = DATA_RAW / "eventos_numericos.parquet"
EVENTS_CSV = DATA_RAW / "eventos_numericos.csv"
DERIVED_OUTPUT = DATA_DERIVED / "derived_dynamic.parquet"
DEFAULT_DERIVED_BUCKET = os.getenv("DERIVED_DYNAMIC_BUCKET", "motor-derived-dynamic")
DEFAULT_DERIVED_PREFIX = os.getenv("DERIVED_DYNAMIC_PREFIX", "")

DEFAULT_LAGS = [1, 2, 3, 7, 14, 30]
DEFAULT_K_VALUES = [1, 2, 5, 10, 50]
DEFAULT_RELATIONS = ["espejo", "complemento", "seq", "sum_mod"]

ALL_NUMBERS = [f"{i:02d}" for i in range(100)]
NUMBER_RANGE = np.arange(100)

PANEL_COLUMN_ORDER = ["fecha", "numero", "e_pos1", "e_pos2", "e_pos3"]
DERIVED_COLUMNS = [
    "fecha",
    "numero",
    "tipo_relacion",
    "lag",
    "k",
    "oportunidades",
    "activaciones",
    "consistencia",
    "oportunidades_historial",
    "datos_suficientes",
]
DERIVED_WINDOW_CONFIG: "OrderedDict[str, int]" = OrderedDict(
    [
        ("short", 90),
        ("mid", 180),
        ("long", 360),
    ]
)
# Pesos (se re-normalizan entre ventanas disponibles); ajustar aquí si se desea.
DERIVED_WINDOW_WEIGHTS: "OrderedDict[str, float]" = OrderedDict(
    [
        ("short", 0.5),
        ("mid", 0.3),
        ("long", 0.2),
    ]
)
DERIVED_REFERENCE_WINDOW = "long"
DEFAULT_MIN_OPPORTUNITIES = int(os.getenv("DERIVED_MIN_OPORTUNIDADES", "30"))

LONG_POSITION_COLUMNS = {"posicion", "posición", "position", "pos"}


@dataclass(frozen=True)
class RelationRule:
    """Encapsula los antecedentes para una combinación (relación, k)."""

    tipo_relacion: str
    k_value: Optional[int]
    antecedent_indexers: Tuple[np.ndarray, ...]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _strip_accents(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_column(name: str) -> str:
    return _strip_accents(name).replace(" ", "").replace("_", "").lower()


def detect_input_format(df: pd.DataFrame) -> str:
    normalized_cols = {_normalize_column(col) for col in df.columns}
    if normalized_cols & {_normalize_column(col) for col in LONG_POSITION_COLUMNS}:
        return "long"
    has_panel_indicators = {
        "epos1",
        "epos2",
        "epos3",
        "pos1",
        "pos2",
        "pos3",
        "primero",
        "segundo",
        "tercero",
    }
    if len(normalized_cols & has_panel_indicators) >= 3:
        return "panel"
    return "panel"


def _read_raw_events() -> Tuple[pd.DataFrame, Path]:
    for path in (EVENTS_PARQUET, EVENTS_CSV):
        if path.exists():
            LOGGER.info("Cargando eventos desde %s", path)
            if path.suffix == ".parquet":
                return pd.read_parquet(path), path
            return pd.read_csv(path), path
    raise FileNotFoundError(
        f"No se encontró eventos_numericos en {EVENTS_PARQUET} ni en {EVENTS_CSV}."
    )


def _warn_on_sparse_dates(wide_df: pd.DataFrame, fmt: str) -> None:
    counts = wide_df.groupby("date")["number"].nunique()
    if counts.empty:
        return
    min_unique = counts.min()
    if min_unique < 100:
        LOGGER.warning(
            "Se detectaron fechas con menos de 100 números (mínimo=%s). "
            "Revisa si el histórico está truncado.",
            min_unique,
        )
    if fmt == "panel":
        expected_rows = len(counts) * 100
        actual_rows = len(wide_df)
        if actual_rows < expected_rows:
            LOGGER.warning(
                "El panel recibido tiene %s filas vs %s esperadas. "
                "Verifica que no haya fechas o números faltantes.",
                actual_rows,
                expected_rows,
            )


def _ensure_full_grid(panel: pd.DataFrame) -> pd.DataFrame:
    dates = panel["fecha"].dropna().sort_values().unique()
    if len(dates) == 0:
        return panel
    grid = pd.MultiIndex.from_product(
        [dates, ALL_NUMBERS], names=["fecha", "numero"]
    )
    filled = (
        panel.set_index(["fecha", "numero"])
        .reindex(grid, fill_value=0)
        .reset_index()
    )
    for col in ["e_pos1", "e_pos2", "e_pos3"]:
        filled[col] = filled[col].fillna(0).astype(int).clip(0, 1)
    return filled.sort_values(["fecha", "numero"]).reset_index(drop=True)


def _validate_panel(panel: pd.DataFrame, fmt: str) -> None:
    unique_dates = panel["fecha"].dropna().nunique()
    if unique_dates == 0:
        raise ValueError("El panel final no contiene fechas válidas.")
    invalid_numbers = panel.loc[~panel["numero"].isin(ALL_NUMBERS), "numero"].unique()
    if len(invalid_numbers) > 0:
        raise ValueError(
            f"Se encontraron números fuera del rango 00-99: {sorted(invalid_numbers)}"
        )
    if fmt == "long":
        per_date = panel.groupby("fecha")[["e_pos1", "e_pos2", "e_pos3"]].sum()
        inconsistent = per_date[(per_date != 1).any(axis=1)]
        if not inconsistent.empty:
            LOGGER.warning(
                "Existen fechas con menos o más de una activación por posición: %s",
                inconsistent.index.strftime("%Y-%m-%d").tolist(),
            )


def load_or_generate_eventos() -> Tuple[pd.DataFrame, Path, str]:
    raw_df, source = _read_raw_events()
    fmt = detect_input_format(raw_df)
    LOGGER.info("Formato detectado para eventos: %s", fmt)

    try:
        canonical = normalize_events_df(raw_df)
    except KeyError as exc:
        raise ValueError(
            "No fue posible normalizar eventos. Revisa las columnas del archivo."
        ) from exc

    _warn_on_sparse_dates(canonical, fmt)

    panel = canonical.rename(
        columns={
            "date": "fecha",
            "number": "numero",
            "pos1": "e_pos1",
            "pos2": "e_pos2",
            "pos3": "e_pos3",
        }
    )
    panel["fecha"] = pd.to_datetime(panel["fecha"], errors="coerce")
    panel = panel.dropna(subset=["fecha"])
    panel["numero"] = panel["numero"].astype(str).str.zfill(2)
    for col in ["e_pos1", "e_pos2", "e_pos3"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    panel = _ensure_full_grid(panel)
    panel = panel[PANEL_COLUMN_ORDER]

    _validate_panel(panel, fmt)

    return panel, source, fmt


def _build_appearance_matrix(panel: pd.DataFrame) -> Tuple[pd.Index, np.ndarray]:
    panel = panel.copy()
    panel["numero_int"] = panel["numero"].astype(int)
    panel["aparece"] = (
        panel[["e_pos1", "e_pos2", "e_pos3"]].sum(axis=1) > 0
    ).astype(np.int8)

    dates = pd.Index(panel["fecha"].dropna().sort_values().unique())
    if len(dates) == 0:
        return dates, np.zeros((0, 100), dtype=np.int8)

    matrix = np.zeros((len(dates), 100), dtype=np.int8)
    grouped = panel.groupby("fecha", sort=True)
    for idx, date in enumerate(dates):
        subset = grouped.get_group(date)
        matrix[idx, subset["numero_int"].to_numpy()] = subset["aparece"].to_numpy()
    return dates, matrix


def _antecedent_hits(rule: RelationRule, previous_values: np.ndarray) -> np.ndarray:
    """Devuelve un vector binario indicando si un antecedente se activó."""
    hits = previous_values[rule.antecedent_indexers[0]].copy()
    for indexer in rule.antecedent_indexers[1:]:
        hits |= previous_values[indexer]
    return hits


def _build_relation_rules(relations: Sequence[str], k_values: Sequence[int]) -> List[RelationRule]:
    requested = set(relations)
    rules: List[RelationRule] = []

    if "espejo" in requested:
        mirror_idx = np.array([int(f"{n:02d}"[::-1]) for n in NUMBER_RANGE], dtype=np.int64)
        rules.append(RelationRule("espejo", None, (mirror_idx,)))

    if "complemento" in requested:
        complement_idx = ((100 - NUMBER_RANGE) % 100).astype(np.int64)
        rules.append(RelationRule("complemento", None, (complement_idx,)))

    if "seq" in requested:
        left_idx = ((NUMBER_RANGE - 1) % 100).astype(np.int64)
        right_idx = ((NUMBER_RANGE + 1) % 100).astype(np.int64)
        rules.append(RelationRule("seq", 1, (left_idx, right_idx)))

    if "sum_mod" in requested:
        valid_k = sorted({int(k) for k in k_values if int(k) > 0})
        if not valid_k:
            LOGGER.warning(
                "Relación 'sum_mod' solicitada sin valores k positivos; la relación se omitirá."
            )
        else:
            for k in valid_k:
                minus_idx = ((NUMBER_RANGE - k) % 100).astype(np.int64)
                plus_idx = ((NUMBER_RANGE + k) % 100).astype(np.int64)
                rules.append(RelationRule("sum_mod", k, (minus_idx, plus_idx)))

    return rules


def _assemble_relation_frame(
    rule: RelationRule,
    lag: int,
    dates: pd.Index,
    number_labels: np.ndarray,
    opportunities: np.ndarray,
    activations: np.ndarray,
) -> pd.DataFrame:
    num_dates = len(dates)
    total_rows = num_dates * len(number_labels)
    data = {
        "fecha": np.repeat(dates.to_numpy(), len(number_labels)),
        "numero": np.tile(number_labels, num_dates),
        "tipo_relacion": np.full(total_rows, rule.tipo_relacion, dtype=object),
        "lag": np.full(total_rows, lag, dtype=int),
        "k": np.full(total_rows, rule.k_value, dtype=object),
        "oportunidades": opportunities.reshape(-1),
        "activaciones": activations.reshape(-1),
    }
    return pd.DataFrame(data)


def _materialize_rule(
    rule: RelationRule,
    dates: pd.Index,
    matrix: np.ndarray,
    lags: Sequence[int],
) -> pd.DataFrame:
    number_labels = np.array(ALL_NUMBERS, dtype=object)
    frames: List[pd.DataFrame] = []
    num_dates = len(dates)

    for lag in lags:
        opportunities = np.zeros((num_dates, 100), dtype=np.int8)
        activations = np.zeros((num_dates, 100), dtype=np.int8)
        for idx in range(num_dates):
            prev_idx = idx - lag
            if prev_idx < 0:
                continue
            antecedent_hits = _antecedent_hits(rule, matrix[prev_idx])
            current_hits = matrix[idx]
            opportunities[idx, :] = 1
            activations[idx, :] = (antecedent_hits & current_hits)
        frames.append(
            _assemble_relation_frame(rule, lag, dates, number_labels, opportunities, activations)
        )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _sliding_window_sum(values: np.ndarray, timestamps: np.ndarray, window_days: int) -> np.ndarray:
    """Suma valores dentro de los últimos `window_days` días excluyendo el día corriente."""

    if len(values) == 0:
        return np.zeros(0, dtype=np.int64)
    prefix = np.concatenate(([0], np.cumsum(values, dtype=np.int64)))
    result = np.zeros(len(values), dtype=np.int64)
    for idx in range(len(values)):
        cutoff = timestamps[idx] - window_days
        start_idx = np.searchsorted(timestamps, cutoff, side="left")
        result[idx] = prefix[idx] - prefix[start_idx]
    return result


def _add_consistency(base_df: pd.DataFrame, min_opportunities: int) -> pd.DataFrame:
    if base_df.empty:
        base_df["consistencia"] = pd.Series(dtype=float)
        base_df["oportunidades_historial"] = pd.Series(dtype=int)
        base_df["datos_suficientes"] = pd.Series(dtype=bool)
        return base_df

    df = base_df.sort_values(
        ["numero", "tipo_relacion", "lag", "k", "fecha"]
    ).reset_index(drop=True)

    group_keys = ["numero", "tipo_relacion", "lag", "k"]
    grouped = df.groupby(group_keys, sort=False)
    window_keys = list(DERIVED_WINDOW_CONFIG.keys())
    for label in window_keys:
        df[f"_opp_hist_{label}"] = 0
        df[f"_act_hist_{label}"] = 0

    for _, indices in grouped.indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) == 0:
            continue
        timestamps = df.loc[idx, "fecha"].to_numpy(dtype="datetime64[D]").astype(np.int64)
        opp_values = df.loc[idx, "oportunidades"].to_numpy(dtype=np.int64)
        act_values = df.loc[idx, "activaciones"].to_numpy(dtype=np.int64)
        for label, window_days in DERIVED_WINDOW_CONFIG.items():
            df.loc[idx, f"_opp_hist_{label}"] = _sliding_window_sum(opp_values, timestamps, int(window_days))
            df.loc[idx, f"_act_hist_{label}"] = _sliding_window_sum(act_values, timestamps, int(window_days))

    opp_matrix = []
    cons_matrix = []
    weight_vector = np.array(
        [float(DERIVED_WINDOW_WEIGHTS.get(label, 0.0)) for label in window_keys],
        dtype=float,
    )
    for label in window_keys:
        opp_col = df[f"_opp_hist_{label}"].to_numpy(dtype=float)
        act_col = df[f"_act_hist_{label}"].to_numpy(dtype=float)
        opp_matrix.append(opp_col)
        cons_matrix.append(
            np.divide(
                act_col,
                opp_col,
                out=np.zeros_like(opp_col),
                where=opp_col > 0,
            )
        )

    opp_matrix_np = np.vstack(opp_matrix).T if opp_matrix else np.zeros((len(df), 0))
    cons_matrix_np = np.vstack(cons_matrix).T if cons_matrix else np.zeros((len(df), 0))
    available_weights = np.where(opp_matrix_np > 0, weight_vector, 0.0)
    weight_sum = available_weights.sum(axis=1)
    weighted_scores = (cons_matrix_np * available_weights).sum(axis=1)
    df["consistencia"] = np.divide(
        weighted_scores,
        weight_sum,
        out=np.zeros(len(df), dtype=float),
        where=weight_sum > 0,
    )

    reference_col = f"_opp_hist_{DERIVED_REFERENCE_WINDOW}"
    if reference_col not in df.columns:
        reference_col = f"_opp_hist_{window_keys[-1]}"
    df["oportunidades_historial"] = df[reference_col].astype(int)
    df["datos_suficientes"] = (df["oportunidades_historial"] >= int(min_opportunities)).astype(bool)
    df.loc[~df["datos_suficientes"], "consistencia"] = 0.0

    drop_cols = [f"_opp_hist_{label}" for label in window_keys] + [
        f"_act_hist_{label}" for label in window_keys
    ]
    df = df.drop(columns=drop_cols)
    return df


def generate_relations(
    panel: pd.DataFrame,
    relations: Sequence[str],
    lags: Sequence[int],
    k_values: Sequence[int],
    min_opportunities: int,
) -> pd.DataFrame:
    dates, matrix = _build_appearance_matrix(panel)
    if len(dates) == 0:
        return pd.DataFrame(columns=DERIVED_COLUMNS)

    rules = _build_relation_rules(relations, k_values)
    if not rules:
        LOGGER.warning("No se definieron reglas para las relaciones solicitadas.")
        return pd.DataFrame(columns=DERIVED_COLUMNS)

    frames: List[pd.DataFrame] = []
    for rule in rules:
        descriptor = (
            rule.tipo_relacion
            if rule.k_value is None
            else f"{rule.tipo_relacion}(k={rule.k_value})"
        )
        LOGGER.info("Iniciando cálculo de relación %s...", descriptor)
        frame = _materialize_rule(rule, dates, matrix, lags)
        frames.append(frame)
        LOGGER.info("Relación %s completada con %s filas.", descriptor, len(frame))

    derived = pd.concat(frames, ignore_index=True)
    derived["fecha"] = pd.to_datetime(derived["fecha"])
    derived["numero"] = derived["numero"].astype(str).str.zfill(2)
    derived["lag"] = derived["lag"].astype(int)
    derived["k"] = pd.array(derived["k"], dtype="Int64")
    derived["oportunidades"] = derived["oportunidades"].astype(int)
    derived["activaciones"] = derived["activaciones"].astype(int)

    enriched = _add_consistency(derived, min_opportunities)
    enriched["fecha"] = pd.to_datetime(enriched["fecha"]).dt.strftime("%Y-%m-%d")
    enriched["numero"] = enriched["numero"].astype(str).str.zfill(2)
    enriched["lag"] = enriched["lag"].astype(int)
    enriched["k"] = pd.array(enriched["k"], dtype="Int64")
    enriched["oportunidades"] = enriched["oportunidades"].astype(int)
    enriched["activaciones"] = enriched["activaciones"].astype(int)
    enriched["consistencia"] = enriched["consistencia"].astype(float)
    enriched["oportunidades_historial"] = enriched["oportunidades_historial"].astype(int)
    enriched["datos_suficientes"] = enriched["datos_suficientes"].astype(bool)

    return enriched.sort_values(
        ["fecha", "numero", "tipo_relacion", "lag", "k"], na_position="last"
    ).reset_index(drop=True)[DERIVED_COLUMNS]


def maybe_run_great_expectations(skip_validation: bool, output_path: Path) -> str:
    if skip_validation:
        LOGGER.info("Validación con Great Expectations omitida por bandera.")
        return "skipped"
    try:
        import great_expectations as gx  # type: ignore

        ctx = gx.get_context()
        checkpoint_name = "derived_dynamic"
        LOGGER.info(
            "Ejecutando checkpoint de Great Expectations '%s' sobre %s...",
            checkpoint_name,
            output_path,
        )
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
    except Exception as exc:
        LOGGER.warning("La validación de Great Expectations falló: %s", exc)
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
        mlflow.set_experiment("derived_dynamic")
        with mlflow.start_run(run_name="derived_dynamic"):
            for key, value in params.items():
                mlflow.log_param(key, value)
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value))
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path="outputs")
    except Exception as exc:
        LOGGER.warning("No se pudieron registrar métricas en MLflow: %s", exc)


def _parse_int_list(raw: str, default: Sequence[int], positive_only: bool = True) -> List[int]:
    if raw is None or raw.strip() == "":
        values = list(default)
    else:
        values = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            values.append(int(token))
    cleaned = sorted(set(values))
    if positive_only and any(value <= 0 for value in cleaned):
        raise ValueError("Todos los valores deben ser enteros positivos.")
    return cleaned


def _parse_relations(raw: str, default: Sequence[str]) -> List[str]:
    if raw is None or raw.strip() == "":
        relations = list(default)
    else:
        relations = [token.strip().lower() for token in raw.split(",") if token.strip()]
    cleaned = []
    seen = set()
    for rel in relations:
        if rel not in {"espejo", "complemento", "seq", "sum_mod"}:
            raise ValueError(f"Relación desconocida '{rel}'.")
        if rel not in seen:
            cleaned.append(rel)
            seen.add(rel)
    return cleaned


def _resolve_run_date(raw: Optional[str], derived: pd.DataFrame) -> date:
    if raw:
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"run-date inválida '{raw}' (formato esperado YYYY-MM-DD).") from exc
    if not derived.empty:
        return pd.to_datetime(derived["fecha"]).max().date()
    return datetime.utcnow().date()


def _build_object_name(prefix: str, run_date: date, filename: str) -> str:
    clean_prefix = prefix.strip("/")
    segments = [seg for seg in [clean_prefix, run_date.strftime("%Y/%m/%d")] if seg]
    segments.append(filename)
    return "/".join(segments)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye el dataset derived_dynamic a partir de eventos numéricos.",
    )
    parser.add_argument(
        "--lags",
        default=",".join(map(str, DEFAULT_LAGS)),
        help="Lista de lags (ej. '1,2,3,7,14,30').",
    )
    parser.add_argument(
        "--k",
        default=",".join(map(str, DEFAULT_K_VALUES)),
        help="Lista de desplazamientos k para sum_mod (ej. '1,2,5,10,50').",
    )
    parser.add_argument(
        "--relaciones",
        default=",".join(DEFAULT_RELATIONS),
        help="Relaciones a calcular (espejo, complemento, seq, sum_mod).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Si se indica, omite la validación con Great Expectations.",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help="URI del servidor MLflow para registrar métricas.",
    )
    parser.add_argument(
        "--run-date",
        default=None,
        help="Fecha (YYYY-MM-DD) que etiquetará el snapshot generado.",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="Bucket S3/MinIO donde subir el parquet resultante.",
    )
    parser.add_argument(
        "--s3-prefix",
        default=None,
        help="Prefijo opcional dentro del bucket (se agregará YYYY/MM/DD/archivo).",
    )
    parser.add_argument(
        "--min-oportunidades",
        type=int,
        default=DEFAULT_MIN_OPPORTUNITIES,
        help="Mínimo de oportunidades históricas (ventana larga) para considerar una relación.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    try:
        lags = _parse_int_list(args.lags, DEFAULT_LAGS, positive_only=True)
        k_values = _parse_int_list(args.k, DEFAULT_K_VALUES, positive_only=False)
        relations = _parse_relations(args.relaciones, DEFAULT_RELATIONS)
    except ValueError as exc:
        LOGGER.error("Error en parámetros de entrada: %s", exc)
        return 2

    min_opportunities = max(int(args.min_oportunidades), 0)
    LOGGER.info(
        "Parámetros -> lags=%s, k=%s, relaciones=%s, min_oportunidades=%s",
        lags,
        k_values,
        relations,
        min_opportunities,
    )

    panel, source_path, fmt = load_or_generate_eventos()
    LOGGER.info(
        "Panel final con %s fechas y %s filas.",
        panel['fecha'].nunique(),
        len(panel),
    )

    derived = generate_relations(panel, relations, lags, k_values, min_opportunities)
    DATA_DERIVED.mkdir(parents=True, exist_ok=True)
    run_date = _resolve_run_date(args.run_date, derived)
    run_date_str = run_date.strftime("%Y-%m-%d")
    dated_path = DATA_DERIVED / f"derived_dynamic_{run_date_str}.parquet"
    derived.to_parquet(DERIVED_OUTPUT, index=False)
    LOGGER.info("Se escribió el dataset derived_dynamic (latest) en %s", DERIVED_OUTPUT)
    if dated_path != DERIVED_OUTPUT:
        shutil.copy2(DERIVED_OUTPUT, dated_path)
        LOGGER.info("Snapshot fechado almacenado en %s", dated_path)
    bucket = args.s3_bucket or DEFAULT_DERIVED_BUCKET
    prefix = args.s3_prefix or DEFAULT_DERIVED_PREFIX
    if bucket:
        object_name = _build_object_name(prefix, run_date, "derived_dynamic.parquet")
        upload_artifact(dated_path, bucket, object_name=object_name)
    else:
        LOGGER.info("Bucket S3 no configurado; se omite carga del snapshot.")

    total_opportunidades = int(derived["oportunidades"].sum()) if not derived.empty else 0
    total_activaciones = int(derived["activaciones"].sum()) if not derived.empty else 0
    ratio_global = (
        total_activaciones / total_opportunidades if total_opportunidades > 0 else 0.0
    )
    LOGGER.info(
        "Métricas -> filas=%s, oportunidades=%s, activaciones=%s, ratio_global=%.4f",
        len(derived),
        total_opportunidades,
        total_activaciones,
        ratio_global,
    )

    gx_status = maybe_run_great_expectations(args.skip_validation, DERIVED_OUTPUT)

    maybe_log_mlflow(
        args.mlflow_uri,
        params={
            "lags": ",".join(map(str, lags)),
            "k": ",".join(map(str, k_values)),
            "relaciones": ",".join(relations),
            "eventos_path": str(source_path),
            "derived_path": str(dated_path),
            "input_format": fmt,
            "run_date": run_date_str,
            "min_oportunidades": str(min_opportunities),
        },
        metrics={
            "filas": float(len(derived)),
            "total_oportunidades": float(total_opportunidades),
            "total_activaciones": float(total_activaciones),
            "ratio_global": float(ratio_global),
        },
        artifact_path=dated_path,
    )

    if gx_status == "failed":
        LOGGER.error("Great Expectations reportó fallas en la validación.")
        return 2

    LOGGER.info("Motor derived_dynamic finalizado correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Transformación de eventos a capa derivada dinámica.

- Normaliza eventos (long/wide) a panel 00-99 y genera derived_dynamic con relaciones (espejo, sum_mod, seq, complemento)
  evaluadas en múltiples lags y ventanas.
- Controla oportunidades/activaciones, consistencia y flags de datos suficientes.
- Exporta parquet canonical y snapshots, con opción de carga a S3.
"""
