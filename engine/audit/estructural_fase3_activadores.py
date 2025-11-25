"""
Fase 3 – Generación de activadores estructurales dinámicos
a partir de los sesgos validados en Fase 2.5.

No recalcula transiciones, sólo transforma los outputs de Fase 2.5
en un catálogo de reglas estructurales con pesos.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("audit.estructural_fase3_activadores")

REPO_ROOT = Path(__file__).resolve().parents[2]

# Directorios de entrada (defaults Fase 2.5)
FASE2_5_DIR = REPO_ROOT / "data" / "audit" / "estructural_fase2_5"
DEFAULT_INPUT_CORE = FASE2_5_DIR / "sesgos_fase2_5_core_y_periodicos.parquet"
DEFAULT_INPUT_PERIODOS = FASE2_5_DIR / "sesgos_fase2_5_por_periodo.parquet"

# Directorio de salida para activadores
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "activadores"
PARQUET_ENGINE = "pyarrow"

PERIODOS = ["2011_2014", "2015_2018", "2019_2022", "2023_2025"]

CLASS_WEIGHT: Dict[str, float] = {
    "core_global": 1.5,
    "extended_global": 1.2,
    "periodico": 1.0,
}

MIN_STABILITY = 0.0
MAX_STABILITY = 1.0


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_int(val, default=None):
    try:
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def _normalize_pos(val):
    if pd.isna(val):
        return "ANY"
    token = str(val).strip().upper()
    if token == "ANY":
        return "ANY"
    try:
        return int(token)
    except Exception:
        return "ANY"


def _coerce_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes", "y"}
    return False


def compute_raw_weight(row: pd.Series) -> float:
    class_factor = CLASS_WEIGHT.get(str(row.get("clasificacion_fase2_5", "")).strip(), 1.0)
    mean_delta = row.get("mean_delta_rel_periods")
    mag_factor = math.log1p(max(mean_delta, 0.0)) if mean_delta is not None and not pd.isna(mean_delta) else 0.0
    stability = row.get("stability_score")
    if stability is None or pd.isna(stability):
        stability_clamped = 0.0
    else:
        stability_clamped = max(MIN_STABILITY, min(MAX_STABILITY, float(stability)))
    stability_factor = 0.5 + 0.5 * stability_clamped
    raw_weight = class_factor * mag_factor * stability_factor
    if pd.isna(raw_weight) or raw_weight < 0:
        return 0.0
    return raw_weight


def normalize_weights(df: pd.DataFrame, weight_col: str = "Peso_Bruto") -> pd.Series:
    weights = df[weight_col].fillna(0).astype(float)
    mean_val = weights.mean()
    if mean_val <= 0 or pd.isna(mean_val):
        return pd.Series([1.0] * len(df), index=df.index)
    return weights / mean_val


def build_period_summary(sesgos_core: pd.DataFrame, sesgos_por_periodo: pd.DataFrame) -> pd.DataFrame:
    # Asegurar tipos y nombres clave
    key_cols = ["tipo_relacion", "numero_base", "numero_destino", "pos_origen", "pos_destino", "lag"]
    for col in key_cols:
        if col not in sesgos_por_periodo.columns:
            raise ValueError(f"Falta la columna {col} en sesgos_por_periodo.csv")

    sesgos_por_periodo = sesgos_por_periodo.copy()
    sesgos_por_periodo["pos_origen"] = sesgos_por_periodo["pos_origen"].apply(_normalize_pos)
    sesgos_por_periodo["pos_destino"] = sesgos_por_periodo["pos_destino"].apply(_normalize_pos)
    sesgos_por_periodo["lag"] = sesgos_por_periodo["lag"].apply(_safe_int)
    sesgos_por_periodo["es_fuerte"] = sesgos_por_periodo["es_fuerte"].apply(_coerce_bool)
    sesgos_por_periodo["es_debil"] = sesgos_por_periodo["es_debil"].apply(_coerce_bool)
    sesgos_por_periodo["tiene_datos"] = sesgos_por_periodo["tiene_datos"].apply(_coerce_bool)

    grouped = sesgos_por_periodo.groupby(key_cols)
    rows: List[Dict[str, object]] = []
    for key, grp in grouped:
        periodo_con_datos = [p for p in grp.loc[grp["tiene_datos"], "periodo"].tolist() if isinstance(p, str)]
        periodos_fuertes = [p for p in grp.loc[grp["es_fuerte"], "periodo"].tolist() if isinstance(p, str)]
        periodos_debiles = [p for p in grp.loc[grp["es_debil"], "periodo"].tolist() if isinstance(p, str)]
        comentario = ""
        if periodos_fuertes:
            comentario = f"Fuerte en periodos: {','.join(periodos_fuertes)}"
        elif periodos_debiles:
            comentario = f"Debil en periodos: {','.join(periodos_debiles)}"

        row_dict = {
            "tipo_relacion": key[0],
            "numero_base": key[1],
            "numero_destino": key[2],
            "pos_origen": key[3],
            "pos_destino": key[4],
            "lag": key[5],
            "Periodos_Con_Datos": ",".join(periodo_con_datos),
            "Periodos_Fuertes": ",".join(periodos_fuertes),
            "Periodos_Debiles": ",".join(periodos_debiles),
            "comentario_periodos": comentario,
        }
        rows.append(row_dict)

    period_summary = pd.DataFrame(rows)
    merged = sesgos_core.merge(period_summary, on=key_cols, how="left")
    return merged


def build_regla_condicional(row: pd.Series) -> str:
    base = _safe_int(row.get("numero_base"), 0)
    dest = _safe_int(row.get("numero_destino"), base)
    lag = _safe_int(row.get("lag"), 0)
    clasif = row.get("clasificacion_fase2_5", "")
    mean_delta = row.get("mean_delta_rel_periods")
    stab = row.get("stability_score")
    periodos_fuertes = row.get("Periodos_Fuertes") or ""

    if row.get("tipo_relacion") == "numero":
        target_desc = f"del mismo número {dest:02d}"
    elif row.get("tipo_relacion") == "espejo":
        target_desc = f"de su espejo {dest:02d}"
    elif row.get("tipo_relacion") == "consecutivo_+1":
        target_desc = f"de su consecutivo superior {dest:02d}"
    elif row.get("tipo_relacion") == "consecutivo_-1":
        target_desc = f"de su consecutivo inferior {dest:02d}"
    else:
        target_desc = f"del número {dest:02d}"

    regla = (
        f"SI el número {base:02d} salió en la posición {row.get('pos_origen')} hace {lag} días, "
        f"ENTONCES aumentar el peso estructural {target_desc} "
        f"(sesgo {clasif}, delta_rel_media≈{mean_delta:.2f} si aplica, estabilidad≈{stab:.2f})."
    )
    if periodos_fuertes:
        regla += f" Activo especialmente en periodos: {periodos_fuertes}."
    return regla


def build_activadores_df(core_path: Path = DEFAULT_INPUT_CORE, periodos_path: Path = DEFAULT_INPUT_PERIODOS) -> pd.DataFrame:
    def _resolve(path: Path, fallbacks: List[str]) -> Path:
        if path.exists():
            return path
        for name in fallbacks:
            alt = path.with_suffix(name)
            if alt.exists():
                return alt
        raise FileNotFoundError(f"No se encontro archivo requerido: {path} ni alternativas {fallbacks}")

    core_file = core_path if core_path.exists() else _resolve(core_path, [".csv", ".parquet"])
    period_file = periodos_path if periodos_path.exists() else _resolve(periodos_path, [".csv", ".parquet"])

    sesgos_core = _read_table(core_file)
    sesgos_periodos = _read_table(period_file)

    # Normalizar claves mínimas
    key_cols = ["tipo_relacion", "numero_base", "numero_destino", "pos_origen", "pos_destino", "lag"]
    for col in key_cols:
        if col not in sesgos_core.columns:
            raise ValueError(f"Falta la columna {col} en sesgos_fase2_5_core_y_periodicos.csv")
    sesgos_core["numero_base"] = sesgos_core["numero_base"].apply(_safe_int)
    sesgos_core["numero_destino"] = sesgos_core["numero_destino"].apply(_safe_int)
    sesgos_core["pos_origen"] = sesgos_core["pos_origen"].apply(_normalize_pos)
    sesgos_core["pos_destino"] = sesgos_core["pos_destino"].apply(_normalize_pos)
    sesgos_core["lag"] = sesgos_core["lag"].apply(_safe_int)

    enriched = build_period_summary(sesgos_core, sesgos_periodos)

    enriched["Peso_Bruto"] = enriched.apply(compute_raw_weight, axis=1)
    enriched["Peso_Normalizado"] = normalize_weights(enriched, "Peso_Bruto")
    enriched["Regla_Condicional"] = enriched.apply(build_regla_condicional, axis=1)

    df_out = enriched.rename(
        columns={
            "numero_destino": "NumeroObjetivo",
            "numero_base": "NumeroCondicionante",
            "tipo_relacion": "TipoRelacion",
            "lag": "Lag",
            "pos_origen": "PosOrigen",
            "pos_destino": "PosDestino",
            "clasificacion_fase2_5": "Clasificacion_Fase2_5",
            "mean_delta_rel_periods": "Mean_DeltaRel",
            "min_delta_rel_periods": "Min_DeltaRel",
            "max_delta_rel_periods": "Max_DeltaRel",
            "stability_score": "Stability_Score",
        }
    )

    return df_out[
        [
            "NumeroObjetivo",
            "NumeroCondicionante",
            "TipoRelacion",
            "Lag",
            "PosOrigen",
            "PosDestino",
            "Clasificacion_Fase2_5",
            "Mean_DeltaRel",
            "Min_DeltaRel",
            "Max_DeltaRel",
            "Stability_Score",
            "Periodos_Con_Datos",
            "Periodos_Fuertes",
            "Periodos_Debiles",
            "Peso_Bruto",
            "Peso_Normalizado",
            "Regla_Condicional",
            "comentario",
            "comentario_periodos",
        ]
    ]


def _export_outputs(df_out: pd.DataFrame, output_dir: Path, fmt: str = "parquet") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    motor_cols = [
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

    do_parquet = fmt in {"parquet", "both"}
    do_csv = fmt in {"csv", "both"}

    if do_parquet:
        df_out.to_parquet(output_dir / "activadores_dinamicos_fase3_raw.parquet", index=False, engine=PARQUET_ENGINE)
        df_out[motor_cols].to_parquet(
            output_dir / "activadores_dinamicos_fase3_para_motor.parquet", index=False, engine=PARQUET_ENGINE
        )
        df_out[df_out["Clasificacion_Fase2_5"].isin(["core_global", "periodico"])][motor_cols].to_parquet(
            output_dir / "activadores_dinamicos_fase3_core_y_periodicos.parquet", index=False, engine=PARQUET_ENGINE
        )

    if do_csv:
        df_out.to_csv(output_dir / "activadores_dinamicos_fase3_raw.csv", index=False)
        df_out[motor_cols].to_csv(output_dir / "activadores_dinamicos_fase3_para_motor.csv", index=False)
        df_out[df_out["Clasificacion_Fase2_5"].isin(["core_global", "periodico"])][motor_cols].to_csv(
            output_dir / "activadores_dinamicos_fase3_core_y_periodicos.csv", index=False
        )


def run_fase3_activadores(
    core_path: Path = DEFAULT_INPUT_CORE,
    periodos_path: Path = DEFAULT_INPUT_PERIODOS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fmt: str = "parquet",
) -> None:
    """
    Orquesta la Fase 3:
    - Carga sesgos core/periodicos de Fase 2.5.
    - Enriquecer con información por periodo.
    - Calcula pesos.
    - Genera DataFrames finales de activadores.
    - Exporta los CSV de salida.
    """
    LOGGER.info("Iniciando Fase 3 con core=%s y periodos=%s", core_path, periodos_path)
    df_out = build_activadores_df(core_path, periodos_path)
    _export_outputs(df_out, output_dir, fmt=fmt)
    LOGGER.info("Activadores generados en %s (formato=%s)", output_dir, fmt)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fase 3: Generación de activadores dinámicos estructurales.")
    parser.add_argument("--core-path", default=str(DEFAULT_INPUT_CORE), help="Ruta al CSV core_y_periodicos de Fase 2.5")
    parser.add_argument("--periodos-path", default=str(DEFAULT_INPUT_PERIODOS), help="Ruta al CSV por periodo de Fase 2.5")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directorio de salida para los CSV de activadores")
    parser.add_argument(
        "--format",
        default="parquet",
        choices=["parquet", "csv", "both"],
        help="Formato de salida (parquet por defecto, csv como cortesía, both para ambos).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        run_fase3_activadores(
            Path(args.core_path),
            Path(args.periodos_path),
            Path(args.output_dir),
            fmt=str(args.format),
        )
    except Exception as exc:
        LOGGER.error("Fase 3 fallo: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
