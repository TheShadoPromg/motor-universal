"""Fase 2.5 - Estabilidad de sesgos estructurales por periodo.

Consume salidas de Fase 2, evalúa delta_rel y estabilidad por bloques
temporales, clasifica en core/periodico/extendido y exporta tablas
para Fase 3. No recalcula transiciones; solo resume y etiqueta.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("audit.estructural_fase2_5")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "audit" / "estructural"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "audit" / "estructural_fase2_5"
PARQUET_ENGINE = "pyarrow"

PERIODOS = ["2011_2014", "2015_2018", "2019_2022", "2023_2025"]
MIN_OPORTUNIDADES_PERIOD = 30
ALPHA_PERIOD = 0.01
DELTA_REL_STRONG = 0.30
DELTA_REL_WEAK = 0.15


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo requerido: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_int(val, default: Optional[int] = None) -> Optional[int]:
    try:
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def _normalize_pos(val) -> str | int:
    if pd.isna(val):
        return "ANY"
    if str(val).strip().upper() == "ANY":
        return "ANY"
    try:
        return int(val)
    except Exception:
        return "ANY"


def _compute_numero_destino(tipo_relacion: str, numero_base: int) -> Optional[int]:
    if tipo_relacion == "numero":
        return numero_base
    if tipo_relacion == "espejo":
        return 99 - numero_base
    if tipo_relacion == "consecutivo_+1":
        return (numero_base + 1) % 100
    if tipo_relacion == "consecutivo_-1":
        return (numero_base - 1) % 100
    return None


def load_sesgos_resumen(path: Path) -> pd.DataFrame:
    df = _read_table(path)
    # Normalizar nombres esperados
    rename_map = {
        "numero": "numero_base",
        "numero_objetivo": "numero_destino",
        "numero_destino": "numero_destino",
        "pos_origen": "pos_origen",
        "pos_destino": "pos_destino",
        "lag": "lag",
    }
    for src, dest in rename_map.items():
        if src in df.columns and dest not in df.columns:
            df = df.rename(columns={src: dest})

    df["numero_base"] = df["numero_base"].apply(_safe_int)
    if "numero_destino" not in df.columns:
        df["numero_destino"] = df.apply(
            lambda row: _compute_numero_destino(str(row["tipo_relacion"]), _safe_int(row["numero_base"], 0)),
            axis=1,
        )
    else:
        df["numero_destino"] = df["numero_destino"].apply(_safe_int)

    # Determinar posiciones (si no existen se asume ANY->ANY)
    df["pos_origen"] = df.get("pos_origen", pd.Series(["ANY"] * len(df))).apply(_normalize_pos)
    df["pos_destino"] = df.get("pos_destino", pd.Series(["ANY"] * len(df))).apply(_normalize_pos)

    # Resolver lag (tomar primer valor de la lista de lags si viene separado por coma)
    if "lag" not in df.columns:
        df["lag"] = (
            df.get("lags", pd.Series([np.nan] * len(df)))
            .astype(str)
            .str.split(",")
            .apply(lambda parts: _safe_int(parts[0], default=None))
        )
    df["lag"] = df["lag"].apply(_safe_int)

    # Métricas globales opcionales
    for col in ["max_delta_rel", "max_z_score", "min_p_value", "n_oportunidades_total", "clasificacion"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df[
        [
            "tipo_relacion",
            "numero_base",
            "numero_destino",
            "pos_origen",
            "pos_destino",
            "lag",
            "max_delta_rel",
            "max_z_score",
            "min_p_value",
            "n_oportunidades_total",
            "clasificacion",
        ]
    ]


def _normalize_transiciones(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "numero" in df.columns:
        df = df.rename(columns={"numero": "numero_base"})
    if "numero_objetivo" in df.columns and "numero_destino" not in df.columns:
        df = df.rename(columns={"numero_objetivo": "numero_destino"})
    df["numero_base"] = df["numero_base"].apply(_safe_int)
    df["numero_destino"] = df["numero_destino"].apply(_safe_int)
    df["pos_origen"] = df["pos_origen"].apply(_normalize_pos)
    df["pos_destino"] = df["pos_destino"].apply(_normalize_pos)
    df["lag"] = df["lag"].apply(_safe_int)
    return df


def load_transiciones(path_dir: Path) -> Dict[str, pd.DataFrame]:
    # Admitimos parquet o csv; preferimos parquet si existe.
    candidates = {
        "numero_any_any": ["transiciones_numero_anypos_anypos.parquet", "transiciones_numero_anypos_anypos.csv"],
        "numero_any_pos": ["transiciones_numero_anypos_pos.parquet", "transiciones_numero_anypos_pos.csv"],
        "numero_pos_pos": ["transiciones_numero_pos_pos.parquet", "transiciones_numero_pos_pos.csv"],
        "espejo_any_any": ["transiciones_espejo_anypos_anypos.parquet", "transiciones_espejo_anypos_anypos.csv"],
        "espejo_pos_pos": ["transiciones_espejo_pos_pos.parquet", "transiciones_espejo_pos_pos.csv"],
        "consec_any_any": ["transiciones_consecutivo_anypos_anypos.parquet", "transiciones_consecutivo_anypos_anypos.csv"],
        "consec_pos_pos": ["transiciones_consecutivo_pos_pos.parquet", "transiciones_consecutivo_pos_pos.csv"],
    }
    result: Dict[str, pd.DataFrame] = {}
    for key, names in candidates.items():
        file_path = next((path_dir / name for name in names if (path_dir / name).exists()), None)
        if file_path is None:
            LOGGER.warning("No se encontro archivo de transiciones para %s; se usara DataFrame vacio.", key)
            result[key] = pd.DataFrame()
            continue
        df = _read_table(file_path)
        result[key] = _normalize_transiciones(df)
    return result


def _select_table_key(tipo_relacion: str, pos_origen, pos_destino) -> str:
    if tipo_relacion == "numero":
        if pos_origen == "ANY" and pos_destino == "ANY":
            return "numero_any_any"
        if pos_origen == "ANY":
            return "numero_any_pos"
        return "numero_pos_pos"
    if tipo_relacion == "espejo":
        if pos_origen == "ANY" and pos_destino == "ANY":
            return "espejo_any_any"
        return "espejo_pos_pos"
    if tipo_relacion.startswith("consecutivo"):
        if pos_origen == "ANY" and pos_destino == "ANY":
            return "consec_any_any"
        return "consec_pos_pos"
    return ""


@dataclass
class PeriodMetrics:
    periodo: str
    n_oportunidades_period: int
    n_exitos_period: int
    p_empirica_period: float
    delta_rel_period: float
    p_value_period: float
    tiene_datos: bool
    es_fuerte: bool
    es_debil: bool


def get_period_metrics_for_sesgo(sesgo_row: pd.Series, transiciones_dict: Dict[str, pd.DataFrame]) -> List[PeriodMetrics]:
    tipo_relacion = sesgo_row["tipo_relacion"]
    numero_base = _safe_int(sesgo_row["numero_base"], 0)
    numero_destino = _safe_int(sesgo_row["numero_destino"], numero_base)
    pos_origen = _normalize_pos(sesgo_row["pos_origen"])
    pos_destino = _normalize_pos(sesgo_row["pos_destino"])
    lag = _safe_int(sesgo_row["lag"], None)

    key = _select_table_key(tipo_relacion, pos_origen, pos_destino)
    df = transiciones_dict.get(key, pd.DataFrame())

    metrics: List[PeriodMetrics] = []
    for periodo in PERIODOS:
        if df.empty or lag is None:
            metrics.append(
                PeriodMetrics(
                    periodo=periodo,
                    n_oportunidades_period=0,
                    n_exitos_period=0,
                    p_empirica_period=float("nan"),
                    delta_rel_period=float("nan"),
                    p_value_period=float("nan"),
                    tiene_datos=False,
                    es_fuerte=False,
                    es_debil=False,
                )
            )
            continue
        subset = df[
            (df["segmento_tipo"] == "PERIODO")
            & (df["segmento_valor"] == periodo)
            & (df["tipo_relacion"] == tipo_relacion)
            & (df["numero_base"] == numero_base)
            & (df["numero_destino"] == numero_destino)
            & (df["pos_origen"] == pos_origen)
            & (df["pos_destino"] == pos_destino)
            & (df["lag"] == lag)
        ]
        if subset.empty:
            metrics.append(
                PeriodMetrics(
                    periodo=periodo,
                    n_oportunidades_period=0,
                    n_exitos_period=0,
                    p_empirica_period=float("nan"),
                    delta_rel_period=float("nan"),
                    p_value_period=float("nan"),
                    tiene_datos=False,
                    es_fuerte=False,
                    es_debil=False,
                )
            )
            continue

        row = subset.iloc[0]
        n_opps = _safe_int(row.get("n_oportunidades"), 0)
        delta_rel = float(row.get("delta_rel", float("nan")))
        p_emp = float(row.get("p_empirica", float("nan")))
        p_val = float(row.get("p_value", float("nan")))

        tiene_datos = n_opps >= MIN_OPORTUNIDADES_PERIOD
        es_fuerte = bool(tiene_datos and delta_rel >= DELTA_REL_STRONG and p_val <= ALPHA_PERIOD)
        es_debil = bool(
            tiene_datos and DELTA_REL_WEAK <= delta_rel < DELTA_REL_STRONG and p_val <= ALPHA_PERIOD
        )

        metrics.append(
            PeriodMetrics(
                periodo=periodo,
                n_oportunidades_period=n_opps,
                n_exitos_period=_safe_int(row.get("n_exitos"), 0),
                p_empirica_period=p_emp,
                delta_rel_period=delta_rel,
                p_value_period=p_val,
                tiene_datos=tiene_datos,
                es_fuerte=es_fuerte,
                es_debil=es_debil,
            )
        )
    return metrics


def classify_sesgo(period_metrics: List[PeriodMetrics]) -> Dict[str, object]:
    with_data = [m for m in period_metrics if m.tiene_datos]
    deltas = [m.delta_rel_period for m in with_data if not pd.isna(m.delta_rel_period)]

    n_periodos_con_datos = len(with_data)
    n_periodos_fuertes = sum(1 for m in with_data if m.es_fuerte)
    n_periodos_debiles = sum(1 for m in with_data if m.es_debil)

    if deltas:
        mean_delta = float(np.mean(deltas))
        min_delta = float(np.min(deltas))
        max_delta = float(np.max(deltas))
        signos_consistentes = all(d >= 0 for d in deltas) or all(d <= 0 for d in deltas)
    else:
        mean_delta = float("nan")
        min_delta = float("nan")
        max_delta = float("nan")
        signos_consistentes = False

    stability_score = (
        (n_periodos_fuertes + 0.5 * n_periodos_debiles) / n_periodos_con_datos if n_periodos_con_datos > 0 else 0.0
    )

    clasificacion = "descartado"
    if signos_consistentes and n_periodos_con_datos >= 3 and n_periodos_fuertes >= 2 and stability_score >= 0.6:
        clasificacion = "core_global"
    elif signos_consistentes and n_periodos_con_datos >= 2 and (n_periodos_fuertes + n_periodos_debiles) >= 2 and stability_score >= 0.4:
        clasificacion = "extended_global"
    elif n_periodos_fuertes >= 1:
        clasificacion = "periodico"

    return {
        "n_periodos_con_datos": n_periodos_con_datos,
        "n_periodos_fuertes": n_periodos_fuertes,
        "n_periodos_debiles": n_periodos_debiles,
        "mean_delta_rel_periods": mean_delta,
        "min_delta_rel_periods": min_delta,
        "max_delta_rel_periods": max_delta,
        "stability_score": stability_score,
        "clasificacion_fase2_5": clasificacion,
        "comentario": "",
    }


def run_structural_fase2_5(input_dir: Path, output_dir: Path) -> None:
    LOGGER.info("Cargando sesgos de Fase 2 desde %s", input_dir)
    sesgos_path = input_dir / "sesgos_resumen_global_fase2.csv"
    sesgos_df = load_sesgos_resumen(sesgos_path)

    LOGGER.info("Cargando tablas de transiciones de Fase 2")
    transiciones = load_transiciones(input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if sesgos_df.empty:
        LOGGER.warning("Fase 2.5: sesgos vacíos; se generan salidas vacías.")
        core_cols = [
            "tipo_relacion",
            "numero_base",
            "numero_destino",
            "pos_origen",
            "pos_destino",
            "lag",
            "max_delta_rel",
            "max_z_score",
            "min_p_value",
            "n_oportunidades_total",
            "clasificacion_inicial",
            "clasificacion_fase2_5",
            "mean_delta_rel_periods",
            "stability_score",
        ]
        period_cols = [
            "tipo_relacion",
            "numero_base",
            "numero_destino",
            "pos_origen",
            "pos_destino",
            "lag",
            "periodo",
            "n_oportunidades_period",
            "n_exitos_period",
            "p_empirica_period",
            "delta_rel_period",
            "p_value_period",
            "tiene_datos",
            "es_fuerte",
            "es_debil",
        ]
        pd.DataFrame(columns=core_cols).to_parquet(
            output_dir / "sesgos_fase2_5_resumen.parquet", index=False, engine=PARQUET_ENGINE
        )
        pd.DataFrame(columns=period_cols).to_parquet(
            output_dir / "sesgos_fase2_5_por_periodo.parquet", index=False, engine=PARQUET_ENGINE
        )
        pd.DataFrame(columns=core_cols).to_parquet(
            output_dir / "sesgos_fase2_5_core_y_periodicos.parquet", index=False, engine=PARQUET_ENGINE
        )
        pd.DataFrame(columns=core_cols).to_csv(output_dir / "sesgos_fase2_5_resumen.csv", index=False)
        pd.DataFrame(columns=period_cols).to_csv(output_dir / "sesgos_fase2_5_por_periodo.csv", index=False)
        pd.DataFrame(columns=core_cols).to_csv(output_dir / "sesgos_fase2_5_core_y_periodicos.csv", index=False)
        return

    resumen_rows: List[Dict[str, object]] = []
    period_rows: List[Dict[str, object]] = []

    for _, sesgo in sesgos_df.iterrows():
        period_metrics = get_period_metrics_for_sesgo(sesgo, transiciones)
        cls = classify_sesgo(period_metrics)

        for pm in period_metrics:
            period_rows.append(
                {
                    "tipo_relacion": sesgo["tipo_relacion"],
                    "numero_base": sesgo["numero_base"],
                    "numero_destino": sesgo["numero_destino"],
                    "pos_origen": sesgo["pos_origen"],
                    "pos_destino": sesgo["pos_destino"],
                    "lag": sesgo["lag"],
                    "periodo": pm.periodo,
                    "n_oportunidades_period": pm.n_oportunidades_period,
                    "n_exitos_period": pm.n_exitos_period,
                    "p_empirica_period": pm.p_empirica_period,
                    "delta_rel_period": pm.delta_rel_period,
                    "p_value_period": pm.p_value_period,
                    "tiene_datos": pm.tiene_datos,
                    "es_fuerte": pm.es_fuerte,
                    "es_debil": pm.es_debil,
                }
            )

        resumen_rows.append(
            {
                "tipo_relacion": sesgo["tipo_relacion"],
                "numero_base": sesgo["numero_base"],
                "numero_destino": sesgo["numero_destino"],
                "pos_origen": sesgo["pos_origen"],
                "pos_destino": sesgo["pos_destino"],
                "lag": sesgo["lag"],
                "max_delta_rel": sesgo.get("max_delta_rel"),
                "max_z_score": sesgo.get("max_z_score"),
                "min_p_value": sesgo.get("min_p_value"),
                "n_oportunidades_total": sesgo.get("n_oportunidades_total"),
                "clasificacion_inicial": sesgo.get("clasificacion"),
                **cls,
            }
        )

    resumen_df = pd.DataFrame(resumen_rows)
    period_df = pd.DataFrame(period_rows)

    # Salida parquet por defecto, CSV como cortesía para inspección rápida
    resumen_df.to_parquet(output_dir / "sesgos_fase2_5_resumen.parquet", index=False, engine=PARQUET_ENGINE)
    period_df.to_parquet(output_dir / "sesgos_fase2_5_por_periodo.parquet", index=False, engine=PARQUET_ENGINE)
    core_periodicos = resumen_df[resumen_df["clasificacion_fase2_5"].isin(["core_global", "periodico"])]
    core_periodicos.to_parquet(
        output_dir / "sesgos_fase2_5_core_y_periodicos.parquet", index=False, engine=PARQUET_ENGINE
    )

    resumen_df.to_csv(output_dir / "sesgos_fase2_5_resumen.csv", index=False)
    period_df.to_csv(output_dir / "sesgos_fase2_5_por_periodo.csv", index=False)
    core_periodicos.to_csv(output_dir / "sesgos_fase2_5_core_y_periodicos.csv", index=False)

    LOGGER.info("Fase 2.5 completada. Archivos escritos en %s (parquet + csv cortesía)", output_dir)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fase 2.5 de auditoria estructural: estabilidad por periodo.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directorio con CSVs de Fase 2.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directorio destino para los CSVs de Fase 2.5.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    try:
        run_structural_fase2_5(input_dir=input_dir, output_dir=output_dir)
    except Exception as exc:
        LOGGER.error("La Fase 2.5 fallo: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Fase 2.5 - Estabilidad de sesgos estructurales por periodo.

Consume salidas de Fase 2 y etiqueta sesgos como core/periodico/extendido según
su estabilidad y delta relativo entre periodos. Genera resúmenes por periodo y
core+periódicos para uso en Fase 3 (activadores dinámicos).
"""
