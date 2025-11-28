"""Fase 3.H - Convierte patrones de hazard/recencia en activadores para el motor.

Entradas (parquet/csv):
- hazard_global_resumen: una fila por bin de recencia con métricas y clasificacion_hazard.
- hazard_numero_resumen: filas por numero+bin con clasificacion_hazard_numero.

Salidas (Parquet+CSV):
- activadores_hazard_global.* (uno por número y bin activo, o por bin si se quiere implícito).
- activadores_hazard_numero.* (activadores específicos por número+bin).
- activadores_hazard_para_motor.* (unificados).

Pesos:
- Peso_Bruto = class_factor * log1p(delta_rel_medio) * (0.5 + 0.5 * stability_score)
  donde class_factor = 1.5 (core), 1.2 (extended), 1.0 (periodico).
- Peso_Normalizado: Peso_Bruto / media(Peso_Bruto>0).
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger("audit.hazard_activadores")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "audit" / "hazard"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "activadores" / "hazard"
PARQUET_ENGINE = "pyarrow"

CLASS_FACTOR = {
    "hazard_core_global": 1.5,
    "hazard_extended_global": 1.2,
    "hazard_periodico": 1.0,
    "hazard_numero_core": 1.5,
    "hazard_numero_periodico": 1.0,
}


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _peso_bruto(delta_rel: float, stability: float, clasif: str) -> float:
    mag = max(delta_rel, 0.0)
    mag_factor = math.log1p(mag)
    stab = max(0.0, min(1.0, stability))
    stab_factor = 0.5 + 0.5 * stab
    class_factor = CLASS_FACTOR.get(clasif, 1.0)
    return class_factor * mag_factor * stab_factor


def _normalize_pesos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base = df["Peso_Bruto"].replace([float("inf"), -float("inf")], 0).fillna(0)
    mean_val = base[base > 0].mean() or 1.0
    df["Peso_Normalizado"] = base / mean_val
    return df


def build_global_activadores(global_df: pd.DataFrame) -> pd.DataFrame:
    active = global_df[global_df["clasificacion_hazard"].isin(["hazard_core_global", "hazard_extended_global", "hazard_periodico"])]
    rows: List[dict] = []
    numbers = [f"{i:02d}" for i in range(100)]
    for _, row in active.iterrows():
        rec_bin = str(row["recencia_bin"])
        clasif = row["clasificacion_hazard"]
        peso = _peso_bruto(float(row.get("delta_rel", 0)), float(row.get("stability_score", 0)), clasif)
        rec_min = row.get("recencia_min")
        rec_max = row.get("recencia_max")
        for num in numbers:
            rows.append(
                {
                    "NumeroObjetivo": num,
                    "RecenciaBin": rec_bin,
                    "RecenciaMin": rec_min,
                    "RecenciaMax": rec_max,
                    "TipoPatron": "hazard_global",
                    "Clasificacion_Hazard": clasif,
                    "Delta_Rel_Medio": row.get("delta_rel"),
                    "Stability_Score": row.get("stability_score"),
                    "Peso_Bruto": peso,
                    "Regla_Condicional": f"SI {num} lleva recencia en bin {rec_bin}, aumentar peso (hazard_global, {clasif}, delta_rel={row.get('delta_rel')}, stab={row.get('stability_score')}).",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["NumeroObjetivo", "RecenciaBin", "RecenciaMin", "RecenciaMax", "TipoPatron", "Clasificacion_Hazard", "Delta_Rel_Medio", "Stability_Score", "Peso_Bruto", "Peso_Normalizado", "Regla_Condicional"])
    return pd.DataFrame(rows)


def build_numero_activadores(num_df: pd.DataFrame) -> pd.DataFrame:
    active = num_df[num_df["clasificacion_hazard_numero"].isin(["hazard_numero_core", "hazard_numero_periodico"])]
    rows: List[dict] = []
    for _, row in active.iterrows():
        num = str(row["numero"]).zfill(2)
        rec_bin = str(row["recencia_bin"])
        clasif = row["clasificacion_hazard_numero"]
        peso = _peso_bruto(float(row.get("delta_rel", 0)), float(row.get("stability_score", 0)), clasif)
        rows.append(
            {
                "NumeroObjetivo": num,
                "RecenciaBin": rec_bin,
                "RecenciaMin": row.get("recencia_min"),
                "RecenciaMax": row.get("recencia_max"),
                "TipoPatron": "hazard_numero",
                "Clasificacion_Hazard": clasif,
                "Delta_Rel_Medio": row.get("delta_rel"),
                "Stability_Score": row.get("stability_score"),
                "Peso_Bruto": peso,
                "Regla_Condicional": f"SI el número {num} lleva recencia en bin {rec_bin}, aumentar peso (hazard_numero, {clasif}, delta_rel={row.get('delta_rel')}, stab={row.get('stability_score')}).",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["NumeroObjetivo", "RecenciaBin", "RecenciaMin", "RecenciaMax", "TipoPatron", "Clasificacion_Hazard", "Delta_Rel_Medio", "Stability_Score", "Peso_Bruto", "Peso_Normalizado", "Regla_Condicional"])
    return pd.DataFrame(rows)


def run_hazard_activadores(input_dir: Path, output_dir: Path, fmt: str) -> None:
    global_path = input_dir / "hazard_global_resumen.parquet"
    num_path = input_dir / "hazard_numero_resumen.parquet"
    global_df = _read_table(global_path)
    num_df = _read_table(num_path)
    act_global_raw = build_global_activadores(global_df)
    act_num_raw = build_numero_activadores(num_df)
    if act_global_raw.empty and act_num_raw.empty:
        act_motor = pd.DataFrame(columns=["NumeroObjetivo", "RecenciaBin", "RecenciaMin", "RecenciaMax", "TipoPatron", "Clasificacion_Hazard", "Delta_Rel_Medio", "Stability_Score", "Peso_Bruto", "Peso_Normalizado", "Regla_Condicional"])
        act_global = act_motor.copy()
        act_num = act_motor.copy()
    else:
        frames = [df for df in [act_global_raw, act_num_raw] if not df.empty]
        act_motor = _normalize_pesos(pd.concat(frames, ignore_index=True))
        act_global = act_motor[act_motor["TipoPatron"] == "hazard_global"].copy()
        act_num = act_motor[act_motor["TipoPatron"] == "hazard_numero"].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    do_parquet = fmt in {"parquet", "both"}
    do_csv = fmt in {"csv", "both"}
    if do_parquet:
        act_global.to_parquet(output_dir / "activadores_hazard_global.parquet", index=False, engine=PARQUET_ENGINE)
        act_num.to_parquet(output_dir / "activadores_hazard_numero.parquet", index=False, engine=PARQUET_ENGINE)
        act_motor.to_parquet(output_dir / "activadores_hazard_para_motor.parquet", index=False, engine=PARQUET_ENGINE)
    if do_csv:
        act_global.to_csv(output_dir / "activadores_hazard_global.csv", index=False)
        act_num.to_csv(output_dir / "activadores_hazard_numero.csv", index=False)
        act_motor.to_csv(output_dir / "activadores_hazard_para_motor.csv", index=False)
    LOGGER.info("Fase 3.H completada en %s", output_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fase 3.H: activadores de hazard/recencia.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directorio con hazard_*_resumen de Fase 2.H.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directorio de salida para activadores.")
    parser.add_argument("--format", choices=["parquet", "csv", "both"], default="parquet", help="Formato de salida.")
    parser.add_argument("--verbose", action="store_true", help="Log verboso.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    run_hazard_activadores(Path(args.input_dir), Path(args.output_dir), args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
