"""Fase 2.H - Hazard/recencia: analiza probabilidad de aparición en función de días desde último hit.

Entradas:
- eventos_numericos (fecha, numero, posicion) filtrable por fechas.

Salidas (Parquet+CSV, sin truncar):
- hazard_global_resumen: métricas por bin de recencia (oportunidades, hits, h_hat, delta_rel, z, p_val, estabilidad).
- hazard_numero_resumen: métricas por número+bin (idem, con estabilidad).
- (opcional) hazard_opportunities: dataset oportunidades/hits por número/fecha/recencia/bin para depuración.

Nota: Diseñada para usarse en pipelines OOS (Train); subventanas internas para estabilidad.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import math

LOGGER = logging.getLogger("audit.hazard_recencia")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "eventos_numericos.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "audit" / "hazard"
PARQUET_ENGINE = "pyarrow"

# Probabilidad base bajo aleatoriedad (3 números de 100)
P0 = 3 / 100

# Config binning
MAX_RECENCIA = 90
DEFAULT_BINS = [
    (1, 5),
    (6, 10),
    (11, 20),
    (21, 30),
    (31, 45),
    (46, 60),
    (61, MAX_RECENCIA),
]

# Umbrales globales
MIN_OPORTUNIDADES_GLOBAL = 3000
DELTA_REL_GLOBAL_STRONG = 0.10
DELTA_REL_GLOBAL_EXT = 0.05
ALPHA_GLOBAL = 0.01
ALPHA_SUBVENTANA_DEFAULT = 0.05

# Umbrales por número
MIN_OPORTUNIDADES_NUMERO = 300
DELTA_REL_NUMERO_STRONG = 0.50
DELTA_REL_NUMERO_MIN = 0.30
ALPHA_NUMERO = 0.005

# Estabilidad (configurable via CLI)
SUBVENTANAS_TRAIN = [
    (date(2011, 10, 19), date(2014, 12, 31)),
    (date(2015, 1, 1), date(2018, 12, 31)),
    (date(2019, 1, 1), date(2019, 12, 31)),
]
MIN_OPORTUNIDADES_SUBVENTANA = 200


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class HazardMetrics:
    oportunidades: int
    hits: int
    h_hat: float
    delta_rel: float
    z: float
    p_val: float


def _read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró eventos en {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    rename_map = {}
    for col in df.columns:
        token = str(col).strip().lower()
        if token == "date":
            rename_map[col] = "fecha"
        elif token == "position":
            rename_map[col] = "posicion"
        elif token == "number":
            rename_map[col] = "numero"
    if rename_map:
        df = df.rename(columns=rename_map)
    missing = [c for c in ["fecha", "posicion", "numero"] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en eventos: {missing}")
    return df


def _filter_dates(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    if start:
        df = df[df["fecha"] >= pd.to_datetime(start)]
    if end:
        df = df[df["fecha"] <= pd.to_datetime(end)]
    return df


def _bin_recencia(r: int, bins: List[Tuple[int, int]], max_recencia: int) -> str:
    for lo, hi in bins:
        if lo <= r <= hi:
            upper = hi if hi < max_recencia else max_recencia
            return f"{lo}-{upper}"
    # fallback: si no cae en ningún bin explícito, etiquetar como el máximo
    return f"{bins[-1][0]}-{max_recencia}"


def _compute_metrics(n_opp: int, n_hits: int) -> HazardMetrics:
    if n_opp <= 0:
        return HazardMetrics(0, 0, 0.0, 0.0, 0.0, 1.0)
    h_hat = n_hits / n_opp
    delta_rel = (h_hat - P0) / P0 if P0 > 0 else 0.0
    mu = n_opp * P0
    sigma = np.sqrt(n_opp * P0 * (1 - P0))
    if sigma <= 0:
        z = 0.0
        p_val = 1.0
    else:
        z = (n_hits - mu) / sigma
        # p_val 1-cola usando aproximación normal sin depender de scipy
        p_val = float(0.5 * math.erfc(z / math.sqrt(2)))
    return HazardMetrics(n_opp, n_hits, h_hat, delta_rel, z, p_val)


def build_opportunities(df: pd.DataFrame, bins: List[Tuple[int, int]], max_recencia: int) -> pd.DataFrame:
    """Construye dataset oportunidades/hits por número y sorteo (fecha) usando recencia real en días."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    numbers = [f"{i:02d}" for i in range(100)]
    # Normaliza formato: long (posicion, numero) o wide (pos1/pos2/pos3, num1/num2/num3)
    draws_by_date: Dict[pd.Timestamp, set] = {}
    if "posicion" in df.columns:
        for fecha, grp in df.groupby("fecha"):
            vals = {str(v).zfill(2) for v in grp["numero"] if str(v).strip().isdigit()}
            draws_by_date[fecha] = {n for n in vals if len(n) == 2}
    else:
        wide_cols = []
        for c in df.columns:
            token = str(c).lower()
            if token.startswith("num") or (token.startswith("pos") and any(ch.isdigit() for ch in token)):
                wide_cols.append(c)
        for _, row in df.iterrows():
            fecha = row["fecha"]
            draws = draws_by_date.setdefault(fecha, set())
            for col in wide_cols:
                if col not in row.index:
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                token = str(val).zfill(2).strip()
                if token.isdigit() and len(token) == 2:
                    draws.add(token)
    sorted_dates = sorted(draws_by_date.keys())
    last_seen = {n: None for n in numbers}
    records: List[Dict[str, object]] = []
    for idx, fecha in enumerate(sorted_dates):
        drawn = draws_by_date.get(fecha, set())
        for n in numbers:
            ls = last_seen[n]
            if ls is None:
                if n in drawn:
                    last_seen[n] = idx
                continue
            rec = idx - ls
            rec = min(rec, max_recencia)
            bin_id = _bin_recencia(rec, bins, max_recencia)
            hit = 1 if n in drawn else 0
            records.append(
                {
                    "fecha": fecha,
                    "t_index": idx,
                    "numero": n,
                    "recencia": rec,
                    "recencia_bin": bin_id,
                    "hit": hit,
                }
            )
            if hit:
                last_seen[n] = idx
    return pd.DataFrame(records)


def _stability(
    subwindows: List[Tuple[date, date]],
    opp_df: pd.DataFrame,
    by_number: bool,
    min_opp_subwindow: int,
    alpha_subwindow: float,
) -> Dict[Tuple, Dict[str, object]]:
    results: Dict[Tuple, Dict[str, object]] = {}
    key_cols = ["recencia_bin"] + (["numero"] if by_number else [])
    groups = opp_df.groupby(key_cols)
    for key, grp in groups:
        opp_total = len(grp)
        hits_total = int(grp["hit"].sum())
        m = _compute_metrics(opp_total, hits_total)
        stability_scores = []
        sign_ok = True
        sub_delta_rel = {}
        sub_p_val = {}
        sub_opp_counts: Dict[str, int] = {}
        for sub in subwindows:
            mask = (grp["fecha"].dt.date >= sub[0]) & (grp["fecha"].dt.date <= sub[1])
            sub_df = grp.loc[mask]
            if len(sub_df) < min_opp_subwindow:
                continue
            sub_m = _compute_metrics(len(sub_df), int(sub_df["hit"].sum()))
            stability_scores.append(sub_m.delta_rel > 0 and sub_m.p_val <= alpha_subwindow)
            if sub_m.delta_rel < -1e-6:
                sign_ok = False
            label = f"{sub[0]}_{sub[1]}"
            sub_delta_rel[label] = sub_m.delta_rel
            sub_p_val[label] = sub_m.p_val
            sub_opp_counts[label] = len(sub_df)
        stability_score = float(np.mean(stability_scores)) if stability_scores else 0.0
        row = {
            "oportunidades": opp_total,
            "hits": hits_total,
            "h_hat": m.h_hat,
            "delta_rel": m.delta_rel,
            "z": m.z,
            "p_val": m.p_val,
            "stability_score": stability_score,
            "signos_consistentes": sign_ok,
            "delta_rel_sub": json.dumps(sub_delta_rel),
            "p_val_sub": json.dumps(sub_p_val),
            "oportunidades_sub": json.dumps(sub_opp_counts),
        }
        # columnas explícitas por subventana (NaN si no aplica)
        for lbl, val in sub_delta_rel.items():
            row[f"delta_rel_{lbl}"] = val
        for lbl, val in sub_p_val.items():
            row[f"p_val_{lbl}"] = val
        for lbl, val in sub_opp_counts.items():
            row[f"oportunidades_{lbl}"] = val
        if by_number:
            rec_bin, num = key
            row["recencia_bin"] = rec_bin
            row["numero"] = num
        else:
            rec_bin = key if isinstance(key, str) else key[0]
            row["recencia_bin"] = rec_bin
        results[key] = row
    return results


def _classify_global(row: Dict[str, object], min_opp: int, delta_strong: float, delta_ext: float, alpha: float) -> str:
    if row["oportunidades"] < min_opp:
        return "ninguno"
    if row["p_val"] <= alpha and row["delta_rel"] >= delta_strong and row["stability_score"] >= 0.6 and row["signos_consistentes"]:
        return "hazard_core_global"
    if (
        row["p_val"] <= alpha
        and row["delta_rel"] >= delta_ext
        and row["stability_score"] >= 0.4
        and row["signos_consistentes"]
    ):
        return "hazard_extended_global"
    return "ninguno"


def _classify_numero(
    row: Dict[str, object], min_opp: int, delta_strong: float, delta_min: float, alpha: float
) -> str:
    if row["oportunidades"] < min_opp:
        return "ninguno"
    if (
        row["p_val"] <= alpha
        and row["delta_rel"] >= delta_strong
        and row["stability_score"] >= 0.6
        and row["signos_consistentes"]
    ):
        return "hazard_numero_core"
    if (
        row["p_val"] <= alpha
        and row["delta_rel"] >= delta_min
        and row["stability_score"] >= 0.4
        and row["signos_consistentes"]
    ):
        return "hazard_numero_periodico"
    return "ninguno"


def run_hazard(
    events_path: Path,
    output_dir: Path,
    start_date: Optional[str],
    end_date: Optional[str],
    bins: List[Tuple[int, int]],
    max_recencia: int,
    include_opportunities: bool,
    min_opp_global: int,
    delta_global_strong: float,
    delta_global_ext: float,
    alpha_global: float,
    min_opp_num: int,
    delta_num_strong: float,
    delta_num_min: float,
    alpha_num: float,
    subwindows: List[Tuple[date, date]],
    min_opp_subwindow: int,
    alpha_subwindow: float,
) -> None:
    events = _read_events(events_path)
    events = _filter_dates(events, start_date, end_date)
    if events.empty:
        raise ValueError("No hay eventos en el rango especificado.")
    opp_df = build_opportunities(events, bins, max_recencia)
    if opp_df.empty:
        LOGGER.warning("Dataset de oportunidades/hits quedó vacío; revisa rango/bins.")
    # Global
    global_stats = _stability(
        subwindows, opp_df, by_number=False, min_opp_subwindow=min_opp_subwindow, alpha_subwindow=alpha_subwindow
    )
    global_rows = []
    for rec_bin, row in global_stats.items():
        # extrae recencia_min/max del label
        try:
            lo, hi = str(rec_bin).split("-")
            row["recencia_min"] = int(lo)
            row["recencia_max"] = None if hi == "MAX" else int(hi)
        except Exception:
            row["recencia_min"] = None
            row["recencia_max"] = None
        row["clasificacion_hazard"] = _classify_global(
            row, min_opp_global, delta_global_strong, delta_global_ext, alpha_global
        )
        global_rows.append(row)
    global_df = pd.DataFrame(global_rows)
    # Numero
    num_stats = _stability(
        subwindows, opp_df, by_number=True, min_opp_subwindow=min_opp_subwindow, alpha_subwindow=alpha_subwindow
    )
    num_rows = []
    for _, row in num_stats.items():
        try:
            lo, hi = str(row["recencia_bin"]).split("-")
            row["recencia_min"] = int(lo)
            row["recencia_max"] = None if hi == "MAX" else int(hi)
        except Exception:
            row["recencia_min"] = None
            row["recencia_max"] = None
        row["clasificacion_hazard_numero"] = _classify_numero(
            row, min_opp_num, delta_num_strong, delta_num_min, alpha_num
        )
        num_rows.append(row)
    num_df = pd.DataFrame(num_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_df.to_parquet(output_dir / "hazard_global_resumen.parquet", index=False, engine=PARQUET_ENGINE)
    global_df.to_csv(output_dir / "hazard_global_resumen.csv", index=False)
    num_df.to_parquet(output_dir / "hazard_numero_resumen.parquet", index=False, engine=PARQUET_ENGINE)
    num_df.to_csv(output_dir / "hazard_numero_resumen.csv", index=False)
    if include_opportunities:
        opp_df.to_parquet(output_dir / "hazard_opportunities.parquet", index=False, engine=PARQUET_ENGINE)
        opp_df.to_csv(output_dir / "hazard_opportunities.csv", index=False)
    LOGGER.info("Fase 2.H completada en %s", output_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fase 2.H - Hazard/recencia sobre eventos.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Ruta a eventos_numericos (csv/parquet).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directorio de salida.")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD (opcional).")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD (opcional).")
    parser.add_argument("--max-recencia", type=int, default=MAX_RECENCIA, help="Recencia máxima (clip).")
    parser.add_argument("--include-opportunities", action="store_true", help="Guardar dataset oportunidades/hits.")
    parser.add_argument(
        "--bins",
        default=None,
        help="Bins de recencia como '1-5,6-10,11-20,21-30,31-45,46-60,61-90'.",
    )
    parser.add_argument("--min-opp-global", type=int, default=MIN_OPORTUNIDADES_GLOBAL, help="Mínimo oportunidades global.")
    parser.add_argument("--delta-global-strong", type=float, default=DELTA_REL_GLOBAL_STRONG, help="Delta rel fuerte global.")
    parser.add_argument("--delta-global-ext", type=float, default=DELTA_REL_GLOBAL_EXT, help="Delta rel extendida global.")
    parser.add_argument("--alpha-global", type=float, default=ALPHA_GLOBAL, help="Alpha p-val global.")
    parser.add_argument("--min-opp-num", type=int, default=MIN_OPORTUNIDADES_NUMERO, help="Mínimo oportunidades por número.")
    parser.add_argument("--delta-num-strong", type=float, default=DELTA_REL_NUMERO_STRONG, help="Delta rel fuerte número.")
    parser.add_argument("--delta-num-min", type=float, default=DELTA_REL_NUMERO_MIN, help="Delta rel mínima número.")
    parser.add_argument("--alpha-num", type=float, default=ALPHA_NUMERO, help="Alpha p-val por número.")
    parser.add_argument(
        "--subwindows",
        default=None,
        help="Subventanas para estabilidad, formato 'YYYY-MM-DD:YYYY-MM-DD;YYYY-MM-DD:YYYY-MM-DD'.",
    )
    parser.add_argument(
        "--min-opp-subwindow",
        type=int,
        default=MIN_OPORTUNIDADES_SUBVENTANA,
        help="Mínimo oportunidades por subventana para estabilidad.",
    )
    parser.add_argument(
        "--alpha-subwindow",
        type=float,
        default=ALPHA_SUBVENTANA_DEFAULT,
        help="Alpha para estabilidad en subventanas (p-val 1-cola).",
    )
    parser.add_argument("--verbose", action="store_true", help="Log verboso.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    # parse bins
    bins = DEFAULT_BINS
    if args.bins:
        bins = []
        for token in args.bins.split(","):
            token = token.strip()
            if not token:
                continue
            lo, hi = token.split("-")
            bins.append((int(lo), int(hi)))
    # parse subwindows
    subwindows = SUBVENTANAS_TRAIN
    if args.subwindows:
        subwindows = []
        for seg in args.subwindows.split(";"):
            seg = seg.strip()
            if not seg:
                continue
            lo, hi = seg.split(":")
            subwindows.append((datetime.strptime(lo, "%Y-%m-%d").date(), datetime.strptime(hi, "%Y-%m-%d").date()))
    run_hazard(
        events_path=Path(args.input),
        output_dir=Path(args.output_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        bins=bins,
        max_recencia=args.max_recencia,
        include_opportunities=args.include_opportunities,
        min_opp_global=args.min_opp_global,
        delta_global_strong=args.delta_global_strong,
        delta_global_ext=args.delta_global_ext,
        alpha_global=args.alpha_global,
        min_opp_num=args.min_opp_num,
        delta_num_strong=args.delta_num_strong,
        delta_num_min=args.delta_num_min,
        alpha_num=args.alpha_num,
        subwindows=subwindows,
        min_opp_subwindow=args.min_opp_subwindow,
        alpha_subwindow=args.alpha_subwindow,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
