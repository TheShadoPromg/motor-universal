"""
Fase 4 - Evaluación predictiva de activadores estructurales (core y periódicos).

- Construye distribuciones diarias sin mirar el futuro (aplica lags de activadores).
- Compara modelos: A (uniforme), B (core), C (core+periódico), con softmax y mezcla opcional.
- Calcula métricas: Hit@K, rank promedio, log-loss (Brier opcional) y lifts vs uniforme.
- Permite grid-search simple sobre beta (temperatura) y lambda (mezcla con uniforme).
- Exporta resúmenes a Parquet/CSV y opcionalmente a Postgres.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas import Timestamp

LOGGER = logging.getLogger("backtesting.phase4")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_PATH = REPO_ROOT / "data" / "raw" / "eventos_numericos.csv"
DEFAULT_ACTIVADORES_PATH = (
    REPO_ROOT / "data" / "activadores" / "activadores_dinamicos_fase3_para_motor.parquet"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "outputs" / "phase4"

# Modelo identifiers
MODEL_UNIFORM = "A_uniforme"
MODEL_CORE = "B_core"
MODEL_FULL = "C_core_periodico"
MODEL_MIX = "D_mezcla"
MODEL_HAZARD = "H_hazard"
MODEL_HAZARD_STRUCT = "H_hazard_struct"


@dataclass(frozen=True)
class Activador:
    numero_obj: int
    numero_cond: int
    lag: int
    peso: float
    clasificacion: str
    pos_origen: Optional[int]  # None means ANY


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def _read_events(db_url: Optional[str], csv_path: Path) -> pd.DataFrame:
    if db_url:
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise SystemExit("sqlalchemy requerido para leer eventos desde DB.") from exc
        engine = create_engine(db_url)
        df = pd.read_sql("SELECT fecha, posicion, numero FROM eventos_numericos", engine)
    else:
        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontró el CSV de eventos en {csv_path}")
        if csv_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(csv_path)
        else:
            df = pd.read_csv(csv_path)
    # Normaliza nombres alternativos
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


def _read_activadores(db_url: Optional[str], path: Path) -> pd.DataFrame:
    if db_url:
        try:
            from sqlalchemy import create_engine, inspect
        except ImportError as exc:
            raise SystemExit("sqlalchemy requerido para leer activadores desde DB.") from exc
        engine = create_engine(db_url)
        insp = inspect(engine)
        if not insp.has_table("activadores_dinamicos_fase3"):
            LOGGER.warning("Tabla activadores_dinamicos_fase3 no encontrada en DB, se usará archivo local.")
        else:
            return pd.read_sql("SELECT * FROM activadores_dinamicos_fase3", engine)
    if not path.exists():
        # intentar csv/parquet alternativos
        if path.with_suffix(".csv").exists():
            path = path.with_suffix(".csv")
        elif path.with_suffix(".parquet").exists():
            path = path.with_suffix(".parquet")
    if not path.exists():
        raise FileNotFoundError(f"No se encontró activadores en {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _normalize_num(val: object) -> int:
    return int(str(val).strip().zfill(2))


def _normalize_pos(val: object) -> Optional[int]:
    if val is None:
        return None
    token = str(val).strip().upper()
    if token in {"", "ANY", "TODAS"}:
        return None
    try:
        pos = int(token)
        return pos if pos in {1, 2, 3} else None
    except Exception:
        return None


def parse_activadores(df: pd.DataFrame) -> List[Activador]:
    required = [
        "NumeroObjetivo",
        "NumeroCondicionante",
        "Lag",
        "Peso_Normalizado",
        "Clasificacion_Fase2_5",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en activadores: {missing}")
    acts: List[Activador] = []
    for _, row in df.iterrows():
        acts.append(
            Activador(
                numero_obj=_normalize_num(row["NumeroObjetivo"]),
                numero_cond=_normalize_num(row["NumeroCondicionante"]),
                lag=int(row["Lag"]),
                peso=float(row["Peso_Normalizado"]),
                clasificacion=str(row["Clasificacion_Fase2_5"]).strip(),
                pos_origen=_normalize_pos(row.get("PosOrigen")),
            )
        )
    return acts


def _build_draws_index(events: pd.DataFrame) -> Dict[date, List[Tuple[int, int]]]:
    """Return mapping fecha -> list of (numero_int, posicion_int)."""
    df = events.copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
    # Posicion puede ser texto, normalizamos a int si posible
    def _pos_to_int(val: object) -> int:
        token = str(val).strip().lower()
        mapping = {"primero": 1, "segundo": 2, "tercero": 3, "1": 1, "2": 2, "3": 3}
        if token in mapping:
            return mapping[token]
        if token.isdigit():
            return int(token)
        return 0

    df["posicion"] = df["posicion"].apply(_pos_to_int)
    index: Dict[date, List[Tuple[int, int]]] = {}
    for fecha, grp in df.groupby("fecha"):
        index[fecha] = [(int(num), int(pos)) for num, pos in zip(grp["numero"], grp["posicion"])]
    return index


def _softmax(scores: np.ndarray, beta: float) -> np.ndarray:
    scaled = scores * beta
    scaled -= scaled.max()  # estabilidad numérica
    exps = np.exp(scaled)
    total = exps.sum()
    if total <= 0 or not math.isfinite(total):
        return np.full_like(scores, 1 / len(scores))
    return exps / total


def _compute_prob_struct(
    acts: Sequence[Activador],
    draw_index: Dict[date, List[Tuple[int, int]]],
    eval_dates: Sequence[date],
    beta: float,
    mix_lambda: float = 1.0,
) -> Dict[date, np.ndarray]:
    probs: Dict[date, np.ndarray] = {}
    max_lag = max((a.lag for a in acts), default=0)
    for current in eval_dates:
        scores = np.zeros(100, dtype=float)
        for act in acts:
            lag_date = current - timedelta(days=act.lag)
            past_draw = draw_index.get(lag_date)
            if not past_draw:
                continue
            # Chequear posición si aplica
            triggered = False
            for num, pos in past_draw:
                if num == act.numero_cond and (act.pos_origen is None or pos == act.pos_origen):
                    triggered = True
                    break
            if triggered:
                scores[act.numero_obj] += act.peso
        if scores.sum() == 0:
            base = np.full(100, 1 / 100)
        else:
            base = _softmax(scores, beta)
        if mix_lambda < 1.0:
            uniform = np.full(100, 1 / 100)
            base = mix_lambda * base + (1 - mix_lambda) * uniform
        probs[current] = base
    return probs


def _compute_prob_hazard(
    hazard_df: pd.DataFrame,
    draw_index: Dict[date, List[Tuple[int, int]]],
    eval_dates: Sequence[date],
    beta: float,
    mix_lambda: float = 1.0,
) -> Dict[date, np.ndarray]:
    """Calcula probabilidades basadas en recencia/hazard usando activadores_hazard_para_motor."""
    # Esperamos columnas: NumeroObjetivo, RecenciaBin, Peso_Normalizado, TipoPatron
    if hazard_df.empty:
        return _compute_uniform(eval_dates)
    hazard_df = hazard_df.copy()
    # Preparar bins -> pesos por numero objetivo (usa propio numero, tanto global como específico)
    weights: Dict[str, Dict[int, float]] = {}
    for _, row in hazard_df.iterrows():
        bin_label = str(row["RecenciaBin"])
        num_obj = int(str(row["NumeroObjetivo"]).zfill(2))
        w = float(row["Peso_Normalizado"])
        weights.setdefault(bin_label, {}).setdefault(num_obj, 0.0)
        weights[bin_label][num_obj] += w

    bin_ranges: List[Tuple[str, int, int]] = []
    for bin_label in hazard_df["RecenciaBin"].unique():
        try:
            lo, hi = bin_label.split("-")
            lo_v = int(lo)
            hi_v = 10**9 if hi == "MAX" else int(hi)
            bin_ranges.append((bin_label, lo_v, hi_v))
        except Exception:
            continue
    bin_ranges.sort(key=lambda x: x[1])

    def bin_recencia(r: int) -> str:
        for label, lo, hi in bin_ranges:
            if lo <= r <= hi:
                return label
        return ""

    prob_map: Dict[date, np.ndarray] = {}
    all_dates = sorted(draw_index.keys())
    idx_map = {d: i for i, d in enumerate(all_dates)}
    numbers = list(range(100))
    last_seen = {n: None for n in numbers}

    for d in all_dates:
        idx = idx_map[d]
        scores = np.zeros(100, dtype=float)
        for n in numbers:
            ls = last_seen[n]
            if ls is None:
                continue
            rec = idx - ls
            bin_id = bin_recencia(rec)
            if bin_id and bin_id in weights:
                w = weights[bin_id].get(n)
                if w:
                    scores[n] += w
        if scores.sum() == 0:
            base = np.full(100, 1 / 100)
        else:
            base = _softmax(scores, beta)
        if mix_lambda < 1.0:
            uniform = np.full(100, 1 / 100)
            base = mix_lambda * base + (1 - mix_lambda) * uniform
        prob_map[d] = base

        drawn = {num for num, _ in draw_index[d]}
        for num in drawn:
            last_seen[num] = idx

    return {d: prob_map[d] for d in eval_dates if d in prob_map}


def _compute_prob_combined_struct_hazard(
    acts_struct: Sequence[Activador],
    hazard_df: pd.DataFrame,
    draw_index: Dict[date, List[Tuple[int, int]]],
    eval_dates: Sequence[date],
    beta_struct: float,
    beta_hazard: float,
    mix_lambda: float,
) -> Dict[date, np.ndarray]:
    """Combina scores estructurales y de hazard antes de softmax."""
    hazard_df = hazard_df.copy()
    weights: Dict[str, Dict[int, float]] = {}
    for _, row in hazard_df.iterrows():
        bin_label = str(row["RecenciaBin"])
        num_obj = int(str(row["NumeroObjetivo"]).zfill(2))
        w = float(row["Peso_Normalizado"])
        weights.setdefault(bin_label, {}).setdefault(num_obj, 0.0)
        weights[bin_label][num_obj] += w
    bin_ranges: List[Tuple[str, int, int]] = []
    for bin_label in hazard_df["RecenciaBin"].unique():
        try:
            lo, hi = bin_label.split("-")
            lo_v = int(lo)
            hi_v = 10**9 if hi == "MAX" else int(hi)
            bin_ranges.append((bin_label, lo_v, hi_v))
        except Exception:
            continue
    bin_ranges.sort(key=lambda x: x[1])

    def bin_recencia(r: int) -> str:
        for label, lo, hi in bin_ranges:
            if lo <= r <= hi:
                return label
        return ""

    numbers = list(range(100))
    all_dates = sorted(draw_index.keys())
    idx_map = {d: i for i, d in enumerate(all_dates)}
    last_seen = {n: None for n in numbers}
    prob_map: Dict[date, np.ndarray] = {}

    # cache struct scores
    acts_by_date: Dict[date, np.ndarray] = _compute_prob_struct(
        acts_struct, draw_index, eval_dates, beta=beta_struct, mix_lambda=1.0
    )
    struct_scores_raw: Dict[date, np.ndarray] = {}
    for d, probs in acts_by_date.items():
        # invert softmax to get scores up to a scale: score = log(prob)
        with np.errstate(divide="ignore"):
            struct_scores_raw[d] = np.log(probs + 1e-12)

    for d in all_dates:
        idx = idx_map[d]
        scores_h = np.zeros(100, dtype=float)
        for n in numbers:
            ls = last_seen[n]
            if ls is None:
                continue
            rec = idx - ls
            bin_id = bin_recencia(rec)
            if bin_id and bin_id in weights:
                w = weights[bin_id].get(n)
                if w:
                    scores_h[n] += w
        scores_struct = struct_scores_raw.get(d, np.zeros(100, dtype=float))
        scores = scores_struct * beta_struct + scores_h * beta_hazard
        if scores.sum() == 0:
            base = np.full(100, 1 / 100)
        else:
            base = _softmax(scores, 1.0)
        if mix_lambda < 1.0:
            uniform = np.full(100, 1 / 100)
            base = mix_lambda * base + (1 - mix_lambda) * uniform
        prob_map[d] = base
        drawn = {num for num, _ in draw_index[d]}
        for num in drawn:
            last_seen[num] = idx
    return {d: prob_map[d] for d in eval_dates if d in prob_map}


def _compute_uniform(eval_dates: Sequence[date]) -> Dict[date, np.ndarray]:
    uniform = np.full(100, 1 / 100)
    return {d: uniform for d in eval_dates}


@dataclass
class Metrics:
    hit_rates: Dict[int, float]
    rank_promedio: float
    log_loss: float
    brier: Optional[float]
    details: Optional[List[Dict[str, object]]] = None


def _expected_random_hit(k: int) -> float:
    # P(al menos un acierto) con 3 ganadores y top-K uniforme sin reposición
    from math import comb

    if k <= 0:
        return 0.0
    return 1 - comb(100 - 3, k) / comb(100, k)


def evaluate_model(
    probs: Dict[date, np.ndarray],
    draw_index: Dict[date, List[Tuple[int, int]]],
    eval_dates: Sequence[date],
    ks: Sequence[int],
    include_brier: bool = False,
    capture_details: bool = False,
    ) -> Metrics:
    total_days = len(eval_dates)
    hit_acc = {k: 0 for k in ks}
    ranks: List[float] = []
    logloss_values: List[float] = []
    brier_accum = 0.0
    details: List[Dict[str, object]] = []

    for day in eval_dates:
        p = probs[day]
        # ranking (desc), ties by numero asc
        order = np.argsort(-p)
        rank = np.empty_like(order)
        rank[order] = np.arange(1, 101)
        drawn_nums = [num for num, _ in draw_index[day]]
        day_detail: Dict[str, object] = {"fecha": day}
        # Hit@K
        for k in ks:
            topk = set(order[:k])
            if any(num in topk for num in drawn_nums):
                hit_acc[k] += 1
                day_detail[f"hit@{k}"] = 1
            else:
                day_detail[f"hit@{k}"] = 0
        # rank promedio de los 3 ganadores
        ranks.append(np.mean([rank[num] for num in drawn_nums]))
        day_detail["rank_promedio"] = float(np.mean([rank[num] for num in drawn_nums]))
        # logloss (toma cada ganador como evento independiente)
        for num in drawn_nums:
            prob = p[num]
            if prob <= 0:
                prob = 1e-12
            logloss_values.append(-math.log(prob))
        day_detail["logloss_sum"] = float(np.sum([-math.log(max(p[num], 1e-12)) for num in drawn_nums]))
        if include_brier:
            y = np.zeros(100, dtype=float)
            for num in drawn_nums:
                y[num] = 1.0
            brier_accum += np.mean((p - y) ** 2)
            day_detail["brier"] = float(np.mean((p - y) ** 2))
        if capture_details:
            # guardar prob y rank de los ganadores
            for idx, num in enumerate(drawn_nums):
                day_detail[f"num_{idx+1}"] = num
                day_detail[f"prob_{idx+1}"] = float(p[num])
                day_detail[f"rank_{idx+1}"] = int(rank[num])
            details.append(day_detail)

    hit_rates = {k: (hit_acc[k] / total_days) if total_days else 0.0 for k in ks}
    rank_prom = float(np.mean(ranks)) if ranks else 0.0
    log_loss = float(np.mean(logloss_values)) if logloss_values else 0.0
    brier = float(brier_accum / total_days) if include_brier and total_days else None
    return Metrics(hit_rates=hit_rates, rank_promedio=rank_prom, log_loss=log_loss, brier=brier, details=details if capture_details else None)


def _filter_dates(draw_index: Dict[date, List[Tuple[int, int]]], start: Optional[str], end: Optional[str]) -> List[date]:
    dates = sorted(draw_index.keys())
    if start:
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        dates = [d for d in dates if d >= start_dt]
    if end:
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()
        dates = [d for d in dates if d <= end_dt]
    return dates


def _segment_periods(
    draw_index: Dict[date, List[Tuple[int, int]]],
    periods: Optional[List[Tuple[str, str, str]]],
) -> Dict[str, List[date]]:
    if not periods:
        return {"ALL": sorted(draw_index.keys())}
    segments: Dict[str, List[date]] = {}
    for label, start, end in periods:
        segments[label] = _filter_dates(draw_index, start, end)
    return segments


def _grid(values: Sequence[str]) -> List[float]:
    result: List[float] = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        result.append(float(v))
    return result


def _summarize(
    period: str,
    model_name: str,
    metrics: Metrics,
    ks: Sequence[int],
    baseline: Metrics,
    beta_used: Optional[float],
    lambda_used: Optional[float],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "Periodo": period,
        "Modelo": model_name,
        "Beta": beta_used,
        "Lambda": lambda_used,
    }
    for k in ks:
        hr = metrics.hit_rates.get(k, 0.0)
        hr_base = baseline.hit_rates.get(k, 0.0)
        row[f"HR@{k}"] = hr
        row[f"Lift_HR@{k}"] = (hr / hr_base) if hr_base > 0 else None
    row["RankPromedio"] = metrics.rank_promedio
    row["Lift_Rank"] = (50.5 / metrics.rank_promedio) if metrics.rank_promedio > 0 else None
    row["LogLoss"] = metrics.log_loss
    row["Lift_LogLoss"] = (baseline.log_loss / metrics.log_loss) if metrics.log_loss > 0 else None
    if metrics.brier is not None:
        row["Brier"] = metrics.brier
    return row


def run_phase4(
    events_db_url: Optional[str],
    events_path: Path,
    activ_db_url: Optional[str],
    activ_path: Path,
    hazard_path: Optional[Path],
    ks: Sequence[int],
    beta_core: float,
    beta_full: float,
    beta_hazard: float,
    beta_hazard_struct: float,
    lambda_core: float,
    lambda_full: float,
    lambda_hazard: float,
    lambda_hazard_struct: float,
    periods: Optional[List[Tuple[str, str, str]]],
    include_brier: bool,
    grid_beta: Optional[Sequence[float]] = None,
    grid_lambda: Optional[Sequence[float]] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_format: str = "parquet",
    output_db_url: Optional[str] = None,
    output_db_table: str = "backtest_phase4_summary",
) -> None:
    events = _read_events(events_db_url, events_path)
    activ_df = _read_activadores(activ_db_url, activ_path)
    draw_index = _build_draws_index(events)
    activadores = parse_activadores(activ_df)

    # Pre-split activadores
    acts_core = [a for a in activadores if a.clasificacion == "core_global"]
    acts_full = activadores  # core + periodico

    # Grids de hiperparámetros (si no se dan, usar valores únicos)
    beta_core_grid = grid_beta if grid_beta else [beta_core]
    beta_full_grid = grid_beta if grid_beta else [beta_full]
    beta_hazard_grid = grid_beta if grid_beta else [beta_hazard]
    beta_hazard_struct_grid = grid_beta if grid_beta else [beta_hazard_struct]
    lambda_core_grid = grid_lambda if grid_lambda else [lambda_core]
    lambda_full_grid = grid_lambda if grid_lambda else [lambda_full]
    lambda_hazard_grid = grid_lambda if grid_lambda else [lambda_hazard]
    lambda_hazard_struct_grid = grid_lambda if grid_lambda else [lambda_hazard_struct]

    segments = _segment_periods(draw_index, periods)
    ks_sorted = sorted(set(int(k) for k in ks))
    hazard_df = pd.read_parquet(hazard_path) if hazard_path and hazard_path.exists() else pd.DataFrame()

    all_rows: List[Dict[str, object]] = []
    all_details: List[Dict[str, object]] = []
    for label, dates in segments.items():
        if not dates:
            LOGGER.warning("Periodo %s sin fechas, se omite.", label)
            continue
        LOGGER.info("Evaluando periodo %s con %s fechas ...", label, len(dates))
        probs_uniform = _compute_uniform(dates)
        m_uniform = evaluate_model(
            probs_uniform, draw_index, dates, ks_sorted, include_brier=include_brier, capture_details=True
        )

        # Modelo B (grid beta/lambda)
        best_core = None
        for b in beta_core_grid:
            for lmb in lambda_core_grid:
                probs_core = _compute_prob_struct(acts_core, draw_index, dates, beta=b, mix_lambda=lmb)
                m_core_tmp = evaluate_model(
                    probs_core, draw_index, dates, ks_sorted, include_brier=include_brier, capture_details=True
                )
                if best_core is None or m_core_tmp.log_loss < best_core[2].log_loss:
                    best_core = (b, lmb, m_core_tmp)
        beta_core_best, lambda_core_best, m_core = best_core

        # Modelo C (grid beta/lambda)
        best_full = None
        for b in beta_full_grid:
            for lmb in lambda_full_grid:
                probs_full = _compute_prob_struct(acts_full, draw_index, dates, beta=b, mix_lambda=lmb)
                m_full_tmp = evaluate_model(
                    probs_full, draw_index, dates, ks_sorted, include_brier=include_brier, capture_details=True
                )
                if best_full is None or m_full_tmp.log_loss < best_full[2].log_loss:
                    best_full = (b, lmb, m_full_tmp)
        beta_full_best, lambda_full_best, m_full = best_full

        # Tabla resumen
        rows = []
        rows.append(_summarize(label, MODEL_UNIFORM, m_uniform, ks_sorted, baseline=m_uniform, beta_used=None, lambda_used=None))
        rows.append(_summarize(label, MODEL_CORE, m_core, ks_sorted, baseline=m_uniform, beta_used=beta_core_best, lambda_used=lambda_core_best))
        rows.append(_summarize(label, MODEL_FULL, m_full, ks_sorted, baseline=m_uniform, beta_used=beta_full_best, lambda_used=lambda_full_best))
        if not hazard_df.empty:
            best_hazard = None
            for b in beta_hazard_grid:
                for lmb in lambda_hazard_grid:
                    probs_h = _compute_prob_hazard(hazard_df, draw_index, dates, beta=b, mix_lambda=lmb)
                    m_h_tmp = evaluate_model(probs_h, draw_index, dates, ks_sorted, include_brier=include_brier)
                    if best_hazard is None or m_h_tmp.log_loss < best_hazard[2].log_loss:
                        best_hazard = (b, lmb, m_h_tmp)
            if best_hazard:
                beta_h_best, lambda_h_best, m_hazard = best_hazard
                rows.append(
                    _summarize(
                        label,
                        MODEL_HAZARD,
                        m_hazard,
                        ks_sorted,
                        baseline=m_uniform,
                        beta_used=beta_h_best,
                        lambda_used=lambda_h_best,
                    )
                )
            # Modelo hazard+estructural (usa acts_full + hazard)
            best_hs = None
            for b in beta_hazard_struct_grid:
                for lmb in lambda_hazard_struct_grid:
                    probs_hs = _compute_prob_combined_struct_hazard(
                        acts_full, hazard_df, draw_index, dates, beta_struct=beta_full_best, beta_hazard=b, mix_lambda=lmb
                    )
                    m_hs_tmp = evaluate_model(probs_hs, draw_index, dates, ks_sorted, include_brier=include_brier)
                    if best_hs is None or m_hs_tmp.log_loss < best_hs[2].log_loss:
                        best_hs = (b, lmb, m_hs_tmp)
            if best_hs:
                beta_hs_best, lambda_hs_best, m_hs = best_hs
                rows.append(
                    _summarize(
                        label,
                        MODEL_HAZARD_STRUCT,
                        m_hs,
                        ks_sorted,
                        baseline=m_uniform,
                        beta_used=beta_hs_best,
                        lambda_used=lambda_hs_best,
                    )
                )
        df = pd.DataFrame(rows)
        print(f"\n=== Resumen periodo: {label} ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(
            f"Mejores hiperparámetros -> Core: beta={beta_core_best}, lambda={lambda_core_best} | "
            f"Full: beta={beta_full_best}, lambda={lambda_full_best}"
        )
        all_rows.extend(rows)
        # Detalle diario para cada modelo (solo mejores)
        if m_uniform.details:
            for d in m_uniform.details:
                d.update({"Periodo": label, "Modelo": MODEL_UNIFORM, "Beta": None, "Lambda": None})
            all_details.extend(m_uniform.details)
        if m_core.details:
            for d in m_core.details:
                d.update({"Periodo": label, "Modelo": MODEL_CORE, "Beta": beta_core_best, "Lambda": lambda_core_best})
            all_details.extend(m_core.details)
        if m_full.details:
            for d in m_full.details:
                d.update({"Periodo": label, "Modelo": MODEL_FULL, "Beta": beta_full_best, "Lambda": lambda_full_best})
            all_details.extend(m_full.details)

    # Persistencia de resultados
    if all_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(all_rows)
        fmt = output_format.lower()
        if fmt in {"parquet", "both"}:
            summary_df.to_parquet(output_dir / "phase4_summary.parquet", index=False)
        if fmt in {"csv", "both"}:
            summary_df.to_csv(output_dir / "phase4_summary.csv", index=False)
        if output_db_url:
            try:
                from sqlalchemy import create_engine
                engine = create_engine(output_db_url)
                summary_df.to_sql(output_db_table, engine, if_exists="append", index=False)
                LOGGER.info("Resumen escrito en DB %s (tabla %s)", output_db_url, output_db_table)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("No se pudo escribir resumen en DB (%s): %s", output_db_url, exc)
        # Detalle diario
        if all_details:
            details_df = pd.DataFrame(all_details)
            if fmt in {"parquet", "both"}:
                details_df.to_parquet(output_dir / "phase4_details.parquet", index=False)
            if fmt in {"csv", "both"}:
                details_df.to_csv(output_dir / "phase4_details.csv", index=False)
            if output_db_url:
                try:
                    from sqlalchemy import create_engine
                    engine = create_engine(output_db_url)
                    details_df.to_sql(f"{output_db_table}_details", engine, if_exists="append", index=False)
                    LOGGER.info(
                        "Detalle diario escrito en DB %s (tabla %s)", output_db_url, f"{output_db_table}_details"
                    )
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("No se pudo escribir detalle en DB (%s): %s", output_db_url, exc)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fase 4: evaluación predictiva de activadores estructurales dinámicos."
    )
    parser.add_argument("--events-db-url", default=None, help="DSN SQLAlchemy para leer eventos_numericos.")
    parser.add_argument(
        "--events-path",
        default=str(DEFAULT_EVENTS_PATH),
        help="CSV de eventos si no se usa DB.",
    )
    parser.add_argument("--activadores-db-url", default=None, help="DSN SQLAlchemy para leer activadores.")
    parser.add_argument(
        "--activadores-path",
        default=str(DEFAULT_ACTIVADORES_PATH),
        help="Parquet/CSV de activadores si no se usa DB.",
    )
    parser.add_argument(
        "--hazard-path",
        default=None,
        help="Parquet/CSV de activadores de hazard (opcional, si se quiere evaluar modelo hazard).",
    )
    parser.add_argument(
        "--ks",
        default="5,10,15,20",
        help="Lista de K para Hit@K, separada por comas.",
    )
    parser.add_argument("--beta-core", type=float, default=1.0, help="Temperatura beta para modelo core.")
    parser.add_argument("--beta-full", type=float, default=1.0, help="Temperatura beta para modelo full.")
    parser.add_argument("--beta-hazard", type=float, default=1.0, help="Temperatura beta para modelo hazard.")
    parser.add_argument("--beta-hazard-struct", type=float, default=1.0, help="Temperatura beta para modelo hazard+struct.")
    parser.add_argument(
        "--lambda-core",
        type=float,
        default=1.0,
        help="Peso de mezcla con uniforme para modelo core (1.0 = no mezcla).",
    )
    parser.add_argument(
        "--lambda-full",
        type=float,
        default=1.0,
        help="Peso de mezcla con uniforme para modelo full (1.0 = no mezcla).",
    )
    parser.add_argument(
        "--lambda-hazard",
        type=float,
        default=1.0,
        help="Peso de mezcla con uniforme para modelo hazard (1.0 = no mezcla).",
    )
    parser.add_argument(
        "--lambda-hazard-struct",
        type=float,
        default=1.0,
        help="Peso de mezcla con uniforme para modelo hazard+struct (1.0 = no mezcla).",
    )
    parser.add_argument(
        "--period",
        action="append",
        nargs=3,
        metavar=("LABEL", "START", "END"),
        help="Periodo a evaluar (label YYYY-MM-DD YYYY-MM-DD). Se puede repetir.",
    )
    parser.add_argument("--include-brier", action="store_true", help="Calcular Brier Score (coste extra).")
    parser.add_argument(
        "--grid-beta",
        default=None,
        help="Lista de beta para grid search (ej. '0.5,1,1.5'); si no se pasa, usa el beta fijo.",
    )
    parser.add_argument(
        "--grid-lambda",
        default=None,
        help="Lista de lambda para grid search (ej. '0.7,0.8,1'); si no se pasa, usa la lambda fija.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directorio donde guardar resúmenes (parquet/csv).",
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv", "both"],
        default="parquet",
        help="Formato de salida de resúmenes.",
    )
    parser.add_argument(
        "--output-db-url",
        default=None,
        help="DSN SQLAlchemy para escribir resumen en Postgres (opcional).",
    )
    parser.add_argument(
        "--output-db-table",
        default="backtest_phase4_summary",
        help="Tabla destino para resumen en DB (si se especifica output-db-url).",
    )
    parser.add_argument("--verbose", action="store_true", help="Log verboso.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    grid_beta = [float(x) for x in args.grid_beta.split(",")] if args.grid_beta else None
    grid_lambda = [float(x) for x in args.grid_lambda.split(",")] if args.grid_lambda else None
    run_phase4(
        events_db_url=args.events_db_url,
        events_path=Path(args.events_path),
        activ_db_url=args.activadores_db_url,
        activ_path=Path(args.activadores_path),
        hazard_path=Path(args.hazard_path) if args.hazard_path else None,
        ks=ks,
        beta_core=args.beta_core,
        beta_full=args.beta_full,
        beta_hazard=args.beta_hazard,
        beta_hazard_struct=args.beta_hazard_struct,
        lambda_core=args.lambda_core,
        lambda_full=args.lambda_full,
        lambda_hazard=args.lambda_hazard,
        lambda_hazard_struct=args.lambda_hazard_struct,
        periods=args.period,
        include_brier=args.include_brier,
        grid_beta=grid_beta,
        grid_lambda=grid_lambda,
        output_dir=Path(args.output_dir),
        output_format=args.output_format,
        output_db_url=args.output_db_url,
        output_db_table=args.output_db_table,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
