"""Fase 4.x - Tuning de hiperparámetros (beta, lambda) del evaluador histórico.

Objetivo:
- Definir splits temporales Train/Valid/Test.
- Explorar un grid de beta/lambda para modelos B (core) y C (core+periódico).
- Seleccionar la mejor combinación en Valid con criterios formales.
- Evaluar en Test (y opcionalmente Train/Valid con los hiperparámetros fijos).
- Persistir resultados completos (sin truncar) en Parquet y CSV.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Reutilizamos utilidades del evaluador Fase 4
from engine.backtesting.phase4 import (
    MODEL_CORE,
    MODEL_FULL,
    MODEL_UNIFORM,
    _build_draws_index,
    _compute_prob_struct,
    _compute_uniform,
    _read_activadores,
    _read_events,
    parse_activadores,
    evaluate_model,
)

LOGGER = logging.getLogger("backtesting.phase4_tune")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_PATH = REPO_ROOT / "data" / "raw" / "eventos_numericos.csv"
DEFAULT_ACTIVADORES_PATH = (
    REPO_ROOT / "data" / "activadores" / "activadores_dinamicos_fase3_para_motor.parquet"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "backtesting"

# Periodos base
P1_START = date(2011, 10, 19)
P1_END = date(2014, 12, 31)
P2_START = date(2015, 1, 1)
P2_END = date(2018, 12, 31)
P3_START = date(2019, 1, 1)
P3_END = date(2022, 12, 31)
P4_START = date(2023, 1, 1)

# Métricas por defecto
DEFAULT_KS = (5, 10, 15, 20)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def _max_lag(activadores: List) -> int:
    return max((act.lag for act in activadores), default=0)


def _filter_dates(draw_index: Dict[date, List[Tuple[int, int]]], start: date, end: date, lag: int) -> List[date]:
    min_date = start + timedelta(days=lag)
    return [d for d in sorted(draw_index.keys()) if min_date <= d <= end]


def _metrics_to_row(
    model_name: str,
    metrics,
    ks: Sequence[int],
    start: date,
    end: date,
    window_id: int,
    beta: Optional[float],
    lambda_mix: Optional[float],
    baseline_metrics,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "WindowId": window_id,
        "Model": model_name,
        "Beta": beta,
        "Lambda": lambda_mix,
        "StartDate": start,
        "EndDate": end,
    }
    for k in ks:
        row[f"HR@{k}"] = metrics.hit_rates.get(k, 0.0)
    row["RankPromedio"] = metrics.rank_promedio
    row["LogLoss"] = metrics.log_loss
    row["BrierScore"] = metrics.brier
    # baseline de uniforme (mismas fechas)
    hr10_base = baseline_metrics.hit_rates.get(10, 0.0)
    row["HR10_uniforme"] = hr10_base
    row["LogLoss_uniforme"] = baseline_metrics.log_loss
    row["Lift_HR10"] = (metrics.hit_rates.get(10, 0.0) / hr10_base) if hr10_base > 0 else None
    row["Delta_LogLoss"] = metrics.log_loss - baseline_metrics.log_loss
    return row


def evaluate_model_on_range(
    model_name: str,
    dates: List[date],
    draw_index: Dict[date, List[Tuple[int, int]]],
    acts_core,
    acts_full,
    beta: Optional[float],
    lambda_mix: Optional[float],
    ks: Sequence[int],
    include_brier: bool,
    window_id: int,
) -> Dict[str, object]:
    if not dates:
        raise ValueError("No hay fechas disponibles en el rango solicitado.")
    # Baseline uniforme para el mismo rango
    probs_uniform = _compute_uniform(dates)
    m_uniform = evaluate_model(probs_uniform, draw_index, dates, ks, include_brier=include_brier)

    if model_name == MODEL_UNIFORM:
        metrics = m_uniform
        beta_used = None
        lambda_used = None
    elif model_name == MODEL_CORE:
        probs = _compute_prob_struct(acts_core, draw_index, dates, beta=beta or 1.0, mix_lambda=lambda_mix or 1.0)
        metrics = evaluate_model(probs, draw_index, dates, ks, include_brier=include_brier)
        beta_used = beta
        lambda_used = lambda_mix
    elif model_name == MODEL_FULL:
        probs = _compute_prob_struct(acts_full, draw_index, dates, beta=beta or 1.0, mix_lambda=lambda_mix or 1.0)
        metrics = evaluate_model(probs, draw_index, dates, ks, include_brier=include_brier)
        beta_used = beta
        lambda_used = lambda_mix
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return _metrics_to_row(
        model_name=model_name,
        metrics=metrics,
        ks=ks,
        start=dates[0],
        end=dates[-1],
        window_id=window_id,
        beta=beta_used,
        lambda_mix=lambda_used,
        baseline_metrics=m_uniform,
    )


def _append_or_create(path_parquet: Path, path_csv: Path, df_new: pd.DataFrame) -> None:
    path_parquet.parent.mkdir(parents=True, exist_ok=True)
    if path_parquet.exists():
        df_old = pd.read_parquet(path_parquet)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_parquet(path_parquet, index=False)
    if path_csv:
        df_new.to_csv(path_csv, index=False)


def _select_best(df: pd.DataFrame, model: str, strict: bool = True) -> pd.Series:
    subset = df[df["Model"] == model].copy()
    if subset.empty:
        raise ValueError(f"No hay filas para el modelo {model} en valid.")
    # Filtros mínimos
    if model == MODEL_FULL and strict:
        subset = subset[(subset["Lift_HR10"] >= 1.10) & (subset["Delta_LogLoss"] <= 0.01)]
    elif model == MODEL_CORE and strict:
        subset = subset[(subset["Lift_HR10"] >= 1.05) & (subset["Delta_LogLoss"] <= 0.02)]
    if subset.empty:
        # Fallback: tomar mejor por HR@10 si los filtros vacían todo
        subset = df[df["Model"] == model].copy()
    # Score compuesto
    subset["Score"] = (
        1.0 * subset["Lift_HR10"].fillna(0)
        + 0.5 * (-subset["Delta_LogLoss"].fillna(0))
        + 0.3 * subset["HR@5"].fillna(0)
    )
    subset = subset.sort_values(
        ["Score", "Lift_HR10", "HR@5", "RankPromedio"], ascending=[False, False, False, True]
    )
    return subset.iloc[0]


def _parse_date_arg(raw: Optional[str], default: Optional[date]) -> Optional[date]:
    if raw:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    return default


def run_tuning(
    events_db_url: Optional[str],
    events_path: Path,
    activ_db_url: Optional[str],
    activ_path: Path,
    ks: Sequence[int],
    beta_grid: Sequence[float],
    lambda_grid: Sequence[float],
    include_brier: bool,
    window_id: int,
    train_start: date,
    train_end: date,
    valid_start: date,
    valid_end: date,
    test_start: date,
    test_end: date,
    output_dir: Path,
) -> None:
    events = _read_events(events_db_url, events_path)
    activ_df = _read_activadores(activ_db_url, activ_path)
    draw_index = _build_draws_index(events)
    max_data_date = max(draw_index.keys()) if draw_index else date.max
    activadores = parse_activadores(activ_df)
    acts_core = [a for a in activadores if a.clasificacion == "core_global"]
    acts_full = activadores
    lag = _max_lag(activadores)

    # Preparar fechas por split respetando lag
    def dates_for_range(start: date, end: date) -> List[date]:
        return _filter_dates(draw_index, start, end, lag)

    train_dates = dates_for_range(train_start, min(train_end, max_data_date))
    valid_dates = dates_for_range(valid_start, min(valid_end, max_data_date))
    test_dates = dates_for_range(test_start, min(test_end, max_data_date))

    # Grid en Valid
    grid_rows: List[Dict[str, object]] = []
    for b in beta_grid:
        for lmb in lambda_grid:
            # Modelo B
            row_b = evaluate_model_on_range(
                MODEL_CORE, valid_dates, draw_index, acts_core, acts_full, b, lmb, ks, include_brier, window_id
            )
            grid_rows.append(row_b)
            # Modelo C
            row_c = evaluate_model_on_range(
                MODEL_FULL, valid_dates, draw_index, acts_core, acts_full, b, lmb, ks, include_brier, window_id
            )
            grid_rows.append(row_c)
    grid_df = pd.DataFrame(grid_rows)

    # Guardar grid Valid
    grid_path_parquet = output_dir / "phase4_grid_valid.parquet"
    grid_path_csv = output_dir / "phase4_grid_valid.csv"
    _append_or_create(grid_path_parquet, grid_path_csv, grid_df)

    # Selección de mejores
    best_b = _select_best(grid_df, MODEL_CORE)
    best_c = _select_best(grid_df, MODEL_FULL)
    best_df = pd.DataFrame([best_b, best_c])
    _append_or_create(output_dir / "best_phase4_params.parquet", output_dir / "best_phase4_params.csv", best_df)

    # Evaluación final en Train/Valid/Test con hiperparámetros óptimos
    final_rows: List[Dict[str, object]] = []
    for split_name, dates in [
        ("TRAIN", train_dates),
        ("VALID", valid_dates),
        ("TEST", test_dates),
    ]:
        # Uniforme
        row_u = evaluate_model_on_range(
            MODEL_UNIFORM, dates, draw_index, acts_core, acts_full, None, None, ks, include_brier, window_id
        )
        row_u.update({"Split": split_name})
        final_rows.append(row_u)
        # Core
        row_b = evaluate_model_on_range(
            MODEL_CORE,
            dates,
            draw_index,
            acts_core,
            acts_full,
            float(best_b["Beta"]),
            float(best_b["Lambda"]),
            ks,
            include_brier,
            window_id,
        )
        row_b.update({"Split": split_name})
        final_rows.append(row_b)
        # Full
        row_c = evaluate_model_on_range(
            MODEL_FULL,
            dates,
            draw_index,
            acts_core,
            acts_full,
            float(best_c["Beta"]),
            float(best_c["Lambda"]),
            ks,
            include_brier,
            window_id,
        )
        row_c.update({"Split": split_name})
        final_rows.append(row_c)

    final_df = pd.DataFrame(final_rows)
    _append_or_create(output_dir / "phase4_results_final.parquet", output_dir / "phase4_results_final.csv", final_df)

    LOGGER.info("Tuning completado. Grid valid=%s filas, resultados finales=%s filas.", len(grid_df), len(final_df))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fase 4.x: tuning de beta/lambda para modelos core y core+periodico con splits Train/Valid/Test."
    )
    parser.add_argument("--events-db-url", default=None, help="DSN SQLAlchemy para leer eventos_numericos.")
    parser.add_argument("--events-path", default=str(DEFAULT_EVENTS_PATH), help="CSV de eventos (fallback).")
    parser.add_argument("--activadores-db-url", default=None, help="DSN SQLAlchemy para leer activadores.")
    parser.add_argument("--activadores-path", default=str(DEFAULT_ACTIVADORES_PATH), help="Parquet/CSV de activadores.")
    parser.add_argument("--ks", default="5,10,15,20", help="Lista de K para Hit@K.")
    parser.add_argument("--beta-grid", default="0.5,1.0,1.5,2.0", help="Grid de beta (coma separada).")
    parser.add_argument("--lambda-grid", default="0.5,0.7,0.85,1.0", help="Grid de lambda (coma separada).")
    parser.add_argument("--include-brier", action="store_true", help="Calcular Brier score.")
    parser.add_argument("--window-id", type=int, default=1, help="Identificador de ventana (para comparar splits).")
    # Rangos opcionales
    parser.add_argument("--train-start", default=None, help="YYYY-MM-DD (por defecto P1 start).")
    parser.add_argument("--train-end", default=None, help="YYYY-MM-DD (por defecto P2 end).")
    parser.add_argument("--valid-start", default=None, help="YYYY-MM-DD (por defecto P3 start).")
    parser.add_argument("--valid-end", default=None, help="YYYY-MM-DD (por defecto P3 end).")
    parser.add_argument("--test-start", default=None, help="YYYY-MM-DD (por defecto P4 start).")
    parser.add_argument("--test-end", default=None, help="YYYY-MM-DD (por defecto max fecha en datos).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directorio de salidas (parquet/csv).")
    parser.add_argument("--verbose", action="store_true", help="Log verboso.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    beta_grid = [float(b) for b in args.beta_grid.split(",") if b.strip()]
    lambda_grid = [float(l) for l in args.lambda_grid.split(",") if l.strip()]

    # Definir rangos
    train_start = _parse_date_arg(args.train_start, P1_START)
    train_end = _parse_date_arg(args.train_end, P2_END)
    valid_start = _parse_date_arg(args.valid_start, P3_START)
    valid_end = _parse_date_arg(args.valid_end, P3_END)
    test_start = _parse_date_arg(args.test_start, P4_START)
    # test_end: se setea luego al máximo de datos en run_tuning si se pasa None, pero aquí necesitamos fecha; se leerá más tarde
    test_end = args.test_end

    # Para test_end, pasamos None y run_tuning usará max fecha de draw_index; para mantener la firma, convertimos a date si string
    test_end_date = datetime.strptime(test_end, "%Y-%m-%d").date() if test_end else date.max

    run_tuning(
        events_db_url=args.events_db_url,
        events_path=Path(args.events_path),
        activ_db_url=args.activadores_db_url,
        activ_path=Path(args.activadores_path),
        ks=ks,
        beta_grid=beta_grid,
        lambda_grid=lambda_grid,
        include_brier=args.include_brier,
        window_id=args.window_id,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=test_end_date,
        output_dir=Path(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
