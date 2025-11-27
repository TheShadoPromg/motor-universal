"""Fase 5 (OOS) - Reentrenar activadores con Train y evaluar out-of-sample.

Pipeline:
1) Filtra eventos a un rango Train_struct y ejecuta Fase 2, 2.5 y 3 sólo con Train.
2) Genera activadores entrenados (parquet/csv) en un directorio aislado por ventana.
3) Llama a phase4_tune con splits Train/Valid/Test y activadores entrenados para tunear beta/lambda sin fuga.
4) Persistencia en Parquet/CSV bajo data/backtesting/oos/window_{id}.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from engine.audit import estructural, estructural_fase2_5, estructural_fase3_activadores
from engine.backtesting.phase4_tune import run_tuning

LOGGER = logging.getLogger("backtesting.phase5_oos")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_PATH = REPO_ROOT / "data" / "raw" / "eventos_numericos.csv"
DEFAULT_BASE_DIR = REPO_ROOT / "data" / "backtesting" / "oos"

# Splits por defecto (P1+P2 / P3 / P4)
TRAIN_DEFAULT = (date(2011, 10, 19), date(2018, 12, 31))
VALID_DEFAULT = (date(2019, 1, 1), date(2022, 12, 31))
TEST_DEFAULT = (date(2023, 1, 1), date(2100, 1, 1))  # se recorta al max de datos

KS_DEFAULT = (5, 10, 15, 20)
DEFAULT_BETA_GRID = (0.5, 1.0, 1.5, 2.0)
DEFAULT_LAMBDA_GRID = (0.5, 0.7, 0.85, 1.0)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def _read_events(db_url: Optional[str], path: Path) -> pd.DataFrame:
    if db_url:
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise SystemExit("sqlalchemy requerido para leer eventos desde DB.") from exc
        engine = create_engine(db_url)
        return pd.read_sql("SELECT fecha, posicion, numero FROM eventos_numericos", engine)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró eventos en {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _parse_date(raw: Optional[str], fallback: date) -> date:
    if raw:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    return fallback


def _run_main(module_main, args: list[str], step: str) -> None:
    code = module_main(args)
    if code not in (0, None):
        raise RuntimeError(f"{step} falló con código {code}")


def run_phase5(
    events_db_url: Optional[str],
    events_path: Path,
    base_dir: Path,
    window_id: int,
    train_start: date,
    train_end: date,
    valid_start: date,
    valid_end: date,
    test_start: date,
    test_end: date,
    ks: Sequence[int],
    beta_grid: Sequence[float],
    lambda_grid: Sequence[float],
    include_brier: bool,
) -> None:
    window_dir = base_dir / f"window_{window_id}"
    inputs_dir = window_dir / "inputs"
    audit_dir = window_dir / "audit"
    activ_dir = window_dir / "activadores"
    bt_dir = window_dir  # se reutiliza para outputs de tuning
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Filtrar eventos a Train_struct
    events = _read_events(events_db_url, events_path)
    events["fecha"] = pd.to_datetime(events["fecha"])
    mask_train = (events["fecha"].dt.date >= train_start) & (events["fecha"].dt.date <= train_end)
    events_train = events.loc[mask_train].copy()
    if events_train.empty:
        raise ValueError("El subset de entrenamiento quedó vacío; revisa las fechas.")
    train_path = inputs_dir / "eventos_train.parquet"
    train_path_csv = inputs_dir / "eventos_train.csv"
    events_train.to_parquet(train_path, index=False)
    events_train.to_csv(train_path_csv, index=False)
    LOGGER.info("Eventos de entrenamiento guardados en %s (%s filas).", train_path, len(events_train))

    # 2) Fase 2 con Train
    struct_out = audit_dir / "estructural_train"
    _run_main(
        estructural.main,
        [
            "--input",
            str(train_path_csv),
            "--output-dir",
            str(struct_out),
            "--start-date",
            train_start.isoformat(),
            "--end-date",
            train_end.isoformat(),
            "--output-format",
            "both",
        ],
        "Fase 2 (estructural)",
    )

    # 3) Fase 2.5 con outputs de Fase 2
    fase25_out = audit_dir / "estructural_fase2_5_train"
    _run_main(
        estructural_fase2_5.main,
        [
            "--input-dir",
            str(struct_out),
            "--output-dir",
            str(fase25_out),
        ],
        "Fase 2.5",
    )

    # 4) Fase 3 (activadores) con outputs de Fase 2.5
    core_path = fase25_out / "sesgos_fase2_5_core_y_periodicos.parquet"
    periodos_path = fase25_out / "sesgos_fase2_5_por_periodo.parquet"
    _run_main(
        estructural_fase3_activadores.main,
        [
            "--core-path",
            str(core_path),
            "--periodos-path",
            str(periodos_path),
            "--output-dir",
            str(activ_dir),
            "--format",
            "parquet",
        ],
        "Fase 3 (activadores)",
    )
    activ_para_motor = activ_dir / "activadores_dinamicos_fase3_para_motor.parquet"
    if not activ_para_motor.exists():
        # fallback csv
        activ_para_motor = activ_dir / "activadores_dinamicos_fase3_para_motor.csv"
    if not activ_para_motor.exists():
        raise FileNotFoundError("No se encontró activadores_dinamicos_fase3_para_motor en salida de Fase 3.")

    # 5) Tuning y evaluación OOS con activadores entrenados
    run_tuning(
        events_db_url=events_db_url,
        events_path=events_path,
        activ_db_url=None,  # usamos el archivo entrenado
        activ_path=activ_para_motor,
        ks=ks,
        beta_grid=beta_grid,
        lambda_grid=lambda_grid,
        include_brier=include_brier,
        window_id=window_id,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=test_end,
        output_dir=bt_dir,
    )

    LOGGER.info("Fase 5 OOS completada en %s", window_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fase 5 (OOS): reentrena activadores con Train y evalúa out-of-sample."
    )
    parser.add_argument("--events-db-url", default=None, help="DSN SQLAlchemy para leer eventos.")
    parser.add_argument("--events-path", default=str(DEFAULT_EVENTS_PATH), help="CSV/Parquet de eventos.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="Directorio base para salidas OOS.")
    parser.add_argument("--window-id", type=int, default=1, help="Identificador de ventana OOS.")
    parser.add_argument("--train-start", default=None, help="YYYY-MM-DD (por defecto 2011-10-19).")
    parser.add_argument("--train-end", default=None, help="YYYY-MM-DD (por defecto 2018-12-31).")
    parser.add_argument("--valid-start", default=None, help="YYYY-MM-DD (por defecto 2019-01-01).")
    parser.add_argument("--valid-end", default=None, help="YYYY-MM-DD (por defecto 2022-12-31).")
    parser.add_argument("--test-start", default=None, help="YYYY-MM-DD (por defecto 2023-01-01).")
    parser.add_argument("--test-end", default=None, help="YYYY-MM-DD (por defecto max fecha).")
    parser.add_argument("--ks", default="5,10,15,20", help="Lista de K para Hit@K.")
    parser.add_argument("--beta-grid", default="0.5,1.0,1.5,2.0", help="Grid de beta (coma separada).")
    parser.add_argument("--lambda-grid", default="0.5,0.7,0.85,1.0", help="Grid de lambda (coma separada).")
    parser.add_argument("--include-brier", action="store_true", help="Calcular Brier score.")
    parser.add_argument("--verbose", action="store_true", help="Log verboso.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    train_start = _parse_date(args.train_start, TRAIN_DEFAULT[0])
    train_end = _parse_date(args.train_end, TRAIN_DEFAULT[1])
    valid_start = _parse_date(args.valid_start, VALID_DEFAULT[0])
    valid_end = _parse_date(args.valid_end, VALID_DEFAULT[1])
    test_start = _parse_date(args.test_start, TEST_DEFAULT[0])
    test_end = _parse_date(args.test_end, TEST_DEFAULT[1])

    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    beta_grid = [float(b) for b in args.beta_grid.split(",") if b.strip()]
    lambda_grid = [float(l) for l in args.lambda_grid.split(",") if l.strip()]

    run_phase5(
        events_db_url=args.events_db_url,
        events_path=Path(args.events_path),
        base_dir=Path(args.base_dir),
        window_id=args.window_id,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=test_end,
        ks=ks,
        beta_grid=beta_grid,
        lambda_grid=lambda_grid,
        include_brier=args.include_brier,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
