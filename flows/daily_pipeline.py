from __future__ import annotations

import datetime as dt
import os
from typing import Optional

try:
    from prefect import flow, task
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Prefect no está instalado en este entorno. Instala 'prefect>=2' para ejecutar el flujo diario."
    ) from exc

from engine.cross import aggregate_daily as cross_daily
from engine.derived_dynamic import aggregate_daily as derived_daily
from engine.derived_dynamic import transform as derived_transform
from engine.fusion import fusionar_3capas, produce_predictions
from engine.structural import aggregate_daily as struct_daily


def _run_main(module_main, args):
    code = module_main(args)
    if code not in (0, None):
        raise RuntimeError(f"El módulo {module_main.__module__} finalizó con código {code}.")


@task
def run_derived_transform(run_date: str):
    _run_main(derived_transform.main, ["--run-date", run_date, "--skip-validation"])


@task
def run_derived_daily(run_date: str):
    _run_main(derived_daily.main, ["--target-date", run_date])


@task
def run_cross_daily(run_date: str):
    _run_main(cross_daily.main, ["--run-date", run_date])


@task
def run_struct_daily(run_date: str):
    _run_main(struct_daily.main, ["--run-date", run_date])


@task
def run_fusion(run_date: str, weights: str):
    _run_main(
        fusionar_3capas.main,
        ["--run-date", run_date, "--weights", weights, "--skip-validation"],
    )


@task
def run_predictions(run_date: str):
    args = ["--run-date", run_date]
    db_url = os.getenv("PREDICTIONS_DB_URL")
    if db_url:
        args += ["--db-url", db_url]
        db_table = os.getenv("PREDICTIONS_DB_TABLE")
        if db_table:
            args += ["--db-table", db_table]
    _run_main(produce_predictions.main, args)


@flow(name="daily_pipeline")
def daily_pipeline(run_date: Optional[str] = None, fusion_weights: str = "0.4,0.3,0.3"):
    if not run_date:
        run_date = dt.datetime.utcnow().strftime("%Y-%m-%d")
    run_derived_transform(run_date)
    run_derived_daily(run_date)
    run_cross_daily(run_date)
    run_struct_daily(run_date)
    run_fusion(run_date, fusion_weights)
    run_predictions(run_date)


if __name__ == "__main__":
    daily_pipeline()
