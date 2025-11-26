from __future__ import annotations

import datetime as dt
import os
from typing import Optional

try:
    from prefect import flow, task
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Prefect no esta instalado en este entorno. Instala 'prefect>=2' para ejecutar el flujo de auditoria estructural."
    ) from exc

from engine.audit import estructural as audit_estructural


def _run_main(module_main, args):
    code = module_main(args)
    if code not in (0, None):
        raise RuntimeError(f"El modulo {module_main.__module__} finalizo con codigo {code}.")


@task(persist_result=True)
def run_audit_structural(run_date: str) -> str:
    args = ["--run-date", run_date, "--output-format", "parquet"]
    audit_dir = os.getenv("AUDIT_ESTRUCTURAL_DIR")
    if audit_dir:
        args += ["--output-dir", audit_dir]
    _run_main(audit_estructural.main, args)
    output_dir = audit_dir or str(audit_estructural.DEFAULT_OUTPUT_DIR)
    return f"Audit artifacts stored in {output_dir} for run_date={run_date}"


@flow(name="audit_structural_pipeline")
def audit_structural_pipeline(run_date: Optional[str] = None):
    if not run_date:
        run_date = dt.datetime.utcnow().strftime("%Y-%m-%d")
    run_audit_structural(run_date)


if __name__ == "__main__":
    audit_structural_pipeline()
"""Prefect flow: corre Fase 2 de auditoría estructural (sesgos origen→destino por lags)."""
