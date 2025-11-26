from __future__ import annotations

import datetime as dt
import os
from typing import Optional

try:
    from prefect import flow, task
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Prefect no esta instalado en este entorno. Instala 'prefect>=2' para ejecutar el flujo de auditoria."
    ) from exc

from engine.audit import estructural_fase2_5


def _run_main(module_main, args):
    code = module_main(args)
    if code not in (0, None):
        raise RuntimeError(f"El modulo {module_main.__module__} finalizo con codigo {code}.")


@task(persist_result=True)
def run_audit_structural_fase2_5(run_date: str) -> str:
    args = []
    input_dir = os.getenv("AUDIT_ESTRUCTURAL_DIR")
    output_dir = os.getenv("AUDIT_ESTRUCTURAL_FASE2_5_DIR")
    if input_dir:
        args += ["--input-dir", input_dir]
    if output_dir:
        args += ["--output-dir", output_dir]
    _run_main(estructural_fase2_5.main, args)
    final_out = output_dir or str(estructural_fase2_5.DEFAULT_OUTPUT_DIR)
    return f"Fase 2.5 completada en {final_out} para run_date={run_date}"


@flow(name="audit_structural_fase2_5_pipeline")
def audit_structural_fase2_5_pipeline(run_date: Optional[str] = None):
    run_audit_structural_fase2_5(run_date or dt.datetime.utcnow().strftime("%Y-%m-%d"))


if __name__ == "__main__":
    audit_structural_fase2_5_pipeline()
"""Prefect flow: corre Fase 2.5 (estabilidad por periodo) usando outputs de Fase 2."""
