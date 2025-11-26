from __future__ import annotations

import datetime as dt
import os
from typing import Optional

try:
    from prefect import flow, task
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Prefect no está instalado en este entorno. Instala 'prefect>=2' para ejecutar el flujo de auditoría."
    ) from exc

from engine.audit import estructural_fase3_activadores as fase3


def _run_main(module_main, args):
    code = module_main(args)
    if code not in (0, None):
        raise RuntimeError(f"El módulo {module_main.__module__} finalizó con código {code}.")


@task(persist_result=True)
def run_audit_structural_fase3(run_date: str) -> str:
    args = ["--format", "parquet"]
    core_dir = os.getenv("AUDIT_ESTRUCTURAL_FASE2_5_DIR")
    output_dir = os.getenv("ACTIVADORES_DIR")

    if core_dir:
        core_path = os.path.join(core_dir, "sesgos_fase2_5_core_y_periodicos.parquet")
        periodos_path = os.path.join(core_dir, "sesgos_fase2_5_por_periodo.parquet")
        args += ["--core-path", core_path, "--periodos-path", periodos_path]
    if output_dir:
        args += ["--output-dir", output_dir]

    _run_main(fase3.main, args)
    final_out = output_dir or str(fase3.DEFAULT_OUTPUT_DIR)
    return f"Fase 3 completada en {final_out} para run_date={run_date}"


@flow(name="audit_structural_fase3_pipeline")
def audit_structural_fase3_pipeline(run_date: Optional[str] = None):
    run_audit_structural_fase3(run_date or dt.datetime.utcnow().strftime("%Y-%m-%d"))


if __name__ == "__main__":
    audit_structural_fase3_pipeline()
"""Prefect flow: ejecuta Fase 3 (activadores dinámicos) a partir de sesgos Fase 2.5."""
