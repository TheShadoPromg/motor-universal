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

from engine.audit import randomness as audit_randomness


def _run_main(module_main, args):
    code = module_main(args)
    if code not in (0, None):
        raise RuntimeError(f"El módulo {module_main.__module__} finalizó con código {code}.")


@task(persist_result=True)
def run_audit_randomness(run_date: str) -> str:
    args = ["--run-date", run_date]
    audit_dir = os.getenv("AUDIT_RANDOMNESS_DIR")
    if audit_dir:
        args += ["--output-dir", audit_dir]
    _run_main(audit_randomness.main, args)
    output_dir = audit_dir or str(audit_randomness.DEFAULT_OUTPUT_DIR)
    return f"Audit artifacts stored in {output_dir} for run_date={run_date}"


@flow(name="audit_randomness_pipeline")
def audit_pipeline(run_date: Optional[str] = None):
    if not run_date:
        run_date = dt.datetime.utcnow().strftime("%Y-%m-%d")
    run_audit_randomness(run_date)


if __name__ == "__main__":
    audit_pipeline()
