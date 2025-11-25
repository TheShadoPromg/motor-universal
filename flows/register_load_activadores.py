from __future__ import annotations

"""
Registro de deployment Prefect para cargar activadores (Fase 3) desde los artefactos.

Uso:
    python -m flows.register_load_activadores

Requisitos:
    - prefect >= 2 instalado en el entorno actual.
    - Variables de entorno opcionales:
        PREFECT_DEPLOYMENT_PATH: ruta del repo a registrar (por defecto, raíz del repo).
        PREFECT_WORK_POOL: nombre del work pool (por defecto: default-process-pool).
"""

import os
import sys
from pathlib import Path


def _load_env_values(keys: set[str]) -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists() or not keys:
        return
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    for key in keys:
        if key not in os.environ and key in values:
            os.environ[key] = values[key]


_load_env_values({"PREFECT_DEPLOYMENT_PATH", "PREFECT_WORK_POOL"})

try:
    from prefect.deployments import Deployment
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Instala prefect>=2.0 para registrar el flujo de activadores.") from exc

sys.path.append(str(Path(__file__).resolve().parents[1]))
from flows.load_activadores_pipeline import load_activadores_pipeline  # noqa: E402


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    deployment_path = os.getenv("PREFECT_DEPLOYMENT_PATH") or os.path.relpath(
        repo_root, start=Path.cwd()
    )

    work_pool = os.getenv("PREFECT_WORK_POOL", "default-process-pool")
    print(f"Registrando deployment 'load_activadores' en pool '{work_pool}' con path='{deployment_path}'")

    deployment = Deployment.build_from_flow(
        flow=load_activadores_pipeline,
        name="load_activadores",
        parameters={"run_date": None},
        work_pool_name=work_pool,
        tags=["motor-universal", "activadores"],
        path=deployment_path,
    )
    print(f"Deployment.path registrado: {deployment.path}")
    deployment.apply()


if __name__ == "__main__":
    main()
