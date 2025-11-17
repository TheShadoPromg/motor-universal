from __future__ import annotations

import os
from pathlib import Path
import sys

try:
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Instala prefect>=2.0 para registrar el flujo (pip install prefect)."
    ) from exc

sys.path.append(str(Path(__file__).resolve().parents[1]))
from flows.daily_pipeline import daily_pipeline  # noqa: E402


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schedule = CronSchedule(
        cron="0 12 * * *",
        timezone=os.getenv("PREFECT_TIMEZONE", "UTC"),
    )
    deployment_path = os.getenv("PREFECT_DEPLOYMENT_PATH") or os.path.relpath(
        repo_root, start=Path.cwd()
    )

    deployment = Deployment.build_from_flow(
        flow=daily_pipeline,
        name="daily-pipeline",
        parameters={
            "run_date": None,
            "fusion_weights": os.getenv("FUSION_WEIGHTS", "0.4,0.3,0.3"),
        },
        work_pool_name=os.getenv("PREFECT_WORK_POOL", "default-agent-pool"),
        tags=["motor-universal"],
        schedules=[schedule],
        path=deployment_path,
    )
    deployment.apply()


if __name__ == "__main__":
    main()
