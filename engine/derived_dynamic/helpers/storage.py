from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("derived_dynamic.storage")


def _client_kwargs() -> dict:
    endpoint = (
        os.getenv("S3_ENDPOINT")
        or os.getenv("AWS_S3_ENDPOINT")
        or os.getenv("MLFLOW_S3_ENDPOINT_URL")
    )
    kwargs: dict = {}
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    verify = os.getenv("AWS_VERIFY_SSL")
    if verify is not None:
        kwargs["verify"] = verify.lower() not in {"0", "false", "no"}
    region = os.getenv("AWS_REGION")
    if region:
        kwargs["region_name"] = region
    return kwargs


def upload_artifact(
    local_path: Path,
    bucket: Optional[str],
    *,
    object_name: Optional[str] = None,
) -> Optional[str]:
    """Sube un archivo a S3/MinIO usando boto3 si está disponible."""
    if not bucket:
        return None
    if not local_path.exists():
        LOGGER.warning("No se encontró el archivo %s para subir a S3.", local_path)
        return None
    try:
        import boto3
    except ImportError:
        LOGGER.warning("boto3 no está instalado; se omite carga a S3.")
        return None

    key = object_name or local_path.name
    key = key.lstrip("/")
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=os.getenv("AWS_REGION"),
    )
    client = session.client("s3", **_client_kwargs())

    try:
        client.upload_file(str(local_path), bucket, key)
    except Exception as exc:  # pragma: no cover - network errors
        LOGGER.warning("No se pudo subir %s a s3://%s/%s (%s)", local_path, bucket, key, exc)
        return None

    uri = f"s3://{bucket}/{key}"
    LOGGER.info("Artefacto subido correctamente a %s", uri)
    return uri
