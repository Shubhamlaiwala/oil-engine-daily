from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import boto3
except Exception:  # pragma: no cover
    boto3 = None


def _env(name: str) -> str:
    return str(os.getenv(name, "")).strip()


def _is_configured() -> bool:
    return all([
        _env("R2_ENDPOINT"),
        _env("R2_ACCESS_KEY_ID"),
        _env("R2_SECRET_ACCESS_KEY"),
        _env("R2_BUCKET"),
        boto3 is not None,
    ])


def _get_client():
    if not _is_configured():
        return None
    return boto3.client(
        "s3",
        endpoint_url=_env("R2_ENDPOINT"),
        aws_access_key_id=_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )


def _candidate_files() -> Iterable[Path]:
    return [
        Path("logs/ml/ml_candidate_dataset.jsonl"),
        Path("logs/ml/portfolio_decisions.jsonl"),
    ]


def upload_ml_logs() -> None:
    client = _get_client()
    if client is None:
        logging.info("[R2] Upload skipped: uploader not fully configured.")
        return

    bucket = _env("R2_BUCKET")
    day_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for local_path in _candidate_files():
        if not local_path.exists():
            logging.info("[R2] Upload skipped: file missing | path=%s", local_path)
            continue

        object_key = f"{day_prefix}/{local_path.as_posix()}"
        try:
            client.upload_file(str(local_path), bucket, object_key)
            logging.info("[R2] Uploaded: %s", object_key)
        except Exception as exc:
            logging.warning("[R2] Upload failed | key=%s | error=%s", object_key, exc)
