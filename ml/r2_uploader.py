import os
from datetime import datetime

try:
    import boto3
except Exception:
    boto3 = None


def get_r2_client():
    if boto3 is None:
        raise RuntimeError("boto3 is not installed")

    endpoint = os.getenv("R2_ENDPOINT")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")

    if not endpoint or not access_key or not secret_key:
        raise RuntimeError("Missing R2 environment variables")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def upload_file(local_path, r2_key):
    bucket = os.getenv("R2_BUCKET")
    if not bucket:
        print("[R2] Missing bucket name")
        return

    if not os.path.exists(local_path):
        print(f"[R2] File not found: {local_path}")
        return

    try:
        client = get_r2_client()
        client.upload_file(local_path, bucket, r2_key)
        print(f"[R2] Uploaded: {r2_key}")
    except Exception as e:
        print(f"[R2 ERROR] {e}")


def upload_ml_logs():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    run_id = os.getenv("RUN_ID", "RUN_UNKNOWN")

    files = [
        "logs/ml/ml_candidate_dataset.jsonl",
        "logs/ml/portfolio_decisions.jsonl",
    ]

    for f in files:
        if os.path.exists(f):
            r2_key = f"{run_id}/{today}/{f}"
            upload_file(f, r2_key)
