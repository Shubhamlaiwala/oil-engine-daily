from __future__ import annotations

import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _logging_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return (config or {}).get("logging", {}) or {}


def _runtime_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return (config or {}).get("runtime", {}) or {}


def _enabled(config: Dict[str, Any]) -> bool:
    logging_cfg = _logging_cfg(config)
    runtime_cfg = _runtime_cfg(config)
    return _safe_bool(
        os.getenv("LOG_ARCHIVE_ENABLED")
        or logging_cfg.get("remote_archive_enabled")
        or runtime_cfg.get("remote_archive_enabled"),
        False,
    )


def _backend(config: Dict[str, Any]) -> str:
    logging_cfg = _logging_cfg(config)
    runtime_cfg = _runtime_cfg(config)
    return _safe_str(
        os.getenv("LOG_ARCHIVE_BACKEND")
        or logging_cfg.get("remote_archive_backend")
        or runtime_cfg.get("remote_archive_backend"),
        "",
    ).lower()


def _interval_seconds(config: Dict[str, Any]) -> int:
    logging_cfg = _logging_cfg(config)
    runtime_cfg = _runtime_cfg(config)
    return max(
        30,
        _safe_int(
            os.getenv("LOG_ARCHIVE_INTERVAL_SECONDS")
            or logging_cfg.get("remote_archive_interval_seconds")
            or runtime_cfg.get("remote_archive_interval_seconds"),
            900,
        ),
    )


def _logs_dir(config: Dict[str, Any]) -> Path:
    logging_cfg = _logging_cfg(config)
    candidate = _safe_str(logging_cfg.get("remote_archive_logs_dir"), "./logs")
    return Path(candidate)


def _include_paths(config: Dict[str, Any]) -> Iterable[Path]:
    logging_cfg = _logging_cfg(config)
    logs_dir = _logs_dir(config)
    include = logging_cfg.get("remote_archive_include_files")
    if isinstance(include, list) and include:
        for item in include:
            text = _safe_str(item)
            if text:
                yield Path(text)
        return

    defaults = [
        logs_dir / "engine_runtime.log",
        logs_dir / "paper_trade_cycles.jsonl",
        logs_dir / "paper_trade_actions.jsonl",
        logs_dir / "execution_state.json",
        logs_dir / "gold_decision_log_kalshi_v6.csv",
        logs_dir / "trade_log_v6.csv",
        logs_dir / "settlement_log_v6.csv",
        logs_dir / "open_positions.csv",
    ]
    for path in defaults:
        yield path


def _export_dir(config: Dict[str, Any]) -> Path:
    logging_cfg = _logging_cfg(config)
    export_dir = _safe_str(logging_cfg.get("remote_archive_export_dir"), "./logs/exports")
    path = Path(export_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _object_prefix(config: Dict[str, Any]) -> str:
    logging_cfg = _logging_cfg(config)
    project_name = _safe_str(((config or {}).get("project", {}) or {}).get("name"), "gold-engine")
    return _safe_str(
        os.getenv("LOG_ARCHIVE_OBJECT_PREFIX")
        or logging_cfg.get("remote_archive_object_prefix"),
        f"{project_name}/runtime-archives",
    ).strip("/")


def _bundle_object_path(config: Dict[str, Any], note: str) -> str:
    stamp = _utc_now_stamp()
    prefix = _object_prefix(config)
    safe_note = _safe_str(note, "cycle").replace(" ", "_")
    return f"{prefix}/{stamp}_{safe_note}.zip"


def _write_manifest(bundle_path: Path, manifest: Dict[str, Any]) -> None:
    with zipfile.ZipFile(bundle_path, mode="a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))


def build_log_bundle(config: Dict[str, Any], note: str, state: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    export_dir = _export_dir(config)
    object_path = _bundle_object_path(config, note)
    bundle_path = export_dir / Path(object_path).name

    existing_files = []
    with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in _include_paths(config):
            path = Path(file_path)
            if not path.exists() or not path.is_file():
                continue
            arcname = f"logs/{path.name}"
            zf.write(path, arcname=arcname)
            existing_files.append({"path": str(path), "arcname": arcname, "size": path.stat().st_size})

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "object_path": object_path,
        "bundle_path": str(bundle_path),
        "files": existing_files,
        "state_summary": {
            "order_intents": len((state or {}).get("order_intents") or {}),
            "execution_tracking": len((state or {}).get("execution_tracking") or {}),
            "paused": bool((state or {}).get("paused", False)),
            "last_execution_mode": (state or {}).get("last_execution_mode"),
        },
        "extra": extra or {},
    }
    _write_manifest(bundle_path, manifest)
    return {
        "bundle_path": str(bundle_path),
        "object_path": object_path,
        "file_count": len(existing_files),
        "manifest": manifest,
    }


def _upload_supabase(config: Dict[str, Any], bundle_path: Path, object_path: str) -> Dict[str, Any]:
    supabase_url = _safe_str(os.getenv("SUPABASE_URL"))
    service_key = _safe_str(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY"))
    bucket = _safe_str(os.getenv("SUPABASE_STORAGE_BUCKET") or _logging_cfg(config).get("supabase_storage_bucket"))
    if not supabase_url or not service_key or not bucket:
        return {"uploaded": False, "reason": "missing_supabase_credentials"}

    upload_url = f"{supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{object_path}"
    headers = {
        "Authorization": f"Bearer {service_key}",
        "apikey": service_key,
        "x-upsert": "true",
        "Content-Type": "application/zip",
    }
    with open(bundle_path, "rb") as fh:
        response = requests.post(upload_url, headers=headers, data=fh, timeout=60)
    if response.status_code >= 400:
        return {"uploaded": False, "reason": f"supabase_http_{response.status_code}", "response": response.text[:500]}

    public_base = _safe_str(os.getenv("SUPABASE_PUBLIC_BASE_URL"))
    public_url = ""
    if public_base:
        public_url = f"{public_base.rstrip('/')}/{bucket}/{object_path}"

    return {"uploaded": True, "public_url": public_url}


def _upload_presigned_put(config: Dict[str, Any], bundle_path: Path, object_path: str) -> Dict[str, Any]:
    put_url = _safe_str(os.getenv("LOG_ARCHIVE_PRESIGNED_PUT_URL"))
    if not put_url:
        return {"uploaded": False, "reason": "missing_presigned_put_url"}
    headers = {"Content-Type": "application/zip", "x-object-path": object_path}
    with open(bundle_path, "rb") as fh:
        response = requests.put(put_url, headers=headers, data=fh, timeout=60)
    if response.status_code >= 400:
        return {"uploaded": False, "reason": f"presigned_put_http_{response.status_code}", "response": response.text[:500]}
    return {"uploaded": True}


def _upload_webhook(config: Dict[str, Any], bundle_path: Path, object_path: str) -> Dict[str, Any]:
    webhook_url = _safe_str(os.getenv("LOG_ARCHIVE_WEBHOOK_URL"))
    if not webhook_url:
        return {"uploaded": False, "reason": "missing_webhook_url"}
    with open(bundle_path, "rb") as fh:
        files = {"file": (Path(object_path).name, fh, "application/zip")}
        data = {"object_path": object_path, "project": _safe_str(((config or {}).get("project", {}) or {}).get("name"), "gold-engine")}
        response = requests.post(webhook_url, data=data, files=files, timeout=60)
    if response.status_code >= 400:
        return {"uploaded": False, "reason": f"webhook_http_{response.status_code}", "response": response.text[:500]}
    return {"uploaded": True, "response": response.text[:500]}


def maybe_upload_log_bundle(
    *,
    config: Dict[str, Any],
    note: str,
    state: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
    tracker: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = {"attempted": False, "uploaded": False, "backend": _backend(config), "reason": "disabled"}
    if not _enabled(config):
        result["reason"] = "disabled"
        return result

    tracker = tracker if isinstance(tracker, dict) else {}
    now_ts = time.time()
    interval = _interval_seconds(config)
    last_upload_ts = float(tracker.get("last_upload_ts") or 0.0)
    if last_upload_ts and (now_ts - last_upload_ts) < interval:
        result.update({"attempted": True, "reason": "interval_not_elapsed"})
        return result

    bundle_info = build_log_bundle(config, note=note, state=state, extra=extra)
    bundle_path = Path(bundle_info["bundle_path"])
    object_path = bundle_info["object_path"]
    backend = _backend(config)

    result.update({
        "attempted": True,
        "backend": backend,
        "bundle_path": str(bundle_path),
        "object_path": object_path,
        "file_count": bundle_info.get("file_count", 0),
    })

    if backend == "supabase":
        upload_result = _upload_supabase(config, bundle_path, object_path)
    elif backend in {"s3_presigned", "presigned_put"}:
        upload_result = _upload_presigned_put(config, bundle_path, object_path)
    elif backend in {"webhook", "gdrive_webhook"}:
        upload_result = _upload_webhook(config, bundle_path, object_path)
    else:
        upload_result = {"uploaded": False, "reason": f"unsupported_backend:{backend or 'none'}"}

    result.update(upload_result)
    if result.get("uploaded"):
        tracker["last_upload_ts"] = now_ts
    return result
