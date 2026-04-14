from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        return str(value)
    except Exception:
        return None


def _ensure_parent(path: str) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: str, payload: Any) -> None:
    _ensure_parent(path)
    file_path = Path(path).expanduser()

    fd, tmp_path = tempfile.mkstemp(
        prefix=file_path.name + ".",
        suffix=".tmp",
        dir=str(file_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def load_json_file(path: str, default=None):
    if not path:
        return default
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload
    except Exception:
        return default


def save_json_file(path: str, payload: Any) -> None:
    if not path:
        return
    safe_payload = _json_safe(payload)
    _atomic_write_json(path, safe_payload)


def _compact_runtime_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "paused": bool((state or {}).get("paused", False)),
        "last_execution_mode": (state or {}).get("last_execution_mode"),
        "paper_cash_balance": (state or {}).get("paper_cash_balance"),
        "order_intents": _json_safe((state or {}).get("order_intents") or {}),
        "execution_tracking": _json_safe((state or {}).get("execution_tracking") or {}),
    }


def load_runtime_state_file(path: str) -> Dict[str, Any]:
    payload = load_json_file(path, default={})
    return payload if isinstance(payload, dict) else {}


def save_runtime_state_file(path: str, state: Dict[str, Any]) -> None:
    compact = _compact_runtime_state(state)
    save_json_file(path, compact)


def load_paper_positions_file(path: str) -> List[Dict[str, Any]]:
    payload = load_json_file(path, default=[])
    if not isinstance(payload, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            out.append(item)
    return out


def save_paper_positions_file(path: str, positions: List[Dict[str, Any]]) -> None:
    safe_positions = positions if isinstance(positions, list) else []
    save_json_file(path, safe_positions)
