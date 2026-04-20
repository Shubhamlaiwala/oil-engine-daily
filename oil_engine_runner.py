from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from pathlib import Path
from datetime import datetime

from ml.ml_data_writer import MLDataWriter
from ml.ml_schema import build_run_id, build_trade_id, trade_outcome_record, config_hash
from ml.r2_uploader import upload_ml_logs

from state_manager import (
    load_paper_positions_file,
    load_runtime_state_file,
    save_paper_positions_file,
    save_runtime_state_file,
)
from persistent_log_uploader import maybe_upload_log_bundle


def _ensure_logs_dir():
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        try:
            if np.isnan(value):
                return None
        except Exception:
            pass
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime, pd.Timestamp)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    if isinstance(value, pd.DataFrame):
        preview = value.head(25).copy()
        preview = preview.replace({np.nan: None})
        return {
            "__type__": "dataframe",
            "rows": int(len(value)),
            "columns": [str(c) for c in value.columns.tolist()],
            "preview": _json_safe(preview.to_dict(orient="records")),
        }

    if isinstance(value, pd.Series):
        preview = value.head(25).replace({np.nan: None})
        return {
            "__type__": "series",
            "rows": int(len(value)),
            "name": str(value.name),
            "preview": _json_safe(preview.tolist()),
        }

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass

    return str(value)


def _write_jsonl_record(file_name: str, **kwargs):
    log_dir = _ensure_logs_dir()
    file_path = log_dir / file_name
    record = {
        "ts": datetime.now().isoformat(),
        **{str(k): _json_safe(v) for k, v in kwargs.items()},
    }
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_paper_trade_cycle_log(**kwargs):
    try:
        _write_jsonl_record("paper_trade_cycles.jsonl", **kwargs)
    except Exception as e:
        print(f"Failed to write cycle log: {e}")


def write_paper_trade_action_log(**kwargs):
    try:
        _write_jsonl_record("paper_trade_actions.jsonl", **kwargs)
    except Exception as e:
        print(f"Failed to write action log: {e}")

from oil_engine_core import (
    current_time_et,
    ensure_parent_dir,
    load_config,
    run_engine_once,
)
from kalshi_account_client import (
    KalshiAuthClient,
    get_kalshi_account_snapshot,
    get_kalshi_live_positions_df,
    submit_kalshi_order,
)
from portfolio_manager import build_micro_allocation_plan
from position_manager import (
    evaluate_exit_rules,
    monitor_open_positions,
)

SUMMARY_INTERVAL_DEFAULT = 60

ALERT_EDGE_THRESHOLD_DEFAULT = 0.15
ALERT_DISTANCE_THRESHOLD_DEFAULT = 80.0
ALERT_COOLDOWN_SECONDS_DEFAULT = 600
ALERT_MIN_EDGE_IMPROVEMENT_DEFAULT = 0.05

PORTFOLIO_ALERT_COOLDOWN_SECONDS_DEFAULT = 600
PORTFOLIO_MIN_ALLOCATION_CHANGE_DEFAULT = 5.0
PORTFOLIO_MIN_CONTRACT_CHANGE_DEFAULT = 1
PORTFOLIO_MIN_EDGE_CHANGE_DEFAULT = 0.01
PORTFOLIO_MIN_ASK_CHANGE_DEFAULT = 0.01

STATUS_ALERT_COOLDOWN_SECONDS_DEFAULT = 600
WATCHLIST_ALERT_COOLDOWN_SECONDS_DEFAULT = 900

TELEGRAM_UPDATES_TIMEOUT_DEFAULT = 0
TELEGRAM_COMMAND_POLL_LIMIT_DEFAULT = 20

POSITION_ACTIVE_EPSILON = 1e-12

EXECUTION_MODE_SIMULATION = "simulation"
EXECUTION_MODE_LIVE = "live"

ORDER_INTENT_STATE_READY_TO_SUBMIT = "READY_TO_SUBMIT"
ORDER_INTENT_STATE_SUBMITTED_SIMULATED = "SUBMITTED_SIMULATED"
ORDER_INTENT_STATE_SUBMITTED_LIVE = "SUBMITTED_LIVE"
ORDER_INTENT_STATE_AWAITING_RECONCILIATION = "AWAITING_RECONCILIATION"
ORDER_INTENT_STATE_RECONCILED_FILLED = "RECONCILED_FILLED"
ORDER_INTENT_STATE_RECONCILED_NOT_FILLED = "RECONCILED_NOT_FILLED"
ORDER_INTENT_STATE_RECONCILED_PARTIAL = "RECONCILED_PARTIAL"
ORDER_INTENT_STATE_SUBMISSION_FAILED = "SUBMISSION_FAILED"
ORDER_INTENT_STATE_CANCELLED_OR_STALE = "CANCELLED_OR_STALE"
ORDER_INTENT_STATE_SKIPPED_DUPLICATE = "SKIPPED_DUPLICATE"
ORDER_INTENT_STATE_SKIPPED_DUPLICATE_PENDING = "SKIPPED_DUPLICATE_PENDING"

EXECUTION_STATE_PLANNED = "PLANNED"
EXECUTION_STATE_READY = "READY_TO_EXECUTE"
EXECUTION_STATE_SKIPPED_DUPLICATE = "SKIPPED_DUPLICATE"
EXECUTION_STATE_SKIPPED_NO_CASH = "SKIPPED_NO_CASH"
EXECUTION_STATE_SKIPPED_NO_ACTIVE_POSITION = "SKIPPED_NO_ACTIVE_POSITION"
EXECUTION_STATE_SKIPPED_CONFLICT = "SKIPPED_CONFLICT"
EXECUTION_STATE_AWAITING_RECON = "AWAITING_FILL_RECONCILIATION"
EXECUTION_STATE_INFORMATIONAL = "INFORMATIONAL_ONLY"
EXECUTION_STATE_RECONCILED = "RECONCILED"

seen_trade_alert_state: Dict[str, Dict[str, Any]] = {}
seen_portfolio_alert_state: Dict[str, Dict[str, Any]] = {}
seen_exit_alert_state: Dict[str, Dict[str, Any]] = {}
seen_status_alert_state: Dict[str, Dict[str, Any]] = {}
seen_watchlist_alert_state: Dict[str, Dict[str, Any]] = {}
seen_execution_plan_alert_state: Dict[str, Dict[str, Any]] = {}

runtime_state: Dict[str, Any] = {
    "paused": False,
    "last_results": None,
    "last_portfolio_plan": None,
    "last_execution_plan": None,
    "last_open_positions_df": pd.DataFrame(),
    "last_exit_df": pd.DataFrame(),
    "last_watchlist_df": pd.DataFrame(),
    "last_account_snapshot": None,
    "last_telegram_update_id": None,
    "execution_tracking": {},
    "order_intents": {},
    "last_execution_plan_signature": None,
    "last_execution_mode": EXECUTION_MODE_SIMULATION,
    "paper_cash_balance": None,
    "paper_positions_cache": [],
    "held_position_state": {},
    "last_trade_ts": None,
    "last_entry_trade_ts": None,
    "last_entry_trade_ts_by_key": {},
    "pending_trade_outcomes": [],
    "recent_exit_ts_by_ticker": {},
    "trade_stats": {
        "completed_trades": 0,
        "wins": 0,
        "losses": 0,
        "breakeven": 0,
        "realized_pnl": 0.0,
    },
}

remote_log_upload_state: Dict[str, Any] = {"last_upload_ts": 0.0}



def get_reentry_cooldown_minutes(config: Dict[str, Any]) -> float:
    portfolio_cfg = config.get("portfolio", {}) or {}
    return max(0.0, safe_float(portfolio_cfg.get("reentry_cooldown_minutes"), 0.0) or 0.0)


def recent_exit_store(state: Dict[str, Any]) -> Dict[str, float]:
    store = state.get("recent_exit_ts_by_ticker")
    if not isinstance(store, dict):
        store = {}
        state["recent_exit_ts_by_ticker"] = store
    cleaned: Dict[str, float] = {}
    for key, value in list(store.items()):
        ticker = safe_str(key)
        ts_value = safe_float(value, None)
        if ticker and ts_value is not None:
            cleaned[ticker] = float(ts_value)
    state["recent_exit_ts_by_ticker"] = cleaned
    return cleaned


def record_recent_exit_ticker(state: Dict[str, Any], ticker: Any, ts: Optional[float] = None) -> None:
    ticker_text = safe_str(ticker)
    if not ticker_text:
        return
    store = recent_exit_store(state)
    store[ticker_text] = float(ts if ts is not None else time.time())


def get_recently_exited_tickers_for_cooldown(state: Dict[str, Any], config: Dict[str, Any], now_ts: Optional[float] = None) -> List[str]:
    cooldown_minutes = get_reentry_cooldown_minutes(config)
    if cooldown_minutes <= 0:
        state["recent_exit_ts_by_ticker"] = {}
        return []

    now_value = float(now_ts if now_ts is not None else time.time())
    cutoff = now_value - (cooldown_minutes * 60.0)
    store = recent_exit_store(state)
    active: Dict[str, float] = {}
    for ticker, ts_value in list(store.items()):
        if ts_value >= cutoff:
            active[ticker] = ts_value
    state["recent_exit_ts_by_ticker"] = active
    return sorted(active.keys())


def attach_recent_exit_cooldown_to_account_snapshot(
    account_snapshot: Optional[Dict[str, Any]],
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    snapshot = dict(account_snapshot or {})
    snapshot["recently_exited_tickers"] = get_recently_exited_tickers_for_cooldown(state, config)
    return snapshot


def load_kalshi_private_key():
    key = os.getenv("KALSHI_PRIVATE_KEY")

    if not key:
        raise Exception("KALSHI_PRIVATE_KEY not set")

    key_path = "/tmp/kalshi_private_key.pem"

    with open(key_path, "w", encoding="utf-8") as f:
        f.write(key)

    return key_path


def configure_logging(config):
    logging_cfg = config.get("logging") or {}
    log_file = logging_cfg.get("engine_runtime_log", "engine_runtime.log")
    ensure_parent_dir(log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def normalize_config_sections(config):
    if not isinstance(config, dict):
        config = {}

    expected_dict_sections = [
        "project",
        "runtime",
        "logging",
        "paper_trading",
        "contract",
        "model",
        "filters",
        "decision",
        "portfolio",
        "portfolio_hold",
        "portfolio_rotation",
        "exit",
        "trading_phases",
        "kalshi",
        "telegram",
        "engine_scope",
    ]

    for section in expected_dict_sections:
        if not isinstance(config.get(section), dict):
            config[section] = {}

    return config


def get_runtime_state_file_path(config: Dict[str, Any]) -> str:
    runtime_cfg = config.get("runtime", {}) or {}
    logging_cfg = config.get("logging", {}) or {}
    return str(
        runtime_cfg.get("runtime_state_file")
        or logging_cfg.get("runtime_state_file")
        or "./logs/execution_state.json"
    )


def load_runtime_state_from_disk(config: Dict[str, Any]) -> Dict[str, Any]:
    state_file = get_runtime_state_file_path(config)
    persisted = load_runtime_state_file(state_file)
    if not persisted:
        logging.info("Runtime state load | file=%s | status=empty_or_missing", state_file)
        return {}
    logging.info(
        "Runtime state load | file=%s | order_intents=%s | execution_tracking=%s | paused=%s",
        state_file,
        len(persisted.get("order_intents") or {}),
        len(persisted.get("execution_tracking") or {}),
        bool(persisted.get("paused", False)),
    )
    return persisted


def persist_runtime_state(config: Dict[str, Any], state: Dict[str, Any], note: str = "") -> None:
    state_file = get_runtime_state_file_path(config)
    save_runtime_state_file(state_file, state)
    logging.info(
        "Runtime state saved | file=%s | order_intents=%s | execution_tracking=%s | note=%s",
        state_file,
        len((state or {}).get("order_intents") or {}),
        len((state or {}).get("execution_tracking") or {}),
        note or "cycle",
    )


def maybe_archive_logs(config: Dict[str, Any], state: Dict[str, Any], note: str = "", extra: Optional[Dict[str, Any]] = None) -> None:
    try:
        result = maybe_upload_log_bundle(
            config=config,
            note=note or "cycle",
            state=state or {},
            extra=extra or {},
            tracker=remote_log_upload_state,
        )
        if not result:
            return
        if result.get("attempted") and result.get("uploaded"):
            logging.info(
                "Remote log archive uploaded | backend=%s | object=%s | note=%s",
                result.get("backend"),
                result.get("object_path"),
                note or "cycle",
            )
        elif result.get("attempted") and result.get("reason"):
            logging.info(
                "Remote log archive skipped | backend=%s | reason=%s | note=%s",
                result.get("backend"),
                result.get("reason"),
                note or "cycle",
            )
    except Exception as exc:
        logging.warning("Remote log archive failed | note=%s | error=%s", note or "cycle", exc)


def get_paper_positions_file_path(config: Dict[str, Any]) -> str:
    paper_cfg = config.get("paper_trading", {}) or {}
    runtime_cfg = config.get("runtime", {}) or {}
    logging_cfg = config.get("logging", {}) or {}
    return str(
        paper_cfg.get("positions_file")
        or runtime_cfg.get("paper_positions_file")
        or logging_cfg.get("paper_positions_file")
        or "./logs/paper_positions.json"
    )


def get_paper_starting_cash(config: Dict[str, Any]) -> float:
    paper_cfg = config.get("paper_trading", {}) or {}
    portfolio_cfg = config.get("portfolio", {}) or {}
    return float(
        paper_cfg.get(
            "starting_cash",
            portfolio_cfg.get("starting_capital", 10.0),
        )
    )


def ensure_paper_cash_balance_initialized(state: Dict[str, Any], config: Dict[str, Any]) -> float:
    current = safe_float(state.get("paper_cash_balance"), None)
    if current is None:
        current = get_paper_starting_cash(config)
        state["paper_cash_balance"] = current
        logging.info("Paper cash initialized | cash_balance=%.2f", current)
    return float(current)


def _normalize_single_paper_position(record: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(record or {})
    item["ticker"] = safe_str(item.get("ticker"))
    item["side"] = safe_upper(item.get("side"))
    item["contracts"] = abs(safe_int(item.get("contracts"), 0) or 0)
    item["entry_price"] = safe_float(item.get("entry_price"), None)
    item["allocation"] = safe_float(item.get("allocation"), 0.0)
    item["status"] = safe_upper(item.get("status") or "OPEN")
    item["source"] = safe_str(item.get("source") or "simulation")
    item["position_id"] = safe_str(item.get("position_id") or item.get("ticker"))
    item["entry_time"] = safe_float(item.get("entry_time"), None)
    item["updated_at"] = safe_float(item.get("updated_at"), None)
    return item


def normalize_paper_positions_records(records: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        normalized = _normalize_single_paper_position(record)
        if not normalized.get("ticker"):
            continue
        out.append(normalized)
    return out


def get_open_paper_positions(records: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    open_records: List[Dict[str, Any]] = []
    for record in normalize_paper_positions_records(records):
        if safe_upper(record.get("status")) != "OPEN":
            continue
        if abs(safe_int(record.get("contracts"), 0) or 0) <= 0:
            continue
        open_records.append(record)
    return open_records


def load_paper_positions_from_disk(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = get_paper_positions_file_path(config)
    records = load_paper_positions_file(path)
    normalized = normalize_paper_positions_records(records)
    open_count = len(get_open_paper_positions(normalized))
    logging.info(
        "Paper positions loaded | file=%s | total_count=%s | open_count=%s",
        path,
        len(normalized),
        open_count,
    )
    return normalized


def persist_paper_positions(config: Dict[str, Any], records: List[Dict[str, Any]], note: str = "") -> None:
    path = get_paper_positions_file_path(config)
    normalized = normalize_paper_positions_records(records)
    save_paper_positions_file(path, normalized)
    logging.info(
        "Paper positions saved | file=%s | total_count=%s | open_count=%s | note=%s",
        path,
        len(normalized),
        len(get_open_paper_positions(normalized)),
        note or "cycle",
    )


def extract_paper_contract_tickers(records: Optional[List[Dict[str, Any]]]) -> List[str]:
    tickers = []
    for record in get_open_paper_positions(records):
        ticker = safe_str(record.get("ticker"))
        if ticker:
            tickers.append(ticker)
    return sorted(set(tickers))


def build_paper_positions_raw_df(records: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    open_records = get_open_paper_positions(records)
    if not open_records:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for record in open_records:
        ticker = safe_str(record.get("ticker"))
        side = safe_upper(record.get("side"))
        contracts = abs(safe_int(record.get("contracts"), 0) or 0)
        entry_price = safe_float(record.get("entry_price"), None)
        allocation = safe_float(record.get("allocation"), 0.0)

        action = "BUY_YES" if side == "YES" else ("BUY_NO" if side == "NO" else "")
        signed_position = contracts if side == "YES" else -contracts

        rows.append(
            {
                "position_id": safe_str(record.get("position_id")) or ticker,
                "contract_ticker": ticker,
                "ticker": ticker,
                "action": action,
                "side_norm": side,
                "contracts": contracts,
                "quantity": contracts,
                "size": contracts,
                "position_numeric": signed_position,
                "position_fp": signed_position,
                "entry_price": entry_price,
                "position_cost": allocation if allocation > 0 else (entry_price * contracts if entry_price is not None else np.nan),
                "cost_basis_total": allocation if allocation > 0 else (entry_price * contracts if entry_price is not None else np.nan),
                "current_price": np.nan,
                "current_position_price": np.nan,
                "market_value": np.nan,
                "unrealized_pnl": np.nan,
                "source": safe_str(record.get("source") or "simulation"),
                "status": safe_upper(record.get("status") or "OPEN"),
                "entry_time": safe_float(record.get("entry_time"), None),
                "updated_at": safe_float(record.get("updated_at"), None),
                "allocation": allocation,
            }
        )

    return pd.DataFrame(rows)


def normalize_paper_positions_for_monitoring(
    records: Optional[List[Dict[str, Any]]],
    ranked_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    raw_df = build_paper_positions_raw_df(records)
    if raw_df.empty:
        return pd.DataFrame()

    normalized = normalize_kalshi_positions_for_monitoring(raw_df, ranked_df if ranked_df is not None else pd.DataFrame())
    logging.info(
        "Paper positions normalized for monitoring | open_count=%s",
        len(normalized),
    )
    return normalized


def build_paper_account_snapshot(
    state: Dict[str, Any],
    open_positions_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    cash_balance = safe_float(state.get("paper_cash_balance"), 0.0)
    raw_snapshot = {
        "cash_balance": cash_balance,
        "paper_mode": True,
        "source": "simulation",
    }
    scoped = build_engine_scoped_account_snapshot(
        raw_account_snapshot=raw_snapshot,
        filtered_positions_df=open_positions_df if open_positions_df is not None else pd.DataFrame(),
    )
    scoped["paper_mode"] = True
    scoped["source"] = "simulation"
    return scoped


def _find_open_paper_position_index(
    records: List[Dict[str, Any]],
    ticker: str,
    side: str,
) -> Optional[int]:
    for idx, record in enumerate(records):
        if safe_upper(record.get("status")) != "OPEN":
            continue
        if safe_str(record.get("ticker")) != safe_str(ticker):
            continue
        if safe_upper(record.get("side")) != safe_upper(side):
            continue
        return idx
    return None


def _pending_trade_outcomes_store(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    store = state.get("pending_trade_outcomes")
    if not isinstance(store, list):
        store = []
        state["pending_trade_outcomes"] = store
    return store


def trade_stats_store(state: Dict[str, Any]) -> Dict[str, Any]:
    stats = state.get("trade_stats")
    if not isinstance(stats, dict):
        stats = {}
    stats.setdefault("completed_trades", 0)
    stats.setdefault("wins", 0)
    stats.setdefault("losses", 0)
    stats.setdefault("breakeven", 0)
    stats.setdefault("realized_pnl", 0.0)
    state["trade_stats"] = stats
    return stats


def _update_trade_stats_for_outcome(state: Dict[str, Any], realized_pnl: Optional[float]) -> None:
    stats = trade_stats_store(state)
    pnl_value = safe_float(realized_pnl, 0.0) or 0.0
    stats["completed_trades"] = safe_int(stats.get("completed_trades"), 0) + 1
    stats["realized_pnl"] = round((safe_float(stats.get("realized_pnl"), 0.0) or 0.0) + pnl_value, 6)
    if pnl_value > 1e-12:
        stats["wins"] = safe_int(stats.get("wins"), 0) + 1
    elif pnl_value < -1e-12:
        stats["losses"] = safe_int(stats.get("losses"), 0) + 1
    else:
        stats["breakeven"] = safe_int(stats.get("breakeven"), 0) + 1


def _et_iso_from_ts(value: Any) -> Optional[str]:
    ts = safe_float(value, None)
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=current_time_et().tzinfo).isoformat()
    except Exception:
        return None


def _append_trade_outcome_for_closed_position(
    *,
    state: Dict[str, Any],
    config: Dict[str, Any],
    position_before_close: Dict[str, Any],
    closed_contracts: int,
    exit_price: Optional[float],
    settlement_value: Optional[float],
    exit_reason: str,
    settled: bool,
    exit_timestamp_ts: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if closed_contracts <= 0:
        return None

    ticker = safe_str(position_before_close.get("ticker"))
    side = safe_upper(position_before_close.get("side"))
    entry_price = safe_float(position_before_close.get("entry_price"), None)
    if entry_price is None:
        return None

    exit_ts = safe_float(exit_timestamp_ts, time.time())
    entry_ts = safe_float(position_before_close.get("entry_time"), None)
    entry_timestamp_et = _et_iso_from_ts(entry_ts) or current_time_et().isoformat()
    exit_timestamp_et = _et_iso_from_ts(exit_ts) or current_time_et().isoformat()

    effective_exit_price = safe_float(exit_price, None)
    if effective_exit_price is None:
        effective_exit_price = safe_float(settlement_value, entry_price)
    if effective_exit_price is None:
        effective_exit_price = entry_price

    realized_pnl = (float(effective_exit_price) - float(entry_price)) * int(closed_contracts)
    realized_return = None
    if entry_price not in (None, 0):
        try:
            realized_return = (float(effective_exit_price) - float(entry_price)) / float(entry_price)
        except Exception:
            realized_return = None

    hold_minutes = None
    if entry_ts is not None and exit_ts is not None:
        hold_minutes = max(0.0, (float(exit_ts) - float(entry_ts)) / 60.0)

    trade_id = build_trade_id(
        contract_ticker=ticker,
        side=side,
        entry_timestamp_et=entry_timestamp_et,
    )

    record = trade_outcome_record(
        trade_id=trade_id,
        contract_ticker=ticker,
        side=side,
        contracts=int(closed_contracts),
        entry_timestamp_et=entry_timestamp_et,
        exit_timestamp_et=exit_timestamp_et,
        entry_price=entry_price,
        exit_price=effective_exit_price,
        settlement_value=safe_float(settlement_value, None),
        realized_pnl=realized_pnl,
        realized_return=realized_return,
        hold_minutes=hold_minutes,
        exit_reason=safe_str(exit_reason),
        settled=bool(settled),
        engine_name="oil_engine_daily",
        run_id=build_run_id("oil_engine_daily", cycle_timestamp_et=exit_timestamp_et),
        config_hash_value=config_hash(config),
        engine_version="v1",
    )
    _pending_trade_outcomes_store(state).append(record)
    _update_trade_stats_for_outcome(state, realized_pnl)
    logging.info(
        "Queued trade outcome | ticker=%s | side=%s | contracts=%s | realized_pnl=%s | exit_reason=%s | settled=%s",
        ticker,
        side,
        closed_contracts,
        realized_pnl,
        exit_reason,
        settled,
    )
    return record


def flush_pending_trade_outcomes(
    *,
    ml_writer: MLDataWriter,
    state: Dict[str, Any],
) -> int:
    pending = _pending_trade_outcomes_store(state)
    if not pending:
        return 0

    written = 0
    remaining: List[Dict[str, Any]] = []
    for record in pending:
        try:
            ml_writer.write_trade_outcome(record)
            written += 1
        except Exception as exc:
            logging.warning(
                "ML trade outcome write failed | trade_id=%s | error=%s",
                record.get("trade_id"),
                exc,
            )
            remaining.append(record)

    state["pending_trade_outcomes"] = remaining
    return written


def detect_and_close_resolved_paper_positions(
    *,
    state: Dict[str, Any],
    config: Dict[str, Any],
    ranked_df: Optional[pd.DataFrame],
) -> int:
    records = normalize_paper_positions_records(state.get("paper_positions_cache") or [])
    open_records = get_open_paper_positions(records)
    if not open_records:
        return 0

    ranked_lookup: Dict[str, Dict[str, Any]] = {}
    if ranked_df is not None and not ranked_df.empty and "contract_ticker" in ranked_df.columns:
        dedup = ranked_df.drop_duplicates(subset=["contract_ticker"], keep="last")
        ranked_lookup = {
            safe_str(row.get("contract_ticker")): (row.to_dict() if hasattr(row, "to_dict") else dict(row))
            for _, row in dedup.iterrows()
        }

    now_et = current_time_et()
    closed_count = 0

    for idx, record in enumerate(list(records)):
        if safe_upper(record.get("status")) != "OPEN":
            continue

        ticker = safe_str(record.get("ticker"))
        side = safe_upper(record.get("side"))
        row = ranked_lookup.get(ticker) or {}

        resolution_time_text = row.get("resolution_time_et")
        hours_left = safe_float(row.get("hours_left"), None)
        strike = safe_float(row.get("strike"), None)
        oil_price = safe_float(row.get("oil_price"), safe_float((state.get("last_results") or {}).get("price"), None))
        decision_state = safe_upper(row.get("decision_state"))
        fair_prob = pick_model_prob_from_row(row, None)

        resolved = False
        if hours_left is not None and hours_left <= 0:
            resolved = True
        elif resolution_time_text:
            try:
                resolution_dt = pd.Timestamp(resolution_time_text)
                if resolution_dt.tzinfo is None:
                    resolution_dt = resolution_dt.tz_localize(now_et.tzinfo)
                else:
                    resolution_dt = resolution_dt.tz_convert(now_et.tzinfo)
                if resolution_dt.to_pydatetime() <= now_et:
                    resolved = True
            except Exception:
                pass
        elif decision_state in {"EXIT_RESOLVED", "CONTRACT_RESOLVED", "RESOLVED"}:
            resolved = True

        if not resolved:
            continue

        settlement_yes = None
        if oil_price is not None and strike is not None:
            settlement_yes = 1.0 if float(oil_price) >= float(strike) else 0.0
        elif fair_prob is not None:
            settlement_yes = 1.0 if float(fair_prob) >= 0.5 else 0.0

        if settlement_yes is None:
            logging.info(
                "Resolved paper position skipped due to missing settlement basis | ticker=%s | strike=%s | oil_price=%s | fair_prob=%s",
                ticker,
                strike,
                oil_price,
                fair_prob,
            )
            continue

        settlement_value = 1.0 if (side == "YES" and settlement_yes == 1.0) or (side == "NO" and settlement_yes == 0.0) else 0.0
        contracts = abs(safe_int(record.get("contracts"), 0) or 0)
        if contracts <= 0:
            continue

        prior_record = dict(record)
        records[idx]["contracts"] = 0
        records[idx]["status"] = "CLOSED"
        records[idx]["updated_at"] = time.time()
        records[idx]["exit_time"] = time.time()
        records[idx]["exit_reason"] = "CONTRACT_RESOLVED"
        records[idx]["settlement_value"] = settlement_value

        current_cash = ensure_paper_cash_balance_initialized(state, config)
        state["paper_cash_balance"] = round(current_cash + (contracts * settlement_value), 2)

        _append_trade_outcome_for_closed_position(
            state=state,
            config=config,
            position_before_close=prior_record,
            closed_contracts=contracts,
            exit_price=settlement_value,
            settlement_value=settlement_value,
            exit_reason="CONTRACT_RESOLVED",
            settled=True,
            exit_timestamp_ts=time.time(),
        )
        record_recent_exit_ticker(state, ticker, time.time())

        logging.info(
            "Paper position resolved and closed | ticker=%s | side=%s | contracts=%s | settlement_value=%s | oil_price=%s | strike=%s",
            ticker,
            side,
            contracts,
            settlement_value,
            oil_price,
            strike,
        )
        closed_count += 1

    if closed_count > 0:
        state["paper_positions_cache"] = records
        persist_paper_positions(config, records, note="resolved_contract_close")

    return closed_count


def _apply_entry_fill_to_paper_positions(
    *,
    intent: Dict[str, Any],
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    records = normalize_paper_positions_records(state.get("paper_positions_cache") or [])
    ticker = safe_str(intent.get("ticker"))
    side = safe_upper(intent.get("side"))
    contracts = abs(safe_int(intent.get("contracts"), 0) or 0)

    # FIX: safely derive entry/ask price from the intent
    ask_price = intent.get("ask_price")
    if ask_price is None:
        ask_price = intent.get("price")
    if ask_price is None:
        ask_price = 0.5

    ask_price = _clip_binary_price(ask_price, ask_price)
    entry_price = safe_float(ask_price, None)

    allocation = safe_float(intent.get("allocation"), 0.0)
    now_ts = time.time()

    if not ticker or side not in {"YES", "NO"} or contracts <= 0:
        return

    if allocation <= 0 and entry_price is not None:
        allocation = contracts * max(entry_price, 0.0)

    idx = _find_open_paper_position_index(records, ticker, side)

    if idx is None:
        position = {
            "position_id": f"paper::{ticker}::{side}",
            "ticker": ticker,
            "side": side,
            "contracts": contracts,
            "entry_price": entry_price,
            "entry_time": safe_float(intent.get("submitted_at"), None) or now_ts,
            "updated_at": now_ts,
            "source": safe_str(intent.get("source") or "simulation"),
            "status": "OPEN",
            "allocation": allocation,
            "reason": safe_str(intent.get("reason")),
            "confidence": safe_str((intent.get("metadata") or {}).get("confidence")),
        }
        records.append(position)

        logging.info(
            "Paper position opened | ticker=%s | side=%s | contracts=%s | entry_price=%s",
            ticker,
            side,
            contracts,
            entry_price,
        )
    else:
        existing = dict(records[idx])

        old_contracts = abs(safe_int(existing.get("contracts"), 0) or 0)
        old_entry_price = safe_float(existing.get("entry_price"), None)
        old_allocation = safe_float(existing.get("allocation"), 0.0)

        new_contracts = old_contracts + contracts
        new_allocation = old_allocation + allocation

        weighted_entry = old_entry_price
        if entry_price is not None and new_contracts > 0:
            prior_weight = 0.0 if old_entry_price is None else (old_entry_price * old_contracts)
            weighted_entry = (prior_weight + (entry_price * contracts)) / new_contracts

        existing["contracts"] = new_contracts
        existing["entry_price"] = weighted_entry
        existing["allocation"] = new_allocation
        existing["updated_at"] = now_ts
        existing["status"] = "OPEN"
        records[idx] = existing

        logging.info(
            "Paper position updated | ticker=%s | side=%s | contracts=%s | entry_price=%s",
            ticker,
            side,
            new_contracts,
            weighted_entry,
        )

    current_cash = ensure_paper_cash_balance_initialized(state, config)
    state["paper_cash_balance"] = round(max(0.0, current_cash - max(allocation, 0.0)), 2)
    state["paper_positions_cache"] = records
    persist_paper_positions(config, records, note="entry_fill_applied")


def _apply_exit_fill_to_paper_positions(
    *,
    intent: Dict[str, Any],
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    records = normalize_paper_positions_records(state.get("paper_positions_cache") or [])
    ticker = safe_str(intent.get("ticker"))
    side = safe_upper(intent.get("side"))
    contracts_to_close = abs(safe_int(intent.get("contracts"), 0) or 0)
    exit_price = safe_float(intent.get("ask_price"), None)
    now_ts = time.time()

    if not ticker or side not in {"YES", "NO"} or contracts_to_close <= 0:
        return

    idx = _find_open_paper_position_index(records, ticker, side)
    if idx is None:
        logging.warning(
            "Paper exit fill had no matching open position | ticker=%s | side=%s | contracts=%s",
            ticker,
            side,
            contracts_to_close,
        )
        return

    existing = dict(records[idx])
    prior_existing = dict(existing)
    held_contracts = abs(safe_int(existing.get("contracts"), 0) or 0)
    if held_contracts <= 0:
        existing["status"] = "CLOSED"
        existing["contracts"] = 0
        existing["updated_at"] = now_ts
        records[idx] = existing
        state["paper_positions_cache"] = records
        persist_paper_positions(config, records, note="exit_fill_zero_cleanup")
        return

    closed_contracts = min(held_contracts, contracts_to_close)
    remaining_contracts = max(0, held_contracts - closed_contracts)

    existing["contracts"] = remaining_contracts
    existing["updated_at"] = now_ts
    if remaining_contracts == 0:
        existing["status"] = "CLOSED"
        existing["exit_time"] = now_ts
        logging.info(
            "Paper position closed | ticker=%s | side=%s | closed_contracts=%s",
            ticker,
            side,
            closed_contracts,
        )
    else:
        existing["status"] = "OPEN"
        logging.info(
            "Paper position updated | ticker=%s | side=%s | remaining_contracts=%s",
            ticker,
            side,
            remaining_contracts,
        )

    entry_price = safe_float(existing.get("entry_price"), None)
    if exit_price is None:
        exit_price = entry_price
    cash_credit = 0.0
    if exit_price is not None:
        cash_credit = closed_contracts * max(exit_price, 0.0)

    current_cash = ensure_paper_cash_balance_initialized(state, config)
    state["paper_cash_balance"] = round(current_cash + cash_credit, 2)

    records[idx] = existing
    state["paper_positions_cache"] = records
    persist_paper_positions(config, records, note="exit_fill_applied")

    if closed_contracts > 0:
        _append_trade_outcome_for_closed_position(
            state=state,
            config=config,
            position_before_close=prior_existing,
            closed_contracts=closed_contracts,
            exit_price=exit_price,
            settlement_value=None,
            exit_reason=safe_str(intent.get("reason") or intent.get("reconciliation_reason") or "EXIT_FILLED"),
            settled=False,
            exit_timestamp_ts=now_ts,
        )
        record_recent_exit_ticker(state, ticker, now_ts)

    if contracts_to_close > held_contracts:
        logging.warning(
            "Paper exit requested more contracts than held | ticker=%s | side=%s | requested=%s | held=%s",
            ticker,
            side,
            contracts_to_close,
            held_contracts,
        )


def maybe_apply_reconciled_simulation_intent_to_paper_ledger(
    *,
    intent: Dict[str, Any],
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    mode = safe_str(intent.get("reconciliation_mode") or intent.get("submission_mode") or get_execution_mode(config)).lower()
    if mode != EXECUTION_MODE_SIMULATION:
        return

    if bool(intent.get("paper_ledger_applied", False)):
        return

    intent_state = safe_upper(intent.get("state"))
    intent_type = safe_upper(intent.get("type"))
    if intent_state not in {
        ORDER_INTENT_STATE_RECONCILED_FILLED,
        ORDER_INTENT_STATE_RECONCILED_PARTIAL,
    }:
        return

    if intent_type == "ENTER":
        _apply_entry_fill_to_paper_positions(intent=intent, state=state, config=config)
    elif intent_type == "EXIT":
        _apply_exit_fill_to_paper_positions(intent=intent, state=state, config=config)
    else:
        return

    intent["paper_ledger_applied"] = True
    intent["paper_ledger_applied_at"] = time.time()


def _get_config_value(config, key, default):
    telegram_cfg = config.get("telegram", {})
    runtime_cfg = config.get("runtime", {})
    if key in telegram_cfg:
        return telegram_cfg[key]
    if key in runtime_cfg:
        return runtime_cfg[key]
    return default


def get_summary_interval(config):
    runtime_cfg = (config.get("runtime") or {}) if isinstance(config, dict) else {}
    return int(runtime_cfg.get("summary_interval_seconds", SUMMARY_INTERVAL_DEFAULT))


def get_max_alerts_per_cycle(config):
    return int(config.get("runtime", {}).get("max_alerts_per_cycle", 1))


def get_alert_edge_threshold(config):
    return float(_get_config_value(config, "alert_edge_threshold", ALERT_EDGE_THRESHOLD_DEFAULT))


def get_alert_distance_threshold(config):
    return float(_get_config_value(config, "alert_distance_threshold", ALERT_DISTANCE_THRESHOLD_DEFAULT))


def get_alert_cooldown_seconds(config):
    return int(_get_config_value(config, "alert_cooldown_seconds", ALERT_COOLDOWN_SECONDS_DEFAULT))


def get_alert_min_edge_improvement(config):
    return float(_get_config_value(config, "alert_min_edge_improvement", ALERT_MIN_EDGE_IMPROVEMENT_DEFAULT))


def get_portfolio_alert_cooldown_seconds(config):
    return int(
        _get_config_value(
            config,
            "portfolio_alert_cooldown_seconds",
            PORTFOLIO_ALERT_COOLDOWN_SECONDS_DEFAULT,
        )
    )


def get_portfolio_min_allocation_change(config):
    return float(
        _get_config_value(
            config,
            "portfolio_min_allocation_change",
            PORTFOLIO_MIN_ALLOCATION_CHANGE_DEFAULT,
        )
    )


def get_portfolio_min_contract_change(config):
    return int(
        _get_config_value(
            config,
            "portfolio_min_contract_change",
            PORTFOLIO_MIN_CONTRACT_CHANGE_DEFAULT,
        )
    )


def get_portfolio_min_edge_change(config):
    return float(
        _get_config_value(
            config,
            "portfolio_min_edge_change",
            PORTFOLIO_MIN_EDGE_CHANGE_DEFAULT,
        )
    )


def get_portfolio_min_ask_change(config):
    return float(
        _get_config_value(
            config,
            "portfolio_min_ask_change",
            PORTFOLIO_MIN_ASK_CHANGE_DEFAULT,
        )
    )


def get_status_alert_cooldown_seconds(config):
    return int(
        _get_config_value(
            config,
            "status_alert_cooldown_seconds",
            STATUS_ALERT_COOLDOWN_SECONDS_DEFAULT,
        )
    )


def get_watchlist_alert_cooldown_seconds(config):
    return int(
        _get_config_value(
            config,
            "watchlist_alert_cooldown_seconds",
            WATCHLIST_ALERT_COOLDOWN_SECONDS_DEFAULT,
        )
    )


def get_telegram_updates_timeout_seconds(config):
    return int(
        _get_config_value(
            config,
            "telegram_updates_timeout_seconds",
            TELEGRAM_UPDATES_TIMEOUT_DEFAULT,
        )
    )


def get_telegram_command_poll_limit(config):
    return int(
        _get_config_value(
            config,
            "telegram_command_poll_limit",
            TELEGRAM_COMMAND_POLL_LIMIT_DEFAULT,
        )
    )


def get_execution_mode(config) -> str:
    runtime_cfg = (config.get("runtime", {}) or {})
    project_cfg = (config.get("project", {}) or {})

    runtime_mode = safe_str(runtime_cfg.get("execution_mode")).lower()
    if runtime_mode in {EXECUTION_MODE_SIMULATION, EXECUTION_MODE_LIVE}:
        return runtime_mode

    project_mode = safe_str(project_cfg.get("mode")).lower()
    if project_mode in {"paper_trading", "paper", "simulation", "sim"}:
        return EXECUTION_MODE_SIMULATION
    if project_mode in {"live", "live_trading", "production_live"}:
        return EXECUTION_MODE_LIVE

    return EXECUTION_MODE_SIMULATION


def get_order_reconciliation_timeout_seconds(config) -> int:
    return int((config.get("runtime", {}) or {}).get("order_reconciliation_timeout_seconds", 600))


def get_simulated_reconciliation_fill_seconds(config) -> int:
    return int((config.get("runtime", {}) or {}).get("simulated_reconciliation_fill_seconds", 5))


def get_order_material_contract_change(config) -> int:
    return int((config.get("runtime", {}) or {}).get("order_material_contract_change", 1))


def get_order_material_allocation_change(config) -> float:
    return float((config.get("runtime", {}) or {}).get("order_material_allocation_change", 5.0))


def get_order_material_ask_change(config) -> float:
    return float((config.get("runtime", {}) or {}).get("order_material_ask_change", 0.01))


def get_min_time_between_trades_seconds(config: Dict[str, Any]) -> int:
    runtime_cfg = config.get("runtime", {}) or {}
    return int(runtime_cfg.get("min_time_between_trades_seconds", 300))


def build_trade_cooldown_key(ticker: Any, side: Any) -> str:
    return "||".join([safe_str(ticker), safe_upper(side)])


def get_last_trade_timestamp(state: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(state, dict):
        return None
    return safe_float(state.get("last_trade_ts"), None)


def get_last_entry_trade_timestamp(state: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(state, dict):
        return None
    return safe_float(state.get("last_entry_trade_ts"), None)


def get_last_entry_trade_timestamp_for_key(
    state: Optional[Dict[str, Any]],
    ticker: Any,
    side: Any,
) -> Optional[float]:
    if not isinstance(state, dict):
        return None
    per_key = state.get("last_entry_trade_ts_by_key")
    if not isinstance(per_key, dict):
        return get_last_entry_trade_timestamp(state)
    key = build_trade_cooldown_key(ticker, side)
    return safe_float(per_key.get(key), get_last_entry_trade_timestamp(state))


def mark_trade_timestamp(
    state: Optional[Dict[str, Any]],
    ts: Optional[float] = None,
    *,
    action_type: Optional[str] = None,
    ticker: Any = None,
    side: Any = None,
) -> None:
    if not isinstance(state, dict):
        return

    ts_value = float(ts if ts is not None else time.time())
    state["last_trade_ts"] = ts_value

    if safe_upper(action_type) != "ENTER":
        return

    state["last_entry_trade_ts"] = ts_value
    per_key = state.get("last_entry_trade_ts_by_key")
    if not isinstance(per_key, dict):
        per_key = {}
        state["last_entry_trade_ts_by_key"] = per_key

    key = build_trade_cooldown_key(ticker, side)
    if key != "||":
        per_key[key] = ts_value


def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def values_materially_different(a, b, tolerance):
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return abs(float(a) - float(b)) >= float(tolerance)


def safe_upper(value: Any) -> str:
    return str(value or "").strip().upper()


def safe_str(value: Any) -> str:
    return str(value or "").strip()


def infer_held_side_from_row(row: Any) -> str:
    if row is None:
        return ""
    held_side = safe_upper(getattr(row, "get", lambda *_: None)("held_side"))
    if held_side in {"YES", "NO"}:
        return held_side
    side_norm = safe_upper(getattr(row, "get", lambda *_: None)("side_norm"))
    if side_norm in {"YES", "NO"}:
        return side_norm
    side = safe_upper(getattr(row, "get", lambda *_: None)("side"))
    if side in {"YES", "NO"}:
        return side
    action = safe_upper(getattr(row, "get", lambda *_: None)("action"))
    if action == "BUY_YES":
        return "YES"
    if action == "BUY_NO":
        return "NO"
    return ""


def format_optional_float(value: Any, fmt: str = ".2f", default: str = "N/A") -> str:
    numeric = safe_float(value, None)
    if numeric is None:
        return default
    return format(numeric, fmt)


def _clean_token(value: Any) -> Optional[str]:
    text = safe_str(value)
    if not text or text.lower() == "nan":
        return None
    return text


def _first_nonempty(*values: Any) -> Optional[str]:
    for value in values:
        cleaned = _clean_token(value)
        if cleaned:
            return cleaned
    return None


def _get_engine_scope(config: Dict[str, Any]) -> Dict[str, Optional[str]]:
    kalshi_cfg = config.get("kalshi", {}) or {}
    runtime_cfg = config.get("runtime", {}) or {}
    contract_cfg = config.get("contract", {}) or {}
    engine_scope_cfg = config.get("engine_scope", {}) or {}

    exact_market_ticker = _first_nonempty(
        engine_scope_cfg.get("exact_market_ticker"),
        kalshi_cfg.get("exact_market_ticker"),
        runtime_cfg.get("exact_market_ticker"),
        contract_cfg.get("exact_market_ticker"),
    )
    event_ticker = _first_nonempty(
        engine_scope_cfg.get("event_ticker"),
        kalshi_cfg.get("event_ticker"),
        runtime_cfg.get("event_ticker"),
        contract_cfg.get("event_ticker"),
    )
    series_ticker = _first_nonempty(
        engine_scope_cfg.get("series_ticker"),
        kalshi_cfg.get("series_ticker"),
        runtime_cfg.get("series_ticker"),
        contract_cfg.get("series_ticker"),
    )

    return {
        "exact_market_ticker": exact_market_ticker,
        "event_ticker": event_ticker,
        "series_ticker": series_ticker,
    }


def pick_model_prob_from_row(row: Any, default=None):
    for col in ["decision_prob", "fair_prob_terminal", "fair_prob_blended"]:
        try:
            value = row.get(col)
        except Exception:
            value = None
        numeric = safe_float(value, None)
        if numeric is not None:
            return numeric
    return default


def get_latest_volatility_from_results(results: Optional[Dict[str, Any]]) -> Optional[float]:
    if not results:
        return None
    vol_stats = results.get("vol_stats") or {}
    return safe_float(vol_stats.get("blended_vol"), None)


def get_latest_oil_price_from_results(results: Optional[Dict[str, Any]]) -> Optional[float]:
    if not results:
        return None
    return safe_float(results.get("price"), None)


def get_exit_trigger_active(exit_df: Optional[pd.DataFrame]) -> bool:
    if exit_df is None or exit_df.empty or "should_exit" not in exit_df.columns:
        return False
    try:
        return bool(exit_df["should_exit"].fillna(False).any())
    except Exception:
        return False


def log_engine_snapshot(results, config):
    if results.get("skipped"):
        logging.info(
            "Snapshot | status=SKIPPED | reason=%s",
            results.get("skip_reason", "unknown"),
        )
        return

    ranked_df = results.get("ranked_df", pd.DataFrame())
    vol_stats = results.get("vol_stats") or {}

    if ranked_df is None or ranked_df.empty:
        logging.info(
            "Snapshot | price=%.2f | vol=%.4f | vol15=%.4f | vol60=%.4f | no ranked trades",
            float(results.get("price", 0) or 0),
            float(vol_stats.get("blended_vol", 0) or 0),
            float(vol_stats.get("short_vol", 0) or 0),
            float(vol_stats.get("medium_vol", 0) or 0),
        )
        return

    top = ranked_df.head(3)

    top_parts = []
    for _, row in top.iterrows():
        action = safe_upper(row.get("action"))
        edge = row.get("edge_yes") if action == "BUY_YES" else row.get("edge_no")
        edge_value = float(edge) if pd.notna(edge) else float("nan")
        model_prob = pick_model_prob_from_row(row, None)

        model_prob_part = ""
        if model_prob is not None:
            model_prob_part = f":p={model_prob:.3f}"

        top_parts.append(
            f"{row.get('contract_ticker')}:{action}:edge={edge_value:.3f}{model_prob_part}"
        )

    logging.info(
        "Snapshot | price=%.2f | vol=%.4f | vol15=%.4f | vol60=%.4f | series=%s | event=%s | top=%s",
        float(results.get("price", 0) or 0),
        float(vol_stats.get("blended_vol", 0) or 0),
        float(vol_stats.get("short_vol", 0) or 0),
        float(vol_stats.get("medium_vol", 0) or 0),
        config.get("kalshi", {}).get("series_ticker"),
        config.get("kalshi", {}).get("event_ticker"),
        " ; ".join(top_parts),
    )


def normalize_portfolio_plan(plan):
    if not plan:
        return {}

    actions = plan.get("actions") or []
    primary_action = actions[0] if actions else {}

    recommendation = str(plan.get("recommendation", "") or "").strip().upper()
    reason = str(plan.get("reason", "") or "").strip()

    normalized = {
        "recommendation": recommendation,
        "reason": reason,
        "capital": safe_float(plan.get("capital"), 0.0),
        "available_cash": safe_float(plan.get("available_cash"), 0.0),
        "reserve_cash_target": safe_float(plan.get("reserve_cash_target"), 0.0),
        "deployable_cash": safe_float(plan.get("deployable_cash"), 0.0),
        "actions": actions,
        "ticker": safe_str(primary_action.get("ticker")),
        "side": safe_upper(primary_action.get("side")),
        "contracts": safe_int(primary_action.get("contracts"), 0),
        "allocation": safe_float(primary_action.get("allocation"), 0.0),
        "ask_price": safe_float(
            primary_action.get("ask_price", primary_action.get("max_price")),
            None,
        ),
        "edge": safe_float(primary_action.get("edge"), None),
        "confidence": safe_upper(primary_action.get("confidence")),
        "action_type": safe_upper(primary_action.get("action")),
        "yes_no_ask_sum": safe_float(primary_action.get("yes_no_ask_sum"), None),
        "overround": safe_float(primary_action.get("overround"), None),
        "market_too_wide": primary_action.get("market_too_wide"),
        "no_trade_reason": safe_str(primary_action.get("no_trade_reason")),
        "decision_prob": safe_float(primary_action.get("decision_prob"), None),
        "fair_prob_terminal": safe_float(primary_action.get("fair_prob_terminal"), None),
        "fair_prob_blended": safe_float(primary_action.get("fair_prob_blended"), None),
    }

    summary = {
        "enter_count": sum(1 for a in actions if safe_upper(a.get("action")) == "ENTER"),
        "hold_count": sum(1 for a in actions if safe_upper(a.get("action")) == "HOLD"),
        "exit_count": sum(1 for a in actions if safe_upper(a.get("action")) == "EXIT"),
        "planned_enter_allocation": sum(
            safe_float(a.get("allocation"), 0.0)
            for a in actions
            if safe_upper(a.get("action")) == "ENTER"
        ),
    }

    normalized["summary"] = summary
    return normalized


def log_portfolio_recommendation(plan):
    if not plan:
        logging.info("Portfolio Recommendation | unavailable")
        return

    p = normalize_portfolio_plan(plan)
    summary = p.get("summary") or {}
    actions = p.get("actions") or []

    logging.info(
        (
            "Portfolio Recommendation | recommendation=%s | reason=%s | "
            "capital=%.2f | available_cash=%.2f | reserve_cash_target=%.2f | "
            "deployable_cash=%.2f | planned_enter_allocation=%.2f | "
            "enter_count=%s hold_count=%s exit_count=%s"
        ),
        p.get("recommendation"),
        p.get("reason"),
        float(p.get("capital", 0.0) or 0.0),
        float(p.get("available_cash", 0.0) or 0.0),
        float(p.get("reserve_cash_target", 0.0) or 0.0),
        float(p.get("deployable_cash", 0.0) or 0.0),
        float(summary.get("planned_enter_allocation", 0.0) or 0.0),
        summary.get("enter_count", 0),
        summary.get("hold_count", 0),
        summary.get("exit_count", 0),
    )

    if not actions:
        logging.info("Portfolio Actions | none")
        return

    for idx, action in enumerate(actions, start=1):
        logging.info(
            (
                "Portfolio Action %s | action=%s | ticker=%s | side=%s | "
                "contracts=%s | ask_price=%s | allocation=%.2f | edge=%s | "
                "confidence=%s | decision_prob=%s | fair_prob_terminal=%s | "
                "fair_prob_blended=%s | yes_no_ask_sum=%s | overround=%s | "
                "market_too_wide=%s | no_trade_reason=%s"
            ),
            idx,
            action.get("action"),
            action.get("ticker"),
            action.get("side"),
            action.get("contracts", 0),
            action.get("ask_price", action.get("max_price")),
            float(action.get("allocation", 0.0) or 0.0),
            action.get("edge"),
            action.get("confidence"),
            action.get("decision_prob"),
            action.get("fair_prob_terminal"),
            action.get("fair_prob_blended"),
            action.get("yes_no_ask_sum"),
            action.get("overround"),
            action.get("market_too_wide"),
            action.get("no_trade_reason"),
        )


def build_kalshi_contract_link(ticker, config):
    series = str(config.get("kalshi", {}).get("series_ticker", "")).lower()
    slug = str(config.get("kalshi", {}).get("market_slug", ""))
    contract = str(ticker).lower()

    if not series or not slug or not contract:
        return ""

    return f"https://kalshi.com/markets/{series}/{slug}/{contract}"


def get_entry_price_from_row(row):
    held_side = infer_held_side_from_row(row)
    action = f"BUY_{held_side}" if held_side in {"YES", "NO"} else str(row.get("action", ""))

    if "entry_price" in row and pd.notna(row.get("entry_price")):
        return float(row.get("entry_price"))

    if action == "BUY_YES" and pd.notna(row.get("ask_yes")):
        return float(row.get("ask_yes"))

    if action == "BUY_NO" and pd.notna(row.get("ask_no")):
        return float(row.get("ask_no"))

    return None


def get_expected_value_from_row(row):
    action = str(row.get("action", ""))

    if action == "BUY_YES" and pd.notna(row.get("ev_yes")):
        return float(row.get("ev_yes"))

    if action == "BUY_NO" and pd.notna(row.get("ev_no")):
        return float(row.get("ev_no"))

    return None


def send_telegram_alert(message, chat_id: Optional[str] = None):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    default_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    target_chat_id = str(chat_id or default_chat_id or "").strip()

    if not bot_token or not target_chat_id:
        logging.info("Telegram credentials not set. Skipping alert send.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": target_chat_id,
        "text": message,
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as exc:
        logging.warning("Telegram alert failed: %s", exc)


# ---------------------------------------------------------------------
# Trade alerts
# ---------------------------------------------------------------------
def build_trade_alert_key(row):
    return "||".join([
        str(row.get("contract_ticker", "")),
        str(row.get("action", "")),
    ])


def build_trade_alert_signature(row):
    return "||".join([
        str(row.get("contract_ticker", "")),
        str(row.get("action", "")),
        str(row.get("confidence", "")),
    ])


def should_send_trade_alert(row, config):
    global seen_trade_alert_state

    trade_key = build_trade_alert_key(row)
    now_ts = time.time()
    best_edge = safe_float(row.get("best_edge"), 0.0)
    current_signature = build_trade_alert_signature(row)

    cooldown_seconds = get_alert_cooldown_seconds(config)
    min_edge_improvement = get_alert_min_edge_improvement(config)

    prior = seen_trade_alert_state.get(trade_key)
    if prior is None:
        return True

    last_sent_ts = safe_float(prior.get("last_sent_ts"), 0.0)
    last_edge = safe_float(prior.get("last_edge"), 0.0)
    last_signature = str(prior.get("signature", ""))

    within_cooldown = (now_ts - last_sent_ts) < cooldown_seconds
    edge_improved = best_edge >= (last_edge + min_edge_improvement)
    signature_changed = current_signature != last_signature

    if signature_changed:
        return True

    if within_cooldown and not edge_improved:
        return False

    return edge_improved


def record_trade_alert_sent(row):
    global seen_trade_alert_state

    trade_key = build_trade_alert_key(row)
    best_edge = safe_float(row.get("best_edge"), 0.0)

    seen_trade_alert_state[trade_key] = {
        "last_sent_ts": time.time(),
        "last_edge": best_edge,
        "signature": build_trade_alert_signature(row),
    }


def format_trade_alert(row, oil_price, config):
    action = str(row.get("action", ""))
    strike = float(row.get("strike", 0))
    ticker = str(row.get("contract_ticker", ""))
    confidence = str(row.get("confidence", ""))

    if action == "BUY_YES":
        edge = float(row.get("edge_yes", 0))
    else:
        edge = float(row.get("edge_no", 0))

    market_prob = float(row.get("market_prob", 0) or 0)
    fair_prob = pick_model_prob_from_row(row, 0.0)

    entry_price = get_entry_price_from_row(row)
    expected_value = get_expected_value_from_row(row)

    distance_to_strike = row.get("distance_to_strike")
    hours_left = row.get("hours_left")

    trade_link = build_kalshi_contract_link(ticker, config)

    entry_price_text = f"{entry_price:.2f}" if entry_price is not None else "N/A"
    ev_text = f"{expected_value:.3f}" if expected_value is not None else "N/A"
    distance_text = f"{float(distance_to_strike):.2f}" if pd.notna(distance_to_strike) else "N/A"
    hours_left_text = f"{float(hours_left):.2f}" if pd.notna(hours_left) else "N/A"

    msg = (
        "🚨 KALSHI OIL TRADE SIGNAL\n\n"
        f"Contract: {ticker}\n"
        f"Action: {action}\n"
        f"Strike: {strike:.2f}\n"
        f"Oil Price: {oil_price:.2f}\n"
        f"Entry Price: {entry_price_text}\n\n"
        f"Market Prob: {market_prob:.3f}\n"
        f"Model Prob: {fair_prob:.3f}\n"
        f"Edge: {edge:.3f}\n"
        f"Expected Value: {ev_text}\n"
        f"Distance to Strike: {distance_text}\n"
        f"Hours Left: {hours_left_text}\n"
        f"Confidence: {confidence}\n"
    )

    optional_lines = []
    for label, col in [
        ("Yes+No Ask Sum", "yes_no_ask_sum"),
        ("Overround", "overround"),
        ("Market Too Wide", "market_too_wide"),
        ("No Trade Reason", "no_trade_reason"),
    ]:
        value = row.get(col)
        if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
            continue
        optional_lines.append(f"{label}: {value}")

    if optional_lines:
        msg += "\n" + "\n".join(optional_lines) + "\n"

    if trade_link:
        msg += f"\nTrade Link:\n{trade_link}"

    return msg


def process_trade_alerts(results, config, alerts_enabled: bool = True):
    if results.get("skipped"):
        logging.info("Trade alerts skipped: engine cycle was skipped.")
        return

    if not alerts_enabled:
        logging.info("Trade alerts blocked: bot is paused.")
        return

    ranked_df = results.get("ranked_df", pd.DataFrame())
    oil_price = float(results.get("price", 0.0) or 0.0)

    if ranked_df is None or ranked_df.empty:
        return

    allowed_conf = set(
        str(x).strip().upper()
        for x in config.get("decision", {}).get("allowed_confidence", ["HIGH", "MEDIUM"])
    )

    actionable = ranked_df.copy()
    actionable["action"] = actionable["action"].astype(str).str.strip().str.upper()
    actionable["confidence_norm"] = actionable["confidence"].astype(str).str.strip().str.upper()

    actionable = actionable[
        (actionable["action"].isin(["BUY_YES", "BUY_NO"])) &
        (actionable["confidence_norm"].isin(allowed_conf))
    ].copy()

    if actionable.empty:
        return

    actionable["best_edge"] = actionable[["edge_yes", "edge_no"]].max(axis=1)
    edge_threshold = get_alert_edge_threshold(config)
    actionable = actionable[actionable["best_edge"] >= edge_threshold]

    if actionable.empty:
        return

    distance_threshold = get_alert_distance_threshold(config)
    actionable = actionable[actionable["distance_to_strike"].abs() <= distance_threshold]

    if actionable.empty:
        return

    actionable = actionable.sort_values(
        by=["best_edge", "distance_to_strike"],
        ascending=[False, True],
    )

    max_alerts = get_max_alerts_per_cycle(config)
    actionable = actionable.head(max_alerts)

    for _, row in actionable.iterrows():
        if not should_send_trade_alert(row, config):
            logging.info(
                "Trade alert suppressed | ticker=%s action=%s edge=%.4f",
                row.get("contract_ticker"),
                row.get("action"),
                float(row.get("best_edge")) if pd.notna(row.get("best_edge")) else float("nan"),
            )
            continue

        message = format_trade_alert(row, oil_price, config)

        logging.info(
            "Trade alert sent | ticker=%s action=%s strike=%s edge=%.4f confidence=%s",
            row.get("contract_ticker"),
            row.get("action"),
            float(row.get("strike")) if pd.notna(row.get("strike")) else float("nan"),
            float(row.get("best_edge")) if pd.notna(row.get("best_edge")) else float("nan"),
            row.get("confidence"),
        )

        send_telegram_alert(message)
        record_trade_alert_sent(row)


# ---------------------------------------------------------------------
# Watchlist alerts
# ---------------------------------------------------------------------
def build_watchlist_df(ranked_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if ranked_df is None or ranked_df.empty:
        return pd.DataFrame()

    working = ranked_df.copy()

    if "decision_state" not in working.columns:
        return pd.DataFrame()

    working["decision_state"] = working["decision_state"].astype(str).str.strip().str.upper()

    watchlist_df = working[
        working["decision_state"].isin([
            "WAIT_FOR_PRICE",
            "PRICE_OK_BUT_SPREAD_TOO_WIDE",
        ])
    ].copy()

    if watchlist_df.empty:
        return watchlist_df

    if "confidence" in watchlist_df.columns:
        watchlist_df["confidence_rank"] = watchlist_df["confidence"].astype(str).str.strip().str.upper().map({
            "HIGH": 3,
            "MEDIUM": 2,
            "LOW": 1,
        }).fillna(0)
    else:
        watchlist_df["confidence_rank"] = 0

    if "distance_to_strike" in watchlist_df.columns:
        watchlist_df["distance_abs"] = pd.to_numeric(
            watchlist_df["distance_to_strike"], errors="coerce"
        ).abs()
    else:
        watchlist_df["distance_abs"] = np.inf

    for col in ["edge_yes", "edge_no"]:
        if col not in watchlist_df.columns:
            watchlist_df[col] = np.nan

    watchlist_df["watchlist_edge"] = watchlist_df[["edge_yes", "edge_no"]].max(axis=1, skipna=True)

    watchlist_df = watchlist_df.sort_values(
        by=["watchlist_edge", "confidence_rank", "distance_abs"],
        ascending=[False, False, True],
        kind="mergesort",
    )

    return watchlist_df


def log_watchlist_snapshot(watchlist_df: Optional[pd.DataFrame], top_n: int = 5):
    if watchlist_df is None or watchlist_df.empty:
        logging.info("Watchlist | no near-actionable watchlist rows.")
        return

    preview_cols = [
        "contract_ticker",
        "decision_state",
        "entry_style",
        "ask_yes",
        "ask_no",
        "target_yes_price",
        "target_no_price",
        "executable_yes_now",
        "executable_no_now",
        "edge_yes",
        "edge_no",
        "yes_no_ask_sum",
        "overround",
        "confidence",
        "distance_to_strike",
    ]
    preview_cols = [c for c in preview_cols if c in watchlist_df.columns]

    logging.info(
        "Watchlist top rows: %s",
        watchlist_df[preview_cols].head(top_n).to_dict(orient="records"),
    )


def build_watchlist_alert_key(row):
    return "||".join([
        safe_str(row.get("contract_ticker")),
        safe_upper(row.get("decision_state")),
    ])


def build_watchlist_alert_signature(row):
    return "||".join([
        safe_str(row.get("contract_ticker")),
        safe_upper(row.get("decision_state")),
        safe_upper(row.get("entry_style")),
        safe_upper(row.get("confidence")),
        format_optional_float(row.get("target_yes_price"), ".4f"),
        format_optional_float(row.get("target_no_price"), ".4f"),
        format_optional_float(row.get("ask_yes"), ".4f"),
        format_optional_float(row.get("ask_no"), ".4f"),
        format_optional_float(row.get("watchlist_edge"), ".4f"),
        format_optional_float(row.get("overround"), ".4f"),
    ])


def should_send_watchlist_alert(row, config):
    global seen_watchlist_alert_state

    alert_key = build_watchlist_alert_key(row)
    current_signature = build_watchlist_alert_signature(row)
    cooldown_seconds = get_watchlist_alert_cooldown_seconds(config)
    now_ts = time.time()

    prior = seen_watchlist_alert_state.get(alert_key)
    if prior is None:
        return True

    last_sent_ts = safe_float(prior.get("last_sent_ts"), 0.0)
    last_signature = safe_str(prior.get("signature"))

    if current_signature != last_signature:
        return True

    within_cooldown = (now_ts - last_sent_ts) < cooldown_seconds
    return not within_cooldown


def record_watchlist_alert_sent(row):
    global seen_watchlist_alert_state

    alert_key = build_watchlist_alert_key(row)
    seen_watchlist_alert_state[alert_key] = {
        "last_sent_ts": time.time(),
        "signature": build_watchlist_alert_signature(row),
        "decision_state": safe_upper(row.get("decision_state")),
        "ticker": safe_str(row.get("contract_ticker")),
    }


def build_watchlist_alert_type(row):
    state = safe_upper(row.get("decision_state"))
    if state == "WAIT_FOR_PRICE":
        return "watchlist_wait_for_price"
    if state == "PRICE_OK_BUT_SPREAD_TOO_WIDE":
        return "watchlist_spread_blocked"
    return "watchlist_other"


def format_watchlist_alert(row, oil_price, config):
    ticker = safe_str(row.get("contract_ticker")) or "N/A"
    decision_state = safe_upper(row.get("decision_state")) or "N/A"
    entry_style = safe_upper(row.get("entry_style")) or "N/A"
    confidence = safe_upper(row.get("confidence")) or "N/A"

    watchlist_edge = safe_float(row.get("watchlist_edge"), None)
    edge_yes = safe_float(row.get("edge_yes"), None)
    edge_no = safe_float(row.get("edge_no"), None)

    ask_yes = safe_float(row.get("ask_yes"), None)
    ask_no = safe_float(row.get("ask_no"), None)
    target_yes = safe_float(row.get("target_yes_price"), None)
    target_no = safe_float(row.get("target_no_price"), None)

    executable_yes_now = row.get("executable_yes_now")
    executable_no_now = row.get("executable_no_now")

    overround = safe_float(row.get("overround"), None)
    ask_sum = safe_float(row.get("yes_no_ask_sum"), None)
    distance = safe_float(row.get("distance_to_strike"), None)
    strike = safe_float(row.get("strike"), None)
    fair_prob = pick_model_prob_from_row(row, None)

    alert_type = build_watchlist_alert_type(row)
    title = "👀 OIL WATCHLIST ALERT"

    if alert_type == "watchlist_wait_for_price":
        reason_line = "Setup is interesting, but current price has not reached your target yet."
    elif alert_type == "watchlist_spread_blocked":
        reason_line = "A fair side exists, but combined market spread/overround is blocking entry."
    else:
        reason_line = "Near-actionable setup detected."

    lines = [
        title,
        "",
        f"Type: {alert_type}",
        f"Contract: {ticker}",
        f"Decision State: {decision_state}",
        f"Entry Style: {entry_style}",
        f"Oil Price: {format_optional_float(oil_price, '.2f')}",
        f"Strike: {format_optional_float(strike, '.2f')}",
        f"Confidence: {confidence}",
        f"Model Prob: {format_optional_float(fair_prob, '.3f')}",
        f"Best Edge: {format_optional_float(watchlist_edge, '.3f')}",
        f"Edge Yes: {format_optional_float(edge_yes, '.3f')}",
        f"Edge No: {format_optional_float(edge_no, '.3f')}",
        f"Ask Yes: {format_optional_float(ask_yes, '.2f')}",
        f"Ask No: {format_optional_float(ask_no, '.2f')}",
        f"Target Yes: {format_optional_float(target_yes, '.2f')}",
        f"Target No: {format_optional_float(target_no, '.2f')}",
        f"Executable Yes Now: {executable_yes_now}",
        f"Executable No Now: {executable_no_now}",
        f"Yes+No Ask Sum: {format_optional_float(ask_sum, '.2f')}",
        f"Overround: {format_optional_float(overround, '.4f')}",
        f"Distance to Strike: {format_optional_float(distance, '.2f')}",
        "",
        f"Why this is on watchlist: {reason_line}",
    ]

    trade_link = build_kalshi_contract_link(ticker, config)
    if trade_link:
        lines.extend(["", f"Trade Link:\n{trade_link}"])

    return "\n".join(lines)


def process_watchlist_alerts(results, config, alerts_enabled: bool = True) -> pd.DataFrame:
    if results.get("skipped"):
        logging.info("Watchlist alerts skipped: engine cycle was skipped.")
        return pd.DataFrame()

    ranked_df = results.get("ranked_df", pd.DataFrame())
    oil_price = float(results.get("price", 0.0) or 0.0)

    if ranked_df is None or ranked_df.empty:
        logging.info("Watchlist alerts skipped: ranked_df is empty.")
        return pd.DataFrame()

    watchlist_df = build_watchlist_df(ranked_df)
    log_watchlist_snapshot(watchlist_df, top_n=5)

    if watchlist_df.empty:
        return watchlist_df

    if not alerts_enabled:
        logging.info("Watchlist alerts blocked: bot is paused.")
        return watchlist_df

    top_row = watchlist_df.iloc[0].copy()

    if not should_send_watchlist_alert(top_row, config):
        logging.info(
            "Watchlist alert suppressed | ticker=%s state=%s edge=%s",
            top_row.get("contract_ticker"),
            top_row.get("decision_state"),
            top_row.get("watchlist_edge"),
        )
        return watchlist_df

    alert_type = build_watchlist_alert_type(top_row)
    message = format_watchlist_alert(top_row, oil_price, config)

    logging.info(
        "Watchlist alert sent | type=%s | ticker=%s | state=%s | confidence=%s | edge=%s",
        alert_type,
        top_row.get("contract_ticker"),
        top_row.get("decision_state"),
        top_row.get("confidence"),
        top_row.get("watchlist_edge"),
    )
    send_telegram_alert(message)
    record_watchlist_alert_sent(top_row)

    return watchlist_df


# ---------------------------------------------------------------------
# Portfolio advisory + portfolio alerts
# ---------------------------------------------------------------------
def build_portfolio_alert_signature(plan):
    p = normalize_portfolio_plan(plan)
    actions = p.get("actions") or []

    action_parts = []
    for a in actions[:10]:
        action_parts.append(
            "~~".join([
                safe_upper(a.get("action")),
                safe_str(a.get("ticker")),
                safe_upper(a.get("side")),
                str(safe_int(a.get("contracts"), 0)),
                f"{safe_float(a.get('allocation'), 0.0):.2f}",
                f"{safe_float(a.get('ask_price', a.get('max_price')), -1.0):.4f}",
                f"{safe_float(a.get('edge'), -999.0):.4f}",
                safe_upper(a.get("confidence")),
                f"{safe_float(a.get('decision_prob'), -1.0):.4f}",
                f"{safe_float(a.get('fair_prob_terminal'), -1.0):.4f}",
                f"{safe_float(a.get('fair_prob_blended'), -1.0):.4f}",
                f"{safe_float(a.get('yes_no_ask_sum'), -1.0):.4f}",
                f"{safe_float(a.get('overround'), -1.0):.4f}",
                safe_str(a.get("market_too_wide")),
                safe_str(a.get("no_trade_reason")),
            ])
        )

    return "||".join([
        p.get("recommendation", ""),
        p.get("reason", ""),
        f"{safe_float(p.get('capital'), 0.0):.2f}",
        f"{safe_float(p.get('available_cash'), 0.0):.2f}",
        f"{safe_float(p.get('reserve_cash_target'), 0.0):.2f}",
        f"{safe_float(p.get('deployable_cash'), 0.0):.2f}",
        *action_parts,
    ])


def portfolio_plan_has_material_change(current_plan, prior_state, config):
    if not prior_state:
        return True

    current_sig = build_portfolio_alert_signature(current_plan)
    prior_sig = str(prior_state.get("signature", "") or "")
    return current_sig != prior_sig


def should_send_portfolio_alert(plan, config):
    global seen_portfolio_alert_state

    if not plan:
        return False

    now_ts = time.time()
    cooldown_seconds = get_portfolio_alert_cooldown_seconds(config)

    prior = seen_portfolio_alert_state.get("portfolio")
    if prior is None:
        return True

    last_sent_ts = safe_float(prior.get("last_sent_ts"), None)

    cooldown_ok = (
        last_sent_ts is None
        or (now_ts - last_sent_ts) >= cooldown_seconds
    )

    has_material_change = portfolio_plan_has_material_change(plan, prior, config)

    if has_material_change:
        return True

    return cooldown_ok


def record_portfolio_alert_sent(plan):
    global seen_portfolio_alert_state

    p = normalize_portfolio_plan(plan)

    seen_portfolio_alert_state["portfolio"] = {
        "last_sent_ts": time.time(),
        "signature": build_portfolio_alert_signature(plan),
        "recommendation": p.get("recommendation"),
        "reason": p.get("reason"),
        "last_plan": p,
    }


def format_portfolio_alert(plan, config):
    p = normalize_portfolio_plan(plan)
    actions = p.get("actions") or []

    lines = [
        "🚀 Oil Portfolio Signal",
        "",
        f"Recommendation: {p.get('recommendation')}",
        f"Reason: {p.get('reason')}",
        "",
    ]

    if not actions:
        lines.append("No portfolio actions generated.")
        return "\n".join(lines)

    for i, a in enumerate(actions, start=1):
        lines.extend([
            f"Action {i}: {a.get('action')}",
            f"Ticker: {a.get('ticker')}",
            f"Side: {a.get('side')}",
            f"Contracts: {a.get('contracts')}",
            f"Allocation: ${a.get('allocation')}",
            f"Ask Price: {a.get('ask_price', a.get('max_price'))}",
            f"Edge: {a.get('edge')}",
            f"Confidence: {a.get('confidence')}",
        ])

        for label, key in [
            ("Decision Prob", "decision_prob"),
            ("Fair Prob Terminal", "fair_prob_terminal"),
            ("Fair Prob Blended", "fair_prob_blended"),
            ("Yes+No Ask Sum", "yes_no_ask_sum"),
            ("Overround", "overround"),
            ("Market Too Wide", "market_too_wide"),
            ("No Trade Reason", "no_trade_reason"),
        ]:
            value = a.get(key)
            if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
                continue
            lines.append(f"{label}: {value}")

        lines.append("")

    return "\n".join(lines).rstrip()


def emit_portfolio_alert(plan, config, alerts_enabled: bool = True):
    if not plan:
        logging.info("Portfolio alert skipped: plan unavailable.")
        return

    p = normalize_portfolio_plan(plan)

    if not alerts_enabled:
        logging.info(
            "Portfolio alert blocked: bot is paused | recommendation=%s",
            p.get("recommendation"),
        )
        return

    if not should_send_portfolio_alert(plan, config):
        logging.info(
            "Portfolio alert suppressed | recommendation=%s | reason=%s",
            p.get("recommendation"),
            p.get("reason"),
        )
        return

    message = format_portfolio_alert(plan, config)
    logging.info(
        "Portfolio alert sent | recommendation=%s | reason=%s | actions=%s",
        p.get("recommendation"),
        p.get("reason"),
        len(p.get("actions") or []),
    )
    send_telegram_alert(message)
    record_portfolio_alert_sent(plan)



def held_position_state_store(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    store = state.get("held_position_state")
    if not isinstance(store, dict):
        store = {}
        state["held_position_state"] = store
    return store


def _held_state_key(ticker: Any, side: Any) -> str:
    return "||".join([safe_str(ticker), safe_upper(side)])


def _count_prior_exit_attempts_for_position(state: Dict[str, Any], ticker: Any, side: Any) -> int:
    intents = order_intent_store(state)
    ticker = safe_str(ticker)
    side = safe_upper(side)
    count = 0
    for intent in intents.values():
        if not isinstance(intent, dict):
            continue
        if safe_upper(intent.get("type")) != "EXIT":
            continue
        if safe_str(intent.get("ticker")) != ticker or safe_upper(intent.get("side")) != side:
            continue
        if safe_upper(intent.get("state")) in {
            ORDER_INTENT_STATE_RECONCILED_NOT_FILLED,
            ORDER_INTENT_STATE_RECONCILED_PARTIAL,
            ORDER_INTENT_STATE_AWAITING_RECONCILIATION,
            ORDER_INTENT_STATE_SUBMITTED_LIVE,
            ORDER_INTENT_STATE_SUBMITTED_SIMULATED,
        }:
            count += 1
    return count


def enrich_exit_df_with_held_state(
    exit_df: Optional[pd.DataFrame],
    state: Dict[str, Any],
    config: Dict[str, Any],
    portfolio_value: Optional[float] = None,
) -> pd.DataFrame:
    if exit_df is None or exit_df.empty:
        return pd.DataFrame() if exit_df is None else exit_df

    out = exit_df.copy()
    store = held_position_state_store(state)
    rotation_cfg = (config.get("portfolio_rotation", {}) or {})
    portfolio_cfg = (config.get("portfolio", {}) or {})
    weak_edge_threshold = float(rotation_cfg.get("stale_held_edge_threshold", portfolio_cfg.get("stale_held_edge_threshold", 0.06)))
    recovery_edge_threshold = float(rotation_cfg.get("stale_held_recovery_edge_threshold", max(weak_edge_threshold + 0.02, 0.08)))

    seen_keys = set()
    for idx, row in out.iterrows():
        ticker = safe_str(row.get("contract_ticker") or row.get("ticker"))
        side = infer_held_side_from_row(row)
        key = _held_state_key(ticker, side)
        seen_keys.add(key)

        rec = dict(store.get(key) or {})
        held_edge = safe_float(row.get("current_edge", row.get("held_edge")), None)
        weak_cycles = safe_int(rec.get("weak_cycles"), 0) or 0

        if held_edge is not None and held_edge <= weak_edge_threshold:
            weak_cycles += 1
        elif held_edge is not None and held_edge >= recovery_edge_threshold:
            weak_cycles = 0

        exit_attempts = max(
            safe_int(rec.get("exit_attempts"), 0) or 0,
            _count_prior_exit_attempts_for_position(state, ticker, side),
        )

        rec.update({
            "ticker": ticker,
            "side": side,
            "weak_cycles": int(weak_cycles),
            "exit_attempts": int(exit_attempts),
            "last_held_edge": held_edge,
            "updated_at": time.time(),
        })
        store[key] = rec

        out.at[idx, "weak_cycles"] = int(weak_cycles)
        out.at[idx, "exit_attempts"] = int(exit_attempts)
        out.at[idx, "portfolio_value"] = safe_float(portfolio_value, None)

    for key in list(store.keys()):
        if key not in seen_keys:
            store.pop(key, None)

    state["held_position_state"] = store
    return out


def build_portfolio_advisory_plan(
    results,
    config,
    open_positions_df=None,
    account_snapshot: Optional[Dict[str, Any]] = None,
):
    if results.get("skipped"):
        logging.info("Portfolio advisory skipped: engine cycle was skipped.")
        return None

    ranked_df = results.get("ranked_df", pd.DataFrame())
    if ranked_df is None or ranked_df.empty:
        logging.info("Portfolio advisory skipped: ranked_df is empty.")
        return None

    plan = build_micro_allocation_plan(
        ranked_df=ranked_df,
        live_positions_df=open_positions_df,
        config=config,
        account_snapshot=account_snapshot,
    )
    return plan


# ---------------------------------------------------------------------
# Exit alerts + exit evaluation
# ---------------------------------------------------------------------
def build_exit_alert_signature(row):
    return "||".join([
        str(row.get("position_id", "")),
        str(row.get("contract_ticker", "")),
        str(row.get("exit_reason", "")),
    ])


def should_send_exit_alert(row):
    global seen_exit_alert_state

    sig = build_exit_alert_signature(row)
    return sig not in seen_exit_alert_state


def record_exit_alert_sent(row):
    global seen_exit_alert_state

    sig = build_exit_alert_signature(row)
    seen_exit_alert_state[sig] = {
        "last_sent_ts": time.time(),
        "ticker": row.get("contract_ticker"),
        "reason": row.get("exit_reason"),
    }


def format_exit_alert(row, config):
    ticker = str(row.get("contract_ticker", ""))
    action = str(row.get("action", ""))
    strike = float(row.get("strike", 0))
    entry_price = row.get("entry_price")
    current_price = row.get("current_position_price")
    unrealized_pnl = row.get("unrealized_pnl")
    fair_now = pick_model_prob_from_row(row, None)
    fair_entry = row.get("entry_fair_prob")
    reason = str(row.get("exit_reason", "Exit condition triggered"))

    trade_link = build_kalshi_contract_link(ticker, config)

    entry_price_text = f"{float(entry_price):.2f}" if pd.notna(entry_price) else "N/A"
    current_price_text = f"{float(current_price):.2f}" if pd.notna(current_price) else "N/A"
    pnl_text = f"{float(unrealized_pnl):.3f}" if pd.notna(unrealized_pnl) else "N/A"
    fair_now_text = f"{float(fair_now):.3f}" if fair_now is not None else "N/A"
    fair_entry_text = f"{float(fair_entry):.3f}" if pd.notna(fair_entry) else "N/A"

    msg = (
        "⚠️ KALSHI OIL EXIT ALERT\n\n"
        f"Contract: {ticker}\n"
        f"Action Held: {action}\n"
        f"Strike: {strike:.2f}\n\n"
        f"Entry Price: {entry_price_text}\n"
        f"Current Position Price: {current_price_text}\n"
        f"Unrealized PnL: {pnl_text}\n"
        f"Entry Fair Prob: {fair_entry_text}\n"
        f"Current Fair Prob: {fair_now_text}\n\n"
        f"Reason: {reason}\n"
    )

    if trade_link:
        msg += f"\nTrade Link:\n{trade_link}"

    return msg


def compute_position_exit_df(results, config, open_positions_df=None) -> pd.DataFrame:
    if results.get("skipped"):
        logging.info("Exit evaluation skipped: engine cycle was skipped.")
        return pd.DataFrame()

    ranked_df = results.get("ranked_df", pd.DataFrame())
    if ranked_df is None or ranked_df.empty:
        logging.info("Exit evaluation skipped: ranked_df is empty.")
        return pd.DataFrame()

    if open_positions_df is None or open_positions_df.empty:
        logging.info("Exit evaluation skipped: no open positions available.")
        return pd.DataFrame()

    monitor_df = monitor_open_positions(open_positions_df, ranked_df)
    if monitor_df is None or monitor_df.empty:
        logging.info("Exit evaluation skipped: monitor_open_positions returned empty.")
        return pd.DataFrame()

    portfolio_value = safe_float((runtime_state.get("last_account_snapshot") or {}).get("portfolio_value"), None)
    monitor_df = enrich_exit_df_with_held_state(
        monitor_df,
        runtime_state,
        config,
        portfolio_value=portfolio_value,
    )

    exit_cfg = config.get("exit", {})
    exit_df = evaluate_exit_rules(
    monitor_df,
    config
    )
    if exit_df is None or exit_df.empty:
        return pd.DataFrame()

    exit_df = enrich_exit_df_with_held_state(
        exit_df,
        runtime_state,
        config,
        portfolio_value=portfolio_value,
    )
    return exit_df


def emit_position_exit_alerts(exit_df: Optional[pd.DataFrame], config):
    if exit_df is None or exit_df.empty:
        logging.info("Exit monitor: no exit dataframe to alert from.")
        return

    flagged = exit_df[exit_df["should_exit"] == True].copy() if "should_exit" in exit_df.columns else pd.DataFrame()

    if flagged.empty:
        logging.info("Exit monitor: no exit triggers active.")
        return

    for _, row in flagged.iterrows():
        if not should_send_exit_alert(row):
            logging.info(
                "Exit alert suppressed | ticker=%s action=%s strike=%s reason=%s",
                row.get("contract_ticker"),
                row.get("action"),
                row.get("strike"),
                row.get("exit_reason"),
            )
            continue

        message = format_exit_alert(row, config)
        logging.info(
            "Exit alert sent | ticker=%s action=%s strike=%s reason=%s",
            row.get("contract_ticker"),
            row.get("action"),
            row.get("strike"),
            row.get("exit_reason"),
        )
        send_telegram_alert(message)
        record_exit_alert_sent(row)


# ---------------------------------------------------------------------
# Execution intent planning / reconciliation
# ---------------------------------------------------------------------
def execution_tracking_store(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    tracker = state.get("execution_tracking")
    if not isinstance(tracker, dict):
        tracker = {}
        state["execution_tracking"] = tracker
    return tracker


def order_intent_store(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    intents = state.get("order_intents")
    if not isinstance(intents, dict):
        intents = {}
        state["order_intents"] = intents
    return intents


def current_live_position_lookup(open_positions_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    if open_positions_df is None or open_positions_df.empty:
        return lookup

    df = open_positions_df.copy()
    if "contract_ticker" not in df.columns:
        return lookup

    for _, row in df.iterrows():
        ticker = safe_str(row.get("contract_ticker"))
        if not ticker:
            continue
        action = safe_upper(row.get("action"))
        held_side = "YES" if action == "BUY_YES" else ("NO" if action == "BUY_NO" else safe_upper(row.get("side_norm")))
        contracts = abs(safe_int(row.get("contracts", row.get("size", row.get("quantity"))), 0) or 0)
        lookup[ticker] = {
            "ticker": ticker,
            "held_side": held_side,
            "contracts": contracts,
            "row": row.to_dict() if hasattr(row, "to_dict") else {},
        }
    return lookup


def build_execution_action_key(action_type: str, ticker: Any, side: Any = None) -> str:
    return "||".join([
        safe_upper(action_type),
        safe_str(ticker),
        safe_upper(side),
    ])


def build_execution_action_signature(action: Dict[str, Any]) -> str:
    return "||".join([
        safe_upper(action.get("action_type")),
        safe_str(action.get("ticker")),
        safe_upper(action.get("side")),
        str(safe_int(action.get("contracts"), 0)),
        f"{safe_float(action.get('allocation'), 0.0):.2f}",
        f"{safe_float(action.get('ask_price'), -1.0):.4f}",
        f"{safe_float(action.get('edge'), -999.0):.4f}",
        safe_str(action.get("reason")),
        safe_str(action.get("source")),
    ])


def build_order_payload_from_execution_action(action: Dict[str, Any]) -> Dict[str, Any]:
    action_type = safe_upper(action.get("action_type"))
    ticker = safe_str(action.get("ticker"))
    side = safe_upper(action.get("side"))
    contracts = safe_int(action.get("contracts"), 0)
    allocation = safe_float(action.get("allocation"), 0.0)
    ask_price = safe_float(action.get("ask_price"), None)
    reason = safe_str(action.get("reason"))
    source = safe_str(action.get("source"))
    created_at = time.time()
    intent_key = build_execution_action_key(action_type, ticker, side)

    return {
        "intent_key": intent_key,
        "ticker": ticker,
        "type": action_type,
        "side": side,
        "contracts": contracts,
        "allocation": allocation,
        "ask_price": ask_price,
        "reason": reason,
        "source": source,
        "state": ORDER_INTENT_STATE_READY_TO_SUBMIT,
        "created_at": created_at,
        "updated_at": created_at,
        "submitted_at": None,
        "last_submission_attempt_at": None,
        "submission_attempts": 0,
        "heartbeat_count": 0,
        "signature": build_execution_action_signature(action),
        "metadata": {
            "confidence": safe_upper(action.get("confidence")),
            "edge": safe_float(action.get("edge"), None),
            "decision_prob": safe_float(action.get("decision_prob"), None),
            "fair_prob_terminal": safe_float(action.get("fair_prob_terminal"), None),
            "fair_prob_blended": safe_float(action.get("fair_prob_blended"), None),
            "market_too_wide": action.get("market_too_wide"),
            "no_trade_reason": safe_str(action.get("no_trade_reason")),
        },
    }


def validate_execution_action(
    action: Dict[str, Any],
    live_lookup: Dict[str, Dict[str, Any]],
    deployable_cash_remaining: float,
    state: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    action_type = safe_upper(action.get("action_type"))
    ticker = safe_str(action.get("ticker"))
    side = safe_upper(action.get("side"))
    allocation = safe_float(action.get("allocation"), 0.0)
    held = live_lookup.get(ticker)

    if action_type == "EXIT":
        if not ticker or held is None:
            return EXECUTION_STATE_SKIPPED_NO_ACTIVE_POSITION, "No active engine-scoped position found to exit."
        return EXECUTION_STATE_READY, safe_str(action.get("reason")) or "Exit action validated."

    if action_type == "ENTER":
        if allocation > max(deployable_cash_remaining, 0.0) + 1e-9:
            return EXECUTION_STATE_SKIPPED_NO_CASH, "Deployable cash unavailable for proposed entry."

        if held is not None and side and held.get("held_side") == side:
            return EXECUTION_STATE_SKIPPED_DUPLICATE, "Same-side position already appears active in live positions."

        existing_intents = (state or {}).get("order_intents") or {}
        for intent in existing_intents.values():
            if not isinstance(intent, dict):
                continue

            intent_ticker = safe_str(intent.get("ticker"))
            intent_type = safe_upper(intent.get("type"))
            intent_side = safe_upper(intent.get("side"))
            intent_state = safe_upper(intent.get("state"))

            if intent_ticker != ticker or intent_type != "ENTER":
                continue

            if side and intent_side and intent_side != side:
                continue

            if intent_state in {
                ORDER_INTENT_STATE_READY_TO_SUBMIT,
                ORDER_INTENT_STATE_SUBMITTED_SIMULATED,
                ORDER_INTENT_STATE_SUBMITTED_LIVE,
                ORDER_INTENT_STATE_AWAITING_RECONCILIATION,
                ORDER_INTENT_STATE_RECONCILED_PARTIAL,
                ORDER_INTENT_STATE_SKIPPED_DUPLICATE_PENDING,
            }:
                return (
                    EXECUTION_STATE_SKIPPED_DUPLICATE,
                    "Duplicate entry intent already exists for this ticker.",
                )

        cooldown_seconds = get_min_time_between_trades_seconds(config or {})
        has_any_active_positions = any(
            abs(safe_int((rec or {}).get("contracts"), 0) or 0) > 0
            for rec in (live_lookup or {}).values()
        )
        last_entry_trade_ts = get_last_entry_trade_timestamp_for_key(state, ticker, side)
        if cooldown_seconds > 0 and has_any_active_positions and last_entry_trade_ts is not None:
            elapsed = max(0.0, time.time() - last_entry_trade_ts)
            if elapsed < cooldown_seconds:
                return (
                    EXECUTION_STATE_SKIPPED_CONFLICT,
                    f"Entry cooldown active for {ticker}/{side} ({elapsed:.0f}s elapsed < {cooldown_seconds}s minimum while positions remain open).",
                )
        return EXECUTION_STATE_READY, safe_str(action.get("reason")) or "Entry action validated."

    if action_type == "HOLD":
        return EXECUTION_STATE_INFORMATIONAL, safe_str(action.get("reason")) or "Hold is informational only."

    return EXECUTION_STATE_PLANNED, safe_str(action.get("reason")) or "Planned informational action."


def mark_execution_action_awaiting_reconciliation(action: Dict[str, Any], tracker: Dict[str, Dict[str, Any]]) -> None:
    key = build_execution_action_key(
        action_type=action.get("action_type"),
        ticker=action.get("ticker"),
        side=action.get("side"),
    )
    tracker[key] = {
        "action_key": key,
        "action_type": safe_upper(action.get("action_type")),
        "ticker": safe_str(action.get("ticker")),
        "side": safe_upper(action.get("side")),
        "signature": build_execution_action_signature(action),
        "state": EXECUTION_STATE_AWAITING_RECON,
        "first_seen_ts": tracker.get(key, {}).get("first_seen_ts", time.time()),
        "last_seen_ts": time.time(),
        "reason": safe_str(action.get("reason")),
        "resolved": False,
    }


def reconcile_execution_tracking(
    tracking_state: Dict[str, Dict[str, Any]],
    open_positions_df: Optional[pd.DataFrame],
) -> Dict[str, Dict[str, Any]]:
    tracker = dict(tracking_state or {})
    live_lookup = current_live_position_lookup(open_positions_df)

    for action_key, rec in list(tracker.items()):
        if not isinstance(rec, dict):
            tracker.pop(action_key, None)
            continue

        ticker = safe_str(rec.get("ticker"))
        action_type = safe_upper(rec.get("action_type"))
        side = safe_upper(rec.get("side"))
        resolved = bool(rec.get("resolved", False))
        live_record = live_lookup.get(ticker)

        if resolved:
            continue

        if action_type == "EXIT":
            if live_record is None:
                rec["resolved"] = True
                rec["state"] = EXECUTION_STATE_RECONCILED
                rec["resolution_reason"] = "Position no longer active in live positions."
                rec["resolved_ts"] = time.time()

        elif action_type == "ENTER":
            if live_record is not None and (not side or live_record.get("held_side") == side):
                rec["resolved"] = True
                rec["state"] = EXECUTION_STATE_RECONCILED
                rec["resolution_reason"] = "Target position now appears active in live positions."
                rec["resolved_ts"] = time.time()

        tracker[action_key] = rec

    return tracker


def build_execution_action_record(
    *,
    action_type: str,
    ticker: Any,
    side: Any = None,
    contracts: Any = None,
    allocation: Any = None,
    ask_price: Any = None,
    edge: Any = None,
    confidence: Any = None,
    reason: Any = None,
    decision_prob: Any = None,
    fair_prob_terminal: Any = None,
    fair_prob_blended: Any = None,
    source: str = "",
    market_too_wide: Any = None,
    no_trade_reason: Any = None,
    execution_state: str = EXECUTION_STATE_PLANNED,
    execution_reason: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action = {
        "action_type": safe_upper(action_type),
        "ticker": safe_str(ticker),
        "side": safe_upper(side),
        "contracts": safe_int(contracts, 0),
        "allocation": safe_float(allocation, 0.0),
        "ask_price": safe_float(ask_price, None),
        "edge": safe_float(edge, None),
        "confidence": safe_upper(confidence),
        "reason": safe_str(reason),
        "decision_prob": safe_float(decision_prob, None),
        "fair_prob_terminal": safe_float(fair_prob_terminal, None),
        "fair_prob_blended": safe_float(fair_prob_blended, None),
        "source": safe_str(source),
        "market_too_wide": market_too_wide,
        "no_trade_reason": safe_str(no_trade_reason),
        "execution_state": safe_upper(execution_state),
        "execution_reason": safe_str(execution_reason),
    }
    if extra:
        action.update(extra)
    action["order_intent"] = build_order_payload_from_execution_action(action)
    action["action_key"] = build_execution_action_key(action["action_type"], action["ticker"], action["side"])
    action["signature"] = build_execution_action_signature(action)
    return action



def _resolve_live_held_side_for_ticker(
    ticker: Any,
    open_positions_df: Optional[pd.DataFrame],
    fallback_side: Any = None,
) -> str:
    ticker_text = safe_str(ticker)
    fallback = safe_upper(fallback_side)
    if not ticker_text:
        return fallback

    if open_positions_df is not None and not open_positions_df.empty:
        working = open_positions_df.copy()
        ticker_col = None
        for candidate in ["contract_ticker", "ticker", "market_ticker"]:
            if candidate in working.columns:
                ticker_col = candidate
                break
        if ticker_col is not None:
            matches = working[working[ticker_col].astype(str).str.strip() == ticker_text]
            if not matches.empty:
                for _, match_row in matches.iterrows():
                    held_side = infer_held_side_from_row(match_row)
                    if held_side in {"YES", "NO"}:
                        return held_side
    return fallback


def _exit_action_priority(action: Dict[str, Any]) -> Tuple[int, float]:
    source = safe_str(action.get("source")).lower()
    ask_price = safe_float(action.get("ask_price"), 0.0)
    reason = safe_str(action.get("reason")).lower()

    priority = 5
    if source == "exit_monitor":
        priority = 0
    elif "probability support deteriorated" in reason:
        priority = 1
    elif "price collapse" in reason:
        priority = 1
    elif source == "portfolio_manager":
        priority = 3

    return (priority, -ask_price)


def collapse_execution_conflicts(
    actions: List[Dict[str, Any]],
    open_positions_df: Optional[pd.DataFrame],
    tracking_state: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    tracking_state = tracking_state or {}
    collapsed: List[Dict[str, Any]] = []

    # -------------------------------------------------
    # 1) Collapse multiple EXIT actions per ticker down
    #    to one true held-side exit.
    # -------------------------------------------------
    exits_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    non_exit_actions: List[Dict[str, Any]] = []

    for action in actions:
        action_type = safe_upper(action.get("action_type"))
        ticker = safe_str(action.get("ticker"))
        if action_type == "EXIT" and ticker:
            exits_by_ticker.setdefault(ticker, []).append(action)
        else:
            non_exit_actions.append(action)

    kept_exit_tickers = set()

    for ticker, exit_group in exits_by_ticker.items():
        true_side = _resolve_live_held_side_for_ticker(
            ticker=ticker,
            open_positions_df=open_positions_df,
            fallback_side=(exit_group[0].get("side") if exit_group else None),
        )

        matching = [a for a in exit_group if safe_upper(a.get("side")) == true_side]
        if not matching:
            matching = exit_group

        chosen = sorted(matching, key=_exit_action_priority)[0]
        chosen["side"] = true_side or safe_upper(chosen.get("side"))
        chosen["action_key"] = build_execution_action_key("EXIT", chosen.get("ticker"), chosen.get("side"))
        chosen["signature"] = build_execution_action_signature(chosen)
        collapsed.append(chosen)
        kept_exit_tickers.add(ticker)

        for other in exit_group:
            if other is chosen:
                continue
            other["execution_state"] = EXECUTION_STATE_SKIPPED_CONFLICT
            other["execution_reason"] = (
                f"Suppressed conflicting exit for ticker={ticker}; retained only one true held-side exit."
            )
            collapsed.append(other)

    # -------------------------------------------------
    # 2) If a ticker already has an unresolved EXIT
    #    intent, suppress any ENTER for that same ticker.
    # -------------------------------------------------
    unresolved_exit_tickers = set()
    for rec in (tracking_state or {}).values():
        if not isinstance(rec, dict):
            continue
        if safe_upper(rec.get("action_type")) != "EXIT":
            continue
        if bool(rec.get("resolved", False)):
            continue
        unresolved_exit_tickers.add(safe_str(rec.get("ticker")))

    # -------------------------------------------------
    # 3) Block same-cycle flip trades:
    #    if EXIT exists for ticker X in this cycle,
    #    suppress ENTER for ticker X until later cycle.
    # -------------------------------------------------
    for action in non_exit_actions:
        action_type = safe_upper(action.get("action_type"))
        ticker = safe_str(action.get("ticker"))

        if action_type == "ENTER" and ticker:
            if ticker in kept_exit_tickers:
                action["execution_state"] = EXECUTION_STATE_SKIPPED_CONFLICT
                action["execution_reason"] = (
                    "Blocked same-cycle re-entry on ticker with active exit action; wait for exit reconciliation first."
                )
            elif ticker in unresolved_exit_tickers:
                action["execution_state"] = EXECUTION_STATE_SKIPPED_CONFLICT
                action["execution_reason"] = (
                    "Blocked entry because an unresolved prior-cycle exit intent still exists for this ticker."
                )

        if action_type == "HOLD" and ticker and ticker in kept_exit_tickers:
            action["execution_state"] = EXECUTION_STATE_SKIPPED_CONFLICT
            action["execution_reason"] = (
                "Suppressed informational hold because ticker is already scheduled for exit this cycle."
            )

        collapsed.append(action)

    return collapsed



def build_exit_execution_actions(
    exit_df: Optional[pd.DataFrame],
    open_positions_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    if exit_df is None or exit_df.empty:
        return actions
    if "should_exit" not in exit_df.columns:
        return actions

    flagged = exit_df[exit_df["should_exit"] == True].copy()
    if flagged.empty:
        return actions

    for _, row in flagged.iterrows():
        resolved_side = _resolve_live_held_side_for_ticker(
            ticker=row.get("contract_ticker"),
            open_positions_df=open_positions_df,
            fallback_side=infer_held_side_from_row(row),
        )
        if resolved_side not in {"YES", "NO"}:
            logging.info(
                "Exit action skipped: could not resolve held side | ticker=%s",
                row.get("contract_ticker"),
            )
            continue

        current_edge = safe_float(row.get("current_edge"), None)
        if current_edge is None:
            current_edge = (
                safe_float(row.get("edge_yes"), None)
                if resolved_side == "YES"
                else safe_float(row.get("edge_no"), None)
            )

        exit_price, tif_override, pricing_note = choose_exit_execution_params(
            row=row,
            resolved_side=resolved_side,
            config=runtime_state.get("last_results_config") or {},
        )
        if exit_price is None:
            exit_price = safe_float(
                row.get("current_position_price", row.get("current_price")),
                None,
            )

        confidence = (
            row.get("current_confidence")
            if row.get("current_confidence") is not None
            else row.get("confidence")
        )

        action = build_execution_action_record(
            action_type="EXIT",
            ticker=row.get("contract_ticker"),
            side=resolved_side,
            contracts=row.get("contracts", row.get("size", row.get("quantity"))),
            allocation=None,
            ask_price=exit_price,
            edge=current_edge,
            confidence=confidence,
            reason=row.get("exit_reason"),
            decision_prob=row.get("decision_prob"),
            fair_prob_terminal=row.get("fair_prob_terminal"),
            fair_prob_blended=row.get("fair_prob_blended"),
            source="exit_monitor",
            market_too_wide=row.get("market_too_wide"),
            no_trade_reason=row.get("no_trade_reason"),
            extra={
                "exit_decision_state": safe_upper(row.get("held_decision_state") or row.get("decision_state")),
                "exit_severity": safe_int(row.get("held_state_severity"), 0),
                "time_in_force": tif_override,
                "execution_pricing_note": pricing_note,
            },
        )
        actions.append(action)

    return actions

def build_portfolio_execution_actions(portfolio_plan: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    if not portfolio_plan:
        return actions

    for row in portfolio_plan.get("actions") or []:
        action_type = safe_upper(row.get("action"))
        if not action_type:
            continue

        if action_type == "ENTER":
            actions.append(
                build_execution_action_record(
                    action_type="ENTER",
                    ticker=row.get("ticker"),
                    side=row.get("side"),
                    contracts=row.get("contracts"),
                    allocation=row.get("allocation"),
                    ask_price=row.get("ask_price", row.get("max_price")),
                    edge=row.get("edge"),
                    confidence=row.get("confidence"),
                    reason=row.get("reason"),
                    decision_prob=row.get("decision_prob"),
                    fair_prob_terminal=row.get("fair_prob_terminal"),
                    fair_prob_blended=row.get("fair_prob_blended"),
                    source="portfolio_manager",
                    market_too_wide=row.get("market_too_wide"),
                    no_trade_reason=row.get("no_trade_reason"),
                )
            )
        elif action_type == "HOLD":
            actions.append(
                build_execution_action_record(
                    action_type="HOLD",
                    ticker=row.get("ticker"),
                    side=row.get("side"),
                    contracts=row.get("contracts"),
                    allocation=row.get("allocation"),
                    ask_price=row.get("ask_price", row.get("max_price")),
                    edge=row.get("edge"),
                    confidence=row.get("confidence"),
                    reason=row.get("reason"),
                    decision_prob=row.get("decision_prob"),
                    fair_prob_terminal=row.get("fair_prob_terminal"),
                    fair_prob_blended=row.get("fair_prob_blended"),
                    source="portfolio_manager",
                    market_too_wide=row.get("market_too_wide"),
                    no_trade_reason=row.get("no_trade_reason"),
                )
            )
        elif action_type == "EXIT":
            actions.append(
                build_execution_action_record(
                    action_type="EXIT",
                    ticker=row.get("ticker"),
                    side=row.get("side"),
                    contracts=row.get("contracts"),
                    allocation=row.get("allocation"),
                    ask_price=row.get("ask_price", row.get("max_price")),
                    edge=row.get("edge"),
                    confidence=row.get("confidence"),
                    reason=row.get("reason"),
                    decision_prob=row.get("decision_prob"),
                    fair_prob_terminal=row.get("fair_prob_terminal"),
                    fair_prob_blended=row.get("fair_prob_blended"),
                    source="portfolio_manager",
                    market_too_wide=row.get("market_too_wide"),
                    no_trade_reason=row.get("no_trade_reason"),
                )
            )

    return actions


def deduplicate_execution_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_keys = set()

    for action in actions:
        key = build_execution_action_key(
            action_type=action.get("action_type"),
            ticker=action.get("ticker"),
            side=action.get("side"),
        )
        if key in seen_keys:
            action["execution_state"] = EXECUTION_STATE_SKIPPED_DUPLICATE
            action["execution_reason"] = "Duplicate action removed during execution-plan construction."
            deduped.append(action)
            continue
        seen_keys.add(key)
        deduped.append(action)

    return deduped


def estimate_exit_cash_proceeds(exit_actions: List[Dict[str, Any]]) -> float:
    estimated = 0.0
    for action in exit_actions:
        contracts = abs(safe_int(action.get("contracts"), 0) or 0)
        ask_price = safe_float(action.get("ask_price"), None)
        if ask_price is None:
            continue
        estimated += contracts * max(ask_price, 0.0)
    return estimated



def apply_execution_lifecycle(
    actions: List[Dict[str, Any]],
    open_positions_df: Optional[pd.DataFrame],
    portfolio_plan: Optional[Dict[str, Any]],
    tracking_state: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    live_lookup = current_live_position_lookup(open_positions_df)
    deployable_cash = safe_float((portfolio_plan or {}).get("deployable_cash"), 0.0)

    executable_exit_actions = [
        a for a in actions
        if safe_upper(a.get("action_type")) == "EXIT"
        and safe_upper(a.get("execution_state")) not in {
            EXECUTION_STATE_SKIPPED_CONFLICT,
            EXECUTION_STATE_SKIPPED_DUPLICATE,
            EXECUTION_STATE_SKIPPED_NO_ACTIVE_POSITION,
            EXECUTION_STATE_SKIPPED_NO_CASH,
        }
    ]
    estimated_exit_proceeds = estimate_exit_cash_proceeds(executable_exit_actions)
    deployable_cash_remaining = deployable_cash + estimated_exit_proceeds
    ready_count = 0
    skipped_count = 0
    awaiting_count = 0

    processed: List[Dict[str, Any]] = []

    for action in actions:
        pre_state = safe_upper(action.get("execution_state"))
        pre_reason = safe_str(action.get("execution_reason"))
        action_type = safe_upper(action.get("action_type"))
        action_key = build_execution_action_key(action_type, action.get("ticker"), action.get("side"))
        signature = build_execution_action_signature(action)
        prior = tracking_state.get(action_key, {})

        if pre_state.startswith("SKIPPED"):
            state = pre_state
            reason = pre_reason or "Action skipped before lifecycle validation."
        else:
            state, reason = validate_execution_action(
                action=action,
                live_lookup=live_lookup,
                deployable_cash_remaining=deployable_cash_remaining,
                state=runtime_state,
                config=runtime_state.get("last_results_config") or {},
            )

        if prior and not prior.get("resolved", False):
            prior_sig = safe_str(prior.get("signature"))
            prior_state = safe_upper(prior.get("state"))
            if prior_sig == signature and prior_state in {EXECUTION_STATE_AWAITING_RECON, EXECUTION_STATE_READY}:
                state = EXECUTION_STATE_AWAITING_RECON
                if action_type == "EXIT":
                    reason = "Prior exit intent already recorded; position still active and awaiting reconciliation."
                elif action_type == "ENTER":
                    reason = "Identical entry intent already exists and remains unresolved; keeping it active while awaiting reconciliation."

        if state == EXECUTION_STATE_READY and action_type == "ENTER":
            allocation = safe_float(action.get("allocation"), 0.0)
            deployable_cash_remaining = max(0.0, deployable_cash_remaining - max(allocation, 0.0))

        action["execution_state"] = state
        action["execution_reason"] = reason
        action["action_key"] = action_key
        action["signature"] = signature
        action["order_intent"] = build_order_payload_from_execution_action(action)

        if state == EXECUTION_STATE_READY:
            ready_count += 1
        elif state == EXECUTION_STATE_AWAITING_RECON:
            awaiting_count += 1
        elif state.startswith("SKIPPED"):
            skipped_count += 1

        processed.append(action)

    stats = {
        "deployable_cash_start": deployable_cash,
        "estimated_exit_proceeds": estimated_exit_proceeds,
        "deployable_cash_effective_start": deployable_cash + estimated_exit_proceeds,
        "deployable_cash_remaining": deployable_cash_remaining,
        "ready_count": ready_count,
        "skipped_count": skipped_count,
        "awaiting_count": awaiting_count,
    }
    return processed, stats



def register_execution_plan_actions(plan: Dict[str, Any], state: Dict[str, Any]) -> None:
    tracker = execution_tracking_store(state)
    for action in plan.get("all_actions", []) or []:
        action_type = safe_upper(action.get("action_type"))
        exec_state = safe_upper(action.get("execution_state"))
        if action_type not in {"EXIT", "ENTER"}:
            continue
        if exec_state not in {EXECUTION_STATE_READY, EXECUTION_STATE_AWAITING_RECON}:
            continue
        mark_execution_action_awaiting_reconciliation(action, tracker)



def build_execution_intent_plan(
    *,
    portfolio_plan: Optional[Dict[str, Any]],
    exit_df: Optional[pd.DataFrame],
    open_positions_df: Optional[pd.DataFrame],
    account_snapshot: Optional[Dict[str, Any]],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    tracker = execution_tracking_store(state)
    tracker = reconcile_execution_tracking(tracker, open_positions_df)
    state["execution_tracking"] = tracker

    exit_actions = build_exit_execution_actions(
        exit_df,
        open_positions_df=open_positions_df,
    )
    portfolio_actions = build_portfolio_execution_actions(portfolio_plan)

    entry_actions = [a for a in portfolio_actions if safe_upper(a.get("action_type")) == "ENTER"]
    hold_actions = [a for a in portfolio_actions if safe_upper(a.get("action_type")) == "HOLD"]
    portfolio_exit_actions = [a for a in portfolio_actions if safe_upper(a.get("action_type")) == "EXIT"]

    raw_actions = exit_actions + portfolio_exit_actions + entry_actions + hold_actions
    raw_actions = deduplicate_execution_actions(raw_actions)
    raw_actions = collapse_execution_conflicts(
        raw_actions,
        open_positions_df=open_positions_df,
        tracking_state=tracker,
    )

    ordered_actions: List[Dict[str, Any]] = []
    ordered_actions.extend([a for a in raw_actions if safe_upper(a.get("action_type")) == "EXIT"])
    ordered_actions.extend([a for a in raw_actions if safe_upper(a.get("action_type")) == "ENTER"])
    ordered_actions.extend([a for a in raw_actions if safe_upper(a.get("action_type")) == "HOLD"])

    lifecycle_actions, lifecycle_stats = apply_execution_lifecycle(
        actions=ordered_actions,
        open_positions_df=open_positions_df,
        portfolio_plan=portfolio_plan,
        tracking_state=tracker,
    )

    exits_final = [a for a in lifecycle_actions if safe_upper(a.get("action_type")) == "EXIT"]
    entries_final = [a for a in lifecycle_actions if safe_upper(a.get("action_type")) == "ENTER"]
    holds_final = [a for a in lifecycle_actions if safe_upper(a.get("action_type")) == "HOLD"]
    skipped_final = [a for a in lifecycle_actions if safe_upper(a.get("execution_state")).startswith("SKIPPED")]
    ready_final = [a for a in lifecycle_actions if safe_upper(a.get("execution_state")) == EXECUTION_STATE_READY]

    plan = {
        "cash_balance": safe_float((account_snapshot or {}).get("cash_balance"), 0.0),
        "portfolio_value": safe_float((account_snapshot or {}).get("portfolio_value"), 0.0),
        "planned_exit_count": len(exits_final),
        "planned_enter_count": len(entries_final),
        "informational_hold_count": len(holds_final),
        "skipped_count": len(skipped_final),
        "ready_count": len(ready_final),
        "awaiting_count": lifecycle_stats.get("awaiting_count", 0),
        "deployable_cash_start": lifecycle_stats.get("deployable_cash_start", 0.0),
        "estimated_exit_proceeds": lifecycle_stats.get("estimated_exit_proceeds", 0.0),
        "deployable_cash_effective_start": lifecycle_stats.get("deployable_cash_effective_start", 0.0),
        "deployable_cash_remaining": lifecycle_stats.get("deployable_cash_remaining", 0.0),
        "ordering": "EXIT_FIRST_THEN_ENTER_THEN_HOLD",
        "exit_actions": exits_final,
        "entry_actions": entries_final,
        "hold_actions": holds_final,
        "all_actions": lifecycle_actions,
        "reconciliation_summary": {
            "tracked_count": len(tracker),
            "active_unresolved_count": sum(1 for rec in tracker.values() if not rec.get("resolved", False)),
            "resolved_count": sum(1 for rec in tracker.values() if rec.get("resolved", False)),
        },
    }

    return plan



def get_order_intent_material_change_reason(existing_intent: Dict[str, Any], new_intent: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    if safe_str(existing_intent.get("ticker")) != safe_str(new_intent.get("ticker")):
        return "ticker changed"
    if safe_upper(existing_intent.get("side")) != safe_upper(new_intent.get("side")):
        return "side changed"
    if safe_upper(existing_intent.get("type")) != safe_upper(new_intent.get("type")):
        return "type changed"
    if values_materially_different(existing_intent.get("contracts"), new_intent.get("contracts"), get_order_material_contract_change(config)):
        return "contracts changed materially"
    if values_materially_different(existing_intent.get("allocation"), new_intent.get("allocation"), get_order_material_allocation_change(config)):
        return "allocation changed materially"
    if values_materially_different(existing_intent.get("ask_price"), new_intent.get("ask_price"), get_order_material_ask_change(config)):
        return "ask price changed materially"
    if safe_str(existing_intent.get("reason")) != safe_str(new_intent.get("reason")):
        return "reason changed"
    if safe_str(existing_intent.get("source")) != safe_str(new_intent.get("source")):
        return "source changed"
    return None


def build_live_client_order_id(intent: Dict[str, Any]) -> str:
    ticker = safe_str(intent.get("ticker")) or "unknown"
    side = safe_upper(intent.get("side")) or "NA"
    intent_type = safe_upper(intent.get("type")) or "NA"
    ts_ms = int(time.time() * 1000)
    entropy = uuid.uuid4().hex[:6]
    return f"{ticker}-{side}-{intent_type}-{ts_ms}-{entropy}"


def get_exit_reprice_after_seconds(config: Dict[str, Any]) -> int:
    runtime_cfg = config.get("runtime", {}) or {}
    return int(runtime_cfg.get("live_exit_reprice_after_seconds", 45))


def get_exit_stale_after_seconds(config: Dict[str, Any]) -> int:
    runtime_cfg = config.get("runtime", {}) or {}
    base = int(runtime_cfg.get("live_exit_stale_after_seconds", 180))
    return max(base, get_exit_reprice_after_seconds(config) + 15)


def get_live_exit_execution_mode(config: Dict[str, Any]) -> str:
    runtime_cfg = config.get("runtime", {}) or {}
    mode = safe_str(runtime_cfg.get("live_exit_execution_mode") or "aggressive").lower()
    return mode if mode in {"aggressive", "balanced", "passive"} else "aggressive"


def get_exit_price_tick(config: Dict[str, Any]) -> float:
    runtime_cfg = config.get("runtime", {}) or {}
    tick = safe_float(runtime_cfg.get("live_exit_price_tick"), 0.01)
    if tick is None or tick <= 0:
        tick = 0.01
    return min(max(tick, 0.01), 0.05)


def _clip_binary_price(value: Any, default: Optional[float] = None) -> Optional[float]:
    numeric = safe_float(value, default)
    if numeric is None:
        return default
    return max(0.01, min(0.99, float(numeric)))
def choose_exit_execution_params(row: Any, resolved_side: str, config: Dict[str, Any]) -> Tuple[Optional[float], str, str]:
    side = safe_upper(resolved_side)

    bid = safe_float(row.get("bid_yes"), None) if side == "YES" else safe_float(row.get("bid_no"), None)
    ask = safe_float(row.get("ask_yes"), None) if side == "YES" else safe_float(row.get("ask_no"), None)

    current_price = safe_float(
        row.get("current_position_price", row.get("current_price")),
        None,
    )
    entry_price = safe_float(row.get("entry_price"), None)
    held_edge = safe_float(row.get("current_edge", row.get("held_edge")), None)
    severity = safe_int(row.get("held_state_severity"), 0) or 0
    exit_reason = safe_str(row.get("exit_reason")).lower()
    decision_state = safe_upper(row.get("held_decision_state") or row.get("decision_state"))
    held_executable_now = bool(row.get("held_executable_now", False))
    market_too_wide = bool(row.get("market_too_wide", False))
    fair_gap = safe_float(row.get("fair_price_gap"), None)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _clip_price(value: Optional[float], fallback: Optional[float] = None) -> Optional[float]:
        base = value if value is not None else fallback
        return _clip_binary_price(base, fallback)

    def _round_binary_price(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return _clip_price(round(float(value), 2), value)

    tick = get_exit_price_tick(config)

    # Compute pnl ratio safely
    pnl_ratio = None
    if entry_price is not None and entry_price > 0 and current_price is not None:
        pnl_ratio = (float(current_price) - float(entry_price)) / float(entry_price)

    # Compute spread characteristics
    spread = None
    if bid is not None and ask is not None:
        spread = max(0.0, float(ask) - float(bid))

    tight_spread = spread is not None and spread <= 0.02
    medium_spread = spread is not None and spread <= 0.04

    # -----------------------------
    # Exit regime classification
    # -----------------------------
    emergency_exit = (
        severity >= 5
        or any(
            token in exit_reason
            for token in [
                "force_stop",
                "hard_stop",
                "stop_loss",
                "price_collapse",
                "probability_collapse",
                "market_deteriorated",
                "support_lost",
                "fair_value_collapsed",
            ]
        )
        or (pnl_ratio is not None and pnl_ratio <= -0.18)
    )

    urgent_exit = (
        emergency_exit
        or severity >= 4
        or any(
            token in exit_reason
            for token in [
                "early_stop",
                "edge_negative",
                "edge_collapsed",
                "losing_and_weak_edge",
                "profit_weakening",
            ]
        )
        or (pnl_ratio is not None and pnl_ratio <= -0.10)
        or (decision_state == "NOT_TRADABLE" and not held_executable_now)
    )

    profit_protect_exit = (
        not emergency_exit
        and pnl_ratio is not None
        and pnl_ratio >= 0.10
    )

    # -----------------------------
    # Price selection
    # -----------------------------
    price = None
    tif = "good_till_cancelled"
    note = "fallback_exit"

    # 1) Emergency exits: prioritize getting out
    if emergency_exit:
        tif = "immediate_or_cancel"
        if bid is not None:
            price = bid
            note = "emergency_ioc_at_bid"
        elif current_price is not None:
            price = current_price
            note = "emergency_ioc_at_current"
        elif ask is not None:
            price = max(0.01, ask - tick)
            note = "emergency_ioc_from_ask_minus_tick"
        else:
            price = 0.01
            note = "emergency_ioc_floor"

    # 2) Urgent exits: still aggressive, but allow minimal improvement if market is tight
    elif urgent_exit:
        tif = "good_till_cancelled"
        if bid is not None and ask is not None:
            if tight_spread:
                price = min(ask, bid + tick)
                note = "urgent_gtc_bid_plus_tick_tight_spread"
            else:
                price = bid
                note = "urgent_gtc_at_bid_wide_spread"
        elif bid is not None:
            price = bid
            note = "urgent_gtc_at_bid"
        elif current_price is not None:
            price = current_price
            note = "urgent_gtc_at_current"
        elif ask is not None:
            price = max(0.01, ask - tick)
            note = "urgent_gtc_from_ask_minus_tick"
        else:
            price = 0.01
            note = "urgent_gtc_floor"

    # 3) Profit protection exits: try to capture better price when market quality allows
    elif profit_protect_exit:
        tif = "good_till_cancelled"
        if bid is not None and ask is not None:
            if tight_spread:
                price = ask
                note = "profit_lock_gtc_at_ask_tight_spread"
            elif medium_spread:
                price = min(ask, bid + tick)
                note = "profit_lock_gtc_bid_plus_tick"
            else:
                price = bid
                note = "profit_lock_gtc_at_bid_wide_spread"
        elif bid is not None:
            price = bid
            note = "profit_lock_gtc_at_bid"
        elif current_price is not None:
            price = current_price
            note = "profit_lock_gtc_at_current"
        elif ask is not None:
            price = ask
            note = "profit_lock_gtc_at_ask_only"
        else:
            price = 0.01
            note = "profit_lock_gtc_floor"

    # 4) Lower urgency / balanced exits
    else:
        tif = "good_till_cancelled"
        if bid is not None and ask is not None:
            if tight_spread:
                price = ask
                note = "balanced_gtc_at_ask_tight_spread"
            elif medium_spread:
                price = min(ask, bid + tick)
                note = "balanced_gtc_bid_plus_tick"
            else:
                price = bid
                note = "balanced_gtc_at_bid_wide_spread"
        elif bid is not None:
            price = bid
            note = "balanced_gtc_at_bid"
        elif current_price is not None:
            price = current_price
            note = "balanced_gtc_at_current"
        elif ask is not None:
            price = max(0.01, ask - tick)
            note = "balanced_gtc_from_ask_minus_tick"
        else:
            price = 0.01
            note = "balanced_gtc_floor"

    # -----------------------------
    # Safety adjustments
    # -----------------------------
    # If market is too wide or fair value is already below market, don't get cute.
    if price is not None and bid is not None and ask is not None:
        if market_too_wide or (fair_gap is not None and fair_gap < 0):
            if urgent_exit or emergency_exit:
                price = bid
                note += "|degraded_to_bid_due_to_market_quality"

    price = _round_binary_price(price)

    return price, tif, note

def cancel_live_order_by_id(order_id: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    order_id_text = safe_str(order_id)
    if not order_id_text:
        return {"ok": False, "status": "missing_order_id", "message": "No order_id available for cancel."}

    client = get_kalshi_auth_client_for_reconciliation(config)
    if client is None:
        return {"ok": False, "status": "missing_auth", "message": "Kalshi auth unavailable for cancel."}

    try:
        response = client.cancel_order(order_id_text)
        return {"ok": True, "status": "cancelled", "message": "Cancel submitted.", "response": response}
    except Exception as exc:
        return {"ok": False, "status": "cancel_failed", "message": str(exc)}


def submit_live_order_stub(intent: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    api_key_id = os.getenv("KALSHI_API_KEY_ID")
    if not api_key_id:
        return {
            "ok": False,
            "submitted": False,
            "message": "Live order submission skipped: KALSHI_API_KEY_ID is missing.",
            "external_order_id": None,
            "status": "skipped_missing_api_key",
        }

    try:
        private_key_path = load_kalshi_private_key()
    except Exception as exc:
        return {
            "ok": False,
            "submitted": False,
            "message": f"Live order submission skipped: private key unavailable ({exc}).",
            "external_order_id": None,
            "status": "skipped_missing_private_key",
        }

    runtime_cfg = config.get("runtime", {}) or {}
    kalshi_cfg = config.get("kalshi", {}) or {}

    intent_type = safe_upper(intent.get("type"))
    intent_side = safe_upper(intent.get("side"))
    ticker = safe_str(intent.get("ticker"))
    contracts = abs(safe_int(intent.get("contracts"), 0) or 0)
    original_contracts = contracts

    ask_price = safe_float(intent.get("ask_price"), None)
    ask_price = _clip_binary_price(ask_price, ask_price)

    live_max_entry_contracts = safe_int(runtime_cfg.get("live_max_entry_contracts"), 999)
    if (
        intent_type == "ENTER"
        and live_max_entry_contracts is not None
        and live_max_entry_contracts > 0
    ):
        contracts = min(contracts, live_max_entry_contracts)

    entry_tif = safe_str(
        runtime_cfg.get("live_order_time_in_force") or "good_till_cancelled"
    ).lower()
    exit_tif = safe_str(
        runtime_cfg.get("live_exit_order_time_in_force") or "good_till_cancelled"
    ).lower()

    tif = safe_str(intent.get("time_in_force")).lower()
    if not tif:
        tif = exit_tif if intent_type == "EXIT" else entry_tif

    post_only = bool(
        intent.get("post_only", runtime_cfg.get("live_order_post_only", False))
    )
    cancel_on_pause = bool(runtime_cfg.get("live_order_cancel_on_pause", True))
    subaccount = safe_int(runtime_cfg.get("kalshi_subaccount"), 0) or 0

    # ENTER = buy same side
    # EXIT  = sell same side
    if intent_type == "ENTER":
        side = intent_side
        order_action = "buy"
    elif intent_type == "EXIT":
        side = intent_side
        order_action = "sell"
    else:
        return {
            "ok": False,
            "submitted": False,
            "message": f"Unsupported intent type: {intent_type}",
            "external_order_id": None,
            "status": "invalid_intent_type",
        }

    if not ticker or side not in {"YES", "NO"} or contracts <= 0:
        return {
            "ok": False,
            "submitted": False,
            "message": "Live order submission skipped: intent missing ticker/side/contracts prerequisites.",
            "external_order_id": None,
            "status": "skipped_invalid_intent",
        }

    if ask_price is None:
        return {
            "ok": False,
            "submitted": False,
            "message": "Live order submission skipped: ask_price is missing or invalid.",
            "external_order_id": None,
            "status": "skipped_invalid_price",
        }

    if contracts != original_contracts:
        logging.info(
            "Live entry contract cap applied | ticker=%s type=%s intent_side=%s submit_side=%s original_contracts=%s capped_contracts=%s max_entry_contracts=%s",
            ticker,
            intent_type,
            intent_side,
            side,
            original_contracts,
            contracts,
            live_max_entry_contracts,
        )

    reduce_only = order_action == "sell"
    client_order_id = safe_str(intent.get("client_order_id")) or build_live_client_order_id(intent)
    base_url = safe_str(kalshi_cfg.get("api_base")) or None

    logging.info(
        "Live submit prepared | ticker=%s type=%s intent_side=%s submit_side=%s order_action=%s contracts=%s ask_price=%s tif=%s post_only=%s reduce_only=%s pricing_note=%s",
        ticker,
        intent_type,
        intent_side,
        side,
        order_action,
        contracts,
        ask_price,
        tif,
        post_only,
        reduce_only,
        safe_str(intent.get("execution_pricing_note")),
    )

    try:
        response = submit_kalshi_order(
            api_key_id=api_key_id,
            private_key_path=private_key_path,
            base_url=base_url,
            ticker=ticker,
            side=side.lower(),
            action=order_action,
            count=contracts,
            max_price=ask_price,
            client_order_id=client_order_id,
            subaccount=subaccount,
            time_in_force=tif,
            post_only=post_only,
            reduce_only=reduce_only,
            cancel_order_on_pause=cancel_on_pause,
        )

        order_payload = response.get("order") if isinstance(response, dict) else None
        order_payload = order_payload if isinstance(order_payload, dict) else {}

        order_id = safe_str(order_payload.get("order_id")) or safe_str(
            response.get("order_id") if isinstance(response, dict) else None
        )
        status = safe_str(order_payload.get("status")) or safe_str(
            response.get("status") if isinstance(response, dict) else None
        ) or "submitted"

        logging.info(
            "Kalshi order submit success | ticker=%s type=%s intent_side=%s submit_side=%s contracts=%s original_contracts=%s order_id=%s status=%s",
            ticker,
            intent_type,
            intent_side,
            side,
            contracts,
            original_contracts,
            order_id,
            status,
        )

        return {
            "ok": True,
            "submitted": True,
            "message": "Kalshi live order submitted successfully.",
            "external_order_id": order_id or None,
            "status": status,
            "client_order_id": client_order_id,
            "response": response,
            "submitted_contracts": contracts,
            "original_contracts": original_contracts,
            "intent_side": intent_side,
            "submitted_side": side,
        }

    except Exception as exc:
        error_text = str(exc)
        error_lower = error_text.lower()
        status = "submission_failed"

        if "fill_or_kill_insufficient_resting_volume" in error_lower:
            status = "insufficient_resting_volume"
        elif "insufficient_balance" in error_lower:
            status = "insufficient_balance"
        elif "order_already_exists" in error_lower:
            status = "order_already_exists"
        elif "invalid" in error_lower and "time" in error_lower and "force" in error_lower:
            status = "invalid_time_in_force"
        elif "post only" in error_lower or "post_only" in error_lower:
            status = "post_only_rejected"

        logging.warning(
            "Kalshi order submit failed | ticker=%s type=%s intent_side=%s submit_side=%s contracts=%s original_contracts=%s status=%s error=%s",
            ticker,
            intent_type,
            intent_side,
            side,
            contracts,
            original_contracts,
            status,
            error_text,
        )

        return {
            "ok": False,
            "submitted": False,
            "message": f"Kalshi order submit failed: {error_text}",
            "external_order_id": None,
            "status": status,
            "client_order_id": client_order_id,
            "submitted_contracts": contracts,
            "original_contracts": original_contracts,
            "intent_side": intent_side,
            "submitted_side": side,
        }

def get_live_reconciliation_lookback_seconds(config: Dict[str, Any]) -> int:
    runtime_cfg = config.get("runtime", {}) or {}
    return int(runtime_cfg.get("live_reconciliation_lookback_seconds", 900))


def _safe_nested_get(payload: Any, path: List[str]) -> Any:
    cur = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _extract_first_nonempty(payload: Dict[str, Any], candidate_paths: List[List[str]]) -> Any:
    for path in candidate_paths:
        value = _safe_nested_get(payload, path)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _extract_first_numeric_value(payload: Dict[str, Any], candidate_paths: List[List[str]]) -> Optional[float]:
    for path in candidate_paths:
        value = _extract_first_nonempty(payload, [path])
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            continue
    return None


def _normalize_order_status(value: Any) -> str:
    return safe_str(value).strip().lower().replace(" ", "_")


def get_kalshi_auth_client_for_reconciliation(config: Dict[str, Any]) -> Optional[KalshiAuthClient]:
    api_key_id = os.getenv("KALSHI_API_KEY_ID")
    if not api_key_id:
        return None
    try:
        private_key_path = load_kalshi_private_key()
    except Exception:
        return None
    base_url = safe_str((config.get("kalshi", {}) or {}).get("api_base")) or None
    return KalshiAuthClient(api_key_id=api_key_id, private_key_path=private_key_path, base_url=base_url)


def fetch_live_reconciliation_snapshot(intent: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "ok": False,
        "order_found": False,
        "order_status": "",
        "filled_contracts": None,
        "remaining_contracts": None,
        "matched_contracts": None,
        "fills_contracts": 0,
        "fills_count": 0,
        "matched_order_id": None,
        "matched_client_order_id": None,
        "reason": "",
        "raw_order": None,
        "raw_fills": None,
    }

    client = get_kalshi_auth_client_for_reconciliation(config)
    if client is None:
        snapshot["reason"] = "Kalshi auth client unavailable for live reconciliation."
        return snapshot

    ticker = safe_str(intent.get("ticker"))
    order_id = safe_str(intent.get("order_id"))
    client_order_id = safe_str(intent.get("client_order_id"))
    submitted_at = safe_float(intent.get("submitted_at"), time.time())
    lookback_seconds = max(60, get_live_reconciliation_lookback_seconds(config))
    min_ts = int(max(0, submitted_at - lookback_seconds) * 1000)

    order_payload: Dict[str, Any] = {}
    orders_payload: Dict[str, Any] = {}
    try:
        if order_id:
            raw = client.get_order(order_id)
            if isinstance(raw, dict):
                order_payload = raw.get("order") if isinstance(raw.get("order"), dict) else raw
    except Exception as exc:
        logging.info("Live reconciliation get_order failed | ticker=%s order_id=%s error=%s", ticker, order_id, exc)

    if not order_payload:
        try:
            raw_orders = client.get_orders(ticker=ticker or None, limit=100)
            if isinstance(raw_orders, dict):
                orders_payload = raw_orders
                records = raw_orders.get("orders")
                if isinstance(records, list):
                    match = None
                    for rec in records:
                        if not isinstance(rec, dict):
                            continue
                        rec_order_id = safe_str(rec.get("order_id"))
                        rec_client_order_id = safe_str(rec.get("client_order_id"))
                        if order_id and rec_order_id == order_id:
                            match = rec
                            break
                        if client_order_id and rec_client_order_id == client_order_id:
                            match = rec
                            break
                    if match is None and len(records) == 1:
                        match = records[0] if isinstance(records[0], dict) else None
                    if isinstance(match, dict):
                        order_payload = match
        except Exception as exc:
            logging.info("Live reconciliation get_orders failed | ticker=%s client_order_id=%s error=%s", ticker, client_order_id, exc)

    fills_payload: Dict[str, Any] = {}
    try:
        raw_fills = client.get_fills(
            ticker=ticker or None,
            order_id=order_id or None,
            min_ts=min_ts,
            limit=100,
        )
        if isinstance(raw_fills, dict):
            fills_payload = raw_fills
    except Exception as exc:
        logging.info("Live reconciliation get_fills failed | ticker=%s order_id=%s error=%s", ticker, order_id, exc)

    if order_payload:
        snapshot["order_found"] = True
        snapshot["raw_order"] = order_payload
        snapshot["matched_order_id"] = safe_str(order_payload.get("order_id")) or order_id or None
        snapshot["matched_client_order_id"] = safe_str(order_payload.get("client_order_id")) or client_order_id or None
        snapshot["order_status"] = _normalize_order_status(
            _extract_first_nonempty(
                order_payload,
                [
                    ["status"],
                    ["order_status"],
                    ["state"],
                    ["order", "status"],
                ],
            )
        )
        snapshot["filled_contracts"] = _extract_first_numeric_value(
            order_payload,
            [
                ["filled_count"],
                ["count_filled"],
                ["filled_contract_count"],
                ["contracts_filled"],
                ["remaining_count"],
                ["count_remaining"],
                ["resting_count"],
                ["open_count"],
            ],
        )
        # Correct remaining separately after above to avoid misuse:
        snapshot["remaining_contracts"] = _extract_first_numeric_value(
            order_payload,
            [
                ["remaining_count"],
                ["count_remaining"],
                ["resting_count"],
                ["open_count"],
                ["unfilled_count"],
            ],
        )
        snapshot["matched_contracts"] = _extract_first_numeric_value(
            order_payload,
            [
                ["matched_count"],
                ["executed_count"],
                ["matched_contract_count"],
            ],
        )

    fills_records = fills_payload.get("fills") if isinstance(fills_payload.get("fills"), list) else []
    filtered_fills = []
    for rec in fills_records:
        if not isinstance(rec, dict):
            continue
        rec_order_id = safe_str(rec.get("order_id"))
        rec_ticker = safe_str(rec.get("ticker")) or ticker
        if order_id and rec_order_id and rec_order_id != order_id:
            continue
        if ticker and rec_ticker and rec_ticker != ticker:
            continue
        filtered_fills.append(rec)

    fills_contracts = 0.0
    for rec in filtered_fills:
        qty = _extract_first_numeric_value(
            rec,
            [
                ["count"],
                ["count_filled"],
                ["qty"],
                ["quantity"],
                ["fill_count"],
            ],
        )
        if qty is not None:
            fills_contracts += max(0.0, qty)

    snapshot["fills_count"] = len(filtered_fills)
    snapshot["fills_contracts"] = fills_contracts
    snapshot["raw_fills"] = filtered_fills
    snapshot["ok"] = bool(order_payload or filtered_fills)
    if not snapshot["reason"]:
        snapshot["reason"] = "Live reconciliation snapshot collected." if snapshot["ok"] else "No live order/fill evidence found yet."
    return snapshot


def log_order_intent(intent: Dict[str, Any], prefix: str = "Order intent") -> None:
    logging.info(
        "%s | ticker=%s type=%s side=%s contracts=%s allocation=%.2f ask_price=%s state=%s reason=%s source=%s intent_key=%s",
        prefix,
        intent.get("ticker"),
        intent.get("type"),
        intent.get("side"),
        intent.get("contracts"),
        safe_float(intent.get("allocation"), 0.0),
        intent.get("ask_price"),
        intent.get("state"),
        intent.get("reason"),
        intent.get("source"),
        intent.get("intent_key"),
    )


def process_order_intents(
    execution_plan: Optional[Dict[str, Any]],
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    summary = {
        "built_count": 0,
        "submitted_count": 0,
        "duplicate_pending_count": 0,
        "awaiting_reconciliation_count": 0,
        "filled_count": 0,
        "not_filled_count": 0,
        "partial_count": 0,
        "submission_failed_count": 0,
        "stale_count": 0,
        "mode": get_execution_mode(config),
    }

    if not execution_plan:
        logging.info("Order submission phase | no execution plan available.")
        return summary

    intents_store = order_intent_store(state)
    tracker = execution_tracking_store(state)
    mode = get_execution_mode(config)
    state["last_execution_mode"] = mode

    logging.info(
        "Order submission phase | mode=%s | exit_actions=%s | entry_actions=%s | hold_actions=%s",
        mode,
        len(execution_plan.get("exit_actions") or []),
        len(execution_plan.get("entry_actions") or []),
        len(execution_plan.get("hold_actions") or []),
    )

    candidate_actions = [
        action for action in (execution_plan.get("all_actions") or [])
        if safe_upper(action.get("action_type")) in {"EXIT", "ENTER"}
        and safe_upper(action.get("execution_state")) == EXECUTION_STATE_READY
    ]

    for action in candidate_actions:
        intent = dict(action.get("order_intent") or {})
        if not intent:
            continue

        intent_key = safe_str(intent.get("intent_key"))
        existing = intents_store.get(intent_key)
        summary["built_count"] += 1
        log_order_intent(intent, prefix="Order intent build")

        if existing:
            existing_state = safe_upper(existing.get("state"))
            material_change_reason = get_order_intent_material_change_reason(existing, intent, config)
            if existing_state in {
                ORDER_INTENT_STATE_SUBMITTED_SIMULATED,
                ORDER_INTENT_STATE_SUBMITTED_LIVE,
                ORDER_INTENT_STATE_AWAITING_RECONCILIATION,
                ORDER_INTENT_STATE_SKIPPED_DUPLICATE,
                ORDER_INTENT_STATE_SKIPPED_DUPLICATE_PENDING,
            } and material_change_reason is None:
                existing["updated_at"] = time.time()
                existing["last_seen_in_plan_at"] = time.time()
                existing["signature"] = safe_str(intent.get("signature")) or safe_str(existing.get("signature"))
                existing["heartbeat_count"] = safe_int(existing.get("heartbeat_count"), 0) + 1
                tracker_state = EXECUTION_STATE_AWAITING_RECON
                tracker_resolved = False
                if existing_state == ORDER_INTENT_STATE_SKIPPED_DUPLICATE_PENDING:
                    existing["state"] = ORDER_INTENT_STATE_AWAITING_RECONCILIATION
                elif existing_state == ORDER_INTENT_STATE_SKIPPED_DUPLICATE:
                    tracker_state = EXECUTION_STATE_SKIPPED_DUPLICATE
                    tracker_resolved = True
                intents_store[intent_key] = existing
                tracker[intent_key] = {
                    "action_key": intent_key,
                    "action_type": safe_upper(existing.get("type")),
                    "ticker": safe_str(existing.get("ticker")),
                    "side": safe_upper(existing.get("side")),
                    "signature": safe_str(existing.get("signature")),
                    "state": tracker_state,
                    "first_seen_ts": tracker.get(intent_key, {}).get("first_seen_ts", existing.get("created_at", time.time())),
                    "last_seen_ts": time.time(),
                    "reason": safe_str(existing.get("reason")),
                    "resolved": tracker_resolved,
                    "baseline_contracts": abs(safe_int((tracker.get(intent_key, {}) or {}).get("baseline_contracts"), 0) or 0),
                }
                logging.info(
                    "Order submission retained pending intent | intent_key=%s state=%s heartbeat_count=%s",
                    intent_key,
                    existing.get("state"),
                    existing.get("heartbeat_count"),
                )
                continue
            if material_change_reason:
                existing["superseded_by_change"] = material_change_reason
                existing["updated_at"] = time.time()
                intents_store[intent_key] = existing
                logging.info(
                    "Order submission | intent_key=%s had material change | reason=%s",
                    intent_key,
                    material_change_reason,
                )

        intent["state"] = ORDER_INTENT_STATE_READY_TO_SUBMIT
        intent["updated_at"] = time.time()
        intent["submission_mode"] = mode
        intent["submission_attempts"] = safe_int((existing or {}).get("submission_attempts"), 0)

        if mode == EXECUTION_MODE_SIMULATION:
            intent["submission_attempts"] = safe_int(intent.get("submission_attempts"), 0) + 1
            intent["submitted_at"] = time.time()
            intent["last_submission_attempt_at"] = time.time()
            intent["reconciliation_mode"] = EXECUTION_MODE_SIMULATION
            intent["state"] = ORDER_INTENT_STATE_RECONCILED_FILLED
            intent["reconciled_at"] = time.time()
            intent["reconciliation_age_seconds"] = 0.0
            intent["reconciliation_reason"] = "Simulation mode fills immediately and applies to the paper ledger in the same cycle."
            maybe_apply_reconciled_simulation_intent_to_paper_ledger(
                intent=intent,
                state=state,
                config=config,
            )
            mark_trade_timestamp(
                state,
                intent["reconciled_at"],
                action_type=intent.get("type"),
                ticker=intent.get("ticker"),
                side=intent.get("side"),
            )
            log_order_intent(intent, prefix="Order submission")
            summary["submitted_count"] += 1
            summary["filled_count"] += 1
        else:
            intent["submission_attempts"] = safe_int(intent.get("submission_attempts"), 0) + 1
            intent["last_submission_attempt_at"] = time.time()
            live_result = submit_live_order_stub(intent, config)
            intent["live_submission_result"] = live_result
            intent["submitted_at"] = time.time()
            intent["submission_status"] = safe_str(live_result.get("status"))
            intent["exchange_response_summary"] = safe_str(live_result.get("message"))
            intent["order_id"] = safe_str(live_result.get("external_order_id")) or None
            intent["client_order_id"] = safe_str(live_result.get("client_order_id")) or safe_str(intent.get("client_order_id")) or None
            intent["submitted_contracts"] = safe_int(live_result.get("submitted_contracts"), None)
            intent["original_contracts"] = safe_int(live_result.get("original_contracts"), safe_int(intent.get("contracts"), 0))
            if live_result.get("submitted"):
                intent["state"] = ORDER_INTENT_STATE_SUBMITTED_LIVE
                summary["submitted_count"] += 1
                logging.info(
                    "Kalshi order submit success | ticker=%s type=%s side=%s contracts=%s submitted_contracts=%s order_id=%s status=%s",
                    intent.get("ticker"),
                    intent.get("type"),
                    intent.get("side"),
                    intent.get("contracts"),
                    intent.get("submitted_contracts"),
                    intent.get("order_id"),
                    intent.get("submission_status"),
                )
            else:
                failure_status = safe_str(live_result.get("status")).lower()
                error_text = safe_str(intent.get("exchange_response_summary"))
                if failure_status == "order_already_exists" or "order_already_exists" in error_text.lower():
                    intent["state"] = ORDER_INTENT_STATE_SKIPPED_DUPLICATE
                    intent["reason"] = "exchange_duplicate_order"
                    summary["duplicate_pending_count"] += 1
                    logging.info(
                        "Duplicate order detected at exchange | ticker=%s type=%s side=%s contracts=%s submitted_contracts=%s client_order_id=%s",
                        intent.get("ticker"),
                        intent.get("type"),
                        intent.get("side"),
                        intent.get("contracts"),
                        intent.get("submitted_contracts"),
                        intent.get("client_order_id"),
                    )
                else:
                    intent["state"] = ORDER_INTENT_STATE_SUBMISSION_FAILED
                    if failure_status == "insufficient_resting_volume":
                        logging.info(
                            "Kalshi order rejected for liquidity | ticker=%s type=%s side=%s contracts=%s submitted_contracts=%s status=%s message=%s",
                            intent.get("ticker"),
                            intent.get("type"),
                            intent.get("side"),
                            intent.get("contracts"),
                            intent.get("submitted_contracts"),
                            intent.get("submission_status"),
                            intent.get("exchange_response_summary"),
                        )
                    else:
                        logging.info(
                            "Kalshi order submit failed | ticker=%s type=%s side=%s contracts=%s submitted_contracts=%s status=%s message=%s",
                            intent.get("ticker"),
                            intent.get("type"),
                            intent.get("side"),
                            intent.get("contracts"),
                            intent.get("submitted_contracts"),
                            intent.get("submission_status"),
                            intent.get("exchange_response_summary"),
                        )
            log_order_intent(intent, prefix="Order submission")

        tracker_state = EXECUTION_STATE_AWAITING_RECON
        tracker_resolved = False
        if safe_upper(intent.get("state")) == ORDER_INTENT_STATE_SUBMITTED_LIVE:
            intent["state"] = ORDER_INTENT_STATE_AWAITING_RECONCILIATION
            intent["awaiting_reconciliation_since"] = time.time()
        elif safe_upper(intent.get("state")) == ORDER_INTENT_STATE_RECONCILED_FILLED:
            tracker_state = EXECUTION_STATE_RECONCILED
            tracker_resolved = True
        elif safe_upper(intent.get("state")) == ORDER_INTENT_STATE_SKIPPED_DUPLICATE:
            tracker_state = EXECUTION_STATE_SKIPPED_DUPLICATE
            tracker_resolved = True

        intents_store[intent_key] = intent
        tracker[intent_key] = {
            "action_key": intent_key,
            "action_type": safe_upper(intent.get("type")),
            "ticker": safe_str(intent.get("ticker")),
            "side": safe_upper(intent.get("side")),
            "signature": safe_str(intent.get("signature")),
            "state": tracker_state,
            "first_seen_ts": tracker.get(intent_key, {}).get("first_seen_ts", time.time()),
            "last_seen_ts": time.time(),
            "reason": safe_str(intent.get("reason")),
            "resolved": tracker_resolved,
            "baseline_contracts": abs(safe_int((current_live_position_lookup(state.get("last_open_positions_df")).get(intent.get("ticker"), {}) or {}).get("contracts"), 0) or 0),
        }

    state["order_intents"] = intents_store
    state["execution_tracking"] = tracker
    return summarize_order_intents(state, config, include_log=True)


def reconcile_order_intents(
    state: Dict[str, Any],
    open_positions_df: Optional[pd.DataFrame],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    intents_store = order_intent_store(state)
    tracker = execution_tracking_store(state)
    live_lookup = current_live_position_lookup(open_positions_df)
    timeout_seconds = get_order_reconciliation_timeout_seconds(config)
    simulated_fill_seconds = max(0, get_simulated_reconciliation_fill_seconds(config))
    now_ts = time.time()

    logging.info("Order reconciliation phase | active_intents=%s", len(intents_store))

    for intent_key, intent in list(intents_store.items()):
        if not isinstance(intent, dict):
            intents_store.pop(intent_key, None)
            continue

        current_state = safe_upper(intent.get("state"))
        if current_state not in {
            ORDER_INTENT_STATE_AWAITING_RECONCILIATION,
            ORDER_INTENT_STATE_SUBMITTED_SIMULATED,
            ORDER_INTENT_STATE_SUBMITTED_LIVE,
            ORDER_INTENT_STATE_SKIPPED_DUPLICATE_PENDING,
        }:
            continue

        ticker = safe_str(intent.get("ticker"))
        intent_type = safe_upper(intent.get("type"))
        intended_side = safe_upper(intent.get("side"))
        intended_contracts = abs(safe_int(intent.get("contracts"), 0) or 0)
        live_record = live_lookup.get(ticker)
        prior_tracker = tracker.get(intent_key, {})
        baseline_contracts = abs(safe_int((prior_tracker.get("baseline_contracts") if isinstance(prior_tracker, dict) else None), 0) or 0)
        if baseline_contracts == 0 and intent_type == "EXIT":
            baseline_contracts = intended_contracts

        submitted_at = safe_float(intent.get("submitted_at"), intent.get("created_at"))
        age_seconds = max(0.0, now_ts - submitted_at)
        submission_mode = safe_str(intent.get("submission_mode") or state.get("last_execution_mode") or get_execution_mode(config)).lower()

        new_state = current_state
        resolution_reason = ""

        if intent_type == "EXIT":
            live_contracts = abs(safe_int((live_record or {}).get("contracts"), 0) or 0)
            if live_record is None or live_contracts == 0:
                new_state = ORDER_INTENT_STATE_RECONCILED_FILLED
                resolution_reason = "Exit intent reconciled: tracked position disappeared from live positions."
            elif baseline_contracts > 0 and live_contracts < baseline_contracts:
                new_state = ORDER_INTENT_STATE_RECONCILED_PARTIAL
                resolution_reason = "Exit intent partially reconciled: tracked position size decreased materially but remains open."
            elif submission_mode == EXECUTION_MODE_SIMULATION and age_seconds >= simulated_fill_seconds:
                new_state = ORDER_INTENT_STATE_RECONCILED_FILLED
                resolution_reason = "Exit intent reconciled in simulation mode after grace period without requiring live exchange evidence."
            elif submission_mode != EXECUTION_MODE_SIMULATION:
                live_snapshot = fetch_live_reconciliation_snapshot(intent, config)
                order_status = _normalize_order_status(live_snapshot.get("order_status"))
                fills_contracts = safe_float(live_snapshot.get("fills_contracts"), 0.0) or 0.0
                filled_contracts = safe_float(live_snapshot.get("filled_contracts"), None)
                matched_contracts = safe_float(live_snapshot.get("matched_contracts"), None)
                effective_filled = max(
                    fills_contracts,
                    filled_contracts or 0.0,
                    matched_contracts or 0.0,
                )
                if order_status in {"filled", "executed", "complete", "completed", "closed"}:
                    new_state = ORDER_INTENT_STATE_RECONCILED_FILLED
                    resolution_reason = f"Exit intent reconciled from live order status={order_status or 'filled'}."
                elif baseline_contracts > 0 and effective_filled >= baseline_contracts:
                    new_state = ORDER_INTENT_STATE_RECONCILED_FILLED
                    resolution_reason = "Exit intent reconciled from live fill evidence covering baseline position size."
                elif effective_filled > 0:
                    new_state = ORDER_INTENT_STATE_RECONCILED_PARTIAL
                    resolution_reason = "Exit intent partially reconciled from live fill evidence while position still remains."
                elif order_status in {"partially_filled", "partial_fill", "partially_executed"}:
                    new_state = ORDER_INTENT_STATE_RECONCILED_PARTIAL
                    resolution_reason = f"Exit intent partially reconciled from live order status={order_status}."
                elif order_status in {"canceled", "cancelled", "rejected", "expired", "failed"}:
                    new_state = ORDER_INTENT_STATE_RECONCILED_NOT_FILLED
                    resolution_reason = f"Exit intent not filled according to live order status={order_status}."
                else:
                    intent["live_reconciliation_snapshot"] = {
                        "order_status": order_status,
                        "fills_contracts": fills_contracts,
                        "filled_contracts": filled_contracts,
                        "matched_contracts": matched_contracts,
                        "order_found": bool(live_snapshot.get("order_found")),
                        "reason": safe_str(live_snapshot.get("reason")),
                    }
                    reprice_after = get_exit_reprice_after_seconds(config)
                    stale_after = get_exit_stale_after_seconds(config)
                    if order_status in {"resting", "open", "pending", "unfilled"} and age_seconds >= reprice_after:
                        cancel_result = cancel_live_order_by_id(intent.get("order_id"), config)
                        intent["cancel_result"] = cancel_result
                        intent["state"] = ORDER_INTENT_STATE_CANCELLED_OR_STALE
                        intent["updated_at"] = now_ts
                        intent["reconciliation_reason"] = (
                            f"Exit intent cancelled for aggressive reprice after {round(age_seconds, 3)}s with order_status={order_status or 'unknown'}."
                        )
                        logging.info(
                            "Order reconciliation | ticker=%s previous_state=%s new_state=%s intent_key=%s reason=%s cancel_ok=%s",
                            ticker,
                            current_state,
                            intent["state"],
                            intent_key,
                            intent["reconciliation_reason"],
                            cancel_result.get("ok"),
                        )
                        tracker[intent_key] = {
                            "action_key": intent_key,
                            "action_type": intent_type,
                            "ticker": ticker,
                            "side": intended_side,
                            "signature": safe_str(intent.get("signature")),
                            "state": EXECUTION_STATE_RECONCILED,
                            "first_seen_ts": tracker.get(intent_key, {}).get("first_seen_ts", intent.get("created_at")),
                            "last_seen_ts": now_ts,
                            "reason": safe_str(intent.get("reason")),
                            "resolved": True,
                            "resolved_ts": now_ts,
                            "resolution_reason": intent["reconciliation_reason"],
                            "baseline_contracts": baseline_contracts,
                            "reconciliation_mode": submission_mode,
                            "reconciliation_age_seconds": round(age_seconds, 3),
                        }
                        intents_store[intent_key] = intent
                        continue
                    if age_seconds >= max(timeout_seconds, stale_after):
                        new_state = ORDER_INTENT_STATE_RECONCILED_NOT_FILLED
                        resolution_reason = "Exit intent not reconciled within timeout window; position still appears active and no live order/fill evidence resolved it."
        elif intent_type == "ENTER":
            live_contracts = abs(safe_int((live_record or {}).get("contracts"), 0) or 0)
            live_side = safe_upper((live_record or {}).get("held_side"))
            if live_record is not None and (not intended_side or live_side == intended_side) and live_contracts >= max(1, intended_contracts):
                new_state = ORDER_INTENT_STATE_RECONCILED_FILLED
                resolution_reason = "Entry intent reconciled: new same-side position now appears active in live positions."
            elif live_record is not None and (not intended_side or live_side == intended_side) and 0 < live_contracts < max(1, intended_contracts):
                new_state = ORDER_INTENT_STATE_RECONCILED_PARTIAL
                resolution_reason = "Entry intent partially reconciled: matching same-side position appeared with fewer contracts than intended."
            elif submission_mode == EXECUTION_MODE_SIMULATION and age_seconds >= simulated_fill_seconds:
                new_state = ORDER_INTENT_STATE_RECONCILED_FILLED
                resolution_reason = "Entry intent reconciled in simulation mode after grace period without requiring live exchange evidence."
            elif submission_mode != EXECUTION_MODE_SIMULATION:
                live_snapshot = fetch_live_reconciliation_snapshot(intent, config)
                order_status = _normalize_order_status(live_snapshot.get("order_status"))
                fills_contracts = safe_float(live_snapshot.get("fills_contracts"), 0.0) or 0.0
                filled_contracts = safe_float(live_snapshot.get("filled_contracts"), None)
                matched_contracts = safe_float(live_snapshot.get("matched_contracts"), None)
                effective_filled = max(
                    fills_contracts,
                    filled_contracts or 0.0,
                    matched_contracts or 0.0,
                )
                if order_status in {"filled", "executed", "complete", "completed", "closed"}:
                    new_state = ORDER_INTENT_STATE_RECONCILED_FILLED
                    resolution_reason = f"Entry intent reconciled from live order status={order_status or 'filled'}."
                elif effective_filled >= max(1, intended_contracts):
                    new_state = ORDER_INTENT_STATE_RECONCILED_FILLED
                    resolution_reason = "Entry intent reconciled from live fill evidence meeting intended contract count."
                elif 0 < effective_filled < max(1, intended_contracts):
                    new_state = ORDER_INTENT_STATE_RECONCILED_PARTIAL
                    resolution_reason = "Entry intent partially reconciled from live fill evidence below intended contract count."
                elif order_status in {"partially_filled", "partial_fill", "partially_executed"}:
                    new_state = ORDER_INTENT_STATE_RECONCILED_PARTIAL
                    resolution_reason = f"Entry intent partially reconciled from live order status={order_status}."
                elif order_status in {"canceled", "cancelled", "rejected", "expired", "failed"}:
                    new_state = ORDER_INTENT_STATE_RECONCILED_NOT_FILLED
                    resolution_reason = f"Entry intent not filled according to live order status={order_status}."
                else:
                    intent["live_reconciliation_snapshot"] = {
                        "order_status": order_status,
                        "fills_contracts": fills_contracts,
                        "filled_contracts": filled_contracts,
                        "matched_contracts": matched_contracts,
                        "order_found": bool(live_snapshot.get("order_found")),
                        "reason": safe_str(live_snapshot.get("reason")),
                    }
                    if age_seconds >= timeout_seconds:
                        new_state = ORDER_INTENT_STATE_RECONCILED_NOT_FILLED
                        resolution_reason = "Entry intent not reconciled within timeout window; no matching live fill/order evidence appeared."

        if new_state in {ORDER_INTENT_STATE_RECONCILED_FILLED, ORDER_INTENT_STATE_RECONCILED_PARTIAL, ORDER_INTENT_STATE_RECONCILED_NOT_FILLED}:
            previous_state = intent.get("state")
            intent["state"] = new_state
            intent["updated_at"] = now_ts
            intent["reconciliation_reason"] = resolution_reason
            intent["reconciled_at"] = now_ts
            intent["reconciliation_age_seconds"] = round(age_seconds, 3)
            intent["reconciliation_mode"] = submission_mode
            logging.info(
                "Order reconciliation | ticker=%s previous_state=%s new_state=%s intent_key=%s reason=%s live_snapshot=%s",
                ticker,
                previous_state,
                new_state,
                intent_key,
                resolution_reason,
                intent.get("live_reconciliation_snapshot"),
            )
            tracker[intent_key] = {
                "action_key": intent_key,
                "action_type": intent_type,
                "ticker": ticker,
                "side": intended_side,
                "signature": safe_str(intent.get("signature")),
                "state": EXECUTION_STATE_RECONCILED,
                "first_seen_ts": tracker.get(intent_key, {}).get("first_seen_ts", intent.get("created_at")),
                "last_seen_ts": now_ts,
                "reason": safe_str(intent.get("reason")),
                "resolved": True,
                "resolved_ts": now_ts,
                "resolution_reason": resolution_reason,
                "baseline_contracts": baseline_contracts,
                "reconciliation_mode": submission_mode,
                "reconciliation_age_seconds": round(age_seconds, 3),
            }

            maybe_apply_reconciled_simulation_intent_to_paper_ledger(
                intent=intent,
                state=state,
                config=config,
            )
            if new_state in {ORDER_INTENT_STATE_RECONCILED_FILLED, ORDER_INTENT_STATE_RECONCILED_PARTIAL}:
                mark_trade_timestamp(
                    state,
                    now_ts,
                    action_type=intent.get("type"),
                    ticker=intent.get("ticker"),
                    side=intent.get("side"),
                )
        else:
            intent["updated_at"] = now_ts
            intent["reconciliation_age_seconds"] = round(age_seconds, 3)
            intent["reconciliation_mode"] = submission_mode
            if submission_mode != EXECUTION_MODE_SIMULATION and age_seconds >= (2 * timeout_seconds):
                intent["state"] = ORDER_INTENT_STATE_CANCELLED_OR_STALE
                intent["updated_at"] = now_ts
                intent["reconciliation_reason"] = "Intent aged beyond stale threshold without reconciliation evidence."
                logging.info(
                    "Order reconciliation | ticker=%s previous_state=%s new_state=%s intent_key=%s reason=%s",
                    ticker,
                    current_state,
                    intent["state"],
                    intent_key,
                    intent["reconciliation_reason"],
                )
                tracker[intent_key] = {
                    "action_key": intent_key,
                    "action_type": intent_type,
                    "ticker": ticker,
                    "side": intended_side,
                    "signature": safe_str(intent.get("signature")),
                    "state": EXECUTION_STATE_RECONCILED,
                    "first_seen_ts": tracker.get(intent_key, {}).get("first_seen_ts", intent.get("created_at")),
                    "last_seen_ts": now_ts,
                    "reason": safe_str(intent.get("reason")),
                    "resolved": True,
                    "resolved_ts": now_ts,
                    "resolution_reason": intent["reconciliation_reason"],
                    "baseline_contracts": baseline_contracts,
                    "reconciliation_mode": submission_mode,
                    "reconciliation_age_seconds": round(age_seconds, 3),
                }

        intents_store[intent_key] = intent

    state["order_intents"] = intents_store
    state["execution_tracking"] = tracker
    return summarize_order_intents(state, config, include_log=True)


def prune_resolved_order_state(state: Dict[str, Any]) -> Dict[str, int]:
    intents_store = order_intent_store(state)
    tracker = execution_tracking_store(state)

    resolved_intent_states = {
        ORDER_INTENT_STATE_RECONCILED_FILLED,
        ORDER_INTENT_STATE_RECONCILED_NOT_FILLED,
        ORDER_INTENT_STATE_RECONCILED_PARTIAL,
        ORDER_INTENT_STATE_CANCELLED_OR_STALE,
        ORDER_INTENT_STATE_SUBMISSION_FAILED,
        ORDER_INTENT_STATE_SKIPPED_DUPLICATE,
    }

    removed_intents = 0
    for intent_key in list(intents_store.keys()):
        intent = intents_store.get(intent_key) or {}
        if safe_upper(intent.get("state")) in resolved_intent_states:
            intents_store.pop(intent_key, None)
            removed_intents += 1

    removed_tracking = 0
    for action_key in list(tracker.keys()):
        rec = tracker.get(action_key) or {}
        if bool(rec.get("resolved", False)):
            tracker.pop(action_key, None)
            removed_tracking += 1

    state["order_intents"] = intents_store
    state["execution_tracking"] = tracker

    logging.info(
        "Pruned runtime order state | removed_intents=%s | removed_tracking=%s | remaining_intents=%s | remaining_tracking=%s",
        removed_intents,
        removed_tracking,
        len(intents_store),
        len(tracker),
    )

    return {
        "removed_intents": removed_intents,
        "removed_tracking": removed_tracking,
        "remaining_intents": len(intents_store),
        "remaining_tracking": len(tracker),
    }


def summarize_order_intents(
    state: Dict[str, Any],
    config: Dict[str, Any],
    include_log: bool = False,
) -> Dict[str, Any]:
    intents_store = order_intent_store(state)
    active_states = {
        ORDER_INTENT_STATE_READY_TO_SUBMIT,
        ORDER_INTENT_STATE_SUBMITTED_SIMULATED,
        ORDER_INTENT_STATE_SUBMITTED_LIVE,
        ORDER_INTENT_STATE_AWAITING_RECONCILIATION,
        ORDER_INTENT_STATE_SKIPPED_DUPLICATE_PENDING,
    }
    summary = {
        "built_count": 0,
        "submitted_count": 0,
        "duplicate_pending_count": 0,
        "awaiting_reconciliation_count": 0,
        "filled_count": 0,
        "not_filled_count": 0,
        "partial_count": 0,
        "submission_failed_count": 0,
        "stale_count": 0,
        "mode": get_execution_mode(config),
        "active_intents_count": 0,
        "stored_intents_count": len(intents_store),
    }

    for intent in intents_store.values():
        state_value = safe_upper(intent.get("state"))
        if state_value in active_states:
            summary["built_count"] += 1
            summary["active_intents_count"] += 1
        if state_value in {ORDER_INTENT_STATE_SUBMITTED_SIMULATED, ORDER_INTENT_STATE_SUBMITTED_LIVE}:
            summary["submitted_count"] += 1
        if state_value in {ORDER_INTENT_STATE_SKIPPED_DUPLICATE, ORDER_INTENT_STATE_SKIPPED_DUPLICATE_PENDING}:
            summary["duplicate_pending_count"] += 1
        if state_value == ORDER_INTENT_STATE_AWAITING_RECONCILIATION:
            summary["awaiting_reconciliation_count"] += 1
        if state_value == ORDER_INTENT_STATE_RECONCILED_FILLED:
            summary["filled_count"] += 1
        if state_value == ORDER_INTENT_STATE_RECONCILED_PARTIAL:
            summary["partial_count"] += 1
        if state_value == ORDER_INTENT_STATE_RECONCILED_NOT_FILLED:
            summary["not_filled_count"] += 1
        if state_value == ORDER_INTENT_STATE_SUBMISSION_FAILED:
            summary["submission_failed_count"] += 1
        if state_value == ORDER_INTENT_STATE_CANCELLED_OR_STALE:
            summary["stale_count"] += 1

    if include_log:
        logging.info(
            "Pending order summary | mode=%s built=%s awaiting=%s filled=%s partial=%s not_filled=%s submission_failed=%s stale=%s duplicate_pending=%s stored=%s",
            summary.get("mode"),
            summary.get("built_count"),
            summary.get("awaiting_reconciliation_count"),
            summary.get("filled_count"),
            summary.get("partial_count"),
            summary.get("not_filled_count"),
            summary.get("submission_failed_count"),
            summary.get("stale_count"),
            summary.get("duplicate_pending_count"),
            summary.get("stored_intents_count"),
        )
    return summary


def log_execution_intent_plan(plan: Optional[Dict[str, Any]]) -> None:
    if not plan:
        logging.info("Execution intent plan | unavailable")
        return

    logging.info(
        (
            "Execution intent plan | planned_exit_count=%s planned_enter_count=%s "
            "informational_hold_count=%s ready_count=%s skipped_count=%s awaiting_count=%s "
            "deployable_cash_start=%.2f estimated_exit_proceeds=%.2f "
            "deployable_cash_effective_start=%.2f deployable_cash_remaining=%.2f ordering=%s"
        ),
        plan.get("planned_exit_count", 0),
        plan.get("planned_enter_count", 0),
        plan.get("informational_hold_count", 0),
        plan.get("ready_count", 0),
        plan.get("skipped_count", 0),
        plan.get("awaiting_count", 0),
        safe_float(plan.get("deployable_cash_start"), 0.0),
        safe_float(plan.get("estimated_exit_proceeds"), 0.0),
        safe_float(plan.get("deployable_cash_effective_start"), 0.0),
        safe_float(plan.get("deployable_cash_remaining"), 0.0),
        plan.get("ordering"),
    )

    actions = plan.get("all_actions") or []
    if not actions:
        logging.info("Execution intent actions | none")
        return

    for idx, action in enumerate(actions, start=1):
        logging.info(
            (
                "Execution action %s | ticker=%s | type=%s | side=%s | contracts=%s | "
                "allocation=%.2f | ask_price=%s | state=%s | reason=%s | source=%s | intent_key=%s"
            ),
            idx,
            action.get("ticker"),
            action.get("action_type"),
            action.get("side"),
            action.get("contracts"),
            safe_float(action.get("allocation"), 0.0),
            action.get("ask_price"),
            action.get("execution_state"),
            action.get("execution_reason"),
            action.get("source"),
            action.get("action_key"),
        )
# ---------------------------------------------------------------------
# Status / control alert scaffold
# ---------------------------------------------------------------------
def build_status_alert_signature(status_type, payload=None):
    payload = payload or {}
    payload_str = "||".join([f"{k}={payload[k]}" for k in sorted(payload.keys())])
    return f"{status_type}||{payload_str}"


def should_send_status_alert(status_type, config, payload=None):
    global seen_status_alert_state

    payload = payload or {}
    sig = build_status_alert_signature(status_type, payload)
    now_ts = time.time()
    cooldown_seconds = get_status_alert_cooldown_seconds(config)

    prior = seen_status_alert_state.get(sig)
    if prior is None:
        return True

    last_sent_ts = safe_float(prior.get("last_sent_ts"), 0.0)
    within_cooldown = (now_ts - last_sent_ts) < cooldown_seconds
    return not within_cooldown


def record_status_alert_sent(status_type, payload=None):
    global seen_status_alert_state

    payload = payload or {}
    sig = build_status_alert_signature(status_type, payload)
    seen_status_alert_state[sig] = {
        "last_sent_ts": time.time(),
        "payload": payload,
    }


def format_status_alert(status_type, payload=None):
    payload = payload or {}
    lines = [f"â„¹ï¸ ENGINE STATUS: {status_type.upper()}"]
    if payload:
        lines.append("")
        for key in sorted(payload.keys()):
            lines.append(f"{key}: {payload[key]}")
    return "\n".join(lines)


def process_status_alert(status_type, config, payload=None):
    if not should_send_status_alert(status_type, config, payload):
        logging.info("Status alert suppressed | type=%s payload=%s", status_type, payload)
        return

    message = format_status_alert(status_type, payload)
    logging.info("Status alert sent | type=%s payload=%s", status_type, payload)
    send_telegram_alert(message)
    record_status_alert_sent(status_type, payload)


# ---------------------------------------------------------------------
# Live positions loading / normalization
# ---------------------------------------------------------------------
def filter_positions_to_engine_scope(
    positions_df: Optional[pd.DataFrame],
    exact_market_ticker: Optional[str] = None,
    event_ticker: Optional[str] = None,
    series_ticker: Optional[str] = None,
) -> pd.DataFrame:
    if positions_df is None or positions_df.empty:
        return pd.DataFrame()

    df = positions_df.copy()

    ticker_col = None
    for candidate in ["contract_ticker", "ticker", "market_ticker"]:
        if candidate in df.columns:
            ticker_col = candidate
            break

    if ticker_col is None:
        logging.warning("Engine scope filter skipped: no ticker column found in positions payload.")
        return df

    df["_scope_contract_ticker"] = df[ticker_col].astype(str).str.strip()

    event_col = "event_ticker" if "event_ticker" in df.columns else None
    series_col = "series_ticker" if "series_ticker" in df.columns else None

    exact_market_ticker = _clean_token(exact_market_ticker)
    event_ticker = _clean_token(event_ticker)
    series_ticker = _clean_token(series_ticker)

    mask = pd.Series(False, index=df.index)

    if exact_market_ticker:
        mask = mask | (df["_scope_contract_ticker"] == exact_market_ticker)

    if event_ticker:
        event_match = df["_scope_contract_ticker"].str.startswith(f"{event_ticker}-", na=False)
        if event_col is not None:
            event_match = event_match | (df[event_col].astype(str).str.strip() == event_ticker)
        mask = mask | event_match

    if series_ticker:
        series_match = df["_scope_contract_ticker"].str.startswith(series_ticker, na=False)
        if series_col is not None:
            series_match = series_match | (df[series_col].astype(str).str.strip() == series_ticker)
        mask = mask | series_match

    if not bool(mask.any()):
        logging.info(
            "Engine scope filter returned zero rows | exact_market_ticker=%s | event_ticker=%s | series_ticker=%s",
            exact_market_ticker,
            event_ticker,
            series_ticker,
        )
        return pd.DataFrame(columns=[c for c in df.columns if c != "_scope_contract_ticker"])

    filtered = df.loc[mask].copy()
    filtered = filtered.drop(columns=["_scope_contract_ticker"], errors="ignore")

    logging.info(
        "Engine scope filter kept %s/%s rows | exact_market_ticker=%s | event_ticker=%s | series_ticker=%s",
        len(filtered),
        len(df),
        exact_market_ticker,
        event_ticker,
        series_ticker,
    )
    return filtered


def _coerce_position_numeric_series(df: pd.DataFrame) -> pd.Series:
    for candidate in ["position_fp", "position_numeric", "position"]:
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _coerce_contracts_series(df: pd.DataFrame) -> pd.Series:
    for candidate in ["contracts", "quantity", "size"]:
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _coerce_market_value_series(df: pd.DataFrame) -> pd.Series:
    if "market_value" in df.columns:
        return pd.to_numeric(df["market_value"], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def filter_to_active_positions(positions_df: Optional[pd.DataFrame], log_prefix: str = "Active position filter") -> pd.DataFrame:
    if positions_df is None or positions_df.empty:
        return pd.DataFrame()

    df = positions_df.copy()

    position_fp = pd.to_numeric(df["position_fp"], errors="coerce") if "position_fp" in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")
    position_numeric = _coerce_position_numeric_series(df)
    contracts = _coerce_contracts_series(df)
    market_value = _coerce_market_value_series(df)

    active_mask = (
        position_fp.abs().fillna(0.0) > POSITION_ACTIVE_EPSILON
    ) | (
        position_numeric.abs().fillna(0.0) > POSITION_ACTIVE_EPSILON
    ) | (
        contracts.abs().fillna(0.0) > POSITION_ACTIVE_EPSILON
    ) | (
        market_value.abs().fillna(0.0) > POSITION_ACTIVE_EPSILON
    )

    active_df = df.loc[active_mask].copy()

    logging.info(
        "%s kept %s/%s rows after removing zero-size remnants.",
        log_prefix,
        len(active_df),
        len(df),
    )

    return active_df


def build_engine_scoped_account_snapshot(
    raw_account_snapshot: Optional[Dict[str, Any]],
    filtered_positions_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    raw_account_snapshot = dict(raw_account_snapshot or {})
    positions_df = filtered_positions_df.copy() if filtered_positions_df is not None else pd.DataFrame()

    cash_balance = safe_float(
        raw_account_snapshot.get("cash_balance", raw_account_snapshot.get("balance")),
        0.0,
    )

    scoped_positions_value = 0.0
    scoped_open_positions_count = 0

    if positions_df is not None and not positions_df.empty:
        active_positions_df = filter_to_active_positions(
            positions_df,
            log_prefix="Account snapshot active-position filter",
        )
        scoped_open_positions_count = len(active_positions_df)

        if not active_positions_df.empty:
            contracts = pd.to_numeric(
                active_positions_df.get("contracts", active_positions_df.get("size", np.nan)),
                errors="coerce",
            )

            current_price = pd.to_numeric(
                active_positions_df.get("current_price", active_positions_df.get("current_position_price", np.nan)),
                errors="coerce",
            )
            market_value = pd.to_numeric(
                active_positions_df.get("market_value", np.nan),
                errors="coerce",
            )
            cost_basis_total = pd.to_numeric(
                active_positions_df.get("cost_basis_total", active_positions_df.get("position_cost", np.nan)),
                errors="coerce",
            )
            entry_price = pd.to_numeric(
                active_positions_df.get("entry_price", np.nan),
                errors="coerce",
            )

            fallback_market_value = current_price * contracts.abs()
            fallback_cost_value = entry_price * contracts.abs()

            effective_position_value = market_value.where(market_value.notna(), fallback_market_value)
            effective_position_value = effective_position_value.where(
                effective_position_value.notna(),
                cost_basis_total,
            )
            effective_position_value = effective_position_value.where(
                effective_position_value.notna(),
                fallback_cost_value,
            )

            scoped_positions_value = float(effective_position_value.fillna(0.0).sum())

    scoped_snapshot = dict(raw_account_snapshot)
    scoped_snapshot["cash_balance"] = cash_balance
    scoped_snapshot["positions_value"] = scoped_positions_value
    scoped_snapshot["portfolio_value"] = cash_balance + scoped_positions_value
    scoped_snapshot["engine_scoped"] = True
    scoped_snapshot["engine_open_positions_count"] = scoped_open_positions_count
    scoped_snapshot["raw_account_snapshot"] = raw_account_snapshot

    return scoped_snapshot


def load_real_kalshi_account_data(config) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    api_key_id = os.getenv("KALSHI_API_KEY_ID")

    if not api_key_id:
        logging.info("KALSHI_API_KEY_ID not set. Skipping live positions/account load.")
        return pd.DataFrame(), None, None

    try:
        private_key_path = load_kalshi_private_key()
    except Exception as exc:
        logging.info(
            "Kalshi private key not available. env_present=%s. Skipping live positions/account load: %s",
            bool(os.getenv("KALSHI_PRIVATE_KEY")),
            exc,
        )
        return pd.DataFrame(), None, None

    scope = _get_engine_scope(config)
    exact_market_ticker = scope.get("exact_market_ticker")
    event_ticker = scope.get("event_ticker")
    series_ticker = scope.get("series_ticker")

    try:
        logging.info(
            "Attempting live positions load | exact_market_ticker=%s | event_ticker=%s | series_ticker=%s",
            exact_market_ticker,
            event_ticker,
            series_ticker,
        )

        live_positions_df = get_kalshi_live_positions_df(
            api_key_id=api_key_id,
            private_key_path=private_key_path,
            ticker=exact_market_ticker,
            event_ticker=event_ticker,
        )
        raw_account_snapshot = get_kalshi_account_snapshot(
            api_key_id=api_key_id,
            private_key_path=private_key_path,
            ticker=exact_market_ticker,
            event_ticker=event_ticker,
        )

        if live_positions_df is not None and not live_positions_df.empty:
            filtered_positions_df = filter_positions_to_engine_scope(
                live_positions_df,
                exact_market_ticker=exact_market_ticker,
                event_ticker=event_ticker,
                series_ticker=series_ticker,
            )
            scoped_account_snapshot = build_engine_scoped_account_snapshot(
                raw_account_snapshot=raw_account_snapshot,
                filtered_positions_df=filtered_positions_df,
            )
            logging.info(
                "Loaded %s raw live Kalshi position rows using exact_market_ticker/event_ticker filter; %s remain after engine scope filtering.",
                len(live_positions_df),
                len(filtered_positions_df),
            )
            return filtered_positions_df, scoped_account_snapshot, raw_account_snapshot

        logging.info("No rows from exact_market_ticker/event_ticker filter. Retrying with event_ticker only.")

        live_positions_df = get_kalshi_live_positions_df(
            api_key_id=api_key_id,
            private_key_path=private_key_path,
            ticker=None,
            event_ticker=event_ticker,
        )
        raw_account_snapshot = get_kalshi_account_snapshot(
            api_key_id=api_key_id,
            private_key_path=private_key_path,
            ticker=None,
            event_ticker=event_ticker,
        )

        if live_positions_df is not None and not live_positions_df.empty:
            filtered_positions_df = filter_positions_to_engine_scope(
                live_positions_df,
                exact_market_ticker=exact_market_ticker,
                event_ticker=event_ticker,
                series_ticker=series_ticker,
            )
            scoped_account_snapshot = build_engine_scoped_account_snapshot(
                raw_account_snapshot=raw_account_snapshot,
                filtered_positions_df=filtered_positions_df,
            )
            logging.info(
                "Loaded %s raw live Kalshi position rows using event_ticker-only filter; %s remain after engine scope filtering.",
                len(live_positions_df),
                len(filtered_positions_df),
            )
            return filtered_positions_df, scoped_account_snapshot, raw_account_snapshot

        logging.info(
            "No live Kalshi positions found after exact_market_ticker/event_ticker attempts. "
            "Skipping unfiltered fallback for this engine."
        )

        scoped_account_snapshot = build_engine_scoped_account_snapshot(
            raw_account_snapshot=raw_account_snapshot,
            filtered_positions_df=pd.DataFrame(),
        )
        return pd.DataFrame(), scoped_account_snapshot, raw_account_snapshot

    except Exception as exc:
        logging.warning("Failed to load live Kalshi account data: %s", exc)
        return pd.DataFrame(), None, None


def extract_live_contract_tickers(positions_df) -> List[str]:
    if positions_df is None or positions_df.empty:
        return []

    ticker_col = None
    for candidate in ["contract_ticker", "ticker", "market_ticker"]:
        if candidate in positions_df.columns:
            ticker_col = candidate
            break

    if ticker_col is None:
        return []

    tickers = (
        positions_df[ticker_col]
        .astype(str)
        .str.strip()
    )
    tickers = tickers[tickers != ""]
    return sorted(tickers.dropna().unique().tolist())


def normalize_kalshi_positions_for_monitoring(positions_df, ranked_df):
    if positions_df is None or positions_df.empty:
        return pd.DataFrame()

    df = positions_df.copy()

    ticker_col = None
    for candidate in ["contract_ticker", "ticker", "market_ticker"]:
        if candidate in df.columns:
            ticker_col = candidate
            break

    if ticker_col is None:
        logging.warning("Could not find ticker column in Kalshi positions payload.")
        return pd.DataFrame()

    df["contract_ticker"] = df[ticker_col].astype(str).str.strip()

    if "position_numeric" not in df.columns:
        if "position" in df.columns:
            df["position_numeric"] = pd.to_numeric(df["position"], errors="coerce")
        elif "position_fp" in df.columns:
            df["position_numeric"] = pd.to_numeric(df["position_fp"], errors="coerce")
        else:
            df["position_numeric"] = np.nan
    else:
        df["position_numeric"] = pd.to_numeric(df["position_numeric"], errors="coerce")

    if "contracts" not in df.columns:
        contracts_col = None
        for candidate in ["contracts", "quantity", "size", "position", "position_fp"]:
            if candidate in df.columns:
                contracts_col = candidate
                break

        if contracts_col is not None:
            df["contracts"] = pd.to_numeric(df[contracts_col], errors="coerce").abs()
        else:
            df["contracts"] = np.nan
    else:
        df["contracts"] = pd.to_numeric(df["contracts"], errors="coerce").abs()

    if "size" not in df.columns:
        df["size"] = df["contracts"]

    if "action" not in df.columns:
        df["action"] = df["position_numeric"].apply(
            lambda x: "BUY_YES" if pd.notna(x) and float(x) > 0 else ("BUY_NO" if pd.notna(x) and float(x) < 0 else None)
        )

    df = df[df["contract_ticker"].notna()].copy()
    df = df[df["contract_ticker"].astype(str).str.strip() != ""].copy()

    df = filter_to_active_positions(
        df,
        log_prefix="Monitoring normalization active-position filter",
    )

    if df.empty:
        logging.info("No nonzero live Kalshi positions remain after filtering.")
        return pd.DataFrame()

    if "contracts" in df.columns:
        df["contracts"] = pd.to_numeric(df["contracts"], errors="coerce").abs()

    for col in ["entry_price", "position_cost", "cost_basis_total", "current_price", "market_value", "unrealized_pnl"]:
        if col not in df.columns:
            df[col] = np.nan

    ranked_merge_cols = [
        "contract_ticker",
        "contract",
        "strike",
        "market_prob",
        "decision_prob",
        "fair_prob_terminal",
        "fair_prob_blended",
        "oil_price",
        "distance_to_strike",
        "hours_left",
        "edge_yes",
        "edge_no",
        "confidence",
        "ask_yes",
        "ask_no",
        "bid_yes",
        "bid_no",
        "last_price_yes",
        "last_price_no",
        "ev_yes",
        "ev_no",
        "yes_no_ask_sum",
        "overround",
        "market_too_wide",
        "no_trade_reason",
    ]
    ranked_merge_cols = [c for c in ranked_merge_cols if c in ranked_df.columns]

    merged = df.merge(
        ranked_df[ranked_merge_cols].drop_duplicates(subset=["contract_ticker"]),
        on="contract_ticker",
        how="left",
        suffixes=("", "_ranked"),
    )

    if "strike_ranked" in merged.columns:
        if "strike" in merged.columns:
            merged["strike"] = merged["strike"].fillna(merged["strike_ranked"])
        else:
            merged["strike"] = merged["strike_ranked"]
        merged = merged.drop(columns=["strike_ranked"])

    yes_mask = merged["action"].astype(str).str.upper() == "BUY_YES"
    no_mask = merged["action"].astype(str).str.upper() == "BUY_NO"

    derived_current_price = pd.Series(np.nan, index=merged.index, dtype="float64")
    if "current_price" in merged.columns:
        derived_current_price = pd.to_numeric(merged["current_price"], errors="coerce")

    if "last_price_yes" in merged.columns:
        derived_current_price = derived_current_price.where(
            ~(yes_mask & derived_current_price.isna()),
            pd.to_numeric(merged["last_price_yes"], errors="coerce"),
        )
    if "last_price_no" in merged.columns:
        derived_current_price = derived_current_price.where(
            ~(no_mask & derived_current_price.isna()),
            pd.to_numeric(merged["last_price_no"], errors="coerce"),
        )
    if "bid_yes" in merged.columns and "ask_yes" in merged.columns:
        yes_mid = (
            pd.to_numeric(merged["bid_yes"], errors="coerce")
            + pd.to_numeric(merged["ask_yes"], errors="coerce")
        ) / 2.0
        derived_current_price = derived_current_price.where(
            ~(yes_mask & derived_current_price.isna()),
            yes_mid,
        )
    if "bid_no" in merged.columns and "ask_no" in merged.columns:
        no_mid = (
            pd.to_numeric(merged["bid_no"], errors="coerce")
            + pd.to_numeric(merged["ask_no"], errors="coerce")
        ) / 2.0
        derived_current_price = derived_current_price.where(
            ~(no_mask & derived_current_price.isna()),
            no_mid,
        )

    merged["current_price"] = pd.to_numeric(merged["current_price"], errors="coerce").where(
        pd.to_numeric(merged["current_price"], errors="coerce").notna(),
        derived_current_price,
    )
    merged["current_position_price"] = merged["current_price"]

    merged["position_cost"] = pd.to_numeric(merged["position_cost"], errors="coerce")
    merged["cost_basis_total"] = pd.to_numeric(merged["cost_basis_total"], errors="coerce")
    merged["entry_price"] = pd.to_numeric(merged["entry_price"], errors="coerce")
    merged["contracts"] = pd.to_numeric(merged["contracts"], errors="coerce").abs()
    merged["market_value"] = pd.to_numeric(merged["market_value"], errors="coerce")

    fallback_cost = merged["entry_price"] * merged["contracts"]
    merged["position_cost"] = merged["position_cost"].where(merged["position_cost"].notna(), fallback_cost)
    merged["cost_basis_total"] = merged["cost_basis_total"].where(merged["cost_basis_total"].notna(), merged["position_cost"])

    fallback_market_value = merged["current_price"] * merged["contracts"]
    merged["market_value"] = merged["market_value"].where(merged["market_value"].notna(), fallback_market_value)

    merged["unrealized_pnl"] = merged["market_value"] - merged["position_cost"]
    merged["entry_fair_prob"] = merged["market_prob"]

    if "position_id" not in merged.columns:
        merged["position_id"] = merged["contract_ticker"]

    logging.info(
        "Normalized live positions preview: %s",
        merged[
            [
                c for c in [
                    "contract_ticker",
                    "action",
                    "contracts",
                    "entry_price",
                    "position_cost",
                    "current_price",
                    "market_value",
                    "unrealized_pnl",
                    "decision_prob",
                    "fair_prob_terminal",
                    "fair_prob_blended",
                ]
                if c in merged.columns
            ]
        ].head(10).to_dict(orient="records"),
    )

    return merged


# ---------------------------------------------------------------------
# Telegram command interface
# ---------------------------------------------------------------------
def get_telegram_bot_token() -> str:
    return str(os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()


def get_default_telegram_chat_id() -> str:
    return str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()


def telegram_api_get(method: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Dict[str, Any]:
    bot_token = get_telegram_bot_token()
    if not bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    response = requests.get(url, params=params or {}, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    if not data.get("ok", False):
        raise RuntimeError(f"Telegram API {method} failed: {data}")

    return data


def fetch_telegram_updates(last_update_id: Optional[int], config) -> List[dict]:
    bot_token = get_telegram_bot_token()
    if not bot_token:
        return []

    params: Dict[str, Any] = {
        "timeout": get_telegram_updates_timeout_seconds(config),
        "limit": get_telegram_command_poll_limit(config),
        "allowed_updates": ["message"],
    }

    if last_update_id is not None:
        params["offset"] = int(last_update_id) + 1

    try:
        payload = telegram_api_get("getUpdates", params=params, timeout=10)
        results = payload.get("result", [])
        if not isinstance(results, list):
            return []
        return results
    except Exception as exc:
        logging.warning("Telegram update fetch failed: %s", exc)
        return []


def is_authorized_telegram_chat(chat_id: Optional[str]) -> bool:
    allowed_chat_id = get_default_telegram_chat_id()
    if not allowed_chat_id:
        return True
    return str(chat_id or "").strip() == allowed_chat_id


def extract_latest_command(updates: List[dict]) -> Tuple[Optional[dict], Optional[int]]:
    if not updates:
        return None, None

    latest_command = None
    latest_update_id = None

    for update in updates:
        update_id = safe_int(update.get("update_id"), None)
        if update_id is not None:
            latest_update_id = update_id

        message = update.get("message") or {}
        text = safe_str(message.get("text"))
        if not text.startswith("/"):
            continue

        chat = message.get("chat") or {}
        chat_id = chat.get("id")

        if not is_authorized_telegram_chat(chat_id):
            logging.info("Ignoring Telegram command from unauthorized chat_id=%s", chat_id)
            continue

        latest_command = update

    return latest_command, latest_update_id


def get_best_actionable_trade(results: Optional[Dict[str, Any]]) -> Optional[pd.Series]:
    if not results:
        return None

    ranked_df = results.get("ranked_df", pd.DataFrame())
    if ranked_df is None or ranked_df.empty:
        return None

    actionable_df = ranked_df.copy()
    actionable_df["action"] = actionable_df["action"].astype(str).str.strip().str.upper()
    actionable_df = actionable_df[actionable_df["action"].isin(["BUY_YES", "BUY_NO"])].copy()

    if actionable_df.empty:
        return None

    actionable_df["confidence_rank"] = actionable_df["confidence"].map({
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
    }).fillna(0)

    actionable_df["selected_edge"] = actionable_df.apply(
        lambda row: safe_float(
            row.get("edge_yes") if safe_upper(row.get("action")) == "BUY_YES" else row.get("edge_no"),
            0.0,
        ),
        axis=1,
    )

    actionable_df["model_prob"] = actionable_df.apply(
        lambda row: pick_model_prob_from_row(row, None),
        axis=1,
    )

    if "distance_to_strike" in actionable_df.columns:
        actionable_df["distance_abs"] = actionable_df["distance_to_strike"].abs()
    else:
        actionable_df["distance_abs"] = float("inf")

    actionable_df = actionable_df.sort_values(
        by=["confidence_rank", "selected_edge", "distance_abs"],
        ascending=[False, False, True],
        kind="mergesort",
    )

    if actionable_df.empty:
        return None

    return actionable_df.iloc[0].copy()


def format_status_message(state: Dict[str, Any]) -> str:
    results = state.get("last_results") or {}
    plan = normalize_portfolio_plan(state.get("last_portfolio_plan"))
    execution_plan = state.get("last_execution_plan") or {}
    exit_df = state.get("last_exit_df", pd.DataFrame())
    watchlist_df = state.get("last_watchlist_df", pd.DataFrame())

    engine_state = "PAUSED" if state.get("paused") else "RUNNING"
    oil_price = format_optional_float(get_latest_oil_price_from_results(results), ".2f")
    volatility = format_optional_float(get_latest_volatility_from_results(results), ".4f")
    available_cash = format_optional_float(plan.get("available_cash"), ".2f")
    recommendation = plan.get("recommendation") or "N/A"

    actions = plan.get("actions") or []
    first_action = actions[0] if actions else {}
    ticker = safe_str(first_action.get("ticker")) or "N/A"
    side = safe_upper(first_action.get("side")) or "N/A"

    exit_active = "YES" if get_exit_trigger_active(exit_df) else "NO"
    watchlist_count = 0 if watchlist_df is None else len(watchlist_df)
    ready_count = execution_plan.get("ready_count", 0)
    awaiting_count = execution_plan.get("awaiting_count", 0)
    ts = current_time_et().strftime("%Y-%m-%d %H:%M:%S %Z")

    return (
        "📍 OIL ENGINE STATUS\n\n"
        f"State: {engine_state}\n"
        f"Oil Price: {oil_price}\n"
        f"Volatility: {volatility}\n"
        f"Portfolio Recommendation: {recommendation}\n"
        f"Ticker: {ticker}\n"
        f"Side: {side}\n"
        f"Available Cash: {available_cash}\n"
        f"Watchlist Rows: {watchlist_count}\n"
        f"Exit Trigger Active: {exit_active}\n"
        f"Execution Ready Count: {ready_count}\n"
        f"Awaiting Reconciliation: {awaiting_count}\n"
        f"Timestamp: {ts}"
    )


def format_positions_message(state: Dict[str, Any]) -> str:
    positions_df = state.get("last_open_positions_df", pd.DataFrame())
    exit_df = state.get("last_exit_df", pd.DataFrame())

    if positions_df is None or positions_df.empty:
        return "📦 LIVE POSITIONS\n\nNo open Kalshi positions currently tracked."

    working_df = positions_df.copy()

    if exit_df is not None and not exit_df.empty:
        merge_cols = [
            col for col in [
                "contract_ticker",
                "current_position_price",
                "unrealized_pnl",
                "should_exit",
                "decision_prob",
                "fair_prob_terminal",
                "fair_prob_blended",
            ]
            if col in exit_df.columns
        ]
        if "contract_ticker" in merge_cols and len(merge_cols) > 1:
            exit_lookup = exit_df[merge_cols].drop_duplicates(subset=["contract_ticker"])
            working_df = working_df.drop(
                columns=[
                    c for c in [
                        "current_position_price",
                        "unrealized_pnl",
                        "should_exit",
                        "decision_prob",
                        "fair_prob_terminal",
                        "fair_prob_blended",
                    ]
                    if c in working_df.columns
                ],
                errors="ignore",
            )
            working_df = working_df.merge(exit_lookup, on="contract_ticker", how="left")

    working_df = working_df.drop_duplicates(subset=["contract_ticker"]).copy()

    lines = ["📦 LIVE POSITIONS", ""]

    for _, row in working_df.iterrows():
        ticker = safe_str(row.get("contract_ticker")) or "N/A"

        action_text = safe_upper(row.get("action"))
        if action_text == "BUY_YES":
            side = "YES"
        elif action_text == "BUY_NO":
            side = "NO"
        else:
            side = safe_upper(row.get("side_norm")) or "N/A"

        contracts = safe_int(
            row.get("contracts", row.get("size", row.get("quantity", 0))),
            0,
        )
        contracts = abs(contracts)

        entry_price = format_optional_float(row.get("entry_price"), ".2f")
        current_price = format_optional_float(
            row.get("current_position_price", row.get("current_price")),
            ".2f",
        )
        unrealized_pnl = format_optional_float(row.get("unrealized_pnl"), ".3f")
        model_prob = format_optional_float(pick_model_prob_from_row(row, None), ".3f")
        should_exit = "YES" if bool(row.get("should_exit", False)) else "NO"

        lines.extend([
            f"Ticker: {ticker}",
            f"Side: {side}",
            f"Contracts: {contracts}",
            f"Entry Price: {entry_price}",
            f"Current Price: {current_price}",
            f"Unrealized PnL: {unrealized_pnl}",
            f"Current Model Prob: {model_prob}",
            f"Should Exit: {should_exit}",
            "",
        ])

    return "\n".join(lines).strip()


def format_latest_trade_message(state: Dict[str, Any]) -> str:
    results = state.get("last_results") or {}
    plan = normalize_portfolio_plan(state.get("last_portfolio_plan"))
    best_trade = get_best_actionable_trade(results)

    if best_trade is None:
        return "📈 LATEST ACTIONABLE TRADE\n\nNo actionable ranked trade candidates available."

    action = safe_upper(best_trade.get("action"))
    edge_value = safe_float(
        best_trade.get("edge_yes") if action == "BUY_YES" else best_trade.get("edge_no"),
        None,
    )
    ask_price_value = safe_float(
        best_trade.get("ask_yes") if action == "BUY_YES" else best_trade.get("ask_no"),
        None,
    )

    strike = format_optional_float(best_trade.get("strike"), ".2f")
    ask_price = format_optional_float(ask_price_value, ".2f")
    edge = format_optional_float(edge_value, ".4f")
    confidence = safe_upper(best_trade.get("confidence")) or "N/A"
    model_prob = format_optional_float(pick_model_prob_from_row(best_trade, None), ".3f")

    allocation = "N/A"
    contracts = "N/A"

    actions = plan.get("actions") or []
    best_trade_side = "YES" if action == "BUY_YES" else "NO"

    for a in actions:
        if (
            safe_str(a.get("ticker")) == safe_str(best_trade.get("contract_ticker"))
            and safe_upper(a.get("side")) == best_trade_side
        ):
            allocation = format_optional_float(a.get("allocation"), ".2f")
            contracts = str(safe_int(a.get("contracts"), 0))
            break

    return (
        "📈 LATEST ACTIONABLE TRADE\n\n"
        f"Ticker: {safe_str(best_trade.get('contract_ticker')) or 'N/A'}\n"
        f"Action: {action or 'N/A'}\n"
        f"Strike: {strike}\n"
        f"Ask Price: {ask_price}\n"
        f"Edge: {edge}\n"
        f"Model Prob: {model_prob}\n"
        f"Confidence: {confidence}\n"
        f"Recommended Allocation: {allocation}\n"
        f"Recommended Contracts: {contracts}"
    )


def handle_telegram_command(command_text: str, chat_id: str, state: Dict[str, Any], config) -> None:
    command = safe_str(command_text).split()[0].lower()

    if command == "/status":
        send_telegram_alert(format_status_message(state), chat_id=chat_id)
        return

    if command == "/positions":
        send_telegram_alert(format_positions_message(state), chat_id=chat_id)
        return

    if command == "/latest":
        send_telegram_alert(format_latest_trade_message(state), chat_id=chat_id)
        return

    if command == "/pause":
        state["paused"] = True
        send_telegram_alert("â¸ï¸ Bot paused successfully.", chat_id=chat_id)
        return

    if command == "/resume":
        state["paused"] = False
        send_telegram_alert("▶️ Bot resumed successfully.", chat_id=chat_id)
        return

    send_telegram_alert(
        "Unknown command.\n\nSupported commands:\n/status\n/positions\n/latest\n/pause\n/resume",
        chat_id=chat_id,
    )


def process_telegram_commands(state: Dict[str, Any], config) -> None:
    updates = fetch_telegram_updates(state.get("last_telegram_update_id"), config)
    if not updates:
        return

    latest_command_update, latest_update_id = extract_latest_command(updates)

    if latest_update_id is not None:
        state["last_telegram_update_id"] = latest_update_id

    if latest_command_update is None:
        return

    message = latest_command_update.get("message") or {}
    text = safe_str(message.get("text"))
    chat = message.get("chat") or {}
    chat_id = str(chat.get("id"))

    logging.info("Telegram command received | chat_id=%s command=%s", chat_id, text)
    handle_telegram_command(text, chat_id, runtime_state, config)


def update_runtime_state(
    state: Dict[str, Any],
    results: Optional[Dict[str, Any]],
    portfolio_plan: Optional[Dict[str, Any]],
    execution_plan: Optional[Dict[str, Any]],
    open_positions_df: Optional[pd.DataFrame],
    exit_df: Optional[pd.DataFrame],
    watchlist_df: Optional[pd.DataFrame],
    account_snapshot: Optional[Dict[str, Any]],
) -> None:
    state["last_results"] = results
    state["last_portfolio_plan"] = portfolio_plan
    state["last_execution_plan"] = execution_plan
    state["last_open_positions_df"] = open_positions_df if open_positions_df is not None else pd.DataFrame()
    state["last_exit_df"] = exit_df if exit_df is not None else pd.DataFrame()
    state["last_watchlist_df"] = watchlist_df if watchlist_df is not None else pd.DataFrame()
    state["last_account_snapshot"] = account_snapshot



def build_trade_stats_snapshot(
    *,
    state: Dict[str, Any],
    open_positions_df: Optional[pd.DataFrame],
    account_snapshot: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    stats = trade_stats_store(state)
    open_count = 0
    unrealized_pnl = 0.0

    if open_positions_df is not None and not open_positions_df.empty:
        open_count = int(len(open_positions_df))
        if "unrealized_pnl" in open_positions_df.columns:
            unrealized_series = pd.to_numeric(open_positions_df["unrealized_pnl"], errors="coerce")
            unrealized_pnl = float(unrealized_series.fillna(0.0).sum())

    completed = safe_int(stats.get("completed_trades"), 0) or 0
    wins = safe_int(stats.get("wins"), 0) or 0
    losses = safe_int(stats.get("losses"), 0) or 0
    breakeven = safe_int(stats.get("breakeven"), 0) or 0
    realized_pnl = safe_float(stats.get("realized_pnl"), 0.0) or 0.0
    decisive_closed = wins + losses
    win_rate = (100.0 * wins / decisive_closed) if decisive_closed > 0 else 0.0

    cash_balance = safe_float((account_snapshot or {}).get("cash_balance"), None)
    portfolio_value = safe_float((account_snapshot or {}).get("portfolio_value"), None)

    return {
        "open_positions": open_count,
        "completed_trades": completed,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": win_rate,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "cash_balance": cash_balance,
        "portfolio_value": portfolio_value,
    }


def log_trade_stats_line(
    *,
    state: Dict[str, Any],
    open_positions_df: Optional[pd.DataFrame],
    account_snapshot: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    snapshot = build_trade_stats_snapshot(
        state=state,
        open_positions_df=open_positions_df,
        account_snapshot=account_snapshot,
    )
    logging.info(
        "TRADE STATS | open=%s | completed=%s | wins=%s | losses=%s | breakeven=%s | win_rate=%.1f%% | realized_pnl=%+.2f | unrealized_pnl=%+.2f | cash=%.2f | portfolio=%.2f",
        snapshot.get("open_positions", 0),
        snapshot.get("completed_trades", 0),
        snapshot.get("wins", 0),
        snapshot.get("losses", 0),
        snapshot.get("breakeven", 0),
        safe_float(snapshot.get("win_rate"), 0.0) or 0.0,
        safe_float(snapshot.get("realized_pnl"), 0.0) or 0.0,
        safe_float(snapshot.get("unrealized_pnl"), 0.0) or 0.0,
        safe_float(snapshot.get("cash_balance"), 0.0) or 0.0,
        safe_float(snapshot.get("portfolio_value"), 0.0) or 0.0,
    )
    return snapshot


def main():
    parser = argparse.ArgumentParser(description="Oil Kalshi Decision Engine runner")
    parser.add_argument("--config", default="settings.yaml", help="Path to YAML config")
    parser.add_argument("--run-once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()

    config = load_config(args.config)
    config = normalize_config_sections(config)
    configure_logging(config)
    runtime_state["last_results_config"] = config

    persisted_state = load_runtime_state_from_disk(config)
    if persisted_state:
        runtime_state["paused"] = bool(
            persisted_state.get("paused", runtime_state.get("paused", False))
        )
        runtime_state["order_intents"] = dict(
            persisted_state.get("order_intents") or {}
        )
        runtime_state["execution_tracking"] = dict(
            persisted_state.get("execution_tracking") or {}
        )
        runtime_state["last_execution_mode"] = (
            safe_str(
                persisted_state.get("last_execution_mode")
                or get_execution_mode(config)
            )
            or get_execution_mode(config)
        )
        if "paper_cash_balance" in persisted_state:
            runtime_state["paper_cash_balance"] = safe_float(
                persisted_state.get("paper_cash_balance"),
                runtime_state.get("paper_cash_balance"),
            )
        runtime_state["held_position_state"] = dict(
            persisted_state.get("held_position_state") or {}
        )
        runtime_state["last_trade_ts"] = safe_float(
            persisted_state.get("last_trade_ts"),
            runtime_state.get("last_trade_ts"),
        )
        runtime_state["last_entry_trade_ts"] = safe_float(
            persisted_state.get("last_entry_trade_ts"),
            runtime_state.get("last_entry_trade_ts"),
        )
        runtime_state["last_entry_trade_ts_by_key"] = dict(
            persisted_state.get("last_entry_trade_ts_by_key") or {}
        )
        runtime_state["trade_stats"] = dict(
            persisted_state.get("trade_stats") or runtime_state.get("trade_stats") or {}
        )
        runtime_state["recent_exit_ts_by_ticker"] = dict(
            persisted_state.get("recent_exit_ts_by_ticker") or {}
        )

    ensure_paper_cash_balance_initialized(runtime_state, config)
    trade_stats_store(runtime_state)
    recent_exit_store(runtime_state)

    runtime_cfg = config.get("runtime", {}) or {}
    poll_seconds = int(
        runtime_cfg.get("poll_seconds", runtime_cfg.get("poll_interval_seconds", 110))
    )
    summary_interval = get_summary_interval(config)
    last_summary_time = 0

    ml_writer = MLDataWriter(
        base_dir="./logs/ml",
        engine_name="oil_engine_daily",
        engine_version="v1",
    )

    logging.info(
        "Starting oil engine. run_once=%s poll=%ss summary_interval=%ss",
        args.run_once,
        poll_seconds,
        summary_interval,
    )

    while True:
        try:
            live_positions_df, account_snapshot, raw_account_snapshot = (
                load_real_kalshi_account_data(config)
            )
            execution_mode = get_execution_mode(config)
            paper_positions_records = load_paper_positions_from_disk(config)
            runtime_state["paper_positions_cache"] = paper_positions_records
            paper_open_records = get_open_paper_positions(paper_positions_records)

            live_positions_active_df = filter_to_active_positions(
                live_positions_df,
                log_prefix="Runner pre-evaluation active-position filter",
            )
            live_held_contract_tickers = extract_live_contract_tickers(
                live_positions_active_df
            )
            paper_held_contract_tickers = extract_paper_contract_tickers(
                paper_open_records
            )

            if execution_mode == EXECUTION_MODE_SIMULATION:
                held_contract_tickers = paper_held_contract_tickers
                if held_contract_tickers:
                    logging.info(
                        "Engine-scoped paper held contracts detected before evaluation | count=%s | tickers=%s",
                        len(held_contract_tickers),
                        held_contract_tickers[:10],
                    )
                else:
                    logging.info(
                        "No engine-scoped paper held contracts detected before evaluation."
                    )
            else:
                held_contract_tickers = live_held_contract_tickers
                if held_contract_tickers:
                    logging.info(
                        "Engine-scoped live held contracts detected before evaluation | count=%s | tickers=%s",
                        len(held_contract_tickers),
                        held_contract_tickers[:10],
                    )
                else:
                    logging.info(
                        "No engine-scoped live held contracts detected before evaluation."
                    )

            if execution_mode == EXECUTION_MODE_SIMULATION:
                account_snapshot = build_paper_account_snapshot(
                    runtime_state,
                    open_positions_df=pd.DataFrame(),
                )
            elif raw_account_snapshot is not None:
                account_snapshot = build_engine_scoped_account_snapshot(
                    raw_account_snapshot=raw_account_snapshot,
                    filtered_positions_df=live_positions_active_df,
                )

            if account_snapshot is not None:
                logging.info(
                    "Engine-scoped account snapshot | mode=%s | cash_balance=%.2f | positions_value=%.2f | portfolio_value=%.2f | engine_open_positions_count=%s",
                    execution_mode,
                    safe_float(account_snapshot.get("cash_balance"), 0.0),
                    safe_float(account_snapshot.get("positions_value"), 0.0),
                    safe_float(account_snapshot.get("portfolio_value"), 0.0),
                    account_snapshot.get("engine_open_positions_count", 0),
                )

            results = run_engine_once(
                config,
                force_include_contract_tickers=held_contract_tickers,
            )

            if not results or not isinstance(results, dict):
                logging.error("Engine output is None or invalid.")
                write_paper_trade_cycle_log(
                    config=config,
                    status="ERROR",
                    error="engine_output_none",
                    force_include_contract_tickers=held_contract_tickers,
                )
                process_status_alert(
                    status_type="engine_output_none",
                    config=config,
                    payload={"force_include_contract_tickers": held_contract_tickers},
                )
                if args.run_once:
                    break
                time.sleep(poll_seconds)
                continue

            now = time.time()
            run_id = build_run_id("oil_engine_daily")
            cycle_timestamp_et = current_time_et().isoformat()

            if now - last_summary_time >= summary_interval:
                log_engine_snapshot(results, config)
                last_summary_time = now

            ranked_df = (results or {}).get("ranked_df", pd.DataFrame())

            try:
                ml_writer.write_candidate_snapshot(
                    ranked_df=ranked_df,
                    run_id=run_id,
                    cycle_timestamp_et=cycle_timestamp_et,
                    config=config,
                    portfolio_plan=None,
                )
            except Exception as exc:
                logging.warning("ML candidate snapshot write failed | error=%s", exc)

            try:
                upload_ml_logs()
            except Exception as exc:
                logging.warning("R2 upload failed after candidate snapshot write | error=%s", exc)

            open_positions_df = pd.DataFrame()
            watchlist_df = pd.DataFrame()
            exit_df = pd.DataFrame()
            portfolio_plan = None
            execution_plan = None

            if (
                not results.get("skipped")
                and ranked_df is not None
                and not ranked_df.empty
            ):
                if execution_mode == EXECUTION_MODE_SIMULATION:
                    open_positions_df = normalize_paper_positions_for_monitoring(
                        paper_open_records,
                        ranked_df,
                    )
                    account_snapshot = build_paper_account_snapshot(
                        runtime_state,
                        open_positions_df=open_positions_df,
                    )
                    logging.info(
                        "Engine paper/live merged positions | mode=%s | paper_count=%s | live_count=%s | operative_count=%s",
                        execution_mode,
                        len(paper_open_records),
                        len(live_positions_active_df)
                        if live_positions_active_df is not None
                        else 0,
                        len(open_positions_df),
                    )
                else:
                    if (
                        live_positions_active_df is not None
                        and not live_positions_active_df.empty
                    ):
                        engine_scope = _get_engine_scope(config)
                        live_positions_active_df = filter_positions_to_engine_scope(
                            live_positions_active_df,
                            exact_market_ticker=engine_scope.get("exact_market_ticker"),
                            event_ticker=engine_scope.get("event_ticker"),
                            series_ticker=engine_scope.get("series_ticker"),
                        )

                        live_positions_active_df = filter_to_active_positions(
                            live_positions_active_df,
                            log_prefix="Runner post-scope active-position filter",
                        )

                        open_positions_df = normalize_kalshi_positions_for_monitoring(
                            live_positions_active_df,
                            ranked_df,
                        )

                        account_snapshot = build_engine_scoped_account_snapshot(
                            raw_account_snapshot=raw_account_snapshot,
                            filtered_positions_df=open_positions_df,
                        )

            if execution_mode == EXECUTION_MODE_SIMULATION:
                resolved_closes = detect_and_close_resolved_paper_positions(
                    state=runtime_state,
                    config=config,
                    ranked_df=ranked_df,
                )
                if resolved_closes > 0:
                    paper_positions_records = normalize_paper_positions_records(
                        runtime_state.get("paper_positions_cache") or []
                    )
                    paper_open_records = get_open_paper_positions(paper_positions_records)
                    open_positions_df = normalize_paper_positions_for_monitoring(
                        paper_open_records,
                        ranked_df,
                    )
                    account_snapshot = build_paper_account_snapshot(
                        runtime_state,
                        open_positions_df=open_positions_df,
                    )

            exit_df = compute_position_exit_df(
                results,
                config,
                open_positions_df=open_positions_df,
            )

            portfolio_input_positions_df = exit_df if exit_df is not None and not exit_df.empty else open_positions_df
            portfolio_account_snapshot = attach_recent_exit_cooldown_to_account_snapshot(
                account_snapshot,
                runtime_state,
                config,
            )
            portfolio_plan = build_portfolio_advisory_plan(
                results,
                config,
                open_positions_df=portfolio_input_positions_df,
                account_snapshot=portfolio_account_snapshot,
            )
            if portfolio_plan is None:
                logging.error("Portfolio plan is None or invalid.")
                portfolio_plan = {
                    "recommendation": "UNAVAILABLE",
                    "reason": "Portfolio plan unavailable because build_micro_allocation_plan returned no plan.",
                    "actions": [],
                    "available_cash": safe_float(
                        (account_snapshot or {}).get("cash_balance"), 0.0
                    ),
                    "capital": safe_float(
                        (account_snapshot or {}).get("portfolio_value"), 0.0
                    ),
                    "reserve_cash_target": 0.0,
                    "deployable_cash": 0.0,
                }
            log_portfolio_recommendation(portfolio_plan)

            try:
                ml_writer.write_portfolio_decision(
                    plan=portfolio_plan,
                    run_id=run_id,
                    cycle_timestamp_et=cycle_timestamp_et,
                    config=config,
                )
            except Exception as exc:
                logging.warning("ML portfolio decision write failed | error=%s", exc)

            try:
                upload_ml_logs()
            except Exception as exc:
                logging.warning("R2 upload failed after portfolio decision write | error=%s", exc)

            execution_plan = build_execution_intent_plan(
                portfolio_plan=portfolio_plan,
                exit_df=exit_df,
                open_positions_df=open_positions_df,
                account_snapshot=account_snapshot,
                state=runtime_state,
            )
            log_execution_intent_plan(execution_plan)
            register_execution_plan_actions(execution_plan, runtime_state)
            write_paper_trade_action_log(
                config=config,
                phase="execution_plan",
                actions=(execution_plan or {}).get("all_actions") or [],
                state=runtime_state,
                extra={
                    "portfolio_recommendation": (portfolio_plan or {}).get(
                        "recommendation"
                    ),
                    "portfolio_reason": (portfolio_plan or {}).get("reason"),
                },
            )

            logging.info("Execution orchestration step | phase=EXITS_FIRST")
            emit_position_exit_alerts(exit_df, config)

            logging.info("Execution orchestration step | phase=ORDER_SUBMISSION")
            order_submission_summary = process_order_intents(
                execution_plan=execution_plan,
                state=runtime_state,
                config=config,
            )
            write_paper_trade_action_log(
                config=config,
                phase="order_submission",
                actions=(execution_plan or {}).get("all_actions") or [],
                state=runtime_state,
                extra={"submission_summary": order_submission_summary or {}},
            )
            persist_runtime_state(config, runtime_state, note="after_order_submission")
            maybe_archive_logs(
                config,
                runtime_state,
                note="after_order_submission",
                extra={"phase": "order_submission"},
            )

            logging.info("Execution orchestration step | phase=ORDER_RECONCILIATION")
            order_reconciliation_summary = reconcile_order_intents(
                state=runtime_state,
                open_positions_df=open_positions_df,
                config=config,
            )
            prune_summary = prune_resolved_order_state(runtime_state)

            paper_positions_records = normalize_paper_positions_records(
                runtime_state.get("paper_positions_cache") or []
            )
            paper_open_records = get_open_paper_positions(paper_positions_records)

            if execution_mode == EXECUTION_MODE_SIMULATION:
                refreshed_open_positions_df = normalize_paper_positions_for_monitoring(
                    paper_open_records,
                    ranked_df,
                )
                open_positions_df = refreshed_open_positions_df
                account_snapshot = build_paper_account_snapshot(
                    runtime_state,
                    open_positions_df=open_positions_df,
                )
                logging.info(
                    "Post-reconciliation simulation refresh | paper_total_count=%s | paper_open_count=%s | operative_open_count=%s | cash_balance=%.2f | positions_value=%.2f | portfolio_value=%.2f",
                    len(paper_positions_records),
                    len(paper_open_records),
                    len(open_positions_df) if open_positions_df is not None else 0,
                    safe_float(account_snapshot.get("cash_balance"), 0.0),
                    safe_float(account_snapshot.get("positions_value"), 0.0),
                    safe_float(account_snapshot.get("portfolio_value"), 0.0),
                )

            trade_stats_snapshot = log_trade_stats_line(
                state=runtime_state,
                open_positions_df=open_positions_df,
                account_snapshot=account_snapshot,
            )

            try:
                written_trade_outcomes = flush_pending_trade_outcomes(
                    ml_writer=ml_writer,
                    state=runtime_state,
                )
                if written_trade_outcomes > 0:
                    upload_ml_logs()
            except Exception as exc:
                logging.warning("Trade outcome flush/upload failed | error=%s", exc)

            write_paper_trade_action_log(
                config=config,
                phase="order_reconciliation",
                actions=(execution_plan or {}).get("all_actions") or [],
                state=runtime_state,
                extra={"reconciliation_summary": order_reconciliation_summary or {}, "prune_summary": prune_summary or {}, "trade_outcomes_written": written_trade_outcomes if 'written_trade_outcomes' in locals() else 0, "trade_stats": trade_stats_snapshot if 'trade_stats_snapshot' in locals() else {}},
            )
            persist_runtime_state(
                config,
                runtime_state,
                note="after_order_reconciliation",
            )
            persist_paper_positions(
                config,
                runtime_state.get("paper_positions_cache") or [],
                note="after_order_reconciliation",
            )
            maybe_archive_logs(
                config,
                runtime_state,
                note="after_order_reconciliation",
                extra={"phase": "order_reconciliation"},
            )

            logging.info(
                "Execution orchestration step | phase=PORTFOLIO_AND_ENTRY_INTENTS"
            )

            post_recon_account_snapshot = attach_recent_exit_cooldown_to_account_snapshot(
                account_snapshot,
                runtime_state,
                config,
            )
            post_recon_portfolio_plan = build_portfolio_advisory_plan(
                results,
                config,
                open_positions_df=open_positions_df,
                account_snapshot=post_recon_account_snapshot,
            )
            if post_recon_portfolio_plan is None:
                post_recon_portfolio_plan = portfolio_plan
            else:
                logging.info(
                    "Post-reconciliation portfolio replan complete | recommendation=%s | reason=%s",
                    (post_recon_portfolio_plan or {}).get("recommendation"),
                    (post_recon_portfolio_plan or {}).get("reason"),
                )
                log_portfolio_recommendation(post_recon_portfolio_plan)

            post_recon_execution_plan = build_execution_intent_plan(
                portfolio_plan=post_recon_portfolio_plan,
                exit_df=pd.DataFrame(),
                open_positions_df=open_positions_df,
                account_snapshot=account_snapshot,
                state=runtime_state,
            )

            post_recon_entry_actions = [
                action for action in ((post_recon_execution_plan or {}).get("all_actions") or [])
                if safe_upper(action.get("action_type")) == "ENTER"
                and safe_upper(action.get("execution_state")) == EXECUTION_STATE_READY
            ]

            if post_recon_entry_actions:
                logging.info(
                    "Execution orchestration step | phase=POST_RECON_ENTRY_SUBMISSION | ready_entries=%s",
                    len(post_recon_entry_actions),
                )
                post_recon_entry_plan = dict(post_recon_execution_plan or {})
                post_recon_entry_plan["all_actions"] = post_recon_entry_actions
                post_recon_entry_plan["entry_actions"] = post_recon_entry_actions
                post_recon_entry_plan["exit_actions"] = []
                post_recon_entry_plan["hold_actions"] = []
                post_recon_entry_plan["planned_exit_count"] = 0
                post_recon_entry_plan["planned_enter_count"] = len(post_recon_entry_actions)
                post_recon_entry_plan["informational_hold_count"] = 0
                post_recon_entry_plan["ready_count"] = len(post_recon_entry_actions)
                post_recon_entry_plan["skipped_count"] = 0
                post_recon_entry_plan["awaiting_count"] = 0

                post_recon_submission_summary = process_order_intents(
                    execution_plan=post_recon_entry_plan,
                    state=runtime_state,
                    config=config,
                )
                write_paper_trade_action_log(
                    config=config,
                    phase="post_reconciliation_entry_submission",
                    actions=post_recon_entry_actions,
                    state=runtime_state,
                    extra={"submission_summary": post_recon_submission_summary or {}},
                )

                if execution_mode == EXECUTION_MODE_SIMULATION:
                    paper_positions_records = normalize_paper_positions_records(
                        runtime_state.get("paper_positions_cache") or []
                    )
                    paper_open_records = get_open_paper_positions(paper_positions_records)
                    open_positions_df = normalize_paper_positions_for_monitoring(
                        paper_open_records,
                        ranked_df,
                    )
                    account_snapshot = build_paper_account_snapshot(
                        runtime_state,
                        open_positions_df=open_positions_df,
                    )
                    trade_stats_snapshot = log_trade_stats_line(
                        state=runtime_state,
                        open_positions_df=open_positions_df,
                        account_snapshot=account_snapshot,
                    )

                prune_summary_post_recon = prune_resolved_order_state(runtime_state)
                logging.info(
                    "Post-reconciliation entry submission complete | submitted=%s | pruned_intents=%s | pruned_tracking=%s",
                    (post_recon_submission_summary or {}).get("submitted_count", 0),
                    (prune_summary_post_recon or {}).get("removed_intents", 0),
                    (prune_summary_post_recon or {}).get("removed_tracking", 0),
                )

                portfolio_plan = post_recon_portfolio_plan
                execution_plan = post_recon_execution_plan
            else:
                portfolio_plan = post_recon_portfolio_plan
                execution_plan = post_recon_execution_plan

            emit_portfolio_alert(
                portfolio_plan,
                config,
                alerts_enabled=(not runtime_state.get("paused", False)),
            )

            watchlist_df = process_watchlist_alerts(
                results,
                config,
                alerts_enabled=(not runtime_state.get("paused", False)),
            )

            process_trade_alerts(
                results,
                config,
                alerts_enabled=(not runtime_state.get("paused", False)),
            )

            final_order_summary = summarize_order_intents(
                runtime_state,
                config,
                include_log=False,
            )
            write_paper_trade_cycle_log(
                config=config,
                results=results,
                portfolio_plan=portfolio_plan,
                execution_plan=execution_plan,
                order_summary=final_order_summary,
                open_positions_df=open_positions_df,
                exit_df=exit_df,
                watchlist_df=watchlist_df,
                account_snapshot=account_snapshot,
            )

            update_runtime_state(
                runtime_state,
                results=results,
                portfolio_plan=portfolio_plan,
                execution_plan=execution_plan,
                open_positions_df=open_positions_df,
                exit_df=exit_df,
                watchlist_df=watchlist_df,
                account_snapshot=account_snapshot,
            )
            persist_runtime_state(config, runtime_state, note="cycle_complete")
            persist_paper_positions(
                config,
                runtime_state.get("paper_positions_cache") or [],
                note="cycle_complete",
            )
            try:
                upload_ml_logs()
            except Exception as exc:
                logging.warning("R2 upload failed at cycle_complete | error=%s", exc)
            maybe_archive_logs(
                config,
                runtime_state,
                note="cycle_complete",
                extra={
                    "phase": "cycle_complete",
                    "portfolio_recommendation": (portfolio_plan or {}).get(
                        "recommendation"
                    ),
                },
            )

            process_telegram_commands(runtime_state, config)
            persist_runtime_state(config, runtime_state, note="post_telegram")
            persist_paper_positions(
                config,
                runtime_state.get("paper_positions_cache") or [],
                note="post_telegram",
            )
            maybe_archive_logs(
                config,
                runtime_state,
                note="post_telegram",
                extra={"phase": "post_telegram"},
            )

        except KeyboardInterrupt:
            logging.info("Interrupted by user. Exiting.")
            break

        except Exception as exc:
            logging.exception("Engine cycle failed: %s", exc)
            process_status_alert(
                status_type="engine_cycle_error",
                config=config,
                payload={"error": str(exc)},
            )

            try:
                process_telegram_commands(runtime_state, config)
            except Exception as telegram_exc:
                logging.warning(
                    "Telegram command processing also failed after engine error: %s",
                    telegram_exc,
                )
            try:
                persist_runtime_state(config, runtime_state, note="error_path")
                persist_paper_positions(
                    config,
                    runtime_state.get("paper_positions_cache") or [],
                    note="error_path",
                )
            except Exception as persist_exc:
                logging.warning(
                    "Runtime state persist failed after engine error: %s",
                    persist_exc,
                )

        if args.run_once:
            break

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
