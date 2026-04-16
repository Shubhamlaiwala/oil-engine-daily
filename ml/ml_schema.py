from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
SCHEMA_VERSION = "1.0.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def et_now_iso() -> str:
    return datetime.now(ET).isoformat()


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        if value.tzinfo is None:
            value = value.tz_localize(ET)
        return value.isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=ET)
        return value.isoformat()
    if isinstance(value, (pd.Series,)):
        return {str(k): safe_value(v) for k, v in value.to_dict().items()}
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_value(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def config_hash(config: Dict[str, Any]) -> str:
    normalized = json.dumps(safe_value(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def build_run_id(engine_name: str, cycle_timestamp_et: Optional[str] = None) -> str:
    ts = cycle_timestamp_et or et_now_iso()
    raw = f"{engine_name}|{ts}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def build_trade_id(contract_ticker: str, side: str, entry_timestamp_et: str) -> str:
    raw = f"{contract_ticker}|{side}|{entry_timestamp_et}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def base_record(
    *,
    engine_name: str,
    run_id: str,
    config_hash_value: str,
    engine_version: str = "paper_v1",
    schema_version: str = SCHEMA_VERSION,
) -> Dict[str, Any]:
    return {
        "schema_version": schema_version,
        "engine_version": engine_version,
        "engine_name": engine_name,
        "run_id": run_id,
        "recorded_at_utc": utc_now_iso(),
        "recorded_at_et": et_now_iso(),
        "config_hash": config_hash_value,
    }


CANDIDATE_FIELDS = [
    "cycle_timestamp_et",
    "event_ticker",
    "series_ticker",
    "contract_ticker",
    "trading_phase",
    "decision_state",
    "entry_style",
    "selected_side",
    "selected_edge",
    "edge_yes",
    "edge_no",
    "decision_prob",
    "fair_prob_terminal",
    "fair_prob_blended",
    "ask_yes",
    "ask_no",
    "bid_yes",
    "bid_no",
    "yes_no_ask_sum",
    "overround",
    "market_too_wide",
    "distance_to_strike",
    "confidence",
    "volatility",
    "dynamic_drift",
    "action",
    "portfolio_action",
    "portfolio_reason",
    "was_top_ranked",
    "was_executed",
]


PORTFOLIO_DECISION_FIELDS = [
    "recommendation",
    "reason",
    "capital",
    "available_cash",
    "reserve_cash_target",
    "deployable_cash",
    "tradable_candidates_count",
    "watchlist_candidates_count",
    "actions",
]


TRADE_OUTCOME_FIELDS = [
    "trade_id",
    "contract_ticker",
    "side",
    "contracts",
    "entry_timestamp_et",
    "exit_timestamp_et",
    "entry_price",
    "exit_price",
    "settlement_value",
    "realized_pnl",
    "realized_return",
    "hold_minutes",
    "exit_reason",
    "settled",
]


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(key, default)
    try:
        return row[key]
    except Exception:
        return default


def candidate_record_from_row(
    *,
    row: Any,
    engine_name: str,
    run_id: str,
    cycle_timestamp_et: str,
    config_hash_value: str,
    portfolio_action: Optional[str] = None,
    portfolio_reason: Optional[str] = None,
    was_top_ranked: bool = False,
    was_executed: bool = False,
    engine_version: str = "paper_v1",
) -> Dict[str, Any]:
    record = base_record(
        engine_name=engine_name,
        run_id=run_id,
        config_hash_value=config_hash_value,
        engine_version=engine_version,
    )
    for field in CANDIDATE_FIELDS:
        record[field] = None

    record.update({
        "cycle_timestamp_et": cycle_timestamp_et,
        "event_ticker": _row_get(row, "event_ticker"),
        "series_ticker": _row_get(row, "series_ticker"),
        "contract_ticker": _row_get(row, "contract_ticker"),
        "trading_phase": _row_get(row, "trading_phase"),
        "decision_state": _row_get(row, "decision_state"),
        "entry_style": _row_get(row, "entry_style"),
        "selected_side": _row_get(row, "selected_side"),
        "selected_edge": _row_get(row, "selected_edge"),
        "edge_yes": _row_get(row, "edge_yes"),
        "edge_no": _row_get(row, "edge_no"),
        "decision_prob": _row_get(row, "decision_prob"),
        "fair_prob_terminal": _row_get(row, "fair_prob_terminal"),
        "fair_prob_blended": _row_get(row, "fair_prob_blended"),
        "ask_yes": _row_get(row, "ask_yes"),
        "ask_no": _row_get(row, "ask_no"),
        "bid_yes": _row_get(row, "bid_yes"),
        "bid_no": _row_get(row, "bid_no"),
        "yes_no_ask_sum": _row_get(row, "yes_no_ask_sum"),
        "overround": _row_get(row, "overround"),
        "market_too_wide": _row_get(row, "market_too_wide"),
        "distance_to_strike": _row_get(row, "distance_to_strike"),
        "confidence": _row_get(row, "confidence"),
        "volatility": _row_get(row, "volatility"),
        "dynamic_drift": _row_get(row, "dynamic_drift"),
        "action": _row_get(row, "action"),
        "portfolio_action": portfolio_action,
        "portfolio_reason": portfolio_reason,
        "was_top_ranked": bool(was_top_ranked),
        "was_executed": bool(was_executed),
    })
    return safe_value(record)


def portfolio_decision_record(
    *,
    plan: Dict[str, Any],
    engine_name: str,
    run_id: str,
    cycle_timestamp_et: str,
    config_hash_value: str,
    engine_version: str = "paper_v1",
) -> Dict[str, Any]:
    record = base_record(
        engine_name=engine_name,
        run_id=run_id,
        config_hash_value=config_hash_value,
        engine_version=engine_version,
    )
    record["cycle_timestamp_et"] = cycle_timestamp_et
    for field in PORTFOLIO_DECISION_FIELDS:
        record[field] = safe_value(plan.get(field))
    return safe_value(record)


def trade_outcome_record(
    *,
    trade_id: str,
    contract_ticker: str,
    side: str,
    contracts: int,
    entry_timestamp_et: str,
    exit_timestamp_et: Optional[str],
    entry_price: Optional[float],
    exit_price: Optional[float],
    settlement_value: Optional[float],
    realized_pnl: Optional[float],
    realized_return: Optional[float],
    hold_minutes: Optional[float],
    exit_reason: Optional[str],
    settled: bool,
    engine_name: str,
    run_id: str,
    config_hash_value: str,
    engine_version: str = "paper_v1",
) -> Dict[str, Any]:
    record = base_record(
        engine_name=engine_name,
        run_id=run_id,
        config_hash_value=config_hash_value,
        engine_version=engine_version,
    )
    record.update({
        "trade_id": trade_id,
        "contract_ticker": contract_ticker,
        "side": side,
        "contracts": contracts,
        "entry_timestamp_et": entry_timestamp_et,
        "exit_timestamp_et": exit_timestamp_et,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "settlement_value": settlement_value,
        "realized_pnl": realized_pnl,
        "realized_return": realized_return,
        "hold_minutes": hold_minutes,
        "exit_reason": exit_reason,
        "settled": settled,
    })
    return safe_value(record)
