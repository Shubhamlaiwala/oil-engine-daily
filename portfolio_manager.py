from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class TradeInstruction:
    action: str
    ticker: Optional[str] = None
    side: Optional[str] = None
    contracts: int = 0
    allocation: float = 0.0
    ask_price: Optional[float] = None
    bid_price: Optional[float] = None
    edge: Optional[float] = None
    confidence: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "ticker": self.ticker,
            "side": self.side,
            "contracts": int(self.contracts),
            "allocation": round(float(self.allocation or 0.0), 2),
            "ask_price": round(float(self.ask_price), 4) if self.ask_price is not None else None,
            "bid_price": round(float(self.bid_price), 4) if self.bid_price is not None else None,
            "edge": round(float(self.edge), 4) if self.edge is not None else None,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class PortfolioPlan:
    recommendation: str
    reason: str
    capital: float = 0.0
    available_cash: float = 0.0
    reserve_cash_target: float = 0.0
    deployable_cash: float = 0.0
    tradable_candidates_count: int = 0
    watchlist_candidates_count: int = 0
    actions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "reason": self.reason,
            "capital": round(float(self.capital or 0.0), 2),
            "available_cash": round(float(self.available_cash or 0.0), 2),
            "reserve_cash_target": round(float(self.reserve_cash_target or 0.0), 2),
            "deployable_cash": round(float(self.deployable_cash or 0.0), 2),
            "tradable_candidates_count": int(self.tradable_candidates_count or 0),
            "watchlist_candidates_count": int(self.watchlist_candidates_count or 0),
            "actions": self.actions,
        }


# =========================================================
# CONSTANTS / METADATA
# =========================================================
ACTION_METADATA_FIELDS = [
    "decision_prob",
    "fair_prob_terminal",
    "fair_prob_blended",
    "yes_no_ask_sum",
    "overround",
    "market_too_wide",
    "no_trade_reason",
    "decision_state",
    "entry_style",
    "target_yes_price",
    "target_no_price",
    "executable_yes_now",
    "executable_no_now",
    "selected_side",
    "selected_edge",
    "selected_ask",
    "selected_bid",
    "edge_yes",
    "edge_no",
    "ask_yes",
    "ask_no",
    "confidence",
    "confidence_norm",
    "distance_to_strike",
    "distance_abs",
    "trading_phase",
    "real_edge",
    "candidate_score",
    "event_ticker",
    "series_ticker",
    "strike",
    "resolution_time_et",
]

DEFAULT_MAX_CAPITAL_DEPLOY = 10.0
DEFAULT_MAX_DOLLARS_PER_TRADE = 3.0
DEFAULT_MAX_CONTRACTS_PER_TRADE = 1
DEFAULT_STOP_TRADING_IF_BALANCE_BELOW = 1.0
DEFAULT_ALLOWED_ENTRY_TRADING_PHASES = ["ACTIVE_TRADING"]
DEFAULT_ALLOWED_ENTRY_TRADING_PHASES_WHEN_FLAT = ["ACTIVE_TRADING"]
PHASE_PREVIEW_COLUMNS = [
    "contract_ticker",
    "event_ticker",
    "decision_state",
    "trading_phase",
    "selected_side",
    "selected_edge",
    "real_edge",
    "selected_ask",
    "overround",
    "market_too_wide",
    "entry_style",
    "candidate_score",
]


# =========================================================
# BASIC HELPERS
# =========================================================
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_str(value: Any, default: str = "") -> str:
    try:
        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        return str(value).strip()
    except Exception:
        return default


def safe_upper(value: Any, default: str = "") -> str:
    return safe_str(value, default).upper()


def safe_row_value(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    try:
        if isinstance(row, pd.Series):
            if key not in row.index:
                return default
            value = row.get(key, default)
        elif isinstance(row, dict):
            value = row.get(key, default)
        else:
            getter = getattr(row, "get", None)
            value = getter(key, default) if callable(getter) else default

        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        return value
    except Exception:
        return default


def _extract_strike_from_ticker(ticker: str) -> Optional[float]:
    if not ticker:
        return None
    match = re.search(r"-T(\d+(?:\.\d+)?)$", str(ticker).strip())
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _extract_event_ticker_from_contract_ticker(ticker: str) -> str:
    value = str(ticker or "").strip()
    if not value:
        return ""
    if "-T" in value:
        return value.rsplit("-T", 1)[0]
    return value


def _safe_preview_records(df: pd.DataFrame, preview_cols: List[str], head_n: int = 5) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    cols = [c for c in preview_cols if c in df.columns]
    if not cols:
        return []
    preview_df = df[cols].head(head_n).copy()
    preview_df = preview_df.replace({np.nan: None})
    return preview_df.to_dict(orient="records")


def _append_watchlist_context(base_reason: str, watchlist_count: int) -> str:
    if watchlist_count <= 0:
        return base_reason
    plural = "candidate" if watchlist_count == 1 else "candidates"
    return f"{base_reason} {watchlist_count} watchlist {plural} identified."


def extract_action_metadata(row: Any) -> Dict[str, Any]:
    return {field: safe_row_value(row, field, None) for field in ACTION_METADATA_FIELDS}


def build_portfolio_action(
    action: str,
    ticker: Optional[str],
    side: Optional[str],
    contracts: int,
    allocation: float,
    source_row: Any = None,
    ask_price: Optional[float] = None,
    bid_price: Optional[float] = None,
    edge: Optional[float] = None,
    confidence: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "action": action,
        "ticker": ticker,
        "side": side,
        "contracts": int(_safe_int(contracts, 0)),
        "allocation": round(float(_safe_float(allocation, 0.0)), 2),
        "ask_price": round(float(ask_price), 4) if ask_price is not None and not pd.isna(ask_price) else None,
        "bid_price": round(float(bid_price), 4) if bid_price is not None and not pd.isna(bid_price) else None,
        "edge": round(float(edge), 4) if edge is not None and not pd.isna(edge) else None,
        "confidence": confidence,
        "reason": reason,
    }
    payload.update(extract_action_metadata(source_row))
    return payload


# =========================================================
# NORMALIZATION
# =========================================================
def _chosen_side(row: pd.Series) -> str:
    edge_yes = _safe_float(row.get("edge_yes"), 0.0)
    edge_no = _safe_float(row.get("edge_no"), 0.0)
    return "YES" if edge_yes >= edge_no else "NO"


def _chosen_edge(row: pd.Series) -> float:
    return _safe_float(row.get("edge_yes"), 0.0) if _chosen_side(row) == "YES" else _safe_float(row.get("edge_no"), 0.0)


def _chosen_ask(row: pd.Series):
    return row.get("ask_yes") if _chosen_side(row) == "YES" else row.get("ask_no")


def _chosen_bid(row: pd.Series):
    return row.get("bid_yes") if _chosen_side(row) == "YES" else row.get("bid_no")


def _candidate_side(row: pd.Series) -> str:
    selected_side = safe_upper(safe_row_value(row, "selected_side", ""))
    if selected_side in {"YES", "NO"}:
        return selected_side
    action_norm = safe_upper(safe_row_value(row, "action_norm", ""))
    if action_norm == "BUY_YES":
        return "YES"
    if action_norm == "BUY_NO":
        return "NO"
    return ""


def _candidate_edge(row: pd.Series) -> float:
    real_edge = safe_row_value(row, "real_edge", None)
    if real_edge is not None:
        return _safe_float(real_edge, 0.0)
    selected_edge = safe_row_value(row, "selected_edge", None)
    if selected_edge is not None:
        return _safe_float(selected_edge, 0.0)
    return _safe_float(safe_row_value(row, "edge_for_trade", 0.0), 0.0)


def _candidate_ask(row: pd.Series):
    selected_ask = safe_row_value(row, "selected_ask", None)
    return selected_ask if selected_ask is not None else safe_row_value(row, "ask_for_trade", None)


def _candidate_bid(row: pd.Series):
    selected_bid = safe_row_value(row, "selected_bid", None)
    return selected_bid if selected_bid is not None else safe_row_value(row, "bid_for_trade", None)


def _normalize_ranked_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    defaults = {
        "contract_ticker": "",
        "event_ticker": "",
        "series_ticker": "",
        "action": "NO_TRADE",
        "confidence": "LOW",
        "edge_yes": 0.0,
        "edge_no": 0.0,
        "ask_yes": np.nan,
        "ask_no": np.nan,
        "bid_yes": np.nan,
        "bid_no": np.nan,
        "decision_prob": np.nan,
        "fair_prob_terminal": np.nan,
        "fair_prob_blended": np.nan,
        "yes_no_ask_sum": np.nan,
        "overround": np.nan,
        "market_too_wide": False,
        "no_trade_reason": "",
        "decision_state": "",
        "entry_style": "",
        "trading_phase": "UNKNOWN",
        "target_yes_price": np.nan,
        "target_no_price": np.nan,
        "executable_yes_now": np.nan,
        "executable_no_now": np.nan,
    }

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    out["contract_ticker"] = out["contract_ticker"].astype(str).str.strip()
    out["event_ticker"] = out.get("event_ticker", "").astype(str).str.strip() if "event_ticker" in out.columns else out["contract_ticker"].apply(_extract_event_ticker_from_contract_ticker)
    out["series_ticker"] = out.get("series_ticker", "").astype(str).str.strip() if "series_ticker" in out.columns else ""
    out["action_norm"] = out["action"].astype(str).str.strip().str.upper()
    out["confidence_norm"] = out["confidence"].astype(str).str.strip().str.upper()
    out["decision_state"] = out["decision_state"].fillna("").astype(str).str.strip().str.upper()
    out["entry_style"] = out["entry_style"].fillna("").astype(str).str.strip()
    out["trading_phase"] = out["trading_phase"].fillna("UNKNOWN").astype(str).str.strip().str.upper()

    numeric_cols = [
        "edge_yes",
        "edge_no",
        "ask_yes",
        "ask_no",
        "bid_yes",
        "bid_no",
        "decision_prob",
        "fair_prob_terminal",
        "fair_prob_blended",
        "yes_no_ask_sum",
        "overround",
        "target_yes_price",
        "target_no_price",
        "distance_to_strike",
        "strike",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["trade_side"] = out.apply(_chosen_side, axis=1)
    out["trade_action"] = out["trade_side"].map(lambda x: f"BUY_{x}")
    out["edge_for_trade"] = out.apply(_chosen_edge, axis=1)
    out["ask_for_trade"] = out.apply(_chosen_ask, axis=1)
    out["bid_for_trade"] = out.apply(_chosen_bid, axis=1)

    out["selected_side"] = np.where(
        out["action_norm"] == "BUY_YES",
        "YES",
        np.where(out["action_norm"] == "BUY_NO", "NO", ""),
    )
    out["selected_edge"] = np.where(
        out["action_norm"] == "BUY_YES",
        out["edge_yes"],
        np.where(out["action_norm"] == "BUY_NO", out["edge_no"], np.nan),
    )
    out["selected_ask"] = np.where(
        out["selected_side"] == "YES",
        out["ask_yes"],
        np.where(out["selected_side"] == "NO", out["ask_no"], np.nan),
    )
    out["selected_bid"] = np.where(
        out["selected_side"] == "YES",
        out["bid_yes"],
        np.where(out["selected_side"] == "NO", out["bid_no"], np.nan),
    )
    out["distance_abs"] = pd.to_numeric(out.get("distance_to_strike"), errors="coerce").abs()
    out["strike"] = pd.to_numeric(out.get("strike"), errors="coerce")
    if out["strike"].isna().any():
        out.loc[out["strike"].isna(), "strike"] = out.loc[out["strike"].isna(), "contract_ticker"].apply(_extract_strike_from_ticker)
    return out


def _normalize_live_positions_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "ticker" not in out.columns:
        out["ticker"] = out.get("contract_ticker", "")
    if "side" not in out.columns:
        out["side"] = out.get("side_norm", "")
    if "contracts" not in out.columns:
        out["contracts"] = out.get("size", out.get("quantity", 0))
    if "allocation" not in out.columns:
        out["allocation"] = out.get("market_value", out.get("position_cost", 0.0))

    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["side"] = out["side"].astype(str).str.strip().str.upper()
    out["contracts"] = pd.to_numeric(out["contracts"], errors="coerce").fillna(0).astype(int)
    out["allocation"] = pd.to_numeric(out["allocation"], errors="coerce").fillna(0.0)
    out = out[out["ticker"] != ""].copy()
    out = out[out["contracts"] > 0].copy()

    if "event_ticker" not in out.columns:
        out["event_ticker"] = out["ticker"].apply(_extract_event_ticker_from_contract_ticker)
    else:
        out["event_ticker"] = out["event_ticker"].astype(str).str.strip()

    if "strike" not in out.columns:
        out["strike"] = pd.to_numeric(
            out["ticker"].apply(_extract_strike_from_ticker),
            errors="coerce",
        )
    else:
        out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
        mask = out["strike"].isna()
        if bool(mask.any()):
            extracted = out.loc[mask, "ticker"].apply(_extract_strike_from_ticker)
            extracted = pd.to_numeric(extracted, errors="coerce")
            out.loc[mask, "strike"] = extracted

    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")

    return out.reset_index(drop=True)


# =========================================================
# CONFIG HELPERS
# =========================================================
def _normalize_allowed_phases(value: Any, default: List[str]) -> List[str]:
    if value is None:
        value = default

    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = [safe_str(item) for item in value]
    else:
        raw_items = list(default)

    phases = [safe_upper(item) for item in raw_items if safe_str(item)]
    return phases or list(default)


def _get_capital_safety_config(config: Dict[str, Any]) -> Dict[str, float]:
    portfolio_cfg = config.get("portfolio", {}) or {}
    return {
        "max_capital_deploy": float(portfolio_cfg.get("max_capital_deploy", DEFAULT_MAX_CAPITAL_DEPLOY)),
        "max_dollars_per_trade": float(portfolio_cfg.get("max_dollars_per_trade", DEFAULT_MAX_DOLLARS_PER_TRADE)),
        "max_contracts_per_trade": int(portfolio_cfg.get("max_contracts_per_trade", DEFAULT_MAX_CONTRACTS_PER_TRADE)),
        "stop_trading_if_balance_below": float(portfolio_cfg.get("stop_trading_if_balance_below", DEFAULT_STOP_TRADING_IF_BALANCE_BELOW)),
    }


def _get_capped_cash_balance(raw_cash: float, config: Dict[str, Any]) -> float:
    safety_cfg = _get_capital_safety_config(config)
    return max(0.0, min(_safe_float(raw_cash, 0.0), safety_cfg["max_capital_deploy"]))


def _get_capped_total_capital(raw_cash: float, positions_val: float, config: Dict[str, Any]) -> float:
    safety_cfg = _get_capital_safety_config(config)
    capped_cash = min(_safe_float(raw_cash, 0.0), safety_cfg["max_capital_deploy"])
    capped_positions = max(0.0, _safe_float(positions_val, 0.0))
    return max(0.0, min(capped_cash + capped_positions, safety_cfg["max_capital_deploy"]))


def _get_effective_min_trade_dollars(capital: float, deployable_cash: float, config: Dict[str, Any]) -> float:
    portfolio_cfg = config.get("portfolio", {}) or {}
    base_min_trade = float(portfolio_cfg.get("min_trade_dollars", 0.0))
    small_account_threshold = float(portfolio_cfg.get("small_account_threshold", 25.0))
    small_account_min_trade = float(portfolio_cfg.get("small_account_min_trade_dollars", 1.0))

    if deployable_cash <= 0:
        return base_min_trade
    if _safe_float(capital, 0.0) <= small_account_threshold:
        baseline = base_min_trade if base_min_trade > 0 else deployable_cash
        return min(baseline, small_account_min_trade, deployable_cash)
    return base_min_trade


def _get_entry_filter_config(config: Dict[str, Any]) -> Dict[str, Any]:
    portfolio_cfg = config.get("portfolio", {}) or {}
    decision_cfg = config.get("decision", {}) or {}

    mid_min = float(portfolio_cfg.get("avoid_midzone_min_price", 0.35))
    mid_max = float(portfolio_cfg.get("avoid_midzone_max_price", 0.65))
    if mid_min > mid_max:
        mid_min, mid_max = mid_max, mid_min

    return {
        "min_edge_to_trade": float(
            portfolio_cfg.get(
                "min_edge_to_trade",
                decision_cfg.get("min_edge_to_trade", 0.06),
            )
        ),
        "min_edge_to_add": float(portfolio_cfg.get("min_edge_to_add", portfolio_cfg.get("min_edge_to_trade", decision_cfg.get("min_edge_to_trade", 0.06)))),
        "max_overround": float(portfolio_cfg.get("max_overround", 0.05)),
        "spread_penalty": float(portfolio_cfg.get("spread_penalty", 0.015)),
        "avoid_midzone_min_price": mid_min,
        "avoid_midzone_max_price": mid_max,
        "allow_midzone_trades": bool(portfolio_cfg.get("allow_midzone_trades", False)),
        "midzone_real_edge_override": float(portfolio_cfg.get("midzone_real_edge_override", 0.10)),
        "allowed_entry_trading_phases": _normalize_allowed_phases(
            portfolio_cfg.get("allowed_entry_trading_phases", DEFAULT_ALLOWED_ENTRY_TRADING_PHASES),
            DEFAULT_ALLOWED_ENTRY_TRADING_PHASES,
        ),
        "allowed_entry_trading_phases_when_flat": _normalize_allowed_phases(
            portfolio_cfg.get(
                "allowed_entry_trading_phases_when_flat",
                portfolio_cfg.get("allowed_entry_trading_phases", DEFAULT_ALLOWED_ENTRY_TRADING_PHASES_WHEN_FLAT),
            ),
            DEFAULT_ALLOWED_ENTRY_TRADING_PHASES_WHEN_FLAT,
        ),
    }


def _select_allowed_entry_phases(config: Dict[str, Any], has_live_positions: bool) -> List[str]:
    entry_filter_cfg = _get_entry_filter_config(config)
    if has_live_positions:
        return list(entry_filter_cfg["allowed_entry_trading_phases"])
    return list(entry_filter_cfg["allowed_entry_trading_phases_when_flat"])


def _compute_dynamic_reserve_cash_target(
    *,
    capital: float,
    available_cash: float,
    ranked_df: pd.DataFrame,
    live_df: pd.DataFrame,
    config: Dict[str, Any],
) -> float:
    portfolio_cfg = config.get("portfolio", {}) or {}
    capital = _safe_float(capital, 0.0)
    available_cash = _safe_float(available_cash, 0.0)
    if capital <= 0 or available_cash <= 0:
        return 0.0

    base_fraction = float(portfolio_cfg.get("reserve_cash_fraction", 0.20))
    min_fraction = float(portfolio_cfg.get("reserve_cash_fraction_min", 0.05))
    max_fraction = float(portfolio_cfg.get("reserve_cash_fraction_max", max(base_fraction, 0.20)))
    reserve_target = capital * base_fraction
    reserve_target = max(capital * min_fraction, reserve_target)
    reserve_target = min(capital * max_fraction, reserve_target, available_cash)
    return max(0.0, reserve_target)


# =========================================================
# SCORING / FILTERING
# =========================================================
def _confidence_sizing_factor(confidence: str) -> float:
    return {"HIGH": 1.00, "MEDIUM": 0.72, "LOW": 0.45}.get(safe_upper(confidence), 0.55)


def _edge_sizing_factor(edge: float, min_edge: float) -> float:
    edge = _safe_float(edge, 0.0)
    min_edge = max(_safe_float(min_edge, 0.03), 1e-6)
    target_edge = max(0.10, min_edge * 3.0)
    return max(0.30, min(1.35, edge / target_edge))


def _overround_sizing_factor(overround: Any, market_too_wide: Any) -> float:
    overround_value = _safe_float(overround, None)
    if overround_value is None:
        base = 1.0
    else:
        base = max(0.30, min(1.00, 1.0 - (max(overround_value, 0.0) / 0.20)))
    if bool(market_too_wide):
        base *= 0.70
    return max(0.25, min(1.00, base))


def _distance_sizing_factor(distance_abs: Any) -> float:
    distance = _safe_float(distance_abs, None)
    if distance is None:
        return 1.0
    if distance <= 3:
        return 0.70
    if distance <= 8:
        return 0.82
    if distance <= 15:
        return 0.92
    return 1.0


def _compute_real_edge(selected_edge: Any, spread_penalty: float) -> float:
    return _safe_float(selected_edge, 0.0) - max(0.0, _safe_float(spread_penalty, 0.0))


def _candidate_score(row: pd.Series, config: Dict[str, Any]) -> float:
    entry_cfg = _get_entry_filter_config(config)
    min_edge = float(entry_cfg["min_edge_to_trade"])

    edge = _candidate_edge(row)
    confidence = safe_row_value(row, "confidence_norm", safe_row_value(row, "confidence", "LOW"))
    overround = safe_row_value(row, "overround", None)
    market_too_wide = safe_row_value(row, "market_too_wide", False)
    distance_abs = safe_row_value(row, "distance_abs", safe_row_value(row, "distance_to_strike", None))

    score = edge
    score *= _confidence_sizing_factor(str(confidence))
    score *= _overround_sizing_factor(overround, market_too_wide)
    score *= _distance_sizing_factor(distance_abs)
    score *= _edge_sizing_factor(edge, min_edge)

    entry_style = safe_upper(safe_row_value(row, "entry_style", ""))
    if entry_style in {"LIMIT_MAKER", "WATCHLIST_LIMIT", "MAKER_TARGET"} and edge >= max(0.045, min_edge * 0.75):
        score += 0.01

    return float(score)


def _score_live_positions_against_ranked(live_df: pd.DataFrame, ranked_df: pd.DataFrame) -> pd.DataFrame:
    if live_df is None or live_df.empty:
        return pd.DataFrame()

    lookup = {}
    if ranked_df is not None and not ranked_df.empty:
        lookup = ranked_df.set_index("contract_ticker", drop=False).to_dict(orient="index")

    records: List[Dict[str, Any]] = []
    for _, pos in live_df.iterrows():
        ticker = safe_str(pos.get("ticker"))
        side = safe_upper(pos.get("side", pos.get("side_norm", "")))
        rec = dict(pos)
        ranked_row = lookup.get(ticker)

        rec["position_event_ticker"] = safe_str(pos.get("event_ticker")) or _extract_event_ticker_from_contract_ticker(ticker)
        rec["matched_ranked"] = ranked_row is not None
        rec["current_edge"] = 0.0
        rec["current_ask"] = None
        rec["current_bid"] = None
        rec["current_confidence"] = safe_upper(pos.get("confidence", "LOW")) or "LOW"

        if ranked_row is not None:
            if side == "YES":
                rec["current_edge"] = _safe_float(ranked_row.get("edge_yes"), 0.0)
                rec["current_ask"] = ranked_row.get("ask_yes")
                rec["current_bid"] = ranked_row.get("bid_yes")
            elif side == "NO":
                rec["current_edge"] = _safe_float(ranked_row.get("edge_no"), 0.0)
                rec["current_ask"] = ranked_row.get("ask_no")
                rec["current_bid"] = ranked_row.get("bid_no")
            rec["current_confidence"] = safe_upper(ranked_row.get("confidence_norm", ranked_row.get("confidence", "LOW")))
            for field in ACTION_METADATA_FIELDS:
                rec[field] = ranked_row.get(field)
        else:
            rec["event_ticker"] = rec.get("event_ticker") or rec["position_event_ticker"]
            rec["decision_state"] = rec.get("decision_state") or "HELD_ONLY"

        records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out = out.sort_values(by=["current_edge", "contracts"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    return out


def _log_tradable_stage(stage_name: str, df: pd.DataFrame, head_n: int = 5) -> None:
    logging.info("%s count: %s", stage_name, 0 if df is None else len(df))
    logging.info("%s preview: %s", stage_name, _safe_preview_records(df, PHASE_PREVIEW_COLUMNS, head_n=head_n))


def _apply_allowed_phase_filter(df: pd.DataFrame, allowed_phases: List[str], context_label: str) -> pd.DataFrame:
    if df is None or df.empty:
        logging.info("%s allowed phases: %s", context_label, allowed_phases)
        return pd.DataFrame() if df is None else df.copy()

    working = df.copy()
    if "trading_phase" not in working.columns:
        raise RuntimeError("CRITICAL: trading_phase missing before configurable phase filter")

    working["trading_phase"] = working["trading_phase"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    allowed = [safe_upper(x) for x in allowed_phases if safe_str(x)]
    logging.info("%s allowed phases: %s", context_label, allowed)
    _log_tradable_stage(f"{context_label} before phase filter", working)

    filtered = working.loc[working["trading_phase"].isin(allowed)].copy()
    _log_tradable_stage(f"{context_label} after phase filter", filtered)
    return filtered


# =========================================================
# POLICY / PORTFOLIO HELPERS
# =========================================================
def _single_position_mode_enabled(config: Dict[str, Any]) -> bool:
    portfolio_cfg = config.get("portfolio", {}) or {}
    allow_multiple_positions = bool(portfolio_cfg.get("allow_multiple_positions", False))
    strict_single_event_mode = bool(portfolio_cfg.get("strict_single_event_mode", True))
    max_open_trades = int(portfolio_cfg.get("max_open_trades", (config.get("decision", {}) or {}).get("max_open_trades", 1)))
    return strict_single_event_mode or (not allow_multiple_positions) or max_open_trades <= 1


def _allow_multiple_positions(config: Dict[str, Any]) -> bool:
    portfolio_cfg = config.get("portfolio", {}) or {}
    if _single_position_mode_enabled(config):
        return False
    return bool(portfolio_cfg.get("allow_multiple_positions", False))


def _get_max_open_trades(config: Dict[str, Any]) -> int:
    portfolio_cfg = config.get("portfolio", {}) or {}
    decision_cfg = config.get("decision", {}) or {}
    if _single_position_mode_enabled(config):
        return 1
    return max(1, int(portfolio_cfg.get("max_open_trades", decision_cfg.get("max_open_trades", 1))))


def _prefer_settlement_over_roll(config: Dict[str, Any]) -> bool:
    portfolio_hold_cfg = config.get("portfolio_hold", {}) or {}
    return bool(portfolio_hold_cfg.get("prefer_settlement_over_roll", False))


def _held_is_near_settlement(held_row: pd.Series, config: Dict[str, Any]) -> bool:
    portfolio_hold_cfg = config.get("portfolio_hold", {}) or {}
    hold_window_hours = float(portfolio_hold_cfg.get("hold_to_expiry_window_hours", 1.0))
    hours_left = _safe_float(
        safe_row_value(held_row, "hours_left", safe_row_value(held_row, "hours_to_expiry", np.nan)),
        np.nan,
    )
    if pd.notna(hours_left):
        return bool(hours_left <= hold_window_hours)
    return False


def _same_event_scope(row_a: Any, row_b: Any) -> bool:
    if row_a is None or row_b is None:
        return False
    a_event = safe_str(safe_row_value(row_a, "event_ticker", safe_row_value(row_a, "position_event_ticker", "")))
    b_event = safe_str(safe_row_value(row_b, "event_ticker", safe_row_value(row_b, "position_event_ticker", "")))
    if a_event and b_event:
        return a_event == b_event
    a_ticker = safe_str(safe_row_value(row_a, "ticker", safe_row_value(row_a, "contract_ticker", "")))
    b_ticker = safe_str(safe_row_value(row_b, "ticker", safe_row_value(row_b, "contract_ticker", "")))
    return _extract_event_ticker_from_contract_ticker(a_ticker) == _extract_event_ticker_from_contract_ticker(b_ticker)


def _held_is_out_of_scope(held_row: pd.Series, candidate_row: Optional[pd.Series]) -> bool:
    if held_row is None or candidate_row is None:
        return False
    return not _same_event_scope(held_row, candidate_row)


def _pick_best_candidate(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    return df.iloc[0]


def _pick_best_held(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    return df.iloc[0]


def _get_rotation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    portfolio_cfg = config.get("portfolio", {}) or {}
    rotation_cfg = config.get("portfolio_rotation", {}) or {}
    return {
        "disable_rotation_exits": bool(rotation_cfg.get("disable_rotation_exits", False)),
        "min_edge_improvement": float(rotation_cfg.get("min_edge_improvement", portfolio_cfg.get("min_edge_improvement_to_rotate", 0.08))),
        "held_min_edge_to_keep": float(rotation_cfg.get("held_min_edge_to_keep", 0.08)),
        "allow_rotation_same_event_only": bool(rotation_cfg.get("allow_rotation_same_event_only", False)),
        "max_rotations_per_cycle": int(rotation_cfg.get("max_rotations_per_cycle", 1)),
        "stale_held_edge_threshold": float(rotation_cfg.get("stale_held_edge_threshold", 0.06)),
        "stale_held_recovery_edge_threshold": float(rotation_cfg.get("stale_held_recovery_edge_threshold", 0.08)),
    }


def _should_hold_strict(held_row: pd.Series, candidate_row: Optional[pd.Series], config: Dict[str, Any]) -> bool:
    if held_row is None:
        return False

    if _prefer_settlement_over_roll(config) and _held_is_near_settlement(held_row, config):
        return True

    if candidate_row is None:
        return True

    if _prefer_settlement_over_roll(config) and _held_is_out_of_scope(held_row, candidate_row):
        return True

    held_decision_state = safe_upper(safe_row_value(held_row, "decision_state", ""))
    held_edge = _safe_float(safe_row_value(held_row, "current_edge", 0.0), 0.0)
    held_fair_prob = _safe_float(
        safe_row_value(
            held_row,
            "decision_prob",
            safe_row_value(held_row, "fair_prob_terminal", safe_row_value(held_row, "fair_prob_blended", np.nan)),
        ),
        np.nan,
    )
    held_ask = _safe_float(safe_row_value(held_row, "current_ask", None), np.nan)
    fair_gap = held_fair_prob - held_ask if pd.notna(held_fair_prob) and pd.notna(held_ask) else np.nan

    candidate_edge = _candidate_edge(candidate_row)
    edge_improvement = candidate_edge - held_edge

    hold_min_edge = float((config.get("portfolio_hold", {}) or {}).get("hold_min_edge", 0.15))
    keep_min_edge = float(_get_rotation_config(config)["held_min_edge_to_keep"])
    min_rotation_improvement = float(_get_rotation_config(config)["min_edge_improvement"])

    if held_decision_state == "ACTIONABLE" and held_edge >= hold_min_edge:
        return True
    if held_edge >= keep_min_edge and pd.notna(fair_gap) and fair_gap > 0 and edge_improvement < min_rotation_improvement:
        return True
    if candidate_edge <= held_edge:
        return True
    if _same_event_scope(held_row, candidate_row) and edge_improvement < min_rotation_improvement:
        return True
    return False


def _should_rotate_strict(held_row: pd.Series, candidate_row: Optional[pd.Series], config: Dict[str, Any]) -> bool:
    if held_row is None or candidate_row is None:
        return False

    if _prefer_settlement_over_roll(config) and _held_is_near_settlement(held_row, config):
        return False

    if _prefer_settlement_over_roll(config) and _held_is_out_of_scope(held_row, candidate_row):
        return False

    rotation_cfg = _get_rotation_config(config)
    if bool(rotation_cfg.get("disable_rotation_exits", False)):
        return False

    held_edge = _safe_float(safe_row_value(held_row, "current_edge", 0.0), 0.0)
    held_decision_state = safe_upper(safe_row_value(held_row, "decision_state", ""))
    candidate_edge = _candidate_edge(candidate_row)
    edge_improvement = candidate_edge - held_edge

    if rotation_cfg["allow_rotation_same_event_only"] and not _same_event_scope(held_row, candidate_row):
        return False

    if candidate_edge <= held_edge:
        return False

    if edge_improvement < float(rotation_cfg["min_edge_improvement"]):
        return False

    keep_min_edge = float(rotation_cfg["held_min_edge_to_keep"])
    stale_threshold = float(rotation_cfg["stale_held_edge_threshold"])

    if held_decision_state == "ACTIONABLE" and held_edge >= keep_min_edge and edge_improvement < max(float(rotation_cfg["min_edge_improvement"]), 0.10):
        return False

    if held_edge > stale_threshold and not _same_event_scope(held_row, candidate_row) and edge_improvement < max(float(rotation_cfg["min_edge_improvement"]), 0.12):
        return False

    return True


def _same_contract_side(held_row: Any, candidate_row: Any) -> bool:
    if held_row is None or candidate_row is None:
        return False
    held_ticker = safe_str(safe_row_value(held_row, "ticker", safe_row_value(held_row, "contract_ticker", "")))
    cand_ticker = safe_str(safe_row_value(candidate_row, "contract_ticker", safe_row_value(candidate_row, "ticker", "")))
    held_side = safe_upper(safe_row_value(held_row, "side", safe_row_value(held_row, "side_norm", "")))
    cand_side = _candidate_side(candidate_row)
    return held_ticker == cand_ticker and held_side == cand_side


def _position_is_protected(held_row: Any, config: Dict[str, Any]) -> bool:
    if held_row is None:
        return False

    portfolio_hold_cfg = config.get("portfolio_hold", {}) or {}
    explicit_protect = bool(safe_row_value(held_row, "protect_from_rotation", False))
    if explicit_protect:
        return True

    if not _prefer_settlement_over_roll(config):
        return False

    require_positive_pnl = bool(portfolio_hold_cfg.get("protect_only_if_profitable", True))
    min_pnl_to_protect = float(portfolio_hold_cfg.get("min_pnl_pct_to_protect", 0.08))
    pnl_pct = _safe_float(
        safe_row_value(held_row, "pnl_pct", safe_row_value(held_row, "unrealized_pnl_pct", np.nan)),
        np.nan,
    )
    if require_positive_pnl and (pd.isna(pnl_pct) or pnl_pct < min_pnl_to_protect):
        return False

    return _held_is_near_settlement(held_row, config)


def _position_event(held_row: Any) -> str:
    return safe_str(safe_row_value(held_row, "event_ticker", safe_row_value(held_row, "position_event_ticker", ""))) or _extract_event_ticker_from_contract_ticker(safe_str(safe_row_value(held_row, "ticker", "")))


def _position_side(held_row: Any) -> str:
    return safe_upper(safe_row_value(held_row, "side", safe_row_value(held_row, "side_norm", "")))


def _candidate_event(candidate_row: Any) -> str:
    return safe_str(safe_row_value(candidate_row, "event_ticker", "")) or _extract_event_ticker_from_contract_ticker(safe_str(safe_row_value(candidate_row, "contract_ticker", "")))


def _candidate_strike(candidate_row: Any) -> Optional[float]:
    strike = _safe_float(safe_row_value(candidate_row, "strike", None), np.nan)
    if pd.notna(strike):
        return float(strike)
    return _extract_strike_from_ticker(safe_str(safe_row_value(candidate_row, "contract_ticker", "")))


def _position_strike(held_row: Any) -> Optional[float]:
    strike = _safe_float(safe_row_value(held_row, "strike", None), np.nan)
    if pd.notna(strike):
        return float(strike)
    return _extract_strike_from_ticker(safe_str(safe_row_value(held_row, "ticker", "")))


def _held_like_from_candidate(row: Any) -> pd.Series:
    return pd.Series({
        "ticker": safe_row_value(row, "contract_ticker", safe_row_value(row, "ticker", "")),
        "event_ticker": _candidate_event(row),
        "side": _candidate_side(row),
        "strike": _candidate_strike(row),
    })


def _portfolio_conflict_for_candidate(
    candidate_row: pd.Series,
    existing_rows: List[pd.Series],
    config: Dict[str, Any],
) -> bool:
    if candidate_row is None:
        return True
    if not existing_rows:
        return False

    portfolio_cfg = config.get("portfolio", {}) or {}
    max_positions_per_event = int(portfolio_cfg.get("max_positions_per_event", 2))
    max_same_side_positions_per_event = int(portfolio_cfg.get("max_same_side_positions_per_event", 1))
    allow_same_event_same_side_add = bool(portfolio_cfg.get("allow_same_event_same_side_add", False))
    min_strike_gap_same_event = float(portfolio_cfg.get("min_strike_gap_same_event", 2.0))

    candidate_ticker = safe_str(safe_row_value(candidate_row, "contract_ticker", safe_row_value(candidate_row, "ticker", "")))
    candidate_event = _candidate_event(candidate_row)
    candidate_side = _candidate_side(candidate_row)
    candidate_strike = _candidate_strike(candidate_row)

    same_event_rows: List[pd.Series] = []
    same_event_same_side_count = 0

    for existing in existing_rows:
        existing_ticker = safe_str(safe_row_value(existing, "ticker", safe_row_value(existing, "contract_ticker", "")))
        if candidate_ticker and existing_ticker and candidate_ticker == existing_ticker:
            return True

        existing_event = _position_event(existing)
        existing_side = _position_side(existing)
        if candidate_event and existing_event and candidate_event == existing_event:
            same_event_rows.append(existing)
            if candidate_side and existing_side == candidate_side:
                same_event_same_side_count += 1

            existing_strike = _position_strike(existing)
            if (
                candidate_strike is not None
                and existing_strike is not None
                and abs(float(candidate_strike) - float(existing_strike)) < min_strike_gap_same_event
            ):
                return True

    if candidate_event and len(same_event_rows) >= max_positions_per_event:
        return True

    if candidate_side and same_event_same_side_count >= max_same_side_positions_per_event:
        return True

    if candidate_side and same_event_same_side_count >= 1 and not allow_same_event_same_side_add:
        return True

    return False


def _candidate_conflicts_with_planned(
    candidate_row: pd.Series,
    planned_entries: List[pd.Series],
    config: Dict[str, Any],
    held_rows: Optional[List[pd.Series]] = None,
) -> bool:
    existing_rows: List[pd.Series] = []
    if held_rows:
        existing_rows.extend(list(held_rows))
    if planned_entries:
        existing_rows.extend([_held_like_from_candidate(row) for row in planned_entries])
    return _portfolio_conflict_for_candidate(candidate_row, existing_rows, config)


def _get_recently_exited_tickers(config: Dict[str, Any], account_snapshot: Optional[Dict[str, Any]] = None) -> List[str]:
    account_snapshot = account_snapshot or {}
    raw = config.get("_recently_exited_tickers", account_snapshot.get("recently_exited_tickers", []))
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return [safe_str(item) for item in raw if safe_str(item)]


def _is_in_reentry_cooldown(candidate_row: pd.Series, recently_exited_tickers: List[str]) -> bool:
    if candidate_row is None:
        return False
    ticker = safe_str(safe_row_value(candidate_row, "contract_ticker", safe_row_value(candidate_row, "ticker", "")))
    if not ticker:
        return False
    return ticker in set(recently_exited_tickers or [])


def _filter_recently_exited_candidates(
    tradable_df: pd.DataFrame,
    recently_exited_tickers: List[str],
    context_label: str = "Portfolio candidates",
) -> pd.DataFrame:
    if tradable_df is None or tradable_df.empty:
        return pd.DataFrame() if tradable_df is None else tradable_df.copy()
    if not recently_exited_tickers:
        return tradable_df.copy()

    exited_set = {safe_str(t) for t in recently_exited_tickers if safe_str(t)}
    if not exited_set:
        return tradable_df.copy()

    working = tradable_df.copy()
    ticker_series = working["contract_ticker"].astype(str).str.strip() if "contract_ticker" in working.columns else pd.Series([""] * len(working), index=working.index)
    mask = ~ticker_series.isin(exited_set)
    removed = working.loc[~mask].copy()
    filtered = working.loc[mask].copy()

    if not removed.empty:
        logging.info(
            "%s skipped by re-entry cooldown | exited_tickers=%s | skipped=%s",
            context_label,
            sorted(exited_set),
            removed["contract_ticker"].astype(str).tolist() if "contract_ticker" in removed.columns else [],
        )

    return filtered


def _filter_candidates_for_existing_positions(tradable_df: pd.DataFrame, scored_live: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if tradable_df is None or tradable_df.empty:
        return pd.DataFrame() if tradable_df is None else tradable_df.copy()
    if scored_live is None or scored_live.empty:
        return tradable_df.copy()

    held_rows = [row for _, row in scored_live.iterrows()]
    keep_rows: List[int] = []
    for idx, row in tradable_df.iterrows():
        if not _portfolio_conflict_for_candidate(row, held_rows, config):
            keep_rows.append(idx)
    return tradable_df.loc[keep_rows].copy()


# =========================================================
# ENTRY INSTRUCTION BUILDING
# =========================================================
def build_new_entry_instruction(
    row: pd.Series,
    cash_budget: float,
    capital: float,
    config: Dict[str, Any],
    live_positions_df: Optional[pd.DataFrame] = None,
) -> Optional[TradeInstruction]:
    if row is None:
        return None

    side = _candidate_side(row)
    ask_price = _candidate_ask(row)
    bid_price = _candidate_bid(row)
    edge = _candidate_edge(row)
    confidence = safe_upper(safe_row_value(row, "confidence_norm", safe_row_value(row, "confidence", "LOW")))

    if side not in {"YES", "NO"}:
        return None

    ask_price = _safe_float(ask_price, None)
    if ask_price is None or ask_price <= 0 or ask_price > 1:
        return None

    safety_cfg = _get_capital_safety_config(config)
    max_fraction = float((config.get("portfolio", {}) or {}).get("max_single_trade_fraction", 0.25))
    min_edge = float((config.get("portfolio", {}) or {}).get("min_edge_to_trade", (config.get("decision", {}) or {}).get("min_edge_to_trade", 0.03)))

    cash_budget = min(cash_budget, safety_cfg["max_capital_deploy"], safety_cfg["max_dollars_per_trade"])
    if cash_budget < ask_price:
        return None

    base_allocation = min(cash_budget, capital * max_fraction)
    desired_allocation = base_allocation * _edge_sizing_factor(edge, min_edge) * _confidence_sizing_factor(confidence)
    desired_allocation = min(desired_allocation, cash_budget, safety_cfg["max_dollars_per_trade"])

    contracts = int(desired_allocation // ask_price)
    contracts = min(contracts, int(safety_cfg["max_contracts_per_trade"]))
    if contracts < 1:
        return None

    allocation = contracts * ask_price
    if allocation > cash_budget:
        return None

    return TradeInstruction(
        action="ENTER",
        ticker=safe_row_value(row, "contract_ticker", None),
        side=side,
        contracts=contracts,
        allocation=allocation,
        ask_price=ask_price,
        bid_price=bid_price,
        edge=edge,
        confidence=confidence,
        reason="STANDARD_ENTRY",
    )


# =========================================================
# PLAN BUILDERS
# =========================================================
def _build_hold_plan(
    held_row: pd.Series,
    capital: float,
    available_cash: float,
    reserve_cash_target: float,
    deployable_cash: float,
    reason: str,
    tradable_candidates_count: int = 0,
    watchlist_candidates_count: int = 0,
) -> PortfolioPlan:
    action_payload = build_portfolio_action(
        action="HOLD",
        ticker=safe_row_value(held_row, "ticker", None),
        side=safe_row_value(held_row, "side", None),
        contracts=_safe_int(safe_row_value(held_row, "contracts", 0), 0),
        allocation=_safe_float(safe_row_value(held_row, "allocation", 0.0), 0.0),
        source_row=held_row,
        ask_price=safe_row_value(held_row, "current_ask", None),
        bid_price=safe_row_value(held_row, "current_bid", None),
        edge=_safe_float(safe_row_value(held_row, "current_edge", 0.0), 0.0),
        confidence=safe_row_value(held_row, "current_confidence", None),
        reason=reason,
    )
    return PortfolioPlan(
        recommendation="HOLD",
        reason=reason,
        capital=capital,
        available_cash=available_cash,
        reserve_cash_target=reserve_cash_target,
        deployable_cash=deployable_cash,
        tradable_candidates_count=tradable_candidates_count,
        watchlist_candidates_count=watchlist_candidates_count,
        actions=[action_payload],
    )


def _build_exit_plan(
    held_row: pd.Series,
    capital: float,
    available_cash: float,
    reserve_cash_target: float,
    deployable_cash: float,
    reason: str,
    tradable_candidates_count: int = 0,
    watchlist_candidates_count: int = 0,
) -> PortfolioPlan:
    action_payload = build_portfolio_action(
        action="EXIT",
        ticker=safe_row_value(held_row, "ticker", None),
        side=safe_row_value(held_row, "side", None),
        contracts=_safe_int(safe_row_value(held_row, "contracts", 0), 0),
        allocation=_safe_float(safe_row_value(held_row, "allocation", 0.0), 0.0),
        source_row=held_row,
        ask_price=safe_row_value(held_row, "current_ask", None),
        bid_price=safe_row_value(held_row, "current_bid", None),
        edge=_safe_float(safe_row_value(held_row, "current_edge", 0.0), 0.0),
        confidence=safe_row_value(held_row, "current_confidence", None),
        reason=reason,
    )
    return PortfolioPlan(
        recommendation="EXIT",
        reason=reason,
        capital=capital,
        available_cash=available_cash,
        reserve_cash_target=reserve_cash_target,
        deployable_cash=deployable_cash,
        tradable_candidates_count=tradable_candidates_count,
        watchlist_candidates_count=watchlist_candidates_count,
        actions=[action_payload],
    )


def _build_enter_plan(
    entry_instr: TradeInstruction,
    entry_row: Any,
    capital: float,
    available_cash: float,
    reserve_cash_target: float,
    deployable_cash: float,
    reason: str,
    tradable_candidates_count: int = 0,
    watchlist_candidates_count: int = 0,
) -> PortfolioPlan:
    action_payload = build_portfolio_action(
        action=entry_instr.action,
        ticker=entry_instr.ticker,
        side=entry_instr.side,
        contracts=entry_instr.contracts,
        allocation=entry_instr.allocation,
        source_row=entry_row,
        ask_price=entry_instr.ask_price,
        bid_price=entry_instr.bid_price,
        edge=entry_instr.edge,
        confidence=entry_instr.confidence,
        reason=reason,
    )
    return PortfolioPlan(
        recommendation="ENTER",
        reason=reason,
        capital=capital,
        available_cash=available_cash,
        reserve_cash_target=reserve_cash_target,
        deployable_cash=deployable_cash,
        tradable_candidates_count=tradable_candidates_count,
        watchlist_candidates_count=watchlist_candidates_count,
        actions=[action_payload],
    )


def _build_rotate_plan(
    held_row: pd.Series,
    entry_instr: TradeInstruction,
    entry_row: Any,
    capital: float,
    available_cash: float,
    reserve_cash_target: float,
    deployable_cash: float,
    reason: str,
    tradable_candidates_count: int = 0,
    watchlist_candidates_count: int = 0,
) -> PortfolioPlan:
    exit_payload = build_portfolio_action(
        action="EXIT",
        ticker=safe_row_value(held_row, "ticker", None),
        side=safe_row_value(held_row, "side", None),
        contracts=_safe_int(safe_row_value(held_row, "contracts", 0), 0),
        allocation=_safe_float(safe_row_value(held_row, "allocation", 0.0), 0.0),
        source_row=held_row,
        ask_price=safe_row_value(held_row, "current_ask", None),
        bid_price=safe_row_value(held_row, "current_bid", None),
        edge=_safe_float(safe_row_value(held_row, "current_edge", 0.0), 0.0),
        confidence=safe_row_value(held_row, "current_confidence", None),
        reason="Rotate out of weaker held position",
    )
    enter_payload = build_portfolio_action(
        action=entry_instr.action,
        ticker=entry_instr.ticker,
        side=entry_instr.side,
        contracts=entry_instr.contracts,
        allocation=entry_instr.allocation,
        source_row=entry_row,
        ask_price=entry_instr.ask_price,
        bid_price=entry_instr.bid_price,
        edge=entry_instr.edge,
        confidence=entry_instr.confidence,
        reason="Rotate into stronger opportunity",
    )
    return PortfolioPlan(
        recommendation="ROTATE",
        reason=reason,
        capital=capital,
        available_cash=available_cash,
        reserve_cash_target=reserve_cash_target,
        deployable_cash=deployable_cash,
        tradable_candidates_count=tradable_candidates_count,
        watchlist_candidates_count=watchlist_candidates_count,
        actions=[exit_payload, enter_payload],
    )


def _build_multi_action_plan(
    recommendation: str,
    reason: str,
    actions: List[Dict[str, Any]],
    capital: float,
    available_cash: float,
    reserve_cash_target: float,
    deployable_cash: float,
    tradable_candidates_count: int = 0,
    watchlist_candidates_count: int = 0,
) -> PortfolioPlan:
    return PortfolioPlan(
        recommendation=recommendation,
        reason=reason,
        capital=capital,
        available_cash=available_cash,
        reserve_cash_target=reserve_cash_target,
        deployable_cash=deployable_cash,
        tradable_candidates_count=tradable_candidates_count,
        watchlist_candidates_count=watchlist_candidates_count,
        actions=actions,
    )


# =========================================================
# MULTI-POSITION PLANNING HELPERS
# =========================================================
def _eligible_tradable_candidates(ranked: pd.DataFrame, config: Dict[str, Any], has_live_positions: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ranked is None or ranked.empty:
        return pd.DataFrame(), pd.DataFrame()

    working = ranked.copy()
    allowed_phases = _select_allowed_entry_phases(config, has_live_positions=has_live_positions)
    working = _apply_allowed_phase_filter(working, allowed_phases, "Portfolio candidates")
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()

    watchlist_df = working[working["decision_state"].isin(["WAIT_FOR_PRICE", "PRICE_OK_BUT_SPREAD_TOO_WIDE"])].copy()

    entry_cfg = _get_entry_filter_config(config)
    spread_penalty = float(entry_cfg["spread_penalty"])
    working["real_edge"] = working.apply(lambda r: _compute_real_edge(_candidate_edge(r), spread_penalty), axis=1)
    working["candidate_score"] = working.apply(lambda row: _candidate_score(row, config), axis=1)
    working["confidence_rank"] = working["confidence_norm"].map({"HIGH": 3, "MEDIUM": 2, "LOW": 1}).fillna(0)

    tradable_df = working[
        working["action_norm"].isin(["BUY_YES", "BUY_NO"])
        & (working["decision_state"] == "ACTIONABLE")
        & (working["real_edge"] >= max(0.12, float(entry_cfg["min_edge_to_add"])))
    ].copy()

    # NEW: STRICT MARKET FILTER
    tradable_df = tradable_df[
        tradable_df["market_too_wide"] == False
    ].copy()

    # NEW: STRICT OVERROUND FILTER
    tradable_df = tradable_df[
        (tradable_df["overround"].isna()) | (tradable_df["overround"] <= 0.05)
    ].copy()

    tradable_df = tradable_df.sort_values(
        by=["candidate_score", "confidence_rank", "real_edge", "selected_edge", "distance_abs"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return tradable_df, watchlist_df


def _build_hold_payload(held_row: pd.Series, reason: str) -> Dict[str, Any]:
    return build_portfolio_action(
        action="HOLD",
        ticker=safe_row_value(held_row, "ticker", None),
        side=safe_row_value(held_row, "side", None),
        contracts=_safe_int(safe_row_value(held_row, "contracts", 0), 0),
        allocation=_safe_float(safe_row_value(held_row, "allocation", 0.0), 0.0),
        source_row=held_row,
        ask_price=safe_row_value(held_row, "current_ask", None),
        bid_price=safe_row_value(held_row, "current_bid", None),
        edge=_safe_float(safe_row_value(held_row, "current_edge", 0.0), 0.0),
        confidence=safe_row_value(held_row, "current_confidence", None),
        reason=reason,
    )


def _build_exit_payload(held_row: pd.Series, reason: str) -> Dict[str, Any]:
    return build_portfolio_action(
        action="EXIT",
        ticker=safe_row_value(held_row, "ticker", None),
        side=safe_row_value(held_row, "side", None),
        contracts=_safe_int(safe_row_value(held_row, "contracts", 0), 0),
        allocation=_safe_float(safe_row_value(held_row, "allocation", 0.0), 0.0),
        source_row=held_row,
        ask_price=safe_row_value(held_row, "current_ask", None),
        bid_price=safe_row_value(held_row, "current_bid", None),
        edge=_safe_float(safe_row_value(held_row, "current_edge", 0.0), 0.0),
        confidence=safe_row_value(held_row, "current_confidence", None),
        reason=reason,
    )


def _build_enter_payload(entry_instr: TradeInstruction, entry_row: pd.Series, reason: str) -> Dict[str, Any]:
    return build_portfolio_action(
        action=entry_instr.action,
        ticker=entry_instr.ticker,
        side=entry_instr.side,
        contracts=entry_instr.contracts,
        allocation=entry_instr.allocation,
        source_row=entry_row,
        ask_price=entry_instr.ask_price,
        bid_price=entry_instr.bid_price,
        edge=entry_instr.edge,
        confidence=entry_instr.confidence,
        reason=reason,
    )


def _select_rotation_pair(scored_live: pd.DataFrame, tradable_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    if scored_live is None or scored_live.empty or tradable_df is None or tradable_df.empty:
        return None, None

    held_candidates = scored_live.sort_values(by=["current_edge", "contracts"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    tradable_sorted = tradable_df.sort_values(by=["candidate_score", "real_edge", "selected_edge"], ascending=[False, False, False], kind="mergesort").reset_index(drop=True)

    for _, held_row in held_candidates.iterrows():
        if _position_is_protected(held_row, config):
            continue
        for _, candidate_row in tradable_sorted.iterrows():
            if _same_contract_side(held_row, candidate_row):
                continue
            if _should_rotate_strict(held_row, candidate_row, config):
                return held_row, candidate_row
    return None, None


def _build_multi_position_plan(
    ranked: pd.DataFrame,
    scored_live: pd.DataFrame,
    capital: float,
    available_cash: float,
    reserve_cash_target: float,
    deployable_cash: float,
    cash_blocked: bool,
    config: Dict[str, Any],
    account_snapshot: Optional[Dict[str, Any]] = None,
) -> PortfolioPlan:
    tradable_df, watchlist_df = _eligible_tradable_candidates(ranked, config, has_live_positions=not scored_live.empty)
    recently_exited_tickers = _get_recently_exited_tickers(config, account_snapshot)
    if recently_exited_tickers:
        logging.info("Portfolio re-entry cooldown active | recently_exited_tickers=%s", sorted(set(recently_exited_tickers)))
    tradable_df = _filter_recently_exited_candidates(tradable_df, recently_exited_tickers, context_label="Portfolio candidates")
    tradable_candidates_count = len(tradable_df)
    watchlist_candidates_count = len(watchlist_df)

    logging.info("Top filtered tradable trades: %s", _safe_preview_records(tradable_df, PHASE_PREVIEW_COLUMNS, head_n=5))

    if tradable_df.empty and scored_live.empty:
        return PortfolioPlan(
            recommendation="WAIT",
            reason=_append_watchlist_context("No tradable candidates.", watchlist_candidates_count),
            capital=capital,
            available_cash=available_cash,
            reserve_cash_target=reserve_cash_target,
            deployable_cash=deployable_cash,
            tradable_candidates_count=0,
            watchlist_candidates_count=watchlist_candidates_count,
            actions=[],
        )

    if not cash_blocked:
        tradable_df = _filter_candidates_for_existing_positions(tradable_df, scored_live, config)

    max_open_trades = _get_max_open_trades(config)
    actions: List[Dict[str, Any]] = []
    planned_entry_rows: List[pd.Series] = []
    deployable_remaining = float(deployable_cash)

    held_rows = [] if scored_live is None or scored_live.empty else [row for _, row in scored_live.iterrows()]
    protected_held = [row for row in held_rows if _position_is_protected(row, config)]
    unprotected_held = [row for row in held_rows if not _position_is_protected(row, config)]

    open_count = len(held_rows)
    protected_count = len(protected_held)

    if open_count >= max_open_trades and not tradable_df.empty:
        held_to_rotate, candidate_to_add = _select_rotation_pair(scored_live, tradable_df, config)
        if held_to_rotate is not None and candidate_to_add is not None:
            estimated_cash_for_rotation = deployable_remaining + max(0.0, _safe_float(safe_row_value(held_to_rotate, "allocation", 0.0), 0.0))
            entry_instr = build_new_entry_instruction(candidate_to_add, estimated_cash_for_rotation, capital, config, live_positions_df=scored_live)
            if entry_instr is not None:
                return _build_rotate_plan(
                    held_row=held_to_rotate,
                    entry_instr=entry_instr,
                    entry_row=candidate_to_add,
                    capital=capital,
                    available_cash=available_cash,
                    reserve_cash_target=reserve_cash_target,
                    deployable_cash=deployable_cash,
                    reason=_append_watchlist_context("Portfolio full; rotating weakest non-protected holding into materially stronger candidate.", watchlist_candidates_count),
                    tradable_candidates_count=tradable_candidates_count,
                    watchlist_candidates_count=watchlist_candidates_count,
                )

    for row in protected_held + unprotected_held:
        hold_reason = "Existing position remains active."
        if _position_is_protected(row, config):
            hold_reason = "Protected high-conviction near-settlement position."
        actions.append(_build_hold_payload(row, hold_reason))

    if cash_blocked:
        if actions:
            return _build_multi_action_plan(
                recommendation="HOLD",
                reason=_append_watchlist_context("Holding current positions; no deployable cash available for adds.", watchlist_candidates_count),
                actions=actions,
                capital=capital,
                available_cash=available_cash,
                reserve_cash_target=reserve_cash_target,
                deployable_cash=deployable_cash,
                tradable_candidates_count=tradable_candidates_count,
                watchlist_candidates_count=watchlist_candidates_count,
            )
        return PortfolioPlan(
            recommendation="WAIT",
            reason=_append_watchlist_context("Actionable trades found, but no deployable cash available.", watchlist_candidates_count),
            capital=capital,
            available_cash=available_cash,
            reserve_cash_target=reserve_cash_target,
            deployable_cash=deployable_cash,
            tradable_candidates_count=tradable_candidates_count,
            watchlist_candidates_count=watchlist_candidates_count,
            actions=[],
        )

    slots_remaining = max(0, max_open_trades - open_count)
    if slots_remaining <= 0:
        if actions:
            return _build_multi_action_plan(
                recommendation="HOLD",
                reason=_append_watchlist_context("Max open trade slots already filled; holding current portfolio.", watchlist_candidates_count),
                actions=actions,
                capital=capital,
                available_cash=available_cash,
                reserve_cash_target=reserve_cash_target,
                deployable_cash=deployable_cash,
                tradable_candidates_count=tradable_candidates_count,
                watchlist_candidates_count=watchlist_candidates_count,
            )
        return PortfolioPlan(
            recommendation="WAIT",
            reason=_append_watchlist_context("Max open trade slots already filled.", watchlist_candidates_count),
            capital=capital,
            available_cash=available_cash,
            reserve_cash_target=reserve_cash_target,
            deployable_cash=deployable_cash,
            tradable_candidates_count=tradable_candidates_count,
            watchlist_candidates_count=watchlist_candidates_count,
            actions=[],
        )

    for _, candidate_row in tradable_df.iterrows():
        if _is_in_reentry_cooldown(candidate_row, recently_exited_tickers):
            logging.info(
                "Portfolio candidate skipped inside multi-position loop due to cooldown | ticker=%s",
                safe_str(safe_row_value(candidate_row, "contract_ticker", safe_row_value(candidate_row, "ticker", ""))),
            )
            continue
        if slots_remaining <= 0:
            break
        if _candidate_conflicts_with_planned(candidate_row, planned_entry_rows, config, held_rows=held_rows):
            continue
        entry_instr = build_new_entry_instruction(candidate_row, deployable_remaining, capital, config, live_positions_df=scored_live)
        if entry_instr is None:
            continue
        planned_entry_rows.append(candidate_row)
        actions.append(_build_enter_payload(entry_instr, candidate_row, "Add high-ranked non-conflicting portfolio position."))
        deployable_remaining -= float(entry_instr.allocation)
        slots_remaining -= 1
        if deployable_remaining <= 0:
            break

    enter_count = sum(1 for a in actions if safe_upper(a.get("action")) == "ENTER")
    hold_count = sum(1 for a in actions if safe_upper(a.get("action")) == "HOLD")

    if enter_count > 0 and hold_count > 0:
        recommendation = "ENTER"
        reason = _append_watchlist_context("Holding current positions and adding best non-conflicting opportunities.", watchlist_candidates_count)
    elif enter_count > 0:
        recommendation = "ENTER"
        reason = _append_watchlist_context("Building multi-position portfolio from top ranked opportunities.", watchlist_candidates_count)
    elif hold_count > 0:
        recommendation = "HOLD"
        reason = _append_watchlist_context("No eligible additions after portfolio conflict checks; holding current positions.", watchlist_candidates_count)
    else:
        recommendation = "WAIT"
        reason = _append_watchlist_context("Tradable candidates exist, but none passed portfolio conflict and sizing rules.", watchlist_candidates_count)

    return _build_multi_action_plan(
        recommendation=recommendation,
        reason=reason,
        actions=actions,
        capital=capital,
        available_cash=available_cash,
        reserve_cash_target=reserve_cash_target,
        deployable_cash=deployable_cash,
        tradable_candidates_count=tradable_candidates_count,
        watchlist_candidates_count=watchlist_candidates_count,
    )


# =========================================================
# MAIN DECISION FUNCTION
# =========================================================
def build_portfolio_decision_plan(ranked_df, live_positions_df, config, account_snapshot=None):
    account_snapshot = account_snapshot or {}
    portfolio_cfg = config.get("portfolio", {}) or {}

    starting_capital = _safe_float(portfolio_cfg.get("starting_capital"), 10.0)

    ranked = _normalize_ranked_df(ranked_df)
    live = _normalize_live_positions_df(live_positions_df)

    raw_cash = _safe_float(account_snapshot.get("cash_balance"), starting_capital)
    raw_positions_val = _safe_float(account_snapshot.get("positions_value"), 0.0)

    capital = _get_capped_total_capital(raw_cash, raw_positions_val, config)
    available_cash = _get_capped_cash_balance(raw_cash, config)
    reserve_cash_target = _compute_dynamic_reserve_cash_target(
        capital=capital,
        available_cash=available_cash,
        ranked_df=ranked,
        live_df=live,
        config=config,
    )
    deployable_cash = max(0.0, available_cash - reserve_cash_target)
    effective_min_trade_dollars = _get_effective_min_trade_dollars(capital, deployable_cash, config)

    safety_cfg = _get_capital_safety_config(config)
    below_balance_floor = available_cash < float(safety_cfg["stop_trading_if_balance_below"])
    cash_blocked = (deployable_cash <= 0) or (effective_min_trade_dollars > 0 and deployable_cash < effective_min_trade_dollars) or below_balance_floor

    if ranked.empty:
        return PortfolioPlan(
            recommendation="WAIT",
            reason="No ranked data",
            capital=capital,
            available_cash=available_cash,
            reserve_cash_target=reserve_cash_target,
            deployable_cash=deployable_cash,
            tradable_candidates_count=0,
            watchlist_candidates_count=0,
            actions=[],
        )

    logging.info("Portfolio raw candidate count: %s", len(ranked))
    logging.info("Portfolio raw live positions count: %s", len(live))
    logging.info(
        "Portfolio cash state | capital=%.2f | available_cash=%.2f | reserve_cash_target=%.2f | deployable_cash=%.2f | min_trade_dollars=%.2f | cash_blocked=%s",
        capital,
        available_cash,
        reserve_cash_target,
        deployable_cash,
        effective_min_trade_dollars,
        cash_blocked,
    )

    scored_live = _score_live_positions_against_ranked(live, ranked)
    if not scored_live.empty:
        logging.info(
            "Portfolio scored live positions: %s",
            _safe_preview_records(
                scored_live,
                [
                    "ticker",
                    "side",
                    "contracts",
                    "current_edge",
                    "decision_state",
                    "current_confidence",
                    "event_ticker",
                    "position_event_ticker",
                ],
                head_n=5,
            ),
        )

    tradable_df, watchlist_df = _eligible_tradable_candidates(ranked, config, has_live_positions=not scored_live.empty)
    recently_exited_tickers = _get_recently_exited_tickers(config, account_snapshot)
    if recently_exited_tickers:
        logging.info("Portfolio re-entry cooldown active | recently_exited_tickers=%s", sorted(set(recently_exited_tickers)))
    tradable_df = _filter_recently_exited_candidates(tradable_df, recently_exited_tickers, context_label="Portfolio candidates")
    tradable_candidates_count = len(tradable_df)
    watchlist_candidates_count = len(watchlist_df)

    if _allow_multiple_positions(config):
        return _build_multi_position_plan(
            ranked=ranked,
            scored_live=scored_live,
            capital=capital,
            available_cash=available_cash,
            reserve_cash_target=reserve_cash_target,
            deployable_cash=deployable_cash,
            cash_blocked=cash_blocked,
            config=config,
            account_snapshot=account_snapshot,
        )

    if tradable_df.empty:
        best_held = _pick_best_held(scored_live)
        if _single_position_mode_enabled(config) and best_held is not None:
            hold_reason = "No new tradable candidates, but an existing held position remains active."
            if _position_is_protected(best_held, config):
                hold_reason = "No new tradable candidates, and the held position is explicitly protected near settlement."
            return _build_hold_plan(
                held_row=best_held,
                capital=capital,
                available_cash=available_cash,
                reserve_cash_target=reserve_cash_target,
                deployable_cash=deployable_cash,
                reason=_append_watchlist_context(hold_reason, watchlist_candidates_count),
                tradable_candidates_count=0,
                watchlist_candidates_count=watchlist_candidates_count,
            )

        return PortfolioPlan(
            recommendation="WAIT",
            reason=_append_watchlist_context("No tradable candidates.", watchlist_candidates_count),
            capital=capital,
            available_cash=available_cash,
            reserve_cash_target=reserve_cash_target,
            deployable_cash=deployable_cash,
            tradable_candidates_count=0,
            watchlist_candidates_count=watchlist_candidates_count,
            actions=[],
        )

    logging.info("Top filtered tradable trades: %s", _safe_preview_records(tradable_df, PHASE_PREVIEW_COLUMNS, head_n=5))
    best_candidate = _pick_best_candidate(tradable_df)
    best_held = _pick_best_held(scored_live)

    if _single_position_mode_enabled(config) and best_held is not None:
        logging.info(
            "STRICT MODE active | held_ticker=%s | held_side=%s | held_edge=%.4f",
            safe_row_value(best_held, "ticker", None),
            safe_row_value(best_held, "side", None),
            _safe_float(safe_row_value(best_held, "current_edge", 0.0), 0.0),
        )

        if _should_hold_strict(best_held, best_candidate, config):
            hold_reason = "Held position remains competitive versus current opportunities."
            if _position_is_protected(best_held, config):
                hold_reason = "Held position is explicitly protected near settlement."
            elif _prefer_settlement_over_roll(config) and _held_is_out_of_scope(best_held, best_candidate):
                hold_reason = "Held position is outside the current event scope, and settlement preference is enabled."
            return _build_hold_plan(
                held_row=best_held,
                capital=capital,
                available_cash=available_cash,
                reserve_cash_target=reserve_cash_target,
                deployable_cash=deployable_cash,
                reason=_append_watchlist_context(hold_reason, watchlist_candidates_count),
                tradable_candidates_count=tradable_candidates_count,
                watchlist_candidates_count=watchlist_candidates_count,
            )

        if _should_rotate_strict(best_held, best_candidate, config):
            estimated_cash_for_rotation = deployable_cash + max(0.0, _safe_float(safe_row_value(best_held, "allocation", 0.0), 0.0))
            entry_instr = build_new_entry_instruction(best_candidate, estimated_cash_for_rotation, capital, config, live_positions_df=scored_live)
            if entry_instr:
                return _build_rotate_plan(
                    held_row=best_held,
                    entry_instr=entry_instr,
                    entry_row=best_candidate,
                    capital=capital,
                    available_cash=available_cash,
                    reserve_cash_target=reserve_cash_target,
                    deployable_cash=deployable_cash,
                    reason=_append_watchlist_context("Held trade weakened and materially better replacement exists.", watchlist_candidates_count),
                    tradable_candidates_count=tradable_candidates_count,
                    watchlist_candidates_count=watchlist_candidates_count,
                )

        return _build_hold_plan(
            held_row=best_held,
            capital=capital,
            available_cash=available_cash,
            reserve_cash_target=reserve_cash_target,
            deployable_cash=deployable_cash,
            reason=_append_watchlist_context("Strict single-position mode blocked additional entries.", watchlist_candidates_count),
            tradable_candidates_count=tradable_candidates_count,
            watchlist_candidates_count=watchlist_candidates_count,
        )

    if cash_blocked:
        return PortfolioPlan(
            recommendation="WAIT",
            reason=_append_watchlist_context("Actionable trades found, but no deployable cash available.", watchlist_candidates_count),
            capital=capital,
            available_cash=available_cash,
            reserve_cash_target=reserve_cash_target,
            deployable_cash=deployable_cash,
            tradable_candidates_count=tradable_candidates_count,
            watchlist_candidates_count=watchlist_candidates_count,
            actions=[],
        )

    entry_instr = build_new_entry_instruction(best_candidate, deployable_cash, capital, config, live_positions_df=live)
    if entry_instr:
        return _build_enter_plan(
            entry_instr=entry_instr,
            entry_row=best_candidate,
            capital=capital,
            available_cash=available_cash,
            reserve_cash_target=reserve_cash_target,
            deployable_cash=deployable_cash,
            reason=_append_watchlist_context("Top ranked single opportunity.", watchlist_candidates_count),
            tradable_candidates_count=tradable_candidates_count,
            watchlist_candidates_count=watchlist_candidates_count,
        )

    return PortfolioPlan(
        recommendation="WAIT",
        reason=_append_watchlist_context("Tradable candidates exist, but none passed sizing constraints.", watchlist_candidates_count),
        capital=capital,
        available_cash=available_cash,
        reserve_cash_target=reserve_cash_target,
        deployable_cash=deployable_cash,
        tradable_candidates_count=tradable_candidates_count,
        watchlist_candidates_count=watchlist_candidates_count,
        actions=[],
    )


# =========================================================
# BACKWARD COMPAT
# =========================================================
def build_micro_allocation_plan(ranked_df, live_positions_df, config, account_snapshot=None):
    return build_portfolio_decision_plan(
        ranked_df,
        live_positions_df,
        config,
        account_snapshot,
    ).to_dict()
