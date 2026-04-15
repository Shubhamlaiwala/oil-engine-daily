from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
]

DEFAULT_MAX_CAPITAL_DEPLOY = 10.0
DEFAULT_MAX_DOLLARS_PER_TRADE = 3.0
DEFAULT_MAX_CONTRACTS_PER_TRADE = 1
DEFAULT_STOP_TRADING_IF_BALANCE_BELOW = 1.0
DEFAULT_ALLOWED_ENTRY_TRADING_PHASES = ["ACTIVE_TRADING"]
DEFAULT_ALLOWED_ENTRY_TRADING_PHASES_WHEN_FLAT = ["ACTIVE_TRADING"]
PHASE_PREVIEW_COLUMNS = [
    "contract_ticker",
    "decision_state",
    "trading_phase",
    "selected_edge",
    "real_edge",
    "selected_ask",
    "overround",
    "market_too_wide",
    "entry_style",
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



def _is_tradeable_price(value: Any) -> bool:
    try:
        return value is not None and 0 < float(value) <= 1
    except Exception:
        return False



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
    out["strike"] = out["contract_ticker"].apply(_extract_strike_from_ticker)
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
    out["strike"] = out["ticker"].apply(_extract_strike_from_ticker)
    return out.reset_index(drop=True)


# =========================================================
# CONFIG HELPERS
# =========================================================
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



def _select_allowed_entry_phases(config: Dict[str, Any], has_live_positions: bool) -> List[str]:
    entry_filter_cfg = _get_entry_filter_config(config)
    if has_live_positions:
        return list(entry_filter_cfg["allowed_entry_trading_phases"])
    return list(entry_filter_cfg["allowed_entry_trading_phases_when_flat"])



def _compute_real_edge(selected_edge: Any, spread_penalty: float) -> float:
    return _safe_float(selected_edge, 0.0) - max(0.0, _safe_float(spread_penalty, 0.0))



def _is_cash_blocked(deployable_cash: float, min_trade_dollars: float) -> bool:
    if deployable_cash <= 0:
        return True
    if min_trade_dollars > 0 and deployable_cash < min_trade_dollars:
        return True
    return False



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



def _candidate_score(row: pd.Series, config: Dict[str, Any]) -> float:
    decision_cfg = config.get("decision", {}) or {}
    portfolio_cfg = config.get("portfolio", {}) or {}
    min_edge = float(portfolio_cfg.get("min_edge_to_trade", decision_cfg.get("min_edge_to_trade", 0.03)))

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

        rec["position_event_ticker"] = _extract_event_ticker_from_contract_ticker(ticker)
        rec["matched_ranked"] = ranked_row is not None
        rec["current_edge"] = 0.0
        rec["current_ask"] = None
        rec["current_bid"] = None
        rec["current_confidence"] = "LOW"

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
# SINGLE-POSITION RULES
# =========================================================
def _single_position_mode_enabled(config: Dict[str, Any]) -> bool:
    portfolio_cfg = config.get("portfolio", {}) or {}
    return bool(portfolio_cfg.get("strict_single_event_mode", True))



def _same_event_scope(held_row: pd.Series, candidate_row: pd.Series) -> bool:
    held_ticker = safe_str(safe_row_value(held_row, "ticker", ""))
    candidate_ticker = safe_str(safe_row_value(candidate_row, "contract_ticker", ""))
    if not held_ticker or not candidate_ticker:
        return False
    return _extract_event_ticker_from_contract_ticker(held_ticker) == _extract_event_ticker_from_contract_ticker(candidate_ticker)



def _get_hold_to_expiry_hours(config: Dict[str, Any]) -> float:
    exit_cfg = config.get("exit", {}) or {}
    return float(exit_cfg.get("hold_to_expiry_hours_left", 2.0))


def _prefer_settlement_over_roll(config: Dict[str, Any]) -> bool:
    daily_cfg = config.get("daily_behavior", {}) or {}
    return bool(daily_cfg.get("prefer_settlement_over_roll", True))


def _held_hours_left(held_row: pd.Series) -> Optional[float]:
    hours_left = _safe_float(safe_row_value(held_row, "hours_left", None), None)
    if hours_left is None:
        return None
    return float(hours_left)


def _held_is_near_settlement(held_row: pd.Series, config: Dict[str, Any]) -> bool:
    hours_left = _held_hours_left(held_row)
    if hours_left is None:
        return False
    return hours_left <= _get_hold_to_expiry_hours(config)


def _held_is_out_of_scope(held_row: pd.Series, candidate_row: Optional[pd.Series]) -> bool:
    if candidate_row is None:
        return False
    return not _same_event_scope(held_row, candidate_row)


def _should_hold_strict(held_row: pd.Series, candidate_row: Optional[pd.Series], config: Dict[str, Any]) -> bool:
    if held_row is None:
        return False

    # Daily-engine profit rule:
    # if we already hold a position and it is close to settlement, do not rotate it
    # out just because the engine is now evaluating a newer event.
    if _prefer_settlement_over_roll(config) and _held_is_near_settlement(held_row, config):
        return True

    if candidate_row is None:
        return True

    # If the held position is outside the currently evaluated event scope, prefer
    # settlement over rotation for daily binaries.
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
    keep_min_edge = float((config.get("portfolio_rotation", {}) or {}).get("held_min_edge_to_keep", 0.08))
    min_rotation_improvement = float((config.get("portfolio_rotation", {}) or {}).get("min_edge_improvement", 0.10))

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

    rotation_cfg = config.get("portfolio_rotation", {}) or {}
    if bool(rotation_cfg.get("disable_rotation_exits", True)):
        return False

    held_edge = _safe_float(safe_row_value(held_row, "current_edge", 0.0), 0.0)
    held_decision_state = safe_upper(safe_row_value(held_row, "decision_state", ""))
    candidate_edge = _candidate_edge(candidate_row)
    edge_improvement = candidate_edge - held_edge

    min_rotation_improvement = float(rotation_cfg.get("min_edge_improvement", 0.10))
    held_weak_threshold = float(rotation_cfg.get("held_weak_edge_threshold", 0.08))

    if candidate_edge <= held_edge:
        return False
    if edge_improvement < min_rotation_improvement:
        return False
    if held_decision_state == "ACTIONABLE" and held_edge >= held_weak_threshold:
        return False
    return True


def _pick_best_candidate(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    return df.iloc[0]


# =========================================================
# ENTRY SIZING
# =========================================================
def build_new_entry_instruction(
    row: pd.Series,
    cash_budget: float,
    capital: float,
    config: Dict[str, Any],
    live_positions_df: Optional[pd.DataFrame] = None,
) -> Optional[TradeInstruction]:
    side = _candidate_side(row)
    ask_price = _candidate_ask(row)
    bid_price = _candidate_bid(row)
    edge = _candidate_edge(row)
    confidence = safe_upper(safe_row_value(row, "confidence_norm", safe_row_value(row, "confidence", "LOW")))

    if side not in {"YES", "NO"}:
        return None
    if not _is_tradeable_price(ask_price):
        return None

    safety_cfg = _get_capital_safety_config(config)
    capital = _safe_float(capital, 0.0)
    cash_budget = _safe_float(cash_budget, 0.0)
    ask_price = _safe_float(ask_price, 0.0)
    edge = _safe_float(edge, 0.0)

    if capital <= 0 or cash_budget <= 0 or ask_price <= 0:
        return None

    small_account_threshold = 15.0
    if capital <= small_account_threshold:
        if cash_budget < ask_price:
            return None
        return TradeInstruction(
            action="ENTER",
            ticker=safe_row_value(row, "contract_ticker", None),
            side=side,
            contracts=1,
            allocation=ask_price,
            ask_price=ask_price,
            bid_price=bid_price,
            edge=edge,
            confidence=confidence,
            reason="SMALL_ACCOUNT_SINGLE_ENTRY",
        )

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
        reason="STANDARD_SINGLE_ENTRY",
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
    effective_min_trade_dollars = _get_effective_min_trade_dollars(
        capital,
        deployable_cash,
        config,
    )

    safety_cfg = _get_capital_safety_config(config)
    below_balance_floor = available_cash < float(safety_cfg["stop_trading_if_balance_below"])
    cash_blocked = _is_cash_blocked(deployable_cash, effective_min_trade_dollars) or below_balance_floor

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

    watchlist_states = {"WAIT_FOR_PRICE", "PRICE_OK_BUT_SPREAD_TOO_WIDE"}
    watchlist_df = ranked[ranked["decision_state"].isin(watchlist_states)].copy()
    watchlist_candidates_count = len(watchlist_df)

    entry_filter_cfg = _get_entry_filter_config(config)
    min_edge = float(entry_filter_cfg["min_edge_to_trade"])
    max_overround = float(entry_filter_cfg["max_overround"])
    spread_penalty = float(entry_filter_cfg["spread_penalty"])
    avoid_midzone_min_price = float(entry_filter_cfg["avoid_midzone_min_price"])
    avoid_midzone_max_price = float(entry_filter_cfg["avoid_midzone_max_price"])
    allow_midzone_trades = bool(entry_filter_cfg["allow_midzone_trades"])
    midzone_real_edge_override = float(entry_filter_cfg["midzone_real_edge_override"])

    tradable_df = ranked.loc[ranked["action_norm"].isin(["BUY_YES", "BUY_NO"])].copy()

    for col, default in [
        ("confidence_norm", "LOW"),
        ("selected_edge", np.nan),
        ("selected_ask", np.nan),
        ("selected_bid", np.nan),
        ("trading_phase", "UNKNOWN"),
        ("overround", np.nan),
        ("market_too_wide", False),
        ("entry_style", ""),
        ("decision_state", ""),
    ]:
        if col not in tradable_df.columns:
            tradable_df[col] = default

    tradable_df["confidence_norm"] = tradable_df["confidence_norm"].fillna("LOW").astype(str).str.strip().str.upper()
    tradable_df["trading_phase"] = tradable_df["trading_phase"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    tradable_df["decision_state"] = tradable_df["decision_state"].fillna("").astype(str).str.strip().str.upper()
    tradable_df["entry_style"] = tradable_df["entry_style"].fillna("").astype(str).str.strip()
    tradable_df["selected_edge"] = pd.to_numeric(tradable_df["selected_edge"], errors="coerce")
    tradable_df["selected_ask"] = pd.to_numeric(tradable_df["selected_ask"], errors="coerce")
    tradable_df["selected_bid"] = pd.to_numeric(tradable_df["selected_bid"], errors="coerce")
    tradable_df["overround"] = pd.to_numeric(tradable_df["overround"], errors="coerce")

    logging.info("Tradable df columns before strict filters: %s", list(tradable_df.columns))
    _log_tradable_stage("Tradable candidates initial", tradable_df)

    tradable_df = tradable_df.loc[tradable_df["confidence_norm"].isin(["HIGH", "MEDIUM"])].copy()
    _log_tradable_stage("Tradable candidates after confidence filter", tradable_df)

    tradable_df = tradable_df.loc[tradable_df["selected_ask"].apply(_is_tradeable_price)].copy()
    _log_tradable_stage("Tradable candidates after price filter", tradable_df)

    tradable_df = tradable_df.loc[(tradable_df["overround"].isna()) | (tradable_df["overround"] <= max_overround)].copy()
    tradable_df = tradable_df.loc[~tradable_df["market_too_wide"].fillna(False).astype(bool)].copy()
    _log_tradable_stage("Tradable candidates after overround filter", tradable_df)

    tradable_df["real_edge"] = tradable_df["selected_edge"].apply(lambda x: _compute_real_edge(x, spread_penalty))
    _log_tradable_stage("Tradable candidates after real_edge compute", tradable_df)

    if not allow_midzone_trades:
        tradable_df = tradable_df.loc[
            (
                (tradable_df["selected_ask"] < avoid_midzone_min_price)
                | (tradable_df["selected_ask"] > avoid_midzone_max_price)
                | (tradable_df["real_edge"] >= midzone_real_edge_override)
            )
        ].copy()
        logging.info(
            "Tradable rows after mid-zone filter (%s-%s removed unless real_edge>=%.3f): %s",
            round(avoid_midzone_min_price, 4),
            round(avoid_midzone_max_price, 4),
            round(midzone_real_edge_override, 4),
            len(tradable_df),
        )
        _log_tradable_stage("Tradable candidates after mid-zone filter", tradable_df)

    tradable_df = tradable_df.loc[tradable_df["real_edge"] >= min_edge].copy()
    _log_tradable_stage("Tradable candidates after real-edge filter", tradable_df)

    allowed_phases = _select_allowed_entry_phases(config, has_live_positions=not live.empty)
    phase_context = "Tradable candidates final phase gate"
    tradable_df = _apply_allowed_phase_filter(tradable_df, allowed_phases, phase_context)

    tradable_candidates_count = len(tradable_df)
    logging.info("Filtered tradable opportunities count: %s", tradable_candidates_count)

    scored_live = _score_live_positions_against_ranked(live, ranked)
    best_held = _pick_best_candidate(scored_live)

    if tradable_df.empty:
        if _single_position_mode_enabled(config) and best_held is not None:
            hold_reason = "No new tradable candidates, but an existing held position remains active."
            if _prefer_settlement_over_roll(config) and _held_is_near_settlement(best_held, config):
                hold_reason = "No new tradable candidates, and held daily position is inside the hold-to-expiry window."
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

    tradable_df["confidence_rank"] = tradable_df["confidence_norm"].map({"HIGH": 3, "MEDIUM": 2, "LOW": 1}).fillna(0)

    tradable_df["candidate_score"] = tradable_df.apply(lambda row: _candidate_score(row, config), axis=1)

    tradable_df = tradable_df.sort_values(
        by=["candidate_score", "confidence_rank", "real_edge", "selected_edge", "distance_abs"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    logging.info(
        "Top filtered tradable trades: %s",
        _safe_preview_records(
            tradable_df,
            [
                "contract_ticker",
                "action",
                "decision_state",
                "trading_phase",
                "selected_side",
                "selected_edge",
                "real_edge",
                "selected_ask",
                "confidence_norm",
                "distance_to_strike",
                "yes_no_ask_sum",
                "overround",
                "entry_style",
                "candidate_score",
            ],
            head_n=5,
        ),
    )

    best_candidate = _pick_best_candidate(tradable_df)

    if _single_position_mode_enabled(config) and best_held is not None:
        logging.info(
            "STRICT MODE active | held_ticker=%s | held_side=%s | held_edge=%.4f",
            safe_row_value(best_held, "ticker", None),
            safe_row_value(best_held, "side", None),
            _safe_float(safe_row_value(best_held, "current_edge", 0.0), 0.0),
        )

        if _should_hold_strict(best_held, best_candidate, config):
            hold_reason = "Held position remains competitive versus current opportunities."
            if _prefer_settlement_over_roll(config) and _held_is_near_settlement(best_held, config):
                hold_reason = "Held daily position is within the hold-to-expiry window; keep it through settlement."
            elif _prefer_settlement_over_roll(config) and _held_is_out_of_scope(best_held, best_candidate):
                hold_reason = "Held position is outside the current event scope, but daily settlement is preferred over forced rotation."
            return _build_hold_plan(
                held_row=best_held,
                capital=capital,
                available_cash=available_cash,
                reserve_cash_target=reserve_cash_target,
                deployable_cash=deployable_cash,
                reason=_append_watchlist_context(
                    hold_reason,
                    watchlist_candidates_count,
                ),
                tradable_candidates_count=tradable_candidates_count,
                watchlist_candidates_count=watchlist_candidates_count,
            )

        if _should_rotate_strict(best_held, best_candidate, config):
            estimated_cash_for_rotation = deployable_cash + max(
                0.0,
                _safe_float(safe_row_value(best_held, "allocation", 0.0), 0.0),
            )

            entry_instr = build_new_entry_instruction(
                best_candidate,
                estimated_cash_for_rotation,
                capital,
                config,
                live_positions_df=scored_live,
            )

            if entry_instr:
                return _build_rotate_plan(
                    held_row=best_held,
                    entry_instr=entry_instr,
                    entry_row=best_candidate,
                    capital=capital,
                    available_cash=available_cash,
                    reserve_cash_target=reserve_cash_target,
                    deployable_cash=deployable_cash,
                    reason=_append_watchlist_context(
                        "Held trade weakened and materially better replacement exists.",
                        watchlist_candidates_count,
                    ),
                    tradable_candidates_count=tradable_candidates_count,
                    watchlist_candidates_count=watchlist_candidates_count,
                )

        return _build_hold_plan(
            held_row=best_held,
            capital=capital,
            available_cash=available_cash,
            reserve_cash_target=reserve_cash_target,
            deployable_cash=deployable_cash,
            reason=_append_watchlist_context(
                "Strict single-position mode blocked additional entries.",
                watchlist_candidates_count,
            ),
            tradable_candidates_count=tradable_candidates_count,
            watchlist_candidates_count=watchlist_candidates_count,
        )

    if cash_blocked:
        return PortfolioPlan(
            recommendation="WAIT",
            reason=_append_watchlist_context(
                "Actionable trades found, but no deployable cash available.",
                watchlist_candidates_count,
            ),
            capital=capital,
            available_cash=available_cash,
            reserve_cash_target=reserve_cash_target,
            deployable_cash=deployable_cash,
            tradable_candidates_count=tradable_candidates_count,
            watchlist_candidates_count=watchlist_candidates_count,
            actions=[],
        )

    entry_instr = build_new_entry_instruction(
        best_candidate,
        deployable_cash,
        capital,
        config,
        live_positions_df=live,
    )

    if entry_instr:
        return _build_enter_plan(
            entry_instr=entry_instr,
            entry_row=best_candidate,
            capital=capital,
            available_cash=available_cash,
            reserve_cash_target=reserve_cash_target,
            deployable_cash=deployable_cash,
            reason=_append_watchlist_context(
                "Top ranked single opportunity.",
                watchlist_candidates_count,
            ),
            tradable_candidates_count=tradable_candidates_count,
            watchlist_candidates_count=watchlist_candidates_count,
        )

    return PortfolioPlan(
        recommendation="WAIT",
        reason=_append_watchlist_context(
            "Tradable candidates exist, but none passed sizing constraints.",
            watchlist_candidates_count,
        ),
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
