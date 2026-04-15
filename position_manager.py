import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


def time_to_expiry_years(hours_left):
    if hours_left is None:
        return 0.0
    try:
        hours_left = float(hours_left)
    except Exception:
        return 0.0
    return max(hours_left, 0.0) / (24.0 * 365.0)


def estimate_terminal_probability_from_inputs(price, strike, hours_left, vol, drift=0.0):
    if pd.isna(price) or pd.isna(strike) or pd.isna(hours_left) or pd.isna(vol):
        return np.nan

    try:
        price = float(price)
        strike = float(strike)
        hours_left = float(hours_left)
        vol = float(vol)
        drift = float(drift)
    except Exception:
        return np.nan

    T = time_to_expiry_years(hours_left)

    if T <= 0:
        return float(price >= strike)

    if strike <= 0 or price <= 0:
        return 0.5

    sigma_t = vol * math.sqrt(T)
    if sigma_t <= 1e-12:
        deterministic_terminal = price * math.exp(drift * T)
        return float(deterministic_terminal > strike)

    d2 = (math.log(price / strike) + (drift - 0.5 * vol**2) * T) / sigma_t
    prob = norm.cdf(d2)
    return float(np.clip(prob, 0.0, 1.0))


def compute_current_position_price(action: str, market_prob: float) -> float:
    if pd.isna(market_prob):
        return np.nan

    action = str(action or "").strip().upper()

    if action == "BUY_YES":
        return float(market_prob)

    if action == "BUY_NO":
        return float(1.0 - market_prob)

    return np.nan


def compute_position_pnl(action: str, entry_price: float, current_market_prob: float, size: float = 1.0) -> float:
    current_price = compute_current_position_price(action, current_market_prob)

    if pd.isna(entry_price) or pd.isna(current_price):
        return np.nan

    if pd.isna(size):
        size = 1.0

    return float((current_price - entry_price) * size)


def _safe_first(series, default=np.nan):
    if series is None or len(series) == 0:
        return default
    try:
        return series.iloc[0]
    except Exception:
        return default


def _coerce_numeric_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = pd.to_numeric(df[col], errors="coerce")


def _coerce_bool_value(value, default=False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    return default


def _safe_upper_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[col].astype(str).str.strip().str.upper()


def _get_current_fair_yes(df: pd.DataFrame) -> pd.Series:
    decision_prob = pd.to_numeric(df.get("decision_prob"), errors="coerce")
    fair_prob_terminal = pd.to_numeric(df.get("fair_prob_terminal"), errors="coerce")
    fair_prob_blended = pd.to_numeric(df.get("fair_prob_blended"), errors="coerce")

    current_fair_yes = decision_prob.copy()
    current_fair_yes = current_fair_yes.where(current_fair_yes.notna(), fair_prob_terminal)
    current_fair_yes = current_fair_yes.where(current_fair_yes.notna(), fair_prob_blended)

    return current_fair_yes


def _safe_selected_edge(row: pd.Series) -> float:
    action = str(row.get("action", "") or "").strip().upper()
    side_norm = str(row.get("side_norm", "") or "").strip().upper()

    edge_yes = pd.to_numeric(pd.Series([row.get("edge_yes")]), errors="coerce").iloc[0]
    edge_no = pd.to_numeric(pd.Series([row.get("edge_no")]), errors="coerce").iloc[0]
    selected_edge = pd.to_numeric(pd.Series([row.get("selected_edge")]), errors="coerce").iloc[0]
    current_edge = pd.to_numeric(pd.Series([row.get("current_edge")]), errors="coerce").iloc[0]

    if pd.notna(current_edge):
        return float(current_edge)
    if pd.notna(selected_edge):
        return float(selected_edge)
    if action == "BUY_YES" or side_norm == "YES":
        return float(edge_yes) if pd.notna(edge_yes) else np.nan
    if action == "BUY_NO" or side_norm == "NO":
        return float(edge_no) if pd.notna(edge_no) else np.nan

    if pd.notna(edge_yes) and pd.notna(edge_no):
        return float(max(edge_yes, edge_no))
    if pd.notna(edge_yes):
        return float(edge_yes)
    if pd.notna(edge_no):
        return float(edge_no)
    return np.nan


def _safe_current_price_from_row(row: pd.Series) -> float:
    current_position_price = pd.to_numeric(pd.Series([row.get("current_position_price")]), errors="coerce").iloc[0]
    if pd.notna(current_position_price):
        return float(current_position_price)

    current_price = pd.to_numeric(pd.Series([row.get("current_price")]), errors="coerce").iloc[0]
    if pd.notna(current_price):
        return float(current_price)

    action = str(row.get("action", "") or "").strip().upper()

    if action == "BUY_YES":
        for col in ["last_price_yes", "ask_yes", "bid_yes", "market_prob"]:
            val = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if pd.notna(val):
                return float(val)

    if action == "BUY_NO":
        for col in ["last_price_no", "ask_no", "bid_no"]:
            val = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if pd.notna(val):
                return float(val)
        market_prob = pd.to_numeric(pd.Series([row.get("market_prob")]), errors="coerce").iloc[0]
        if pd.notna(market_prob):
            return float(1.0 - market_prob)

    return np.nan


def _safe_position_size(row: pd.Series) -> float:
    for col in ["contracts", "size"]:
        val = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if pd.notna(val):
            return float(val)
    return np.nan


def _resolve_entry_price(row: pd.Series) -> float:
    """
    FINAL SAFE VERSION — ALWAYS CORRECT COST BASIS FIRST
    """

    cost = row.get("position_cost")
    contracts = row.get("contracts")

    try:
        if cost is not None and contracts:
            cost = float(cost)
            contracts = float(contracts)
            if contracts > 0:
                inferred = cost / contracts
                if inferred > 1:
                    inferred = inferred / 100.0
                if 0 < inferred <= 1:
                    return inferred
    except Exception:
        pass

    val = row.get("entry_price")
    try:
        val = float(val)
        if 0 < val <= 1:
            return val
    except Exception:
        pass

    for col in ["avg_entry_price", "fill_price", "price"]:
        val = row.get(col)
        try:
            val = float(val)
            if 0 < val <= 1:
                return val
        except Exception:
            pass

    current_price = _safe_current_price_from_row(row)
    if pd.notna(current_price) and 0 < current_price <= 1:
        return float(current_price)

    return np.nan


def _infer_fair_price_for_held_side(row: pd.Series) -> float:
    fair_yes = pd.to_numeric(pd.Series([_get_current_fair_yes(pd.DataFrame([row])).iloc[0]]), errors="coerce").iloc[0]
    action = str(row.get("action", "") or "").strip().upper()

    if pd.isna(fair_yes):
        return np.nan

    if action == "BUY_YES":
        return float(fair_yes)
    if action == "BUY_NO":
        return float(1.0 - fair_yes)
    return np.nan


def _format_pct(val: Optional[float]) -> str:
    if val is None or pd.isna(val):
        return "n/a"
    return f"{float(val):.3f}"


def _format_price(val: Optional[float]) -> str:
    if val is None or pd.isna(val):
        return "n/a"
    return f"{float(val):.3f}"


def _held_side_context(row: pd.Series) -> dict:
    action = str(row.get("action", "") or "").strip().upper()
    fair_yes = pd.to_numeric(pd.Series([_get_current_fair_yes(pd.DataFrame([row])).iloc[0]]), errors="coerce").iloc[0]

    edge_yes = pd.to_numeric(pd.Series([row.get("edge_yes")]), errors="coerce").iloc[0]
    edge_no = pd.to_numeric(pd.Series([row.get("edge_no")]), errors="coerce").iloc[0]

    target_yes_price = pd.to_numeric(pd.Series([row.get("target_yes_price")]), errors="coerce").iloc[0]
    target_no_price = pd.to_numeric(pd.Series([row.get("target_no_price")]), errors="coerce").iloc[0]

    executable_yes_now = _coerce_bool_value(row.get("executable_yes_now"), default=False)
    executable_no_now = _coerce_bool_value(row.get("executable_no_now"), default=False)

    ask_yes = pd.to_numeric(pd.Series([row.get("ask_yes")]), errors="coerce").iloc[0]
    ask_no = pd.to_numeric(pd.Series([row.get("ask_no")]), errors="coerce").iloc[0]

    decision_prob = pd.to_numeric(pd.Series([row.get("decision_prob")]), errors="coerce").iloc[0]

    if action == "BUY_YES":
        held_side = "YES"
        held_edge = edge_yes
        held_target_price = target_yes_price
        held_executable_now = executable_yes_now
        held_ask = ask_yes
        held_fair_side_price = fair_yes
        held_prob_support = decision_prob
    elif action == "BUY_NO":
        held_side = "NO"
        held_edge = edge_no
        held_target_price = target_no_price
        held_executable_now = executable_no_now
        held_ask = ask_no
        held_fair_side_price = (1.0 - fair_yes) if pd.notna(fair_yes) else np.nan
        held_prob_support = (1.0 - decision_prob) if pd.notna(decision_prob) else np.nan
    else:
        held_side = ""
        held_edge = _safe_selected_edge(row)
        held_target_price = np.nan
        held_executable_now = False
        held_ask = np.nan
        held_fair_side_price = _infer_fair_price_for_held_side(row)
        held_prob_support = np.nan

    current_position_price = pd.to_numeric(pd.Series([_safe_current_price_from_row(row)]), errors="coerce").iloc[0]

    return {
        "held_side": held_side,
        "held_edge": held_edge,
        "held_target_price": held_target_price,
        "held_executable_now": held_executable_now,
        "held_ask": held_ask,
        "held_fair_side_price": held_fair_side_price,
        "held_prob_support": held_prob_support,
        "current_position_price": current_position_price,
    }


def _build_classification_payload(
    row: pd.Series,
    held_side: str,
    state: str,
    severity: int,
    should_exit: bool,
    rotate_candidate: bool,
    exit_reason: str,
    held_reason: str,
    held_edge: float,
    held_target_price: float,
    held_executable_now: bool,
    held_prob_support: float,
    current_fair_side_price: float,
    fair_price_gap: float,
    size: float,
    unrealized_pnl: float,
    market_value: float,
    position_cost: float,
    current_price: float,
) -> dict:
    weak_cycles = pd.to_numeric(pd.Series([row.get("weak_cycles")]), errors="coerce").fillna(0).iloc[0]
    exit_attempts = pd.to_numeric(pd.Series([row.get("exit_attempts")]), errors="coerce").fillna(0).iloc[0]
    portfolio_value = pd.to_numeric(pd.Series([row.get("portfolio_value")]), errors="coerce").iloc[0]
    position_weight = np.nan
    if pd.notna(portfolio_value) and float(portfolio_value) > 0 and pd.notna(market_value):
        position_weight = float(market_value) / float(portfolio_value)

    return {
        "held_side": held_side,
        "held_decision_state": state,
        "held_state_severity": severity,
        "should_exit": bool(should_exit),
        "rotate_candidate": bool(rotate_candidate),
        "exit_reason": exit_reason if should_exit else "",
        "held_reason": held_reason or "Held position evaluated.",
        "current_edge": held_edge,
        "held_edge": held_edge,
        "held_target_price": held_target_price,
        "held_executable_now": held_executable_now,
        "held_prob_support": held_prob_support,
        "current_fair_side_price": current_fair_side_price,
        "fair_price_gap": fair_price_gap,
        "position_size": size,
        "unrealized_pnl": unrealized_pnl,
        "market_value": market_value,
        "position_cost": position_cost,
        "current_position_price": current_price,
        "weak_cycles": int(weak_cycles),
        "exit_attempts": int(exit_attempts),
        "position_weight": position_weight,
    }


def _classify_held_position(row: pd.Series) -> dict:
    action = str(row.get("action", "") or "").strip().upper()
    confidence = str(row.get("confidence", "") or "").strip().upper()
    no_trade_reason = str(row.get("no_trade_reason", "") or "").strip()
    decision_state = str(row.get("decision_state", "") or "").strip().upper()

    position_cost = pd.to_numeric(pd.Series([row.get("position_cost")]), errors="coerce").iloc[0]
    market_value = pd.to_numeric(pd.Series([row.get("market_value")]), errors="coerce").iloc[0]
    fair_prob_drop = pd.to_numeric(pd.Series([row.get("fair_prob_drop")]), errors="coerce").iloc[0]
    size = pd.to_numeric(pd.Series([_safe_position_size(row)]), errors="coerce").iloc[0]
    distance_to_strike = pd.to_numeric(pd.Series([row.get("distance_to_strike")]), errors="coerce").iloc[0]
    unrealized_pnl = pd.to_numeric(pd.Series([row.get("unrealized_pnl")]), errors="coerce").iloc[0]
    market_too_wide = _coerce_bool_value(row.get("market_too_wide"), default=False)

    ctx = _held_side_context(row)
    held_side = ctx["held_side"]
    held_edge = pd.to_numeric(pd.Series([ctx["held_edge"]]), errors="coerce").iloc[0]
    held_target_price = pd.to_numeric(pd.Series([ctx["held_target_price"]]), errors="coerce").iloc[0]
    held_executable_now = bool(ctx["held_executable_now"])
    held_ask = pd.to_numeric(pd.Series([ctx["held_ask"]]), errors="coerce").iloc[0]
    current_fair_side_price = pd.to_numeric(pd.Series([ctx["held_fair_side_price"]]), errors="coerce").iloc[0]
    held_prob_support = pd.to_numeric(pd.Series([ctx["held_prob_support"]]), errors="coerce").iloc[0]
    current_price = pd.to_numeric(pd.Series([ctx["current_position_price"]]), errors="coerce").iloc[0]

    if pd.isna(current_fair_side_price):
        current_fair_side_price = _infer_fair_price_for_held_side(row)

    contracts = pd.to_numeric(pd.Series([_safe_position_size(row)]), errors="coerce").iloc[0]
    if pd.isna(contracts) or contracts <= 0:
        contracts = 1.0

    if pd.isna(current_price) or current_price <= 0:
        if pd.notna(market_value) and pd.notna(contracts) and contracts > 0:
            raw_market_value = float(market_value)
            if raw_market_value > float(contracts):
                current_price = (raw_market_value / 100.0) / float(contracts)
            else:
                current_price = raw_market_value / float(contracts)

    entry_price = _resolve_entry_price(row)
    if pd.isna(entry_price) or entry_price <= 0:
        entry_price = current_price

    fair_price_gap = np.nan
    if pd.notna(current_fair_side_price) and pd.notna(current_price):
        fair_price_gap = float(current_fair_side_price - current_price)

    pnl_ratio = np.nan
    if pd.notna(entry_price) and entry_price > 0 and pd.notna(current_price):
        pnl_ratio = (float(current_price) - float(entry_price)) / float(entry_price)

    if pd.notna(entry_price) and entry_price > 0 and pd.notna(current_price) and pd.notna(contracts) and contracts > 0:
        unrealized_pnl = (float(current_price) - float(entry_price)) * float(contracts)

    force_stop_loss_pct = -0.25
    hard_stop_loss_pct = -0.12
    early_stop_loss_pct = -0.08

    full_profit_lock_pct = 0.20
    soft_profit_lock_pct = 0.14

    severe_probability_collapse_threshold = 0.40
    support_low_threshold = 0.06

    hard_negative_edge_threshold = -0.04
    soft_negative_edge_threshold = -0.01
    weak_positive_edge_threshold = 0.02
    healthy_edge_threshold = 0.08

    fair_collapse_hard_threshold = 0.12
    fair_collapse_soft_threshold = 0.08

    pnl_hard_loss = pd.notna(pnl_ratio) and pnl_ratio <= hard_stop_loss_pct
    pnl_soft_loss = pd.notna(pnl_ratio) and pnl_ratio <= early_stop_loss_pct

    edge_hard_negative = pd.notna(held_edge) and held_edge <= hard_negative_edge_threshold
    edge_nonpositive = pd.notna(held_edge) and held_edge <= soft_negative_edge_threshold
    edge_weak = pd.notna(held_edge) and held_edge <= weak_positive_edge_threshold
    edge_healthy = pd.notna(held_edge) and held_edge >= healthy_edge_threshold

    fair_collapse_hard = pd.notna(fair_prob_drop) and fair_prob_drop >= fair_collapse_hard_threshold
    fair_collapse_soft = pd.notna(fair_prob_drop) and fair_prob_drop >= fair_collapse_soft_threshold

    support_low = pd.notna(held_prob_support) and held_prob_support <= support_low_threshold

    target_support_lost = pd.notna(held_target_price) and pd.notna(held_ask) and held_ask > held_target_price
    fair_gap_negative = pd.notna(fair_price_gap) and fair_price_gap < 0
    fair_gap_clearly_negative = pd.notna(fair_price_gap) and fair_price_gap <= -0.02

    hard_market_deterioration = (
        decision_state == "NOT_TRADABLE"
        and bool(market_too_wide)
        and pd.notna(row.get("overround"))
        and float(row.get("overround")) > 0.25
    )

    near_strike_risk = pd.notna(distance_to_strike) and abs(float(distance_to_strike)) <= 15.0 and confidence != "HIGH"

    if pd.notna(pnl_ratio) and pnl_ratio >= full_profit_lock_pct and (
        edge_nonpositive or fair_gap_clearly_negative or fair_collapse_hard
    ):
        return _build_classification_payload(row, held_side, "EXIT_PROFIT_LOCK", 5, True, False,
            f"Profit lock triggered: pnl_ratio={_format_pct(pnl_ratio)} with weakened support.",
            "Strong unrealized profit captured after thesis weakened.", held_edge, held_target_price,
            held_executable_now, held_prob_support, current_fair_side_price, fair_price_gap,
            contracts, unrealized_pnl, market_value, position_cost, current_price)

    if pd.notna(pnl_ratio) and pnl_ratio >= soft_profit_lock_pct and (
        edge_hard_negative or fair_collapse_hard or fair_gap_clearly_negative or (not held_executable_now and support_low)
    ):
        return _build_classification_payload(row, held_side, "EXIT_PROFIT_WEAKENING", 4, True, False,
            f"Profit weakening exit: pnl_ratio={_format_pct(pnl_ratio)}, held_edge={_format_pct(held_edge)}, fair_gap={_format_pct(fair_price_gap)}.",
            "Profitable position exited only after clear thesis deterioration.", held_edge, held_target_price,
            held_executable_now, held_prob_support, current_fair_side_price, fair_price_gap,
            contracts, unrealized_pnl, market_value, position_cost, current_price)

    if pd.notna(pnl_ratio) and pnl_ratio <= force_stop_loss_pct:
        return _build_classification_payload(row, held_side, "EXIT_FORCE_STOP_LOSS", 6, True, False,
            f"Forced stop loss triggered: pnl_ratio={_format_pct(pnl_ratio)}.",
            "Emergency exit triggered from hard capital protection stop.", held_edge, held_target_price,
            held_executable_now, held_prob_support, current_fair_side_price, fair_price_gap,
            contracts, unrealized_pnl, market_value, position_cost, current_price)

    if pd.notna(size) and float(size) >= 20:
        return _build_classification_payload(row, held_side, "EXIT_OVERSIZED", 6, True, False,
            f"Forced size exit: contracts={int(float(size))}.",
            "Emergency exit triggered because position breached size safety ceiling.", held_edge, held_target_price,
            held_executable_now, held_prob_support, current_fair_side_price, fair_price_gap,
            size, unrealized_pnl, market_value, position_cost, current_price)

    if pd.notna(held_prob_support) and held_prob_support <= severe_probability_collapse_threshold and (
        edge_nonpositive or fair_gap_clearly_negative
    ):
        return _build_classification_payload(row, held_side, "EXIT_PROBABILITY_COLLAPSE", 5, True, False,
            f"Probability collapse triggered: held_prob_support={_format_pct(held_prob_support)}.",
            "Held-side probability support collapsed materially.", held_edge, held_target_price,
            held_executable_now, held_prob_support, current_fair_side_price, fair_price_gap,
            contracts, unrealized_pnl, market_value, position_cost, current_price)

    if edge_hard_negative and (fair_gap_clearly_negative or fair_collapse_soft or support_low or not held_executable_now):
        return _build_classification_payload(row, held_side, "EXIT_EDGE_NEGATIVE", 5, True, False,
            f"Held {action} lost edge support: held_edge={_format_pct(held_edge)}, executable_now={held_executable_now}, held_prob_support={_format_pct(held_prob_support)}, fair_gap={_format_pct(fair_price_gap)}.",
            "Held position lost edge support on its own side.", held_edge, held_target_price,
            held_executable_now, held_prob_support, current_fair_side_price, fair_price_gap,
            contracts, unrealized_pnl, market_value, position_cost, current_price)

    if fair_collapse_hard and (edge_nonpositive or fair_gap_negative):
        return _build_classification_payload(row, held_side, "EXIT_FAIR_VALUE_COLLAPSED", 5, True, False,
            f"Held {action} fair value collapsed: fair_drop={_format_pct(fair_prob_drop)}, held_edge={_format_pct(held_edge)}.",
            "Held-side fair value collapsed materially.", held_edge, held_target_price,
            held_executable_now, held_prob_support, current_fair_side_price, fair_price_gap,
            contracts, unrealized_pnl, market_value, position_cost, current_price)

    if pnl_hard_loss and (edge_nonpositive or fair_collapse_soft or fair_gap_negative):
        return _build_classification_payload(row, held_side, "EXIT_HARD_STOP", 5, True, False,
            f"Hard stop loss triggered: pnl_ratio={_format_pct(pnl_ratio)}, held_edge={_format_pct(held_edge)}, fair_drop={_format_pct(fair_prob_drop)}.",
            "Loss threshold breached with confirming weakness.", held_edge, held_target_price,
            held_executable_now, held_prob_support, current_fair_side_price, fair_price_gap,
            contracts, unrealized_pnl, market_value, position_cost, current_price)

    if pnl_soft_loss and (edge_hard_negative or fair_collapse_hard or fair_gap_clearly_negative):
        return _build_classification_payload(row, held_side, "EXIT_EARLY_STOP", 4, True, False,
            f"Early stop triggered: pnl_ratio={_format_pct(pnl_ratio)}, held_edge={_format_pct(held_edge)}, fair_gap={_format_pct(fair_price_gap)}.",
            "Moderate loss exited only because support clearly broke.", held_edge, held_target_price,
            held_executable_now, held_prob_support, current_fair_side_price, fair_price_gap,
            contracts, unrealized_pnl, market_value, position_cost, current_price)

    if hard_market_deterioration and (edge_nonpositive or support_low):
        return _build_classification_payload(row, held_side, "EXIT_MARKET_DETERIORATED", 5, True, False,
            f"Held {action} no longer acceptable in deteriorated market: decision_state={decision_state}, overround={_format_pct(row.get('overround'))}, held_edge={_format_pct(held_edge)}.",
            "Market structure deteriorated enough to exit.", held_edge, held_target_price,
            held_executable_now, held_prob_support, current_fair_side_price, fair_price_gap,
            contracts, unrealized_pnl, market_value, position_cost, current_price)

    state = "HOLD_MONITOR"
    severity = 1
    should_exit = False
    rotate_candidate = False
    exit_reason = ""

    detail_bits = []
    if held_side:
        detail_bits.append(f"held_side={held_side}")
    if pd.notna(held_edge):
        detail_bits.append(f"held_edge={_format_pct(held_edge)}")
    detail_bits.append(f"executable_now={held_executable_now}")
    if pd.notna(held_prob_support):
        detail_bits.append(f"held_prob_support={_format_pct(held_prob_support)}")
    if pd.notna(held_target_price):
        detail_bits.append(f"target={_format_price(held_target_price)}")
    if pd.notna(held_ask):
        detail_bits.append(f"ask={_format_price(held_ask)}")
    if pd.notna(fair_price_gap):
        detail_bits.append(f"fair_gap={_format_pct(fair_price_gap)}")
    if pd.notna(fair_prob_drop):
        detail_bits.append(f"fair_drop={_format_pct(fair_prob_drop)}")
    if pd.notna(pnl_ratio):
        detail_bits.append(f"pnl_ratio={_format_pct(pnl_ratio)}")

    healthy_hold = ((edge_healthy or (pd.notna(held_edge) and held_edge > weak_positive_edge_threshold)) and
                    not fair_gap_clearly_negative and not fair_collapse_soft and not support_low and not hard_market_deterioration)

    if healthy_hold:
        state = "HOLD_HEALTHY"
        severity = 0
        held_reason = "Held position remains healthy: " + ", ".join(detail_bits) + "."
    else:
        monitor_bits = []
        if edge_nonpositive:
            monitor_bits.append(f"held_edge={_format_pct(held_edge)}")
        elif edge_weak:
            monitor_bits.append(f"held_edge modest at {_format_pct(held_edge)}")
        if not held_executable_now:
            monitor_bits.append("not executable now")
        if target_support_lost:
            monitor_bits.append(f"ask above target ({_format_price(held_ask)} > {_format_price(held_target_price)})")
        if support_low:
            monitor_bits.append(f"held_prob_support={_format_pct(held_prob_support)}")
        if fair_collapse_soft:
            monitor_bits.append(f"fair_drop={_format_pct(fair_prob_drop)}")
        if pnl_soft_loss:
            monitor_bits.append(f"pnl_ratio={_format_pct(pnl_ratio)}")
        if near_strike_risk:
            monitor_bits.append("near strike with non-high confidence")
        if market_too_wide:
            monitor_bits.append("market too wide")
        if decision_state and decision_state not in {"", "HELD_POSITION_MONITORING_ONLY"}:
            monitor_bits.append(f"decision_state={decision_state}")
        if no_trade_reason:
            monitor_bits.append(f"market_note={no_trade_reason}")
        if not monitor_bits:
            monitor_bits = detail_bits
        held_reason = "Monitor held position: " + ", ".join(monitor_bits) + "."

    rotate_conditions = [
        edge_weak or edge_nonpositive,
        fair_gap_clearly_negative,
        fair_collapse_soft,
        support_low,
        pnl_soft_loss,
        decision_state in {"WAIT_FOR_PRICE", "PRICE_OK_BUT_SPREAD_TOO_WIDE", "NOT_TRADABLE", "NO_TRADE"},
    ]
    weak_enough_for_rotate = sum(bool(x) for x in rotate_conditions) >= 3
    if weak_enough_for_rotate and state != "HOLD_HEALTHY":
        rotate_candidate = True
        severity = max(severity, 2)
        held_reason = held_reason.rstrip(".") + " Rotation is reasonable only if a materially superior tradable replacement exists."

    return _build_classification_payload(row, held_side, state, severity, should_exit, rotate_candidate, exit_reason,
        held_reason, held_edge, held_target_price, held_executable_now, held_prob_support,
        current_fair_side_price, fair_price_gap, size, unrealized_pnl, market_value, position_cost, current_price)



def monitor_open_positions(open_positions_df: pd.DataFrame, ranked_df: pd.DataFrame) -> pd.DataFrame:
    if open_positions_df is None or open_positions_df.empty:
        return pd.DataFrame()

    monitored = open_positions_df.copy()

    if ranked_df is not None and not ranked_df.empty:
        merge_cols = [
            col for col in [
                "contract_ticker", "market_prob", "decision_prob", "fair_prob_terminal", "fair_prob_blended",
                "gold_price", "oil_price", "distance_to_strike", "hours_left", "edge_yes", "edge_no", "selected_edge",
                "confidence", "volatility", "adjusted_volatility", "ask_yes", "ask_no", "bid_yes", "bid_no",
                "last_price_yes", "last_price_no", "strike", "action", "side_norm", "decision_state",
                "no_trade_reason", "current_price", "market_value", "target_yes_price", "target_no_price",
                "executable_yes_now", "executable_no_now", "market_too_wide",
            ] if col in ranked_df.columns
        ]

        if "contract_ticker" in merge_cols and len(merge_cols) > 1:
            ranked_lookup = ranked_df[merge_cols].drop_duplicates(subset=["contract_ticker"]).copy()
            ranked_lookup = ranked_lookup.set_index("contract_ticker")

            if "contract_ticker" not in monitored.columns:
                monitored["contract_ticker"] = np.nan

            for col in merge_cols:
                if col == "contract_ticker":
                    continue

                mapped_series = monitored["contract_ticker"].map(ranked_lookup[col])

                if col in monitored.columns:
                    monitored[col] = mapped_series.combine_first(monitored[col])
                else:
                    monitored[col] = mapped_series

    latest_price = np.nan
    for candidate in ["gold_price", "oil_price"]:
        if ranked_df is not None and candidate in getattr(ranked_df, "columns", []):
            latest_price = _safe_first(ranked_df[candidate])
            if pd.notna(latest_price):
                break

    latest_hours_left = _safe_first(ranked_df["hours_left"]) if ranked_df is not None and "hours_left" in ranked_df.columns else np.nan
    latest_adjusted_vol = _safe_first(ranked_df["adjusted_volatility"]) if ranked_df is not None and "adjusted_volatility" in ranked_df.columns else np.nan
    latest_raw_vol = _safe_first(ranked_df["volatility"]) if ranked_df is not None and "volatility" in ranked_df.columns else np.nan

    required_cols = [
        "gold_price", "oil_price", "hours_left", "distance_to_strike", "market_prob", "decision_prob", "fair_prob_terminal",
        "fair_prob_blended", "edge_yes", "edge_no", "selected_edge", "confidence", "adjusted_volatility",
        "volatility", "entry_price", "position_cost", "market_value", "contracts", "size", "current_price",
        "current_position_price", "entry_fair_prob", "fair_prob_drop", "current_edge", "held_edge",
        "unrealized_pnl", "decision_state", "no_trade_reason", "side_norm", "target_yes_price",
        "target_no_price", "executable_yes_now", "executable_no_now", "market_too_wide", "held_side",
        "held_target_price", "held_executable_now", "held_prob_support", "held_decision_state", "held_reason",
        "held_state_severity", "rotate_candidate", "should_exit", "exit_reason", "current_fair_side_price",
        "fair_price_gap", "pnl_pct",
    ]
    for col in required_cols:
        if col not in monitored.columns:
            monitored[col] = np.nan

    if "gold_price" in monitored.columns:
        monitored["gold_price"] = monitored["gold_price"].fillna(latest_price)
    if "oil_price" in monitored.columns:
        monitored["oil_price"] = monitored["oil_price"].fillna(latest_price)

    monitored["hours_left"] = monitored["hours_left"].fillna(latest_hours_left)
    monitored["adjusted_volatility"] = monitored["adjusted_volatility"].fillna(latest_adjusted_vol)
    monitored["volatility"] = monitored["volatility"].fillna(latest_raw_vol)

    for col in [
        "contracts", "size", "entry_price", "position_cost", "market_value", "market_prob", "decision_prob",
        "fair_prob_terminal", "fair_prob_blended", "current_price", "gold_price", "oil_price", "hours_left", "distance_to_strike",
        "adjusted_volatility", "volatility", "edge_yes", "edge_no", "selected_edge", "strike", "entry_fair_prob",
        "current_position_price", "target_yes_price", "target_no_price", "unrealized_pnl", "pnl_pct",
    ]:
        _coerce_numeric_column(monitored, col)

    price_col = "gold_price" if "gold_price" in monitored.columns and monitored["gold_price"].notna().any() else "oil_price"

    if "strike" in monitored.columns and price_col in monitored.columns:
        monitored["distance_to_strike"] = np.where(
            monitored["distance_to_strike"].isna(),
            pd.to_numeric(monitored["strike"], errors="coerce") - pd.to_numeric(monitored[price_col], errors="coerce"),
            monitored["distance_to_strike"],
        )

    unmatched_mask = monitored["market_prob"].isna()
    if unmatched_mask.any():
        inferred_yes_prob = monitored.loc[unmatched_mask].apply(
            lambda r: estimate_terminal_probability_from_inputs(
                price=r.get(price_col),
                strike=r.get("strike"),
                hours_left=r.get("hours_left"),
                vol=r.get("adjusted_volatility") if pd.notna(r.get("adjusted_volatility")) else r.get("volatility"),
                drift=0.0,
            ),
            axis=1,
        )
        monitored.loc[unmatched_mask, "market_prob"] = inferred_yes_prob
        monitored.loc[unmatched_mask, "decision_prob"] = monitored.loc[unmatched_mask, "decision_prob"].where(
            monitored.loc[unmatched_mask, "decision_prob"].notna(),
            inferred_yes_prob,
        )
        monitored.loc[unmatched_mask, "fair_prob_terminal"] = monitored.loc[unmatched_mask, "fair_prob_terminal"].where(
            monitored.loc[unmatched_mask, "fair_prob_terminal"].notna(),
            inferred_yes_prob,
        )
        monitored.loc[unmatched_mask, "fair_prob_blended"] = monitored.loc[unmatched_mask, "fair_prob_blended"].where(
            monitored.loc[unmatched_mask, "fair_prob_blended"].notna(),
            inferred_yes_prob,
        )
        monitored.loc[unmatched_mask, "confidence"] = monitored.loc[unmatched_mask, "confidence"].fillna("LOW")

        action_upper_unmatched = _safe_upper_series(monitored.loc[unmatched_mask], "action")
        buy_yes_idx = action_upper_unmatched[action_upper_unmatched == "BUY_YES"].index
        buy_no_idx = action_upper_unmatched[action_upper_unmatched == "BUY_NO"].index

        if len(buy_yes_idx) > 0:
            monitored.loc[buy_yes_idx, "edge_yes"] = (
                pd.to_numeric(monitored.loc[buy_yes_idx, "decision_prob"], errors="coerce")
                - pd.to_numeric(monitored.loc[buy_yes_idx, "market_prob"], errors="coerce")
            )
        if len(buy_no_idx) > 0:
            monitored.loc[buy_no_idx, "edge_no"] = (
                (1.0 - pd.to_numeric(monitored.loc[buy_no_idx, "decision_prob"], errors="coerce"))
                - (1.0 - pd.to_numeric(monitored.loc[buy_no_idx, "market_prob"], errors="coerce"))
            )

    action_upper = _safe_upper_series(monitored, "action")
    yes_mask = action_upper == "BUY_YES"
    no_mask = action_upper == "BUY_NO"

    derived_current_price = pd.to_numeric(monitored["current_position_price"], errors="coerce").copy()
    current_price_series = pd.to_numeric(monitored["current_price"], errors="coerce")
    derived_current_price = derived_current_price.where(derived_current_price.notna(), current_price_series)

    if "last_price_yes" in monitored.columns:
        derived_current_price = derived_current_price.where(
            ~(yes_mask & derived_current_price.isna()),
            pd.to_numeric(monitored["last_price_yes"], errors="coerce"),
        )
    if "last_price_no" in monitored.columns:
        derived_current_price = derived_current_price.where(
            ~(no_mask & derived_current_price.isna()),
            pd.to_numeric(monitored["last_price_no"], errors="coerce"),
        )
    if "bid_yes" in monitored.columns and "ask_yes" in monitored.columns:
        yes_mid = (
            pd.to_numeric(monitored["bid_yes"], errors="coerce")
            + pd.to_numeric(monitored["ask_yes"], errors="coerce")
        ) / 2.0
        derived_current_price = derived_current_price.where(
            ~(yes_mask & derived_current_price.isna()),
            yes_mid,
        )
    if "bid_no" in monitored.columns and "ask_no" in monitored.columns:
        no_mid = (
            pd.to_numeric(monitored["bid_no"], errors="coerce")
            + pd.to_numeric(monitored["ask_no"], errors="coerce")
        ) / 2.0
        derived_current_price = derived_current_price.where(
            ~(no_mask & derived_current_price.isna()),
            no_mid,
        )

    monitored["current_position_price"] = pd.to_numeric(derived_current_price, errors="coerce")

    original_entry_price = pd.to_numeric(monitored["entry_price"], errors="coerce").copy()
    resolved_entry_price = monitored.apply(_resolve_entry_price, axis=1)
    monitored["entry_price"] = original_entry_price.where(
        original_entry_price.notna() & (original_entry_price > 0),
        resolved_entry_price,
    )

    entry_price_series = pd.to_numeric(monitored["entry_price"], errors="coerce")
    contracts_series = pd.to_numeric(monitored["contracts"], errors="coerce")
    current_price_series = pd.to_numeric(monitored["current_position_price"], errors="coerce")

    computed_unrealized_pnl = np.where(
        entry_price_series.notna()
        & (entry_price_series > 0)
        & current_price_series.notna()
        & contracts_series.notna()
        & (contracts_series > 0),
        (current_price_series - entry_price_series) * contracts_series,
        np.nan,
    )

    existing_unrealized_pnl = pd.to_numeric(monitored["unrealized_pnl"], errors="coerce")
    monitored["unrealized_pnl"] = existing_unrealized_pnl.where(
        existing_unrealized_pnl.notna(),
        pd.Series(computed_unrealized_pnl, index=monitored.index),
    )

    monitored["entry_fair_prob"] = pd.to_numeric(monitored.get("entry_fair_prob"), errors="coerce")
    monitored["entry_fair_prob"] = monitored["entry_fair_prob"].where(
        monitored["entry_fair_prob"].notna(),
        pd.to_numeric(monitored.get("market_prob"), errors="coerce"),
    )

    current_fair_yes = _get_current_fair_yes(monitored)
    entry_fair_yes = pd.to_numeric(monitored.get("entry_fair_prob"), errors="coerce")

    monitored["fair_prob_drop"] = np.where(
        action_upper == "BUY_YES",
        entry_fair_yes - current_fair_yes,
        np.where(action_upper == "BUY_NO", current_fair_yes - entry_fair_yes, np.nan),
    )
    monitored["current_edge"] = np.where(
        action_upper == "BUY_YES",
        pd.to_numeric(monitored.get("edge_yes"), errors="coerce"),
        np.where(
            action_upper == "BUY_NO",
            pd.to_numeric(monitored.get("edge_no"), errors="coerce"),
            pd.to_numeric(monitored.get("selected_edge"), errors="coerce"),
        ),
    )
    monitored["current_fair_side_price"] = np.where(
        action_upper == "BUY_YES",
        current_fair_yes,
        np.where(action_upper == "BUY_NO", 1.0 - current_fair_yes, np.nan),
    )
    monitored["fair_price_gap"] = (
        pd.to_numeric(monitored["current_fair_side_price"], errors="coerce")
        - pd.to_numeric(monitored["current_position_price"], errors="coerce")
    )

    existing_pnl_pct = pd.to_numeric(monitored["pnl_pct"], errors="coerce")
    computed_pnl_pct = np.where(
        entry_price_series.notna()
        & current_price_series.notna()
        & (entry_price_series > 0),
        (current_price_series - entry_price_series) / entry_price_series,
        np.nan,
    )
    monitored["pnl_pct"] = existing_pnl_pct.where(
        existing_pnl_pct.notna(),
        pd.Series(computed_pnl_pct, index=monitored.index),
    )

    classifications = monitored.apply(_classify_held_position, axis=1, result_type="expand")
    for col in classifications.columns:
        monitored[col] = classifications[col]

    return monitored


def evaluate_exit_rules(monitored_positions_df, config):
    import numpy as np
    import pandas as pd

    if monitored_positions_df is None or len(monitored_positions_df) == 0:
        return pd.DataFrame()

    df = monitored_positions_df.copy()
    exit_cfg = config.get("exit", {}) or {}

    hold_to_expiry_enabled = bool(exit_cfg.get("hold_to_expiry_enabled", True))
    hold_to_expiry_hours_left = float(exit_cfg.get("hold_to_expiry_hours_left", 2.0))
    emergency_stop_loss_pct = float(exit_cfg.get("emergency_stop_loss_pct", -0.85))
    emergency_price_move = float(exit_cfg.get("emergency_price_move", -0.35))
    max_exit_spread = float(exit_cfg.get("max_exit_spread", 0.04))

    def _safe_float(v, default=np.nan):
        try:
            if v is None:
                return default
            v = float(v)
            if np.isnan(v):
                return default
            return v
        except Exception:
            return default

    def _safe_bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        if isinstance(v, str):
            return v.strip().lower() in {"true", "1", "yes", "y", "t"}
        try:
            return bool(v)
        except Exception:
            return False

    def _resolve_effective_entry_price(row):
        entry_price = _safe_float(row.get("entry_price"), np.nan)
        if pd.notna(entry_price) and 0 < entry_price <= 1:
            return entry_price

        position_cost = _safe_float(row.get("position_cost"), np.nan)
        contracts = _safe_float(row.get("contracts"), np.nan)
        if pd.notna(position_cost) and pd.notna(contracts) and contracts > 0:
            inferred = position_cost / contracts
            if inferred > 1:
                inferred = inferred / 100.0
            if 0 < inferred <= 1:
                return inferred

        for col in ["avg_entry_price", "fill_price", "price"]:
            val = _safe_float(row.get(col), np.nan)
            if pd.notna(val) and val > 1:
                val = val / 100.0
            if pd.notna(val) and 0 < val <= 1:
                return val

        current_price = _safe_float(row.get("current_position_price", row.get("current_price")), np.nan)
        if pd.notna(current_price) and 0 < current_price <= 1:
            return current_price
        return np.nan

    def _safe_side_from_row(row):
        selected_side = str(row.get("selected_side", "") or "").strip().upper()
        side_norm = str(row.get("side_norm", "") or "").strip().upper()
        action = str(row.get("action", "") or "").strip().upper()
        if selected_side in {"YES", "NO"}:
            return selected_side
        if side_norm in {"YES", "NO"}:
            return side_norm
        if action == "BUY_YES":
            return "YES"
        if action == "BUY_NO":
            return "NO"
        return ""

    def _exit_prices_from_row(row, held_side):
        if held_side == "YES":
            bid_price = _safe_float(row.get("selected_bid", row.get("current_bid", row.get("bid_yes"))), np.nan)
            ask_price = _safe_float(row.get("selected_ask", row.get("current_ask", row.get("ask_yes"))), np.nan)
        elif held_side == "NO":
            bid_price = _safe_float(row.get("selected_bid", row.get("current_bid", row.get("bid_no"))), np.nan)
            ask_price = _safe_float(row.get("selected_ask", row.get("current_ask", row.get("ask_no"))), np.nan)
        else:
            bid_price = _safe_float(row.get("selected_bid", row.get("current_bid")), np.nan)
            ask_price = _safe_float(row.get("selected_ask", row.get("current_ask")), np.nan)
        return bid_price, ask_price

    def _rule_eval(row):
        contracts = _safe_float(row.get("contracts"), 0.0)
        current_price = _safe_float(row.get("current_position_price", row.get("current_price")), np.nan)
        entry_price = _resolve_effective_entry_price(row)
        hours_left = _safe_float(row.get("hours_left"), np.nan)
        held_side = _safe_side_from_row(row)

        held_state = str(row.get("held_decision_state", "") or "").strip().upper()
        exit_reason_existing = str(row.get("exit_reason", "") or "").strip()
        should_exit_existing = _safe_bool(row.get("should_exit"))

        pnl_ratio = _safe_float(row.get("pnl_pct"), np.nan)
        if pd.isna(pnl_ratio) and pd.notna(entry_price) and entry_price > 0 and pd.notna(current_price):
            pnl_ratio = (current_price - entry_price) / entry_price

        unrealized_pnl = _safe_float(row.get("unrealized_pnl"), np.nan)
        if pd.isna(unrealized_pnl) and pd.notna(entry_price) and entry_price > 0 and pd.notna(current_price) and pd.notna(contracts) and contracts > 0:
            unrealized_pnl = (current_price - entry_price) * contracts

        price_move = np.nan
        if pd.notna(entry_price) and pd.notna(current_price):
            price_move = current_price - entry_price

        current_edge = _safe_float(row.get("current_edge"), np.nan)
        fair_gap = _safe_float(row.get("fair_price_gap"), np.nan)

        bid_price, ask_price = _exit_prices_from_row(row, held_side)
        spread = ask_price - bid_price if pd.notna(bid_price) and pd.notna(ask_price) else np.nan

        out = {
            "should_exit": False,
            "exit_reason": "HOLD_TO_EXPIRY",
            "exit_state": "HOLD",
            "exit_priority": 0,
            "pnl_ratio": pnl_ratio,
            "unrealized_pnl_real": unrealized_pnl,
            "effective_entry_price": entry_price,
            "held_current_edge": current_edge,
            "held_fair_gap": fair_gap,
            "exit_bid_price": bid_price,
            "exit_ask_price": ask_price,
            "exit_spread": spread,
            "max_exit_spread": max_exit_spread,
            "hours_left": hours_left,
            "price_move": price_move,
        }

        # Respect upstream emergency classifications if they already fired.
        if should_exit_existing or held_state.startswith("EXIT_"):
            out.update({
                "should_exit": True,
                "exit_reason": exit_reason_existing or held_state or "CLASSIFICATION_EXIT",
                "exit_state": held_state or "CLASSIFICATION_EXIT",
                "exit_priority": 100,
            })
            return pd.Series(out)

        if pd.isna(current_price) or pd.isna(entry_price) or entry_price <= 0:
            out.update({
                "should_exit": False,
                "exit_reason": "INVALID_EXIT_DATA_HOLD",
                "exit_state": "HOLD_INVALID_DATA",
            })
            return pd.Series(out)

        # Expiry / settlement handoff.
        if pd.notna(hours_left) and hours_left <= 0:
            out.update({
                "should_exit": True,
                "exit_reason": "CONTRACT_RESOLVED",
                "exit_state": "EXIT_RESOLVED",
                "exit_priority": 110,
            })
            return pd.Series(out)

        # For daily binaries, the default behavior is to hold through the final window.
        if hold_to_expiry_enabled and pd.notna(hours_left) and hours_left <= hold_to_expiry_hours_left:
            # Only break glass for extreme damage.
            if pd.notna(pnl_ratio) and pnl_ratio <= emergency_stop_loss_pct:
                out.update({
                    "should_exit": True,
                    "exit_reason": f"EXIT_EMERGENCY_STOP pnl_ratio={pnl_ratio:.4f}",
                    "exit_state": "EXIT_EMERGENCY_STOP",
                    "exit_priority": 95,
                })
                return pd.Series(out)

            if pd.notna(price_move) and price_move <= emergency_price_move:
                out.update({
                    "should_exit": True,
                    "exit_reason": f"EXIT_EMERGENCY_PRICE_MOVE price_move={price_move:.4f}",
                    "exit_state": "EXIT_EMERGENCY_PRICE_MOVE",
                    "exit_priority": 94,
                })
                return pd.Series(out)

            out.update({
                "should_exit": False,
                "exit_reason": "HOLD_TO_EXPIRY_WINDOW",
                "exit_state": "HOLD_TO_EXPIRY_WINDOW",
                "exit_priority": 0,
            })
            return pd.Series(out)

        # Outside the final hold window, still be very conservative.
        if pd.notna(pnl_ratio) and pnl_ratio <= emergency_stop_loss_pct:
            out.update({
                "should_exit": True,
                "exit_reason": f"EXIT_EMERGENCY_STOP pnl_ratio={pnl_ratio:.4f}",
                "exit_state": "EXIT_EMERGENCY_STOP",
                "exit_priority": 95,
            })
            return pd.Series(out)

        if pd.notna(price_move) and price_move <= emergency_price_move:
            out.update({
                "should_exit": True,
                "exit_reason": f"EXIT_EMERGENCY_PRICE_MOVE price_move={price_move:.4f}",
                "exit_state": "EXIT_EMERGENCY_PRICE_MOVE",
                "exit_priority": 94,
            })
            return pd.Series(out)

        # Do not force exits into bad spreads unless this is truly catastrophic.
        if pd.notna(spread) and spread > max_exit_spread:
            out.update({
                "should_exit": False,
                "exit_reason": f"HOLD_SPREAD_TOO_WIDE spread={spread:.4f}",
                "exit_state": "HOLD_SPREAD_TOO_WIDE",
                "exit_priority": 0,
            })
            return pd.Series(out)

        out.update({
            "should_exit": False,
            "exit_reason": "HOLD_TO_EXPIRY_DEFAULT",
            "exit_state": "HOLD",
            "exit_priority": 0,
        })
        return pd.Series(out)

    exit_eval = df.apply(_rule_eval, axis=1)
    for col in exit_eval.columns:
        df[col] = exit_eval[col]
    return df

