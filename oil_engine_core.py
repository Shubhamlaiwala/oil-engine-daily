import logging
import math
import os
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf
from scipy.stats import norm

ET = ZoneInfo("America/New_York")

def _dedup_signature_frame(df):
    if df is None or df.empty:
        return pd.Series([], dtype=str)

    cols = [
        "contract_ticker",
        "selected_side",
    ]

    cols = [c for c in cols if c in df.columns]

    if not cols:
        return pd.Series([], dtype=str)

    return df[cols].astype(str).agg("|".join, axis=1)

def load_config(path: str = "settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def current_time_et() -> datetime:
    return datetime.now(ET)


def format_et(dt_obj):
    if dt_obj is None:
        return None

    if isinstance(dt_obj, pd.Timestamp):
        dt_obj = dt_obj.to_pydatetime()

    if not isinstance(dt_obj, datetime):
        return str(dt_obj)

    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=ET)
    else:
        dt_obj = dt_obj.astimezone(ET)

    return dt_obj.strftime("%Y-%m-%d %H:%M:%S %Z")


def parse_iso_to_et(value):
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()

    if isinstance(value, datetime):
        dt_obj = value
    else:
        s = str(value).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        try:
            dt_obj = datetime.fromisoformat(s)
        except Exception:
            return None

    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=ZoneInfo("UTC"))

    return dt_obj.astimezone(ET)


def parse_unix_ts_to_et(value):
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=ZoneInfo("UTC")).astimezone(ET)
    except Exception:
        return None


def get_fallback_resolution_time(config):
    now = current_time_et()
    mode = config["contract"].get("fallback_resolution_mode", "today_5pm_et")

    if mode == "today_5pm_et":
        dt_obj = now.replace(hour=17, minute=0, second=0, microsecond=0)
        if now >= dt_obj:
            dt_obj += timedelta(days=1)
        return dt_obj

    if mode == "today_11pm_et":
        dt_obj = now.replace(hour=23, minute=0, second=0, microsecond=0)
        if now >= dt_obj:
            dt_obj += timedelta(days=1)
        return dt_obj

    return now + timedelta(hours=4)


def get_hours_left(resolution_time_et):
    if resolution_time_et is None:
        return 0.0

    if isinstance(resolution_time_et, pd.Timestamp):
        resolution_time_et = resolution_time_et.to_pydatetime()

    if isinstance(resolution_time_et, str):
        resolution_time_et = parse_iso_to_et(resolution_time_et)

    if resolution_time_et is None:
        return 0.0

    if resolution_time_et.tzinfo is None:
        resolution_time_et = resolution_time_et.replace(tzinfo=ET)
    else:
        resolution_time_et = resolution_time_et.astimezone(ET)

    return max((resolution_time_et - current_time_et()).total_seconds() / 3600.0, 0.0)


def classify_trading_phase(hours_left, config):
    phase_cfg = config.get("trading_phases", {}) or {}

    observe_only_above_hours = float(phase_cfg.get("observe_only_above_hours", 72.0))
    active_trading_above_hours = float(phase_cfg.get("active_trading_above_hours", 12.0))

    try:
        hours_left = float(hours_left)
    except Exception:
        return "UNKNOWN"

    if hours_left > observe_only_above_hours:
        return "OBSERVE_ONLY"

    if hours_left > active_trading_above_hours:
        return "ACTIVE_TRADING"

    return "CLOSE_ONLY"


class TTLCache:
    def __init__(self):
        self.value = None
        self.timestamp = None
        self.source = None
        self.extra = None


LAST_PRICE_CACHE = TTLCache()
LAST_VOL_CACHE = TTLCache()


# =========================================================
# MARKET DATA HELPERS
# =========================================================
def standardize_history_df(df, datetime_col, close_col, source_name):
    if df is None or df.empty:
        return None

    out = df.copy()
    out["datetime"] = pd.to_datetime(out[datetime_col], errors="coerce")
    out["close"] = pd.to_numeric(out[close_col], errors="coerce")
    out = out[["datetime", "close"]].dropna().sort_values("datetime").reset_index(drop=True)

    if out.empty:
        return None

    try:
        if out["datetime"].dt.tz is None:
            out["datetime"] = out["datetime"].dt.tz_localize("UTC").dt.tz_convert(ET)
        else:
            out["datetime"] = out["datetime"].dt.tz_convert(ET)
    except Exception:
        return None

    out["datetime_source_tz"] = out["datetime"]
    out["source"] = source_name
    return out[["datetime", "datetime_source_tz", "close", "source"]]


# =========================================================
# OIL PRICE SOURCES
# =========================================================
OIL_YAHOO_SYMBOL_CANDIDATES = [
    "CL=F",
    "USO",
]


def _extract_close_series_from_yahoo_history(df):
    if df is None or df.empty:
        return pd.Series(dtype=float)

    working = df.copy()
    if isinstance(working.columns, pd.MultiIndex):
        working.columns = [c[0] if isinstance(c, tuple) else c for c in working.columns]

    if "Close" not in working.columns and "Adj Close" in working.columns:
        working["Close"] = working["Adj Close"]

    if "Close" not in working.columns:
        return pd.Series(dtype=float)

    close_series = pd.to_numeric(working["Close"], errors="coerce").dropna()
    return close_series



def get_live_oil_price_yahoo(symbols=None):
    symbols = symbols or OIL_YAHOO_SYMBOL_CANDIDATES
    errors = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d", interval="1m")
            close_series = _extract_close_series_from_yahoo_history(data)

            if close_series.empty:
                data = ticker.history(period="1d", interval="5m")
                close_series = _extract_close_series_from_yahoo_history(data)

            if close_series.empty:
                data = ticker.history(period="1mo", interval="1d")
                close_series = _extract_close_series_from_yahoo_history(data)

            if close_series.empty:
                raise ValueError(f"Yahoo Finance returned no usable close data for {symbol}")

            return float(close_series.iloc[-1]), f"YAHOO:{symbol}"
        except Exception as e:
            errors.append(f"{symbol} -> {e}")
            print(f"Warning: YAHOO live oil price fetch failed for {symbol}: {e}")

    raise RuntimeError("Yahoo live oil price unavailable from all symbol fallbacks: " + " | ".join(errors))



def get_live_oil_price_twelvedata():
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        raise ValueError("TWELVEDATA_API_KEY is not set.")

    url = "https://api.twelvedata.com/price"
    params = {
        "symbol": "USO",
        "apikey": api_key,
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    if "price" not in data:
        raise ValueError(f"Unexpected Twelve Data price response: {data}")

    return float(data["price"]), "TWELVEDATA:USO"



def get_live_oil_price_from_chain(config):
    providers = [
        ("YAHOO", get_live_oil_price_yahoo),
        ("TWELVEDATA", get_live_oil_price_twelvedata),
    ]

    for source_name, fn in providers:
        try:
            price, source_detail = fn()
            return float(price), source_detail or source_name
        except Exception as e:
            print(f"Warning: {source_name} live oil price fetch failed: {e}")

    raise RuntimeError("Live oil price unavailable from all providers.")



def get_live_oil_price_cached(config):
    now = time.time()
    ttl = config["runtime"]["price_cache_seconds"]

    if (
        LAST_PRICE_CACHE.value is not None
        and LAST_PRICE_CACHE.timestamp is not None
        and (now - LAST_PRICE_CACHE.timestamp <= ttl)
    ):
        return LAST_PRICE_CACHE.value, LAST_PRICE_CACHE.source or "CACHE"

    price, source = get_live_oil_price_from_chain(config)

    LAST_PRICE_CACHE.value = price
    LAST_PRICE_CACHE.timestamp = now
    LAST_PRICE_CACHE.source = source
    LAST_PRICE_CACHE.extra = None

    return price, source



def get_oil_price_history_yahoo(config, period="1d", interval="1m"):
    symbols = config.get("market_data", {}).get("yahoo_oil_symbols", OIL_YAHOO_SYMBOL_CANDIDATES)

    normalized_periods = [period, "5d", "1mo", "3mo"]
    normalized_intervals = [interval, "5m", "1d"]

    for symbol in symbols:
        for candidate_period in normalized_periods:
            for candidate_interval in normalized_intervals:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=candidate_period, interval=candidate_interval)

                    if df is None or df.empty:
                        continue

                    df = df.reset_index().copy()
                    time_col = "Datetime" if "Datetime" in df.columns else "Date"

                    out = standardize_history_df(df, time_col, "Close", f"YAHOO:{symbol}")
                    if out is None or out.empty:
                        continue

                    return out

                except Exception as e:
                    print(
                        f"Warning: Yahoo historical oil fetch failed for {symbol} period={candidate_period} interval={candidate_interval}: {e}"
                    )

    print("Warning: Yahoo Finance returned no usable historical oil data from any symbol fallback")
    return None



def get_oil_price_history_twelvedata(config, interval="1min", outputsize=240):
    try:
        api_key = os.getenv("TWELVEDATA_API_KEY")
        if not api_key:
            raise ValueError("TWELVEDATA_API_KEY is not set.")

        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": "USO",
            "interval": interval,
            "outputsize": outputsize,
            "timezone": "America/New_York",
            "apikey": api_key,
        }

        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        values = data.get("values")
        if not values:
            print(f"Warning: Twelve Data returned no historical values: {data}")
            return None

        df = pd.DataFrame(values)
        out = standardize_history_df(df, "datetime", "close", "TWELVEDATA:USO")
        if out is None or out.empty:
            print("Warning: Twelve Data history could not be standardized")
            return None

        return out

    except Exception as e:
        print(f"Warning: Twelve Data historical fetch failed: {e}")
        return None



def get_oil_history_from_chain(config):
    providers = [
        ("YAHOO", lambda: get_oil_price_history_yahoo(config)),
        (
            "TWELVEDATA",
            lambda: get_oil_price_history_twelvedata(
                config,
                interval=config["model"].get("vol_interval", "1min"),
                outputsize=config["model"].get("vol_lookback_bars", 240),
            ),
        ),
    ]

    for source_name, fn in providers:
        try:
            history_df = fn()
            if history_df is not None and not history_df.empty:
                derived_source = None
                if "source" in history_df.columns:
                    non_null_sources = history_df["source"].dropna().astype(str)
                    if not non_null_sources.empty:
                        derived_source = non_null_sources.iloc[-1]
                return history_df, derived_source or source_name
        except Exception as e:
            print(f"Warning: {source_name} historical fetch failed: {e}")

    return None, None

def annualize_realized_std(realized_std, config):
    if pd.isna(realized_std) or realized_std <= 0:
        return config["model"]["annualized_volatility_fallback"]
    return float(realized_std * math.sqrt(config["model"]["vol_trading_periods_per_year"]))


def compute_realized_volatility(history_df, config):
    df = history_df.copy()

    if len(df) < 3:
        fallback = config["model"]["annualized_volatility_fallback"]
        return {
            "short_vol": fallback,
            "medium_vol": fallback,
            "blended_vol": fallback,
            "history_df": df,
        }

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    returns = df["log_return"].dropna().reset_index(drop=True)

    short_window = min(config["model"]["vol_short_window"], len(returns))
    medium_window = min(config["model"]["vol_medium_window"], len(returns))

    short_std = returns.tail(short_window).std() if short_window >= 2 else np.nan
    medium_std = returns.tail(medium_window).std() if medium_window >= 2 else np.nan

    short_vol = annualize_realized_std(short_std, config)
    medium_vol = annualize_realized_std(medium_std, config)

    blended_vol = (
        config["model"]["vol_short_weight"] * short_vol
        + config["model"]["vol_medium_weight"] * medium_vol
    )
    blended_vol = float(
        np.clip(
            blended_vol,
            config["model"]["min_annualized_vol"],
            config["model"]["max_annualized_vol"],
        )
    )

    return {
        "short_vol": float(short_vol),
        "medium_vol": float(medium_vol),
        "blended_vol": blended_vol,
        "history_df": df,
    }


def compute_dynamic_drift(history_df, config):
    if history_df is None or history_df.empty or len(history_df) < 3:
        return float(config["model"].get("drift", 0.0))

    df = history_df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    returns = df["log_return"].dropna().reset_index(drop=True)

    if returns.empty:
        return float(config["model"].get("drift", 0.0))

    drift_window = min(int(config["model"].get("drift_window", 30)), len(returns))
    if drift_window < 2:
        return float(config["model"].get("drift", 0.0))

    mean_return = returns.tail(drift_window).mean()
    annualized_drift = float(mean_return * config["model"]["vol_trading_periods_per_year"])

    min_drift = float(config["model"].get("min_annualized_drift", -0.25))
    max_drift = float(config["model"].get("max_annualized_drift", 0.25))

    return float(np.clip(annualized_drift, min_drift, max_drift))


def compute_momentum_features(price, history_df, config):
    model_cfg = config.get("model", {}) or {}
    short_window = max(2, int(model_cfg.get("momentum_short_window", 3)))
    medium_window = max(short_window + 1, int(model_cfg.get("momentum_medium_window", 5)))

    default = {
        "oil_momentum_short": 0.0,
        "oil_momentum_medium": 0.0,
        "oil_momentum_regime": "NEUTRAL",
    }

    try:
        if history_df is None or history_df.empty or "close" not in history_df.columns:
            return default

        closes = pd.to_numeric(history_df["close"], errors="coerce").dropna().reset_index(drop=True)
        if len(closes) < medium_window:
            return default

        latest_price = float(price) if price is not None and not pd.isna(price) else float(closes.iloc[-1])
        short_base = float(closes.iloc[-short_window])
        medium_base = float(closes.iloc[-medium_window])

        if short_base <= 0 or medium_base <= 0:
            return default

        short_mom = (latest_price / short_base) - 1.0
        medium_mom = (latest_price / medium_base) - 1.0

        regime_threshold = float(model_cfg.get("momentum_regime_threshold", 0.0015))
        if short_mom >= regime_threshold and medium_mom >= 0:
            regime = "BULLISH"
        elif short_mom <= -regime_threshold and medium_mom <= 0:
            regime = "BEARISH"
        else:
            regime = "NEUTRAL"

        return {
            "oil_momentum_short": float(short_mom),
            "oil_momentum_medium": float(medium_mom),
            "oil_momentum_regime": regime,
        }
    except Exception:
        return default


def get_realized_volatility_cached(config):
    now = time.time()
    ttl = config["runtime"]["vol_cache_seconds"]

    if (
        LAST_VOL_CACHE.value is not None
        and LAST_VOL_CACHE.timestamp is not None
        and (now - LAST_VOL_CACHE.timestamp <= ttl)
    ):
        return LAST_VOL_CACHE.value, "CACHE", LAST_VOL_CACHE.extra

    history_df, source = get_oil_history_from_chain(config)
    if history_df is None or history_df.empty:
        raise RuntimeError("Live oil history unavailable from all providers.")

    vol_stats = compute_realized_volatility(history_df, config)

    LAST_VOL_CACHE.value = vol_stats
    LAST_VOL_CACHE.timestamp = now
    LAST_VOL_CACHE.source = source
    LAST_VOL_CACHE.extra = vol_stats["history_df"]

    return vol_stats, source, vol_stats["history_df"]


# =========================================================
# MODEL
# =========================================================
def time_to_expiry_years(hours_left):
    if hours_left is None:
        return 0.0
    try:
        hours_left = float(hours_left)
    except Exception:
        return 0.0
    return max(hours_left, 0.0) / (24.0 * 365.0)


def estimate_terminal_probability(price, strike, resolution_time_et, vol, drift, n_paths=None):
    hours_left = get_hours_left(resolution_time_et)
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
    prob = float(np.clip(prob, 0.0, 1.0))
    return prob


def estimate_touch_probability(price, strike, resolution_time_et, vol, drift, n_paths=2000, n_steps=40):
    hours_left = get_hours_left(resolution_time_et)
    T = time_to_expiry_years(hours_left)

    if T <= 0:
        return float(price >= strike)

    if strike <= 0 or price <= 0:
        return 0.5

    barrier = math.log(strike / price)
    if barrier <= 0:
        return 1.0

    sigma = max(vol, 1e-12)
    mu = drift - 0.5 * vol**2
    sigma_t = sigma * math.sqrt(T)

    term_1 = 1.0 - norm.cdf((barrier - mu * T) / sigma_t)
    term_2 = math.exp((2.0 * mu * barrier) / (sigma**2)) * norm.cdf((-barrier - mu * T) / sigma_t)
    prob = term_1 + term_2
    prob = float(np.clip(prob, 0.0, 1.0))

    terminal_prob = estimate_terminal_probability(price, strike, resolution_time_et, vol, drift)
    return float(max(prob, terminal_prob))


def detect_vol_regime(vol):
    if vol < 0.10:
        return "LOW_VOL"
    if vol < 0.25:
        return "NORMAL_VOL"
    return "HIGH_VOL"


def adjust_volatility_by_regime(vol, regime):
    if regime == "LOW_VOL":
        return vol * 0.95
    if regime == "HIGH_VOL":
        return vol * 1.10
    return vol


def compute_confidence(edge):
    edge = pd.to_numeric(edge, errors="coerce")
    if pd.isna(edge):
        return "LOW"
    if edge >= 0.20:
        return "HIGH"
    if edge >= 0.10:
        return "MEDIUM"
    return "LOW"


def _coalesce_price(primary, fallback):
    primary = pd.to_numeric(primary, errors="coerce")
    fallback = pd.to_numeric(fallback, errors="coerce")
    if pd.notna(primary):
        return float(primary)
    if pd.notna(fallback):
        return float(fallback)
    return np.nan


def _clip_price_01_99(value):
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return np.nan
    return float(np.clip(value, 0.01, 0.99))


def _has_sane_side_quote(ask_price):
    ask_price = pd.to_numeric(ask_price, errors="coerce")
    return bool(pd.notna(ask_price) and 0.0 <= float(ask_price) <= 1.0)


def compute_expected_values_from_row(row):
    decision_prob = pd.to_numeric(row.get("decision_prob"), errors="coerce")
    if pd.isna(decision_prob):
        return pd.Series({"ev_yes": np.nan, "ev_no": np.nan})

    decision_prob = float(np.clip(decision_prob, 0.0, 1.0))
    fair_yes = decision_prob
    fair_no = 1.0 - decision_prob

    market_yes = pd.to_numeric(row.get("market_prob_yes"), errors="coerce")
    market_no = pd.to_numeric(row.get("market_prob_no"), errors="coerce")

    if pd.isna(market_yes):
        market_yes = pd.to_numeric(row.get("ask_yes"), errors="coerce")
    if pd.isna(market_no):
        market_no = pd.to_numeric(row.get("ask_no"), errors="coerce")

    ev_yes = fair_yes - market_yes if pd.notna(market_yes) else np.nan
    ev_no = fair_no - market_no if pd.notna(market_no) else np.nan

    return pd.Series(
        {
            "ev_yes": ev_yes,
            "ev_no": ev_no,
        }
    )


# =========================================================
# MARKET / LIQUIDITY
# =========================================================
def compute_quote_spread(row, side="yes"):
    bid = row.get(f"bid_{side}")
    ask = row.get(f"ask_{side}")
    if bid is None or ask is None or pd.isna(bid) or pd.isna(ask):
        return np.nan
    return float(max(ask - bid, 0.0))


def apply_liquidity_filters(df, config):
    if df is None or df.empty:
        return df

    out = df.copy()
    out["yes_spread"] = out.apply(lambda r: compute_quote_spread(r, "yes"), axis=1)
    out["no_spread"] = out.apply(lambda r: compute_quote_spread(r, "no"), axis=1)

    min_bid = float(config["filters"]["min_bid"])
    min_ask = float(config["filters"]["min_ask"])
    require_two_sided = bool(config["filters"]["require_two_sided_quotes"])
    max_yes_spread = float(config["filters"]["max_yes_spread"])
    max_no_spread = float(config["filters"]["max_no_spread"])

    out["has_valid_yes_quote"] = (
        out["bid_yes"].fillna(0).ge(min_bid)
        & out["ask_yes"].fillna(0).ge(min_ask)
    )
    out["has_valid_no_quote"] = (
        out["bid_no"].fillna(0).ge(min_bid)
        & out["ask_no"].fillna(0).ge(min_ask)
    )

    out["yes_spread_ok"] = out["yes_spread"].fillna(np.inf).le(max_yes_spread)
    out["no_spread_ok"] = out["no_spread"].fillna(np.inf).le(max_no_spread)

    out["tradable_yes"] = out["has_valid_yes_quote"] & out["yes_spread_ok"]
    out["tradable_no"] = out["has_valid_no_quote"] & out["no_spread_ok"]

    if require_two_sided:
        out["is_quoted"] = out["has_valid_yes_quote"] & out["has_valid_no_quote"]
        out["is_liquid"] = out["tradable_yes"] & out["tradable_no"]
    else:
        out["is_quoted"] = out["has_valid_yes_quote"] | out["has_valid_no_quote"]
        out["is_liquid"] = out["tradable_yes"] | out["tradable_no"]

    return out


def enforce_monotonic_probabilities(df, prob_cols=None):
    if df is None or df.empty:
        return df

    out = df.sort_values("strike").reset_index(drop=True).copy()
    prob_cols = prob_cols or ["fair_prob_terminal", "fair_prob_touch", "fair_prob_blended", "decision_prob"]

    for col in prob_cols:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce").astype(float).to_numpy()
        values = np.minimum.accumulate(values)
        values = np.clip(values, 0.0, 1.0)
        out[col] = values

    return out


def _normalized_candidate_score(row):
    selected_edge = pd.to_numeric(pd.Series([row.get("selected_edge")]), errors="coerce").iloc[0]
    selected_ev = pd.to_numeric(pd.Series([row.get("selected_ev")]), errors="coerce").iloc[0]
    side_prob_support = pd.to_numeric(pd.Series([row.get("side_prob_support")]), errors="coerce").iloc[0]

    actionable = bool(str(row.get("action", "")).strip().upper() in {"BUY_YES", "BUY_NO"})
    executable = bool(row.get("selected_executable_now", False))
    market_too_wide = bool(row.get("market_too_wide", False))
    entry_style = str(row.get("entry_style", "") or "").strip().upper()
    confidence = str(row.get("confidence", "") or "").strip().upper()

    edge_component = float(np.clip(selected_edge if pd.notna(selected_edge) else 0.0, 0.0, 1.0))
    ev_component = float(np.clip(selected_ev if pd.notna(selected_ev) else 0.0, 0.0, 1.0))
    prob_component = float(np.clip(side_prob_support if pd.notna(side_prob_support) else 0.0, 0.0, 1.0))

    score = (
        0.50 * edge_component
        + 0.20 * ev_component
        + 0.20 * prob_component
        + (0.08 if actionable else 0.0)
        + (0.05 if executable else 0.0)
    )

    if entry_style in {"LIMIT_MAKER", "WATCHLIST_LIMIT", "MAKER_TARGET"}:
        score -= 0.08
    if market_too_wide:
        score -= 0.15
    if confidence == "LOW":
        score -= 0.08
    elif confidence == "MEDIUM":
        score -= 0.02

    return float(np.clip(score, 0.0, 1.0))


def rank_trade_candidates(df):
    if df is None or df.empty:
        return df

    ranked = df.copy()

    if "action" not in ranked.columns:
        ranked["action"] = "NO_TRADE"

    if "decision_state" not in ranked.columns:
        ranked["decision_state"] = "NOT_TRADABLE"

    if "selected_side" not in ranked.columns:
        ranked["selected_side"] = ""

    if "is_liquid" not in ranked.columns:
        ranked["is_liquid"] = True

    if "confidence" not in ranked.columns:
        ranked["confidence"] = "LOW"

    if "ev_yes" not in ranked.columns:
        ranked["ev_yes"] = 0.0
    if "ev_no" not in ranked.columns:
        ranked["ev_no"] = 0.0
    if "ev_yes_exec" not in ranked.columns:
        ranked["ev_yes_exec"] = ranked.get("ev_yes", 0.0)
    if "ev_no_exec" not in ranked.columns:
        ranked["ev_no_exec"] = ranked.get("ev_no", 0.0)
    if "selected_ev" not in ranked.columns:
        ranked["selected_ev"] = np.nan

    if "no_trade_reason" not in ranked.columns:
        ranked["no_trade_reason"] = ""

    if "market_too_wide" not in ranked.columns:
        ranked["market_too_wide"] = False

    if "executable_yes_now" not in ranked.columns:
        ranked["executable_yes_now"] = False
    if "executable_no_now" not in ranked.columns:
        ranked["executable_no_now"] = False
    if "selected_executable_now" not in ranked.columns:
        ranked["selected_executable_now"] = False

    if "decision_prob" not in ranked.columns:
        ranked["decision_prob"] = np.nan

    ranked["action_upper"] = ranked["action"].astype(str).str.upper()
    ranked["decision_state_upper"] = ranked["decision_state"].astype(str).str.upper()
    ranked["entry_style_upper"] = ranked.get("entry_style", "").astype(str).str.upper() if "entry_style" in ranked.columns else ""

    ranked["actionable"] = ranked["action_upper"].isin(["BUY_YES", "BUY_NO"])
    ranked["watchlist"] = ranked["decision_state_upper"].isin(["WAIT_FOR_PRICE", "PRICE_OK_BUT_SPREAD_TOO_WIDE"])

    ranked["selected_edge"] = pd.to_numeric(ranked.get("selected_edge"), errors="coerce")
    ranked["edge_yes"] = pd.to_numeric(ranked.get("edge_yes"), errors="coerce")
    ranked["edge_no"] = pd.to_numeric(ranked.get("edge_no"), errors="coerce")
    ranked["selected_ev"] = pd.to_numeric(ranked.get("selected_ev"), errors="coerce")
    ranked["decision_prob"] = pd.to_numeric(ranked.get("decision_prob"), errors="coerce")

    ranked["best_edge"] = ranked[["edge_yes", "edge_no"]].max(axis=1)

    if "distance_to_strike" in ranked.columns:
        ranked["abs_distance_to_strike"] = pd.to_numeric(ranked["distance_to_strike"], errors="coerce").abs()
    else:
        ranked["abs_distance_to_strike"] = np.nan

    confidence_rank_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    ranked["confidence_rank"] = ranked["confidence"].astype(str).str.upper().map(confidence_rank_map).fillna(0)

    side_prob_support = np.where(
        ranked["action_upper"] == "BUY_YES",
        ranked["decision_prob"],
        np.where(ranked["action_upper"] == "BUY_NO", 1.0 - ranked["decision_prob"], np.nan),
    )
    ranked["side_prob_support"] = pd.to_numeric(pd.Series(side_prob_support, index=ranked.index), errors="coerce")

    ranked["candidate_score"] = ranked.apply(_normalized_candidate_score, axis=1)

    ranked = ranked.sort_values(
        by=[
            "actionable",
            "selected_executable_now",
            "candidate_score",
            "selected_edge",
            "confidence_rank",
            "side_prob_support",
            "abs_distance_to_strike",
        ],
        ascending=[False, False, False, False, False, False, True],
    ).reset_index(drop=True)

    return ranked.drop(columns=["confidence_rank", "action_upper", "decision_state_upper", "entry_style_upper"], errors="ignore")


# =========================================================
# KALSHI HELPERS
# =========================================================
def parse_kalshi_url_parts(url):
    if not url:
        return None, None, None

    s = str(url).strip()
    m = re.search(r"/markets/([^/?#]+)/([^/?#]+)(?:/([^/?#]+))?", s, flags=re.IGNORECASE)
    if not m:
        return None, None, None

    series_ticker_raw = m.group(1).strip() if m.group(1) else None
    slug = m.group(2).strip() if m.group(2) else None
    maybe_event = m.group(3).strip() if m.group(3) else None

    series_ticker = series_ticker_raw.upper() if series_ticker_raw else None
    exact_market_ticker = maybe_event.upper() if maybe_event else None

    return series_ticker, slug, exact_market_ticker


def normalize_kalshi_price(value):
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if v < 0:
        return None
    if v > 1 and v <= 100:
        v = v / 100.0
    return max(0.0, min(1.0, v))


def first_non_null(d, keys):
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def slugify_text(s):
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def parse_kalshi_dt_to_et(value):
    dt_obj = parse_iso_to_et(value)
    if dt_obj is not None:
        return dt_obj
    dt_obj = parse_unix_ts_to_et(value)
    if dt_obj is not None:
        return dt_obj
    return None


def extract_kalshi_resolution_time_et(market, config):
    for key in [
        "close_time",
        "expiration_time",
        "latest_expiration_time",
        "expected_expiration_time",
        "settlement_time",
        "settlement_ts",
        "close_ts",
        "expiration_ts",
        "open_time",
    ]:
        dt_obj = parse_kalshi_dt_to_et(market.get(key))
        if dt_obj is not None:
            return dt_obj
    return get_fallback_resolution_time(config)


def extract_kalshi_strike(market):
    floor_strike = first_non_null(market, ["floor_strike"])
    cap_strike = first_non_null(market, ["cap_strike"])
    functional_strike = first_non_null(market, ["functional_strike"])

    for value in [floor_strike, cap_strike, functional_strike]:
        try:
            if value is not None and str(value).strip() != "":
                return float(value)
        except Exception:
            pass

    text = " ".join(
        [
            str(first_non_null(market, ["ticker"]) or ""),
            str(first_non_null(market, ["title"]) or ""),
            str(first_non_null(market, ["subtitle"]) or ""),
            str(first_non_null(market, ["yes_sub_title"]) or ""),
            str(first_non_null(market, ["yes_subtitle"]) or ""),
            str(first_non_null(market, ["rules_primary"]) or ""),
        ]
    )

    matches = re.findall(r"(\d{4,6}(?:\.\d+)?)", text)
    if matches:
        try:
            return float(matches[-1])
        except Exception:
            return None
    return None


def get_market_price_snapshot(market):
    yes_bid_raw = first_non_null(
        market,
        [
            "yes_bid_dollars",
            "yes_bid",
            "best_yes_bid",
            "bid_yes",
        ],
    )

    yes_ask_raw = first_non_null(
        market,
        [
            "yes_ask_dollars",
            "yes_ask",
            "best_yes_ask",
            "ask_yes",
        ],
    )

    no_bid_raw = first_non_null(
        market,
        [
            "no_bid_dollars",
            "no_bid",
            "best_no_bid",
            "bid_no",
        ],
    )

    no_ask_raw = first_non_null(
        market,
        [
            "no_ask_dollars",
            "no_ask",
            "best_no_ask",
            "ask_no",
        ],
    )

    last_yes_raw = first_non_null(
        market,
        [
            "yes_price_dollars",
            "yes_price",
            "last_price_dollars",
            "last_price",
            "last_price_yes",
            "price",
        ],
    )

    yes_bid = normalize_kalshi_price(yes_bid_raw)
    yes_ask = normalize_kalshi_price(yes_ask_raw)
    no_bid = normalize_kalshi_price(no_bid_raw)
    no_ask = normalize_kalshi_price(no_ask_raw)
    last_yes = normalize_kalshi_price(last_yes_raw)

    if yes_ask is None and last_yes is not None:
        yes_ask = last_yes

    if no_ask is None and last_yes is not None:
        no_ask = 1.0 - last_yes

    if yes_ask is None and no_ask is None:
        yes_ask = 0.5
        no_ask = 0.5

    if yes_bid is not None and yes_ask is not None:
        market_yes_prob = (yes_bid + yes_ask) / 2.0
        source = "mid_from_bid_ask"
    elif last_yes is not None:
        market_yes_prob = last_yes
        source = "last_price"
    elif yes_ask is not None:
        market_yes_prob = yes_ask
        source = "ask_only"
    elif yes_bid is not None:
        market_yes_prob = yes_bid
        source = "bid_only"
    else:
        market_yes_prob = 0.50
        source = "fallback_0.50"

    return {
        "bid_yes": yes_bid,
        "ask_yes": yes_ask,
        "bid_no": no_bid,
        "ask_no": no_ask,
        "last_price_yes": last_yes if last_yes is not None else market_yes_prob,
        "last_price_no": (1.0 - last_yes) if last_yes is not None else (1.0 - market_yes_prob),
        "market_yes_prob": market_yes_prob,
        "market_no_prob": 1.0 - market_yes_prob,
        "pricing_source": source,
    }


def kalshi_get(config, path, params=None):
    base = config["kalshi"]["api_base"].rstrip("/")
    url = f"{base}/{path.lstrip('/')}"

    runtime_cfg = config.get("runtime", {})
    timeout = int(runtime_cfg.get("request_timeout_seconds", 20))
    max_attempts = int(runtime_cfg.get("kalshi_retry_attempts", 4))
    base_backoff = float(runtime_cfg.get("kalshi_retry_backoff_seconds", 2.0))

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params or {}, timeout=timeout)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait_s = max(float(retry_after), base_backoff)
                    except Exception:
                        wait_s = base_backoff * attempt
                else:
                    wait_s = base_backoff * attempt

                logging.warning(
                    "Kalshi 429 rate limit | path=%s | params=%s | attempt=%s/%s | sleeping=%.1fs",
                    path,
                    params,
                    attempt,
                    max_attempts,
                    wait_s,
                )
                time.sleep(wait_s)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            last_error = e
            status_code = getattr(e.response, "status_code", None)

            if status_code in {500, 502, 503, 504} and attempt < max_attempts:
                wait_s = base_backoff * attempt
                logging.warning(
                    "Kalshi HTTP error | status=%s | path=%s | attempt=%s/%s | sleeping=%.1fs",
                    status_code,
                    path,
                    attempt,
                    max_attempts,
                    wait_s,
                )
                time.sleep(wait_s)
                continue

            raise

        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_attempts:
                wait_s = base_backoff * attempt
                logging.warning(
                    "Kalshi request exception | path=%s | attempt=%s/%s | sleeping=%.1fs | error=%s",
                    path,
                    attempt,
                    max_attempts,
                    wait_s,
                    e,
                )
                time.sleep(wait_s)
                continue
            raise

    if last_error:
        raise last_error

    raise RuntimeError(f"Kalshi GET failed for path={path}")


def kalshi_paginated_get(config, path, params=None, results_key=None, max_pages=10):
    params = dict(params or {})
    out = []
    cursor = None
    pages = 0
    page_pause = float(config.get("runtime", {}).get("kalshi_page_pause_seconds", 0.25))

    while pages < max_pages:
        call_params = dict(params)
        if cursor:
            call_params["cursor"] = cursor

        data = kalshi_get(config, path, params=call_params)
        batch = data.get(results_key, []) if results_key else []

        if batch:
            out.extend(batch)

        cursor = data.get("cursor")
        pages += 1

        if not cursor:
            break

        if page_pause > 0:
            time.sleep(page_pause)

    return out


def _select_active_event_ticker_from_markets(raw_markets, config):
    if not raw_markets:
        return None

    now_et = current_time_et()
    today_date = now_et.date()

    same_day_candidates = []
    future_candidates = []

    for market in raw_markets:
        event_ticker = market.get("event_ticker")
        if not event_ticker:
            continue

        status = str(market.get("status", "")).strip().lower()
        if status in {"settled", "closed", "expired", "finalized"}:
            continue

        resolution_time_et = extract_kalshi_resolution_time_et(market, config)
        if resolution_time_et is None:
            continue

        if resolution_time_et < now_et:
            continue

        if resolution_time_et.date() == today_date:
            same_day_candidates.append((resolution_time_et, event_ticker))
        else:
            future_candidates.append((resolution_time_et, event_ticker))

    if same_day_candidates:
        same_day_candidates.sort(key=lambda x: x[0])
        return same_day_candidates[0][1]

    if future_candidates:
        future_candidates.sort(key=lambda x: x[0])
        return future_candidates[0][1]

    return None


def get_kalshi_market_contracts(config):
    kalshi_cfg = config["kalshi"]

    series_ticker = kalshi_cfg.get("series_ticker")
    exact_market_ticker = kalshi_cfg.get("exact_market_ticker")

    if not series_ticker and not exact_market_ticker:
        url = kalshi_cfg.get("market_url")
        parsed_series, parsed_slug, parsed_exact = parse_kalshi_url_parts(url)
        kalshi_cfg["series_ticker"] = kalshi_cfg.get("series_ticker") or parsed_series
        kalshi_cfg["market_slug"] = kalshi_cfg.get("market_slug") or parsed_slug
        kalshi_cfg["exact_market_ticker"] = kalshi_cfg.get("exact_market_ticker") or parsed_exact
        series_ticker = kalshi_cfg.get("series_ticker")
        exact_market_ticker = kalshi_cfg.get("exact_market_ticker")

    if not series_ticker and not exact_market_ticker:
        raise ValueError("No Kalshi series_ticker or exact_market_ticker found.")

    if exact_market_ticker:
        raw_markets = kalshi_paginated_get(
            config,
            "markets",
            params={"tickers": exact_market_ticker, "limit": 100},
            results_key="markets",
            max_pages=1,
        )
    else:
        raw_markets = kalshi_paginated_get(
            config,
            "markets",
            params={
                "series_ticker": series_ticker,
                "limit": int(config.get("runtime", {}).get("market_fetch_limit", 100)),
            },
            results_key="markets",
            max_pages=config["runtime"].get("max_market_pages", 3),
        )

    if not raw_markets:
        kalshi_cfg["event_ticker"] = None
        return pd.DataFrame()

    selected_event_ticker = None

    if exact_market_ticker:
        selected_event_ticker = exact_market_ticker
    else:
        selected_event_ticker = _select_active_event_ticker_from_markets(raw_markets, config)

    if selected_event_ticker:
        filtered_for_event = []
        for market in raw_markets:
            event_ticker = market.get("event_ticker")
            ticker = market.get("ticker")
            if event_ticker == selected_event_ticker or ticker == selected_event_ticker:
                filtered_for_event.append(market)

        if filtered_for_event:
            raw_markets = filtered_for_event

    filtered_markets = []
    for m in raw_markets:
        status = str(m.get("status", "")).lower()
        if status in ["active", "open", "initialized", ""]:
            filtered_markets.append(m)

    if filtered_markets:
        raw_markets = filtered_markets

    if selected_event_ticker is None:
        unique_events = sorted({m.get("event_ticker") for m in raw_markets if m.get("event_ticker")})
        selected_event_ticker = unique_events[0] if unique_events else None

    kalshi_cfg["event_ticker"] = selected_event_ticker

    debug_market_logs = bool(config.get("runtime", {}).get("debug_market_logs", False))

    records = []
    for idx, market in enumerate(raw_markets):
        if debug_market_logs and idx < 3:
            logging.info("MARKET RAW SAMPLE %s", {k: market.get(k) for k in market.keys()})

        ticker = market.get("ticker")
        if not ticker:
            continue

        strike = extract_kalshi_strike(market)
        if strike is None:
            continue

        px = get_market_price_snapshot(market)
        resolution_time_et = extract_kalshi_resolution_time_et(market, config)
        contract_name = market.get("title") or market.get("subtitle") or ticker

        records.append(
            {
                "contract_ticker": ticker,
                "event_ticker": market.get("event_ticker"),
                "series_ticker": series_ticker,
                "contract_name": contract_name,
                "status": market.get("status"),
                "strike": float(strike),
                "strike_type": market.get("strike_type"),
                "floor_strike": market.get("floor_strike"),
                "cap_strike": market.get("cap_strike"),
                "market_yes_prob": float(px["market_yes_prob"]),
                "market_no_prob": float(px["market_no_prob"]),
                "bid_yes": px["bid_yes"],
                "ask_yes": px["ask_yes"],
                "bid_no": px["bid_no"],
                "ask_no": px["ask_no"],
                "last_price_yes": px["last_price_yes"],
                "last_price_no": px["last_price_no"],
                "pricing_source": px["pricing_source"],
                "resolution_time_et": format_et(resolution_time_et),
                "resolution_time_dt_et": resolution_time_et,
                "cash_settlement_time_et": format_et(resolution_time_et),
                "source": "kalshi_api_markets",
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    return (
        df.sort_values(["resolution_time_dt_et", "strike", "contract_ticker"])
        .drop_duplicates(subset=["contract_ticker"])
        .reset_index(drop=True)
    )


def build_contracts_from_market_df(market_contracts_df):
    if market_contracts_df is None or market_contracts_df.empty:
        return []

    contracts = []
    for _, row in market_contracts_df.iterrows():
        contracts.append(
            {
                "contract_ticker": row["contract_ticker"],
                "contract_name": row["contract_name"],
                "event_ticker": row.get("event_ticker"),
                "series_ticker": row.get("series_ticker"),
                "strike": float(row["strike"]),
                "strike_type": row.get("strike_type"),
                "floor_strike": row.get("floor_strike"),
                "cap_strike": row.get("cap_strike"),
                "market_yes_probability": float(row["market_yes_prob"]),
                "market_no_probability": float(row["market_no_prob"]),
                "bid_yes": row.get("bid_yes"),
                "ask_yes": row.get("ask_yes"),
                "bid_no": row.get("bid_no"),
                "ask_no": row.get("ask_no"),
                "last_price_yes": row.get("last_price_yes"),
                "last_price_no": row.get("last_price_no"),
                "yes_spread": row.get("yes_spread", np.nan),
                "no_spread": row.get("no_spread", np.nan),
                "tradable_yes": bool(row.get("tradable_yes", True)),
                "tradable_no": bool(row.get("tradable_no", True)),
                "is_liquid": bool(row.get("is_liquid", True)),
                "pricing_source": row.get("pricing_source"),
                "status": row.get("status"),
                "source": row.get("source", "kalshi_api"),
                "resolution_time_et": row.get("resolution_time_et"),
                "resolution_time_dt_et": row.get("resolution_time_dt_et"),
                "cash_settlement_time_et": row.get("cash_settlement_time_et"),
                "force_include_for_monitoring": bool(row.get("force_include_for_monitoring", False)),
                "entry_candidate": bool(row.get("entry_candidate", True)),
            }
        )
    return contracts


def filter_contracts_near_spot(market_contracts_df, spot, config, vol_stats=None):
    if market_contracts_df is None or market_contracts_df.empty:
        return market_contracts_df

    out = market_contracts_df.copy()
    out["distance_to_strike"] = pd.to_numeric(out["strike"], errors="coerce") - float(spot)
    out["abs_distance_to_strike"] = out["distance_to_strike"].abs()

    filters_cfg = (config.get("filters") or {})
    model_cfg = (config.get("model") or {})

    strike_band_below = float(filters_cfg.get("strike_band_below", 10.0))
    strike_band_above = float(filters_cfg.get("strike_band_above", 10.0))
    hard_band_cap = max(strike_band_below, strike_band_above, 10.0)

    static_cap = float(filters_cfg.get("max_abs_distance", hard_band_cap))
    use_dynamic_distance = bool(filters_cfg.get("use_dynamic_distance", False))

    blended_vol = None
    if isinstance(vol_stats, dict):
        blended_vol = vol_stats.get("blended_vol")
    if blended_vol is None:
        blended_vol = filters_cfg.get(
            "distance_vol_reference",
            model_cfg.get("annualized_volatility_fallback", 0.25),
        )
    try:
        blended_vol = float(blended_vol)
    except Exception:
        blended_vol = float(model_cfg.get("annualized_volatility_fallback", 0.25))

    if use_dynamic_distance:
        distance_vol_multiplier = float(filters_cfg.get("distance_vol_multiplier", 1.0))
        min_distance = float(filters_cfg.get("min_distance", 5.0))
        max_distance = float(filters_cfg.get("max_distance", static_cap))

        dynamic_distance = float(spot) * max(blended_vol, 0.0) * max(distance_vol_multiplier, 0.0)
        max_abs_distance = max(min_distance, min(dynamic_distance, max_distance))
    else:
        max_abs_distance = static_cap

    max_abs_distance = min(max_abs_distance, hard_band_cap)

    logging.info(
        "Strike distance filter | spot=%.2f | blended_vol=%.4f | dynamic=%s | max_abs_distance=%.2f | band_cap=%.2f",
        float(spot),
        float(blended_vol),
        use_dynamic_distance,
        float(max_abs_distance),
        float(hard_band_cap),
    )

    out = out[out["abs_distance_to_strike"] <= max_abs_distance].copy()
    return out.reset_index(drop=True)


def force_include_contracts_for_monitoring(
    all_event_markets_df,
    candidate_markets_df,
    force_include_contract_tickers,
):
    if all_event_markets_df is None or all_event_markets_df.empty:
        return candidate_markets_df if candidate_markets_df is not None else pd.DataFrame()

    force_include_contract_tickers = force_include_contract_tickers or []
    force_include_contract_tickers = [
        str(t).strip() for t in force_include_contract_tickers if str(t).strip()
    ]

    all_df = all_event_markets_df.copy()
    all_df["contract_ticker"] = all_df["contract_ticker"].astype(str).str.strip()
    all_df["force_include_for_monitoring"] = False
    all_df["entry_candidate"] = True

    if candidate_markets_df is None or candidate_markets_df.empty:
        candidate_df = all_df.iloc[0:0].copy()
    else:
        candidate_df = candidate_markets_df.copy()
        candidate_df["contract_ticker"] = candidate_df["contract_ticker"].astype(str).str.strip()
        candidate_df["force_include_for_monitoring"] = False
        candidate_df["entry_candidate"] = True

    if not force_include_contract_tickers:
        return candidate_df.drop_duplicates(subset=["contract_ticker"]).reset_index(drop=True)

    forced_df = all_df[all_df["contract_ticker"].isin(force_include_contract_tickers)].copy()
    if forced_df.empty:
        logging.info(
            "No held contract tickers from live positions were found in the active event market set. requested=%s",
            sorted(set(force_include_contract_tickers)),
        )
        return candidate_df.drop_duplicates(subset=["contract_ticker"]).reset_index(drop=True)

    forced_df["force_include_for_monitoring"] = True
    forced_df["entry_candidate"] = False

    out = pd.concat([candidate_df, forced_df], ignore_index=True, sort=False)
    out = out.drop_duplicates(subset=["contract_ticker"], keep="first").reset_index(drop=True)

    logging.info(
        "Force-including held contracts for monitoring | requested=%s | matched=%s | added_outside_band=%s",
        len(force_include_contract_tickers),
        len(forced_df),
        int((~out["entry_candidate"].fillna(False)).sum()),
    )

    return out


# =========================================================
# EVALUATION / STATE / LOGGING
# =========================================================
def _safe_float_config(value, default):
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _compute_dynamic_edge_threshold(overround, decision_cfg):
    base_threshold = _safe_float_config(
        decision_cfg.get(
            "dynamic_threshold_base",
            decision_cfg.get(
                "buy_threshold",
                decision_cfg.get("min_edge_to_trade", 0.03),
            ),
        ),
        0.03,
    )
    dynamic_enabled = bool(decision_cfg.get("dynamic_threshold_enabled", True))
    if not dynamic_enabled or pd.isna(overround):
        return base_threshold

    overround_multiplier = _safe_float_config(
        decision_cfg.get("dynamic_threshold_overround_mult", 1.50),
        1.50,
    )
    threshold = base_threshold + max(0.0, float(overround)) * overround_multiplier
    return max(base_threshold, threshold)


def _compute_target_price(fair_price, buffer_total, dynamic_threshold, executable_aggression=0.0):
    if pd.isna(fair_price):
        return np.nan
    return _clip_price_01_99(fair_price - buffer_total - dynamic_threshold + executable_aggression)


def evaluate_ladder(price, contracts, vol_stats, config):
    rows = []

    if isinstance(vol_stats, dict):
        blended_vol = vol_stats.get(
            "blended_vol",
            vol_stats.get("short_vol", vol_stats.get("medium_vol", 0.15)),
        )
        history_df = vol_stats.get("history_df")
    else:
        blended_vol = float(vol_stats)
        history_df = None

    run_time_et = current_time_et()
    vol_regime = detect_vol_regime(blended_vol)
    adjusted_vol = adjust_volatility_by_regime(blended_vol, vol_regime)
    adjusted_vol = float(
        np.clip(
            adjusted_vol,
            config["model"]["min_annualized_vol"],
            config["model"]["max_annualized_vol"],
        )
    )
    dynamic_drift = compute_dynamic_drift(history_df, config)
    momentum_features = compute_momentum_features(price, history_df, config)

    prob_floor = float(config["model"].get("prob_floor", 0.001))
    prob_cap = float(config["model"].get("prob_cap", 0.999))

    model_cfg = config.get("model", {}) or {}
    touch_paths = int(model_cfg.get("touch_paths", 2000))
    touch_steps = int(model_cfg.get("touch_steps", 40))

    decision_cfg = config.get("decision", {})
    fee_buffer = float(decision_cfg.get("fee_buffer", 0.0))
    spread_buffer = float(decision_cfg.get("spread_buffer", 0.0))
    safety_buffer = float(decision_cfg.get("safety_buffer", 0.0))
    buy_threshold = float(
        decision_cfg.get(
            "buy_threshold",
            decision_cfg.get("min_edge_to_trade", 0.03),
        )
    )
    near_positive_edge_tolerance = float(
        decision_cfg.get("near_positive_edge_tolerance", 0.01)
    )
    clean_market_aggressive_entry_enabled = bool(
        decision_cfg.get("clean_market_aggressive_entry_enabled", True)
    )
    clean_market_max_overround = float(
        decision_cfg.get("clean_market_max_overround", 0.05)
    )
    clean_market_min_edge = float(
        decision_cfg.get("clean_market_min_edge", 0.08)
    )
    clean_market_target_price_improvement = float(
        decision_cfg.get("clean_market_target_price_improvement", 0.01)
    )
    buffer_total = fee_buffer + spread_buffer + safety_buffer

    filters_cfg = config.get("filters", {})
    max_yes_no_ask_sum = float(filters_cfg.get("max_yes_no_ask_sum", 1.08))
    max_overround = float(filters_cfg.get("max_overround", 0.08))
    hard_skip_overround = float(filters_cfg.get("hard_skip_overround", max_overround))
    hard_skip_yes_no_ask_sum = float(
        filters_cfg.get("hard_skip_yes_no_ask_sum", max_yes_no_ask_sum)
    )
    near_executable_overround = float(
        filters_cfg.get("near_executable_overround", 0.12)
    )

    # Profit-mode override: allow strong, executable trades to pass even when
    # spreads are a bit wider than the standard hard skip thresholds.
    allow_wide_market_edge_override = bool(
        decision_cfg.get("allow_wide_market_edge_override", True)
    )
    wide_market_edge_override_threshold = float(
        decision_cfg.get("wide_market_edge_override_threshold", 0.14)
    )
    wide_market_max_overround = float(
        decision_cfg.get(
            "wide_market_max_overround",
            max(near_executable_overround, hard_skip_overround),
        )
    )
    wide_market_max_yes_no_ask_sum = float(
        decision_cfg.get(
            "wide_market_max_yes_no_ask_sum",
            max(hard_skip_yes_no_ask_sum, 1.10),
        )
    )
    wide_market_require_executable = bool(
        decision_cfg.get("wide_market_require_executable", True)
    )
    wide_market_force_active_phase = bool(
        decision_cfg.get("wide_market_force_active_phase", True)
    )

    for c in contracts:
        strike = float(c["strike"])

        resolution_time_et = c.get("resolution_time_dt_et")
        if resolution_time_et is None:
            resolution_time_et = get_fallback_resolution_time(config)

        hours_left = get_hours_left(resolution_time_et)
        trading_phase = classify_trading_phase(hours_left, config)

        fair_prob_terminal = estimate_terminal_probability(
            price,
            strike,
            resolution_time_et,
            adjusted_vol,
            dynamic_drift,
            config["model"].get("monte_carlo_paths"),
        )
        fair_prob_terminal = float(np.clip(fair_prob_terminal, prob_floor, prob_cap))

        fair_prob_touch = estimate_touch_probability(
            price,
            strike,
            resolution_time_et,
            adjusted_vol,
            dynamic_drift,
            n_paths=touch_paths,
            n_steps=touch_steps,
        )
        fair_prob_touch = float(np.clip(fair_prob_touch, prob_floor, prob_cap))

        fair_prob_blended = (
            config["model"].get("blend_terminal_weight", 1.0) * fair_prob_terminal
            + config["model"].get("blend_touch_weight", 0.0) * fair_prob_touch
        )
        fair_prob_blended = float(np.clip(fair_prob_blended, prob_floor, prob_cap))

        decision_prob = float(np.clip(fair_prob_blended, prob_floor, prob_cap))
        fair_yes = decision_prob
        fair_no = 1.0 - decision_prob

        ask_yes = _coalesce_price(c.get("ask_yes"), c.get("market_yes_probability"))
        ask_no = _coalesce_price(c.get("ask_no"), c.get("market_no_probability"))

        market_prob_yes = ask_yes
        market_prob_no = ask_no

        yes_no_ask_sum = (
            ask_yes + ask_no
            if pd.notna(ask_yes) and pd.notna(ask_no)
            else np.nan
        )
        overround = (
            yes_no_ask_sum - 1.0
            if pd.notna(yes_no_ask_sum)
            else np.nan
        )

        market_too_wide = False
        if pd.notna(yes_no_ask_sum) and yes_no_ask_sum > max_yes_no_ask_sum:
            market_too_wide = True
        if pd.notna(overround) and overround > max_overround:
            market_too_wide = True

        hard_market_skip = False
        if pd.notna(yes_no_ask_sum) and yes_no_ask_sum > hard_skip_yes_no_ask_sum:
            hard_market_skip = True
        if pd.notna(overround) and overround > hard_skip_overround:
            hard_market_skip = True

        raw_edge_yes = fair_yes - market_prob_yes if pd.notna(market_prob_yes) else np.nan
        raw_edge_no = fair_no - market_prob_no if pd.notna(market_prob_no) else np.nan

        edge_yes = raw_edge_yes - buffer_total if pd.notna(raw_edge_yes) else np.nan
        edge_no = raw_edge_no - buffer_total if pd.notna(raw_edge_no) else np.nan

        dynamic_buy_threshold = _compute_dynamic_edge_threshold(overround, decision_cfg)
        clean_market_regime = bool(
            pd.notna(overround)
            and overround <= clean_market_max_overround
        )
        clean_market_aggression = (
            clean_market_target_price_improvement
            if clean_market_aggressive_entry_enabled and clean_market_regime
            else 0.0
        )

        target_yes_price = _compute_target_price(
            fair_yes,
            buffer_total,
            dynamic_buy_threshold,
            executable_aggression=clean_market_aggression,
        )
        target_no_price = _compute_target_price(
            fair_no,
            buffer_total,
            dynamic_buy_threshold,
            executable_aggression=clean_market_aggression,
        )

        executable_yes_now = bool(
            pd.notna(ask_yes) and pd.notna(target_yes_price) and ask_yes <= target_yes_price
        )
        executable_no_now = bool(
            pd.notna(ask_no) and pd.notna(target_no_price) and ask_no <= target_no_price
        )

        tradable_yes = bool(c.get("tradable_yes", c.get("is_liquid", True))) and pd.notna(edge_yes)
        tradable_no = bool(c.get("tradable_no", c.get("is_liquid", True))) and pd.notna(edge_no)

        # =====================================================
        # VALID-SIDE CLASSIFICATION
        # Economic validity must stay separate from immediate executability.
        # A side is economically valid if it:
        #   * has a sane quote
        #   * is tradable on that side
        #   * clears the edge threshold
        # Immediate executability is handled separately through
        # executable_yes_now / executable_no_now.
        # =====================================================
        yes_edge_qualifies = bool(
            pd.notna(edge_yes)
            and (
                edge_yes >= dynamic_buy_threshold
                or (
                    clean_market_regime
                    and edge_yes >= clean_market_min_edge
                )
            )
        )
        no_edge_qualifies = bool(
            pd.notna(edge_no)
            and (
                edge_no >= dynamic_buy_threshold
                or (
                    clean_market_regime
                    and edge_no >= clean_market_min_edge
                )
            )
        )

        # =====================================================
        # STRICT LIVE ENTRY FILTER (FINAL)
        # Tighten future live entries so only stronger, high-confidence
        # setups pass into the actionable path. This does not affect
        # monitoring of existing held positions.
        # =====================================================
        strict_edge = float(decision_cfg.get("strict_live_entry_edge", 0.10))
        strict_prob = float(decision_cfg.get("strict_live_entry_prob", 0.60))
        strict_high_conf_only = bool(decision_cfg.get("strict_live_entry_high_conf_only", False))

        yes_conf = str(compute_confidence(edge_yes)).upper() if pd.notna(edge_yes) else "LOW"
        no_conf = str(compute_confidence(edge_no)).upper() if pd.notna(edge_no) else "LOW"

        yes_prob_ok = pd.notna(decision_prob) and decision_prob >= strict_prob
        no_prob_ok = pd.notna(decision_prob) and (1.0 - decision_prob) >= strict_prob

        yes_strict_ok = bool(
            pd.notna(edge_yes)
            and edge_yes >= strict_edge
            and yes_prob_ok
            and ((not strict_high_conf_only) or yes_conf == "HIGH")
        )

        no_strict_ok = bool(
            pd.notna(edge_no)
            and edge_no >= strict_edge
            and no_prob_ok
            and ((not strict_high_conf_only) or no_conf == "HIGH")
        )

        yes_side_valid = bool(
            pd.notna(ask_yes)
            and tradable_yes
            and yes_edge_qualifies
            and yes_strict_ok
            and fair_yes >= 0.55
        )
        no_side_valid = bool(
            pd.notna(ask_no)
            and tradable_no
            and no_edge_qualifies
            and no_strict_ok
            and fair_no >= 0.55
        )
        any_valid_side = bool(yes_side_valid or no_side_valid)

        # =====================================================
        # SIDE SELECTION MUST BE DECOUPLED FROM EXECUTABILITY.
        # Pick the economically better side first, then decide whether
        # it is actionable now, actionable via limit maker, watchlist,
        # or truly not tradable.
        # =====================================================
        selected_side = ""
        if tradable_yes and tradable_no:
            if pd.notna(edge_yes) and pd.notna(edge_no):
                selected_side = "YES" if float(edge_yes) >= float(edge_no) else "NO"
            elif pd.notna(raw_edge_yes) and pd.notna(raw_edge_no):
                selected_side = "YES" if float(raw_edge_yes) >= float(raw_edge_no) else "NO"
            elif pd.notna(edge_yes) or pd.notna(raw_edge_yes):
                selected_side = "YES"
            elif pd.notna(edge_no) or pd.notna(raw_edge_no):
                selected_side = "NO"
        elif tradable_yes and (pd.notna(edge_yes) or pd.notna(raw_edge_yes)):
            selected_side = "YES"
        elif tradable_no and (pd.notna(edge_no) or pd.notna(raw_edge_no)):
            selected_side = "NO"

        if selected_side == "YES":
            selected_edge = edge_yes
            selected_raw_edge = raw_edge_yes
            selected_target_price = target_yes_price
            selected_market_price = ask_yes
            selected_executable_now = executable_yes_now
        elif selected_side == "NO":
            selected_edge = edge_no
            selected_raw_edge = raw_edge_no
            selected_target_price = target_no_price
            selected_market_price = ask_no
            selected_executable_now = executable_no_now
        else:
            selected_edge = np.nan
            selected_raw_edge = np.nan
            selected_target_price = np.nan
            selected_market_price = np.nan
            selected_executable_now = False

        selected_side_tradeable = bool(
            (selected_side == "YES" and tradable_yes)
            or (selected_side == "NO" and tradable_no)
        )
        selected_side_valid = bool(
            (selected_side == "YES" and yes_side_valid)
            or (selected_side == "NO" and no_side_valid)
        )
        side_prob_support = fair_yes if selected_side == "YES" else fair_no if selected_side == "NO" else np.nan

        # Final coherence check between modeled support and quoted price for the selected side.
        selected_quote = ask_yes if selected_side == "YES" else ask_no if selected_side == "NO" else np.nan
        selected_dynamic_edge = (
            (side_prob_support - selected_quote)
            if pd.notna(side_prob_support) and pd.notna(selected_quote)
            else np.nan
        )
        selected_side_consistent = bool(
            selected_side in {"YES", "NO"}
            and pd.notna(side_prob_support)
            and side_prob_support >= 0.60
            and pd.notna(selected_dynamic_edge)
            and selected_dynamic_edge >= dynamic_buy_threshold
        )
        if not selected_side_consistent:
            selected_side_valid = False

        momentum_short = float(momentum_features.get("oil_momentum_short", 0.0))
        momentum_medium = float(momentum_features.get("oil_momentum_medium", 0.0))
        momentum_regime = str(momentum_features.get("oil_momentum_regime", "NEUTRAL"))
        momentum_pass = True
        momentum_block_reason = ""

        if selected_side == "YES":
            if not (momentum_short > 0.0 and momentum_medium >= 0.0):
                momentum_pass = False
                if momentum_short <= 0.0 and momentum_medium < 0.0:
                    momentum_block_reason = "yes_block_short_and_medium_bearish"
                elif momentum_short <= 0.0:
                    momentum_block_reason = "yes_block_short_not_bullish"
                else:
                    momentum_block_reason = "yes_block_medium_bearish"
        elif selected_side == "NO":
            if not (momentum_short < 0.0 and momentum_medium <= 0.0):
                momentum_pass = False
                if momentum_short >= 0.0 and momentum_medium > 0.0:
                    momentum_block_reason = "no_block_short_and_medium_bullish"
                elif momentum_short >= 0.0:
                    momentum_block_reason = "no_block_short_not_bearish"
                else:
                    momentum_block_reason = "no_block_medium_bullish"

        if not momentum_pass:
            selected_side_valid = False

        watchlist_candidate = bool(
            selected_side_tradeable
            and pd.notna(selected_edge)
            and selected_edge >= 0.06
            and pd.notna(side_prob_support)
            and side_prob_support >= 0.60
            and not selected_executable_now
            and not market_too_wide
        )
        limit_entry_allowed = bool(
            selected_side_tradeable
            and pd.notna(selected_edge)
            and selected_edge >= 0.08
            and pd.notna(side_prob_support)
            and side_prob_support >= 0.65
            and not selected_executable_now
            and not market_too_wide
            and not hard_market_skip
        )

        # Disable permissive wide-market promotion unless a setup is both executable and very strong.
        wide_market_override_yes = bool(
            allow_wide_market_edge_override
            and tradable_yes
            and yes_side_valid
            and executable_yes_now
            and pd.notna(edge_yes)
            and edge_yes >= max(wide_market_edge_override_threshold, strict_edge + 0.05)
            and fair_yes >= max(strict_prob, 0.75)
            and (
                (pd.notna(overround) and overround <= wide_market_max_overround)
                or (pd.notna(yes_no_ask_sum) and yes_no_ask_sum <= wide_market_max_yes_no_ask_sum)
            )
        )
        wide_market_override_no = bool(
            allow_wide_market_edge_override
            and tradable_no
            and no_side_valid
            and executable_no_now
            and pd.notna(edge_no)
            and edge_no >= max(wide_market_edge_override_threshold, strict_edge + 0.05)
            and fair_no >= max(strict_prob, 0.75)
            and (
                (pd.notna(overround) and overround <= wide_market_max_overround)
                or (pd.notna(yes_no_ask_sum) and yes_no_ask_sum <= wide_market_max_yes_no_ask_sum)
            )
        )

        held_for_monitoring_only = bool(c.get("force_include_for_monitoring", False)) and not bool(
            c.get("entry_candidate", True)
        )

        has_contract_ticker = bool(str(c.get("contract_ticker") or "").strip())
        has_valid_strike = pd.notna(strike) and strike > 0
        has_any_quote = any(
            pd.notna(v)
            for v in [
                c.get("bid_yes"),
                c.get("ask_yes"),
                c.get("bid_no"),
                c.get("ask_no"),
                c.get("market_yes_probability"),
                c.get("market_no_probability"),
            ]
        )

        invalid_reason = None
        if not has_contract_ticker:
            invalid_reason = "MALFORMED_CONTRACT"
        elif not has_valid_strike:
            invalid_reason = "INVALID_STRIKE"
        elif not has_any_quote:
            invalid_reason = "MISSING_QUOTES"
        elif not any_valid_side and not (tradable_yes or tradable_no):
            invalid_reason = "NO_VALID_SIDE"

        invalid_row = invalid_reason is not None

        interesting_yes = bool(
            yes_side_valid
            or (
                tradable_yes
                and (
                    executable_yes_now
                    or (pd.notna(raw_edge_yes) and raw_edge_yes > 0)
                    or (pd.notna(edge_yes) and edge_yes >= -near_positive_edge_tolerance)
                )
            )
        )
        interesting_no = bool(
            no_side_valid
            or (
                tradable_no
                and (
                    executable_no_now
                    or (pd.notna(raw_edge_no) and raw_edge_no > 0)
                    or (pd.notna(edge_no) and edge_no >= -near_positive_edge_tolerance)
                )
            )
        )

        spread_blocked_interesting = bool(
            market_too_wide
            and (
                interesting_yes
                or interesting_no
                or any_valid_side
                or selected_executable_now
                or (pd.notna(selected_raw_edge) and selected_raw_edge > 0)
                or (pd.notna(selected_edge) and selected_edge >= -near_positive_edge_tolerance)
            )
        )

        actionable_yes = bool(
            (not market_too_wide)
            and (not hard_market_skip)
            and yes_side_valid
            and selected_side == "YES"
            and tradable_yes
            and executable_yes_now
            and pd.notna(edge_yes)
            and yes_edge_qualifies
            and fair_yes >= 0.65
            and yes_conf == "HIGH"
            and selected_side_consistent
            and momentum_pass
        )
        actionable_no = bool(
            (not market_too_wide)
            and (not hard_market_skip)
            and no_side_valid
            and selected_side == "NO"
            and tradable_no
            and executable_no_now
            and pd.notna(edge_no)
            and no_edge_qualifies
            and fair_no >= 0.65
            and no_conf == "HIGH"
            and selected_side_consistent
            and momentum_pass
        )

        if held_for_monitoring_only:
            decision_state = "HELD_POSITION_MONITORING_ONLY"
            action = "NO_TRADE"
            no_trade_reason = "HELD_POSITION_MONITORING_ONLY"
            entry_style = "MONITOR_ONLY"

        elif invalid_row:
            decision_state = "NOT_TRADABLE"
            action = "NO_TRADE"
            no_trade_reason = invalid_reason
            entry_style = "NONE"

        elif not momentum_pass and selected_side in {"YES", "NO"}:
            decision_state = "TIMING_BLOCKED"
            action = "NO_TRADE"
            no_trade_reason = momentum_block_reason or "MOMENTUM_BLOCK"
            entry_style = "TIMING_FILTER"

        elif actionable_yes:
            decision_state = "ACTIONABLE"
            action = "BUY_YES"
            no_trade_reason = ""
            entry_style = "TAKER"

        elif actionable_no:
            decision_state = "ACTIONABLE"
            action = "BUY_NO"
            no_trade_reason = ""
            entry_style = "TAKER"

        elif wide_market_override_yes:
            decision_state = "ACTIONABLE"
            action = "BUY_YES"
            no_trade_reason = "ACTIONABLE_WIDE_MARKET_OVERRIDE"
            entry_style = "TAKER_WIDE_MARKET_OVERRIDE"

        elif wide_market_override_no:
            decision_state = "ACTIONABLE"
            action = "BUY_NO"
            no_trade_reason = "ACTIONABLE_WIDE_MARKET_OVERRIDE"
            entry_style = "TAKER_WIDE_MARKET_OVERRIDE"

        elif hard_market_skip and (any_valid_side or spread_blocked_interesting):
            decision_state = "PRICE_OK_BUT_SPREAD_TOO_WIDE"
            action = "NO_TRADE"
            no_trade_reason = "HARD_SKIP_OVERROUND"
            entry_style = "BLOCKED_BY_SPREAD"

        elif hard_market_skip:
            decision_state = "MARKET_TOO_WIDE"
            action = "NO_TRADE"
            no_trade_reason = "HARD_SKIP_OVERROUND"
            entry_style = "BLOCKED_BY_SPREAD"

        elif any_valid_side and market_too_wide:
            decision_state = "PRICE_OK_BUT_SPREAD_TOO_WIDE"
            action = "NO_TRADE"
            no_trade_reason = (
                "NEAR_EXECUTABLE_SPREAD_BLOCKED"
                if pd.notna(overround) and overround <= near_executable_overround
                else "PRICE_OK_BUT_SPREAD_TOO_WIDE"
            )
            entry_style = "BLOCKED_BY_SPREAD"

        elif spread_blocked_interesting:
            decision_state = "PRICE_OK_BUT_SPREAD_TOO_WIDE"
            action = "NO_TRADE"
            no_trade_reason = (
                "NEAR_EXECUTABLE_SPREAD_BLOCKED"
                if pd.notna(overround) and overround <= near_executable_overround
                else "PRICE_OK_BUT_SPREAD_TOO_WIDE"
            )
            entry_style = "BLOCKED_BY_SPREAD"

        elif market_too_wide:
            decision_state = "MARKET_TOO_WIDE"
            action = "NO_TRADE"
            no_trade_reason = "MARKET_TOO_WIDE"
            entry_style = "BLOCKED_BY_SPREAD"

        elif limit_entry_allowed and selected_side == "YES":
            decision_state = "WAIT_FOR_PRICE"
            action = "WAIT_FOR_PRICE"
            no_trade_reason = "LIMIT_ENTRY_REQUIRED"
            entry_style = "LIMIT_MAKER"

        elif limit_entry_allowed and selected_side == "NO":
            decision_state = "WAIT_FOR_PRICE"
            action = "WAIT_FOR_PRICE"
            no_trade_reason = "LIMIT_ENTRY_REQUIRED"
            entry_style = "LIMIT_MAKER"

        elif watchlist_candidate and selected_side:
            decision_state = "WAIT_FOR_PRICE"
            action = "WAIT_FOR_PRICE"
            no_trade_reason = "WATCHLIST_LIMIT_REQUIRED"
            entry_style = "WATCHLIST_LIMIT"

        elif yes_side_valid and selected_side == "YES":
            decision_state = "WAIT_FOR_PRICE"
            action = "WAIT_FOR_PRICE"
            no_trade_reason = "PRICE_ABOVE_TARGET" if not executable_yes_now else "EDGE_BELOW_THRESHOLD"
            entry_style = "MAKER_TARGET"

        elif no_side_valid and selected_side == "NO":
            decision_state = "WAIT_FOR_PRICE"
            action = "WAIT_FOR_PRICE"
            no_trade_reason = "PRICE_ABOVE_TARGET" if not executable_no_now else "EDGE_BELOW_THRESHOLD"
            entry_style = "MAKER_TARGET"

        elif any_valid_side or (selected_side_tradeable and pd.notna(selected_edge) and selected_edge > 0):
            decision_state = "WAIT_FOR_PRICE"
            action = "WAIT_FOR_PRICE"
            no_trade_reason = "NO_SELECTED_SIDE_BUT_VALID_ECONOMICS"
            entry_style = "MAKER_TARGET"

            if not selected_side:
                if yes_side_valid and not no_side_valid:
                    selected_side = "YES"
                    selected_edge = edge_yes
                    selected_raw_edge = raw_edge_yes
                    selected_target_price = target_yes_price
                    selected_market_price = ask_yes
                    selected_executable_now = executable_yes_now
                elif no_side_valid and not yes_side_valid:
                    selected_side = "NO"
                    selected_edge = edge_no
                    selected_raw_edge = raw_edge_no
                    selected_target_price = target_no_price
                    selected_market_price = ask_no
                    selected_executable_now = executable_no_now

        else:
            decision_state = "NOT_TRADABLE"
            action = "NO_TRADE"
            no_trade_reason = "NO_VALID_SIDE"
            entry_style = "NONE"

        market_too_wide_but_monitorable = decision_state == "PRICE_OK_BUT_SPREAD_TOO_WIDE"

        effective_trading_phase = trading_phase
        if (
            wide_market_force_active_phase
            and trading_phase == "OBSERVE_ONLY"
            and decision_state == "ACTIONABLE"
        ):
            effective_trading_phase = "ACTIVE_TRADING"

        ev_yes_exec = fair_yes - ask_yes if pd.notna(ask_yes) else np.nan
        ev_no_exec = fair_no - ask_no if pd.notna(ask_no) else np.nan

        row = {
            "run_timestamp_et": format_et(run_time_et),
            "contract_ticker": c.get("contract_ticker"),
            "event_ticker": c.get("event_ticker"),
            "contract": c.get("contract_name"),
            "strike": strike,
            "oil_price": price,
            "distance_to_strike": strike - price,
            "hours_left": hours_left,
            "trading_phase": effective_trading_phase,
            "market_prob": c.get("market_yes_probability"),
            "market_no_prob_mid": c.get("market_no_probability"),
            "market_prob_yes": market_prob_yes,
            "market_prob_no": market_prob_no,
            "decision_prob": decision_prob,
            "fair_prob_terminal": fair_prob_terminal,
            "fair_prob_touch": fair_prob_touch,
            "fair_prob_blended": fair_prob_blended,
            "volatility": blended_vol,
            "adjusted_volatility": adjusted_vol,
            "vol_regime": vol_regime,
            "dynamic_drift": dynamic_drift,
            "oil_momentum_short": momentum_short,
            "oil_momentum_medium": momentum_medium,
            "oil_momentum_regime": momentum_regime,
            "momentum_pass": momentum_pass,
            "momentum_block_reason": momentum_block_reason,
            "fee_buffer": fee_buffer,
            "spread_buffer": spread_buffer,
            "safety_buffer": safety_buffer,
            "buffer_total": buffer_total,
            "dynamic_buy_threshold": dynamic_buy_threshold,
            "clean_market_regime": clean_market_regime,
            "hard_market_skip": hard_market_skip,
            "raw_edge_yes": raw_edge_yes,
            "raw_edge_no": raw_edge_no,
            "edge_yes": edge_yes,
            "edge_no": edge_no,
            "target_yes_price": target_yes_price,
            "target_no_price": target_no_price,
            "executable_yes_now": executable_yes_now,
            "executable_no_now": executable_no_now,
            "selected_market_price": selected_market_price,
            "selected_side_tradeable": selected_side_tradeable,
            "selected_side_valid": selected_side_valid,
            "watchlist_candidate": watchlist_candidate,
            "limit_entry_allowed": limit_entry_allowed,
            "selected_target_price": selected_target_price,
            "selected_raw_edge": selected_raw_edge,
            "selected_executable_now": selected_executable_now,
            "side_prob_support": side_prob_support,
            "candidate_score": np.nan,
            "decision_state": decision_state,
            "entry_style": entry_style,
            "yes_no_ask_sum": yes_no_ask_sum,
            "overround": overround,
            "market_too_wide": market_too_wide,
            "market_too_wide_but_monitorable": market_too_wide_but_monitorable,
            "no_trade_reason": no_trade_reason,
            "selected_side": selected_side,
            "selected_edge": selected_edge,
            "ev_yes_exec": ev_yes_exec,
            "ev_no_exec": ev_no_exec,
            "action": action,
            "bid_yes": c.get("bid_yes"),
            "ask_yes": ask_yes,
            "bid_no": c.get("bid_no"),
            "ask_no": ask_no,
            "tradable_yes": tradable_yes,
            "tradable_no": tradable_no,
            "is_liquid": bool(tradable_yes or tradable_no),
            "resolution_time_et": format_et(resolution_time_et),
            "resolution_time_dt_et": resolution_time_et,
            "force_include_for_monitoring": bool(c.get("force_include_for_monitoring", False)),
            "entry_candidate": bool(c.get("entry_candidate", True)),
            "held_for_monitoring_only": held_for_monitoring_only,
        }
        row["confidence"] = compute_confidence(selected_edge)
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    ev_df = df.apply(compute_expected_values_from_row, axis=1)
    df["ev_yes"] = ev_df["ev_yes"]
    df["ev_no"] = ev_df["ev_no"]

    logging.info(
        "EDGE DEBUG SAMPLE: %s",
        df[
            [
                "contract_ticker",
                "decision_prob",
                "market_prob_yes",
                "market_prob_no",
                "ask_yes",
                "ask_no",
                "target_yes_price",
                "target_no_price",
                "executable_yes_now",
                "executable_no_now",
                "yes_no_ask_sum",
                "overround",
                "raw_edge_yes",
                "raw_edge_no",
                "edge_yes",
                "edge_no",
                "selected_side",
                "selected_edge",
                "decision_state",
                "trading_phase",
                "oil_momentum_short",
                "oil_momentum_medium",
                "momentum_pass",
                "momentum_block_reason",
                "entry_style",
                "action",
                "market_too_wide",
                "market_too_wide_but_monitorable",
                "no_trade_reason",
            ]
        ].head(5).to_dict(orient="records"),
    )

    return enforce_monotonic_probabilities(df)

def log_trade_candidates(ranked_df, market_contracts_df, price, price_src, vol_stats, vol_src, config):
    trade_log_file = config["logging"].get("trade_log_file", "trade_log_v6.csv")

    if ranked_df is None or ranked_df.empty:
        return pd.DataFrame()

    candidates = ranked_df[ranked_df["action"].isin(["BUY_YES", "BUY_NO"])].copy()
    if candidates.empty:
        return pd.DataFrame()

    merge_cols = [
        "contract_ticker",
        "bid_yes",
        "ask_yes",
        "bid_no",
        "ask_no",
        "yes_spread",
        "no_spread",
        "tradable_yes",
        "tradable_no",
        "resolution_time_et",
        "resolution_time_dt_et",
    ]
    available_merge_cols = [c for c in merge_cols if c in market_contracts_df.columns]
    market_subset = market_contracts_df[available_merge_cols].copy()

    out = candidates.merge(market_subset, on="contract_ticker", how="left", suffixes=("", "_market"))

    out["entry_price"] = np.where(
        out["action"] == "BUY_YES",
        out.get("ask_yes"),
        out.get("ask_no"),
    )
    out["model_prediction"] = np.where(out["decision_prob"] >= 0.5, 1, 0)
    out["actual_outcome"] = np.nan
    out["pnl"] = np.nan
    out["calibration_error"] = np.nan
    out["resolved"] = False

    current_et = current_time_et()
    out["log_written_timestamp_et"] = format_et(current_et)
    out["oil_price"] = price
    out["oil_price_source"] = price_src
    out["blended_vol"] = vol_stats.get("blended_vol")
    out["short_vol"] = vol_stats.get("short_vol")
    out["medium_vol"] = vol_stats.get("medium_vol")
    out["volatility_source"] = vol_src

    desired_cols = [
        "run_timestamp_et",
        "log_written_timestamp_et",
        "contract_ticker",
        "event_ticker",
        "contract",
        "strike",
        "oil_price",
        "distance_to_strike",
        "hours_left",
        "trading_phase",
        "market_prob",
        "market_prob_yes",
        "market_prob_no",
        "decision_prob",
        "fair_prob_terminal",
        "fair_prob_touch",
        "fair_prob_blended",
        "dynamic_drift",
        "oil_momentum_short",
        "oil_momentum_medium",
        "oil_momentum_regime",
        "momentum_pass",
        "momentum_block_reason",
        "raw_edge_yes",
        "raw_edge_no",
        "target_yes_price",
        "target_no_price",
        "executable_yes_now",
        "executable_no_now",
        "selected_market_price",
        "selected_target_price",
        "selected_raw_edge",
        "decision_state",
        "entry_style",
        "yes_no_ask_sum",
        "overround",
        "market_too_wide",
        "market_too_wide_but_monitorable",
        "no_trade_reason",
        "model_prediction",
        "action",
        "selected_side",
        "confidence",
        "edge_yes",
        "edge_no",
        "selected_edge",
        "ev_yes",
        "ev_no",
        "ev_yes_exec",
        "ev_no_exec",
        "entry_price",
        "bid_yes",
        "ask_yes",
        "bid_no",
        "ask_no",
        "yes_spread",
        "no_spread",
        "tradable_yes",
        "tradable_no",
        "resolution_time_et",
        "oil_price_source",
        "blended_vol",
        "short_vol",
        "medium_vol",
        "volatility_source",
        "actual_outcome",
        "pnl",
        "calibration_error",
        "resolved",
    ]
    out = out[[c for c in desired_cols if c in out.columns]]

    ensure_parent_dir(trade_log_file)

    if os.path.exists(trade_log_file):
        existing = pd.read_csv(trade_log_file)
        if not existing.empty:
            existing_sig = set(_dedup_signature_frame(existing).astype(str).tolist())
            new_sig = _dedup_signature_frame(out).astype(str)
            out = out.loc[~new_sig.isin(existing_sig)].copy()

    if out.empty:
        return out

    file_exists = os.path.exists(trade_log_file)
    out.to_csv(trade_log_file, mode="a", header=not file_exists, index=False)
    return out


def run_engine_once(config: dict, force_include_contract_tickers=None) -> dict:
    force_include_contract_tickers = force_include_contract_tickers or []
    force_include_contract_tickers = [
        str(t).strip() for t in force_include_contract_tickers if str(t).strip()
    ]

    try:
        price, price_src = get_live_oil_price_cached(config)
        vol_stats, vol_src, history_df = get_realized_volatility_cached(config)
    except Exception as e:
        print(f"Live oil data unavailable from all providers â€” skipping cycle: {e}")
        return {
            "skipped": True,
            "skip_reason": str(e),
            "price": None,
            "price_src": None,
            "vol_stats": None,
            "vol_src": None,
            "history_df": pd.DataFrame(),
            "all_event_markets_df": pd.DataFrame(),
            "market_contracts_df": pd.DataFrame(),
            "candidate_contracts_df": pd.DataFrame(),
            "evaluated_df": pd.DataFrame(),
            "ranked_df": pd.DataFrame(),
            "open_positions_df": pd.DataFrame(),
            "trade_events": [],
            "logged_trades_df": pd.DataFrame(),
            "force_include_contract_tickers": force_include_contract_tickers,
        }

    all_event_markets_df = get_kalshi_market_contracts(config)
    candidate_contracts_df = filter_contracts_near_spot(all_event_markets_df, price, config, vol_stats=vol_stats)
    candidate_contracts_df = apply_liquidity_filters(candidate_contracts_df, config)

    kalshi_cfg = config.get("kalshi", {})
    current_event_ticker = kalshi_cfg.get("event_ticker")
    current_series_ticker = kalshi_cfg.get("series_ticker")

    if not current_event_ticker and all_event_markets_df is not None and not all_event_markets_df.empty:
        event_series = all_event_markets_df.get("event_ticker")
        if event_series is not None:
            event_values = event_series.dropna().astype(str).str.strip()
            if not event_values.empty:
                current_event_ticker = event_values.iloc[0]

    if not current_series_ticker and all_event_markets_df is not None and not all_event_markets_df.empty:
        series_series = all_event_markets_df.get("series_ticker")
        if series_series is not None:
            series_values = series_series.dropna().astype(str).str.strip()
            if not series_values.empty:
                current_series_ticker = series_values.iloc[0]

    # Keep all held contract tickers available for monitoring workflows.
    # Daily positions may belong to the immediately prior event while still
    # being economically valid to hold through settlement. Do not pre-filter
    # them away here just because the active event scope has advanced.
    filtered_force_include_contract_tickers = [
        str(t).strip() for t in force_include_contract_tickers if str(t).strip()
    ]

    if len(filtered_force_include_contract_tickers) != len(force_include_contract_tickers):
        logging.info(
            "Normalized held contract tickers for monitoring | original=%s | normalized=%s | event_ticker=%s | series_ticker=%s",
            len(force_include_contract_tickers),
            len(filtered_force_include_contract_tickers),
            current_event_ticker,
            current_series_ticker,
        )

    monitored_contracts_df = force_include_contracts_for_monitoring(
        all_event_markets_df=all_event_markets_df,
        candidate_markets_df=candidate_contracts_df,
        force_include_contract_tickers=filtered_force_include_contract_tickers,
    )
    monitored_contracts_df = apply_liquidity_filters(monitored_contracts_df, config)

    # ðŸ”¥ FIXED BLOCK (NO CRASH)
    if monitored_contracts_df.empty:
        logging.warning(
            "No Kalshi contracts matched the current active event after strike/liquidity filters. Skipping cycle."
        )
        return {
            "skipped": True,
            "skip_reason": "No Kalshi contracts matched after filters",
            "price": price,
            "price_src": price_src,
            "vol_stats": vol_stats,
            "vol_src": vol_src,
            "history_df": history_df,
            "all_event_markets_df": all_event_markets_df,
            "market_contracts_df": pd.DataFrame(),
            "candidate_contracts_df": candidate_contracts_df,
            "evaluated_df": pd.DataFrame(),
            "ranked_df": pd.DataFrame(),
            "open_positions_df": pd.DataFrame(),
            "trade_events": [],
            "logged_trades_df": pd.DataFrame(),
            "force_include_contract_tickers": filtered_force_include_contract_tickers,
        }

    contracts = build_contracts_from_market_df(monitored_contracts_df)
    vol_stats["history_df"] = history_df
    evaluated_df = evaluate_ladder(price, contracts, vol_stats, config)
    ranked_df = rank_trade_candidates(evaluated_df)
    open_positions_df = pd.DataFrame()
    trade_events = []
    logged_trades_df = log_trade_candidates(
        ranked_df,
        monitored_contracts_df,
        price,
        price_src,
        vol_stats,
        vol_src,
        config,
    )

    return {
        "skipped": False,
        "skip_reason": None,
        "price": price,
        "price_src": price_src,
        "vol_stats": vol_stats,
        "vol_src": vol_src,
        "history_df": history_df,
        "all_event_markets_df": all_event_markets_df,
        "market_contracts_df": monitored_contracts_df,
        "candidate_contracts_df": candidate_contracts_df,
        "evaluated_df": evaluated_df,
        "ranked_df": ranked_df,
        "open_positions_df": open_positions_df,
        "trade_events": trade_events,
        "logged_trades_df": logged_trades_df,
        "force_include_contract_tickers": filtered_force_include_contract_tickers,
    }


# ---------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------
get_live_gold_price_yahoo = get_live_oil_price_yahoo
get_live_gold_price_twelvedata = get_live_oil_price_twelvedata
get_live_gold_price_from_chain = get_live_oil_price_from_chain
get_live_gold_price_cached = get_live_oil_price_cached
get_gold_price_history_yahoo = get_oil_price_history_yahoo
get_gold_price_history_twelvedata = get_oil_price_history_twelvedata
get_gold_history_from_chain = get_oil_history_from_chain
