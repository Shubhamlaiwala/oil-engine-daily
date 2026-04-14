from __future__ import annotations

import base64
import logging
import time
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_ORDER_TIME_IN_FORCE_DEFAULT = "good_till_canceled"
KALSHI_VALID_TIME_IN_FORCE = {
    "good_till_canceled",
    "immediate_or_cancel",
    "fill_or_kill",
}


def _normalize_time_in_force(value: Optional[str]) -> str:
    """Normalize common aliases to the exact Kalshi REST v2 enum strings."""
    if value is None:
        return KALSHI_ORDER_TIME_IN_FORCE_DEFAULT

    raw = str(value).strip().lower()
    if not raw:
        return KALSHI_ORDER_TIME_IN_FORCE_DEFAULT

    if raw in {"gtc", "good_till_canceled", "good_till_cancelled"}:
        return "good_till_canceled"

    if raw in {"ioc", "immediate_or_cancel"}:
        return "immediate_or_cancel"

    if raw in {"fok", "fill_or_kill"}:
        return "fill_or_kill"

    return KALSHI_ORDER_TIME_IN_FORCE_DEFAULT

class KalshiAuthClient:
    def __init__(self, api_key_id: str, private_key_path: str, base_url: str = KALSHI_API_BASE):
        self.api_key_id = api_key_id
        self.private_key_path = private_key_path
        self.base_url = (base_url or KALSHI_API_BASE).rstrip("/")
        self.private_key = self._load_private_key()

    def _load_private_key(self):
        with open(self.private_key_path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    def _sign_request(self, timestamp_ms: str, method: str, path: str) -> str:
        path_without_query = path.split("?")[0]
        message = f"{timestamp_ms}{method.upper()}{path_without_query}".encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
        json: Optional[dict] = None,
        data: Any = None,
        headers: Optional[dict] = None,
        timeout: int = 20,
    ):
        """Backward-compatible authenticated request helper."""
        method = method.upper()
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        full_url = self.base_url + endpoint
        timestamp_ms = str(int(time.time() * 1000))
        sign_path = urlparse(full_url).path

        signature = self._sign_request(
            timestamp_ms=timestamp_ms,
            method=method,
            path=sign_path,
        )

        request_headers = {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }
        if headers:
            request_headers.update(headers)

        body_json = json if json is not None else json_body
        if body_json is not None and "Content-Type" not in request_headers:
            request_headers["Content-Type"] = "application/json"

        response = requests.request(
            method=method,
            url=full_url,
            headers=request_headers,
            params=params,
            json=body_json,
            data=data,
            timeout=timeout,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            response_text = ""
            try:
                response_text = response.text
            except Exception:
                response_text = "<unable to read response text>"

            logging.warning(
                "Kalshi HTTP error | method=%s endpoint=%s status=%s response=%s",
                method,
                endpoint,
                getattr(response, "status_code", None),
                response_text[:1000],
            )
            raise exc

        try:
            return response.json()
        except ValueError:
            logging.warning(
                "Kalshi response was not valid JSON | method=%s endpoint=%s status=%s",
                method,
                endpoint,
                getattr(response, "status_code", None),
            )
            return {"raw_text": response.text}

    def get_balance(self):
        return self._request("GET", "/portfolio/balance")

    def get_positions(
        self,
        ticker: str | None = None,
        event_ticker: str | None = None,
        settlement_status: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ):
        params = {}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if settlement_status:
            params["settlement_status"] = settlement_status
        if limit is not None:
            params["limit"] = int(limit)
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/portfolio/positions", params=params or None)

    def get_fills(
        self,
        ticker: str | None = None,
        order_id: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ):
        params = {"limit": int(limit)}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id
        if min_ts is not None:
            params["min_ts"] = int(min_ts)
        if max_ts is not None:
            params["max_ts"] = int(max_ts)
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/portfolio/fills", params=params)

    def get_orders(
        self,
        ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ):
        params = {"limit": int(limit)}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/portfolio/orders", params=params)

    def get_order(self, order_id: str):
        return self._request("GET", f"/portfolio/orders/{order_id}")

    def cancel_order(self, order_id: str):
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    def submit_order(
        self,
        *,
        ticker: str,
        side: str,
        action: str,
        count: int,
        max_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        subaccount: int = 0,
        time_in_force: str = KALSHI_ORDER_TIME_IN_FORCE_DEFAULT,
        post_only: bool = False,
        reduce_only: bool = False,
        cancel_order_on_pause: bool = True,
    ) -> dict:
        side_norm = str(side or "").strip().lower()
        action_norm = str(action or "").strip().lower()
        tif_norm = _normalize_time_in_force(time_in_force)
        ticker_norm = str(ticker or "").strip()

        if side_norm not in {"yes", "no"}:
            raise ValueError(f"Unsupported side for Kalshi order submit: {side}")
        if action_norm not in {"buy", "sell"}:
            raise ValueError(f"Unsupported order action for Kalshi order submit: {action}")
        if not ticker_norm:
            raise ValueError("Ticker is required for Kalshi order submit.")

        count_int = int(float(count))
        if count_int <= 0:
            raise ValueError(f"Order count must be positive, got {count}")

        price_cents: Optional[int] = None
        limit_price_decimal: Optional[float] = None
        if max_price is not None:
            limit_price_decimal = float(max_price)
            if not (0 < limit_price_decimal < 1):
                raise ValueError(f"max_price must be within (0, 1), got {limit_price_decimal}")
            price_cents = max(1, min(99, int(round(limit_price_decimal * 100))))

        payload = {
            "ticker": ticker_norm,
            "action": action_norm,
            "side": side_norm,
            "type": "limit",
            "count": count_int,
            "time_in_force": tif_norm,
        }

        # NOTE:
        # elections API has been rejecting orders with `client_order_id`
        # in this workflow, so we intentionally do not send it for now.
        _unused_client_order_id = str(client_order_id) if client_order_id else None

        if subaccount:
            payload["subaccount"] = int(subaccount)

        if price_cents is not None:
            if side_norm == "yes":
                payload["yes_price"] = int(price_cents)
            else:
                payload["no_price"] = int(price_cents)

        logging.info(
            "Kalshi submit order request | ticker=%s side=%s action=%s count=%s "
            "limit_price=%s price_cents=%s tif=%s post_only=%s reduce_only=%s "
            "cancel_order_on_pause=%s client_order_id_supplied=%s payload_keys=%s payload_preview=%s",
            ticker_norm,
            side_norm,
            action_norm,
            count_int,
            limit_price_decimal,
            price_cents,
            tif_norm,
            bool(post_only),
            bool(reduce_only),
            bool(cancel_order_on_pause),
            bool(_unused_client_order_id),
            sorted(payload.keys()),
            payload,
        )

        return self._request("POST", "/portfolio/orders", json_body=payload)


def _to_dict(payload):
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "to_dict"):
        return payload.to_dict()
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    return {}


def _extract_records(payload: dict, preferred_keys: list[str]) -> list[dict]:
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, list):
            normalized = []
            for item in value:
                if isinstance(item, dict):
                    normalized.append(item)
                else:
                    normalized.append(_to_dict(item))
            return normalized
    return []


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _empty_object_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def _empty_float_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _first_present_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = _empty_object_series(df)
    for col in candidates:
        if col in df.columns:
            candidate = df[col]
            out = out.where(out.notna(), candidate)
    return out


def _normalize_price_series(
    df: pd.DataFrame,
    dollar_candidates: list[str],
    raw_candidates: list[str],
) -> pd.Series:
    out = _empty_float_series(df)

    for col in dollar_candidates:
        if col in df.columns:
            numeric = _to_numeric_series(df[col])
            out = out.where(out.notna(), numeric)

    for col in raw_candidates:
        if col in df.columns:
            numeric = _to_numeric_series(df[col])
            if numeric.notna().any():
                normalized = numeric.where(numeric <= 1.0, numeric / 100.0)
                out = out.where(out.notna(), normalized)

    return out


def _normalize_total_dollar_series(
    df: pd.DataFrame,
    dollar_candidates: list[str],
    raw_candidates: list[str],
) -> pd.Series:
    out = _empty_float_series(df)

    for col in dollar_candidates:
        if col in df.columns:
            numeric = _to_numeric_series(df[col])
            out = out.where(out.notna(), numeric)

    for col in raw_candidates:
        if col in df.columns:
            numeric = _to_numeric_series(df[col])
            if numeric.notna().any():
                normalized = numeric.where(numeric <= 1.0, numeric / 100.0)
                out = out.where(out.notna(), normalized)

    return out


def _infer_action_from_row(row: pd.Series) -> Optional[str]:
    explicit_side_candidates = ["side", "position_side", "market_side", "yes_no"]
    for col in explicit_side_candidates:
        if col in row.index and pd.notna(row[col]):
            side_text = str(row[col]).strip().upper()
            if side_text in {"YES", "BUY_YES", "LONG_YES"}:
                return "BUY_YES"
            if side_text in {"NO", "BUY_NO", "LONG_NO"}:
                return "BUY_NO"

    signed_position = row.get("position_numeric")
    if pd.notna(signed_position):
        try:
            signed_position = float(signed_position)
            if signed_position > 0:
                return "BUY_YES"
            if signed_position < 0:
                return "BUY_NO"
        except Exception:
            return None

    return None


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _select_scope_ticker_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="string")

    candidates = []
    for col in ["contract_ticker", "ticker"]:
        if col in df.columns:
            candidates.append(_normalize_text_series(df[col]))

    if not candidates:
        return pd.Series([""] * len(df), index=df.index, dtype="string")

    out = candidates[0].copy()
    for candidate in candidates[1:]:
        out = out.where(out != "", candidate)
    return out.astype("string")


def _compute_positions_value_from_df(scoped_positions_df: pd.DataFrame) -> float:
    if scoped_positions_df is None or scoped_positions_df.empty:
        return 0.0

    df = scoped_positions_df.copy()

    if "contracts" in df.columns:
        contracts = pd.to_numeric(df["contracts"], errors="coerce").fillna(0.0).abs()
        df = df[contracts > 0].copy()
        if df.empty:
            return 0.0

    if "market_value" in df.columns:
        market_value = pd.to_numeric(df["market_value"], errors="coerce")
        if market_value.notna().any():
            return float(market_value.fillna(0.0).sum())

    current_price = (
        pd.to_numeric(df["current_price"], errors="coerce")
        if "current_price" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    entry_price = (
        pd.to_numeric(df["entry_price"], errors="coerce")
        if "entry_price" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    contracts = (
        pd.to_numeric(df["contracts"], errors="coerce").fillna(0.0).abs()
        if "contracts" in df.columns
        else pd.Series(0.0, index=df.index, dtype="float64")
    )

    effective_price = current_price.where(current_price.notna(), entry_price)
    if effective_price.notna().any():
        return float((effective_price.fillna(0.0) * contracts).sum())

    position_cost = (
        pd.to_numeric(df["position_cost"], errors="coerce")
        if "position_cost" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    if position_cost.notna().any():
        return float(position_cost.fillna(0.0).sum())

    return 0.0


def positions_to_dataframe(payload) -> pd.DataFrame:
    payload_dict = _to_dict(payload)

    records = _extract_records(payload_dict, ["market_positions", "positions"])
    if not records and "event_positions" in payload_dict and isinstance(payload_dict["event_positions"], list):
        records = _extract_records(payload_dict, ["event_positions"])

    df = pd.DataFrame(records)

    if df.empty:
        logging.info("Kalshi positions dataframe empty after payload extraction.")
        return df

    numeric_candidates = [
        "position",
        "position_fp",
        "quantity",
        "contracts",
        "size",
        "total_traded",
        "total_traded_fp",
        "total_traded_dollars",
        "fees_paid",
        "fees_paid_fp",
        "fees_paid_dollars",
        "realized_pnl",
        "realized_pnl_fp",
        "realized_pnl_dollars",
        "resting_orders_count",
        "average_price",
        "average_price_dollars",
        "avg_price",
        "cost_basis",
        "cost_basis_dollars",
        "market_value",
        "market_value_dollars",
        "market_exposure",
        "market_exposure_fp",
        "market_exposure_dollars",
        "mark_price",
        "mark_price_dollars",
        "last_price",
        "last_price_dollars",
        "current_price",
        "current_price_dollars",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ticker_series = _first_present_series(df, ["ticker", "market_ticker", "contract_ticker"])
    df["contract_ticker"] = ticker_series.astype("string").fillna("").str.strip()
    df["ticker"] = df["contract_ticker"]

    df["position_numeric"] = _to_numeric_series(_first_present_series(df, ["position", "position_fp"]))
    df["action"] = df.apply(_infer_action_from_row, axis=1)
    df["side_norm"] = df["action"].map({"BUY_YES": "YES", "BUY_NO": "NO"})

    contracts_series = _to_numeric_series(
        _first_present_series(df, ["contracts", "quantity", "size"])
    )
    df["contracts"] = contracts_series.abs()
    df["size"] = df["contracts"]

    # Keep signed informational position separately, but do NOT use it as contract count.
    df["position_numeric"] = _to_numeric_series(_first_present_series(df, ["position", "position_fp"]))

    df["entry_price"] = _normalize_price_series(
        df,
        dollar_candidates=["average_price_dollars"],
        raw_candidates=["average_price", "avg_price"],
    )

    df["current_price"] = _normalize_price_series(
        df,
        dollar_candidates=["current_price_dollars", "mark_price_dollars", "last_price_dollars"],
        raw_candidates=["current_price", "mark_price", "last_price"],
    )

    df["position_cost"] = _normalize_total_dollar_series(
        df,
        dollar_candidates=["cost_basis_dollars", "total_traded_dollars"],
        raw_candidates=["cost_basis", "total_traded", "total_traded_fp"],
    )
    df["cost_basis_total"] = df["position_cost"]

    fallback_cost = df["entry_price"] * df["contracts"]
    df["position_cost"] = df["position_cost"].where(df["position_cost"].notna(), fallback_cost)
    df["cost_basis_total"] = df["cost_basis_total"].where(df["cost_basis_total"].notna(), fallback_cost)

    df["market_value"] = _normalize_total_dollar_series(
        df,
        dollar_candidates=["market_value_dollars", "market_exposure_dollars"],
        raw_candidates=["market_value", "market_exposure", "market_exposure_fp"],
    )

    fallback_market_value = df["current_price"] * df["contracts"]
    df["market_value"] = df["market_value"].where(df["market_value"].notna(), fallback_market_value)

    df["unrealized_pnl"] = df["market_value"] - df["position_cost"]

    if "position_id" not in df.columns:
        df["position_id"] = df["contract_ticker"]

    df["contract_ticker"] = df["contract_ticker"].replace({"": pd.NA})
    df = df[df["contract_ticker"].notna()].copy()

    logging.info("Kalshi normalized positions row count: %s", len(df))
    return df


def fills_to_dataframe(payload) -> pd.DataFrame:
    payload_dict = _to_dict(payload)
    records = _extract_records(payload_dict, ["fills"])
    return pd.DataFrame(records)


def orders_to_dataframe(payload) -> pd.DataFrame:
    payload_dict = _to_dict(payload)
    records = _extract_records(payload_dict, ["orders"])
    return pd.DataFrame(records)


def submit_kalshi_order(
    *,
    api_key_id: str,
    private_key_path: str,
    ticker: str,
    side: str,
    action: str,
    count: int,
    max_price: Optional[float] = None,
    client_order_id: Optional[str] = None,
    subaccount: int = 0,
    time_in_force: str = KALSHI_ORDER_TIME_IN_FORCE_DEFAULT,
    post_only: bool = False,
    reduce_only: bool = False,
    cancel_order_on_pause: bool = True,
    base_url: str = KALSHI_API_BASE,
) -> dict:
    client = KalshiAuthClient(api_key_id=api_key_id, private_key_path=private_key_path, base_url=base_url)
    return client.submit_order(
        ticker=ticker,
        side=side,
        action=action,
        count=count,
        max_price=max_price,
        client_order_id=client_order_id,
        subaccount=subaccount,
        time_in_force=time_in_force,
        post_only=post_only,
        reduce_only=reduce_only,
        cancel_order_on_pause=cancel_order_on_pause,
    )


def get_kalshi_balance_payload(api_key_id: str, private_key_path: str):
    client = KalshiAuthClient(api_key_id=api_key_id, private_key_path=private_key_path)
    return client.get_balance()


def _extract_first_numeric(payload_dict: dict, candidates: list[str]) -> Optional[float]:
    for key in candidates:
        if key in payload_dict:
            try:
                value = payload_dict.get(key)
                if value is None:
                    continue
                if pd.isna(value):
                    continue
                return float(value)
            except Exception:
                continue
    return None


def _normalize_balance_number(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None

    try:
        value = float(value)
    except Exception:
        return None

    if value >= 10:
        return value / 100.0

    return value


def balance_payload_to_account_snapshot(payload) -> dict:
    payload_dict = _to_dict(payload)
    logging.info("Raw balance payload: %s", payload_dict)

    cash_balance = _extract_first_numeric(
        payload_dict,
        ["cash_balance", "cash", "balance", "available_cash", "free_cash"],
    )
    portfolio_value = _extract_first_numeric(
        payload_dict,
        ["portfolio_value", "account_value", "equity", "total_value", "net_liquidation_value"],
    )
    positions_value = _extract_first_numeric(
        payload_dict,
        ["positions_value", "market_exposure", "open_positions_value"],
    )

    cash_balance = _normalize_balance_number(cash_balance)
    portfolio_value = _normalize_balance_number(portfolio_value)
    positions_value = _normalize_balance_number(positions_value)

    return {
        "cash_balance": None if cash_balance is None else round(float(cash_balance), 2),
        "portfolio_value": None if portfolio_value is None else round(float(portfolio_value), 2),
        "positions_value": None if positions_value is None else round(float(positions_value), 2),
        "raw_balance_payload": payload_dict,
    }


def filter_positions_to_scope(
    positions_df: pd.DataFrame,
    event_ticker: str | None = None,
    series_ticker: str | None = None,
    exact_market_ticker: str | None = None,
) -> pd.DataFrame:
    if positions_df is None:
        logging.info("Scope filter received positions_df=None; returning empty DataFrame.")
        return pd.DataFrame()

    if positions_df.empty:
        logging.info("Scope filter received empty positions DataFrame.")
        return positions_df.copy()

    scoped_df = positions_df.copy()
    scope_series = _select_scope_ticker_series(scoped_df)

    if scope_series.empty:
        logging.info("Scope filter could not find ticker-like columns; returning original DataFrame.")
        return scoped_df

    applied_scope = "unfiltered"
    mask = pd.Series(True, index=scoped_df.index, dtype="bool")

    exact_market_ticker_norm = (exact_market_ticker or "").strip()
    event_ticker_norm = (event_ticker or "").strip()
    series_ticker_norm = (series_ticker or "").strip()

    if exact_market_ticker_norm:
        applied_scope = f"exact_market_ticker={exact_market_ticker_norm}"
        mask = scope_series.eq(exact_market_ticker_norm)
    elif event_ticker_norm:
        applied_scope = f"event_ticker={event_ticker_norm}"
        mask = scope_series.str.startswith(event_ticker_norm, na=False)
    elif series_ticker_norm:
        applied_scope = f"series_ticker={series_ticker_norm}"
        mask = scope_series.str.startswith(series_ticker_norm, na=False)

    filtered_df = scoped_df.loc[mask].copy()

    logging.info(
        "Kalshi scope filter applied (%s): input_rows=%s output_rows=%s",
        applied_scope,
        len(scoped_df),
        len(filtered_df),
    )

    if not filtered_df.empty and "contract_ticker" in filtered_df.columns:
        preview = filtered_df["contract_ticker"].astype("string").fillna("").head(5).tolist()
        logging.info("Kalshi scope filter preview tickers: %s", preview)

    return filtered_df


def build_scoped_account_snapshot(
    raw_account_snapshot: dict | None,
    scoped_positions_df: pd.DataFrame | None,
) -> dict:
    raw_snapshot = raw_account_snapshot or {}
    cash_balance = raw_snapshot.get("cash_balance")

    try:
        cash_balance = None if cash_balance is None else float(cash_balance)
    except Exception:
        cash_balance = None

    positions_value = _compute_positions_value_from_df(scoped_positions_df)
    portfolio_value = None if cash_balance is None else cash_balance + positions_value
    updated_ts = int(time.time())

    scoped_snapshot = {
        "cash_balance": None if cash_balance is None else round(float(cash_balance), 2),
        "portfolio_value": None if portfolio_value is None else round(float(portfolio_value), 2),
        "positions_value": round(float(positions_value), 2),
        "updated_ts": updated_ts,
        "raw_account_snapshot": raw_snapshot,
    }

    logging.info(
        "Kalshi scoped account snapshot: cash_balance=%s portfolio_value=%s positions_value=%s rows=%s",
        scoped_snapshot.get("cash_balance"),
        scoped_snapshot.get("portfolio_value"),
        scoped_snapshot.get("positions_value"),
        0 if scoped_positions_df is None else len(scoped_positions_df),
    )

    return scoped_snapshot


def get_kalshi_account_snapshot(
    api_key_id: str,
    private_key_path: str,
    ticker: str | None = None,
    event_ticker: str | None = None,
) -> dict:
    client = KalshiAuthClient(api_key_id=api_key_id, private_key_path=private_key_path)

    balance_payload = client.get_balance()
    raw_account_snapshot = balance_payload_to_account_snapshot(balance_payload)

    try:
        positions_payload = client.get_positions(ticker=ticker, event_ticker=event_ticker)
        positions_df = positions_to_dataframe(positions_payload)

        if not positions_df.empty:
            active_df = positions_df[positions_df["contracts"].fillna(0).abs() > 0].copy()

            if raw_account_snapshot.get("positions_value") is None:
                positions_value = _compute_positions_value_from_df(active_df)
                raw_account_snapshot["positions_value"] = round(float(positions_value), 2)

            if raw_account_snapshot.get("portfolio_value") is None:
                cash_balance = raw_account_snapshot.get("cash_balance")
                positions_value = raw_account_snapshot.get("positions_value")
                if cash_balance is not None and positions_value is not None:
                    raw_account_snapshot["portfolio_value"] = round(
                        float(cash_balance) + float(positions_value),
                        2,
                    )

    except Exception as exc:
        logging.warning("Failed to enrich Kalshi raw/global account snapshot with positions fallback: %s", exc)

    logging.info(
        "Kalshi raw/global account snapshot: cash_balance=%s portfolio_value=%s positions_value=%s",
        raw_account_snapshot.get("cash_balance"),
        raw_account_snapshot.get("portfolio_value"),
        raw_account_snapshot.get("positions_value"),
    )

    return raw_account_snapshot


def get_kalshi_live_positions_df(
    api_key_id: str,
    private_key_path: str,
    ticker: str | None = None,
    event_ticker: str | None = None,
    subaccount: int = 0,
) -> pd.DataFrame:
    del subaccount

    client = KalshiAuthClient(api_key_id=api_key_id, private_key_path=private_key_path)
    payload = client.get_positions(ticker=ticker, event_ticker=event_ticker)

    payload_dict = _to_dict(payload)
    logging.info("Kalshi positions payload keys: %s", sorted(list(payload_dict.keys())))

    df = positions_to_dataframe(payload)

    if df.empty:
        logging.info("Kalshi live positions dataframe is empty.")
        return df

    preview_cols = [
        col
        for col in [
            "contract_ticker",
            "ticker",
            "position",
            "position_fp",
            "position_numeric",
            "action",
            "side_norm",
            "contracts",
            "entry_price",
            "position_cost",
            "market_value",
            "current_price",
            "unrealized_pnl",
            "average_price",
            "average_price_dollars",
            "cost_basis",
            "cost_basis_dollars",
            "total_traded_dollars",
            "market_exposure_dollars",
        ]
        if col in df.columns
    ]

    logging.info("Kalshi live positions columns: %s", list(df.columns))
    logging.info(
        "Kalshi live positions preview: %s",
        df[preview_cols].head(10).to_dict(orient="records"),
    )

    return df
