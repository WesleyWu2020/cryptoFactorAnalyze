"""Binance USDT-M perpetual market-data adapters."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from .http import JsonHttpClient

BINANCE_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
DAY_MS = 86_400_000


def _datetime_from_ms(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, unit="ms", utc=True).tz_localize(None)


def _cast_columns(frame: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    for column, dtype in dtypes.items():
        frame[column] = frame[column].astype(dtype)
    return frame


def normalize_exchange_info(payload: object, fetched_at: pd.Timestamp) -> pd.DataFrame:
    symbols = payload["symbols"]  # type: ignore[index]
    rows = [
        {
            "binance_symbol": item["symbol"],
            "base_asset": item["baseAsset"],
            "quote_asset": item["quoteAsset"],
            "contract_type": item["contractType"],
            "onboard_date": _datetime_from_ms(item["onboardDate"]),
            "status": item["status"],
            "fetched_at_utc": pd.to_datetime(fetched_at, utc=True).tz_localize(None),
        }
        for item in symbols
        if item.get("quoteAsset") == "USDT" and item.get("contractType") == "PERPETUAL"
    ]
    columns = ["binance_symbol", "base_asset", "quote_asset", "contract_type", "onboard_date", "status", "fetched_at_utc"]
    return _cast_columns(
        pd.DataFrame(rows, columns=columns),
        {"onboard_date": "datetime64[ns]", "fetched_at_utc": "datetime64[ns]"},
    )


def fetch_exchange_info(client: JsonHttpClient, fetched_at: pd.Timestamp | None = None) -> pd.DataFrame:
    payload = client.get_json(BINANCE_EXCHANGE_INFO_URL)
    return normalize_exchange_info(payload, fetched_at or pd.Timestamp.now(tz="UTC"))


def _normalize_klines(rows: list[list[Any]], symbol: str) -> pd.DataFrame:
    columns = ["date", "symbol", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume"]
    normalized = []
    for row in rows:
        normalized.append([
            _datetime_from_ms(row[0]), symbol, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]),
            _datetime_from_ms(row[6]), float(row[7]), int(row[8]), float(row[9]), float(row[10]),
        ])
    return _cast_columns(
        pd.DataFrame(normalized, columns=columns),
        {
            "date": "datetime64[ns]", "close_time": "datetime64[ns]",
            "open": "float64", "high": "float64", "low": "float64",
            "close": "float64", "volume": "float64",
            "quote_asset_volume": "float64", "trade_count": "int64",
            "taker_buy_base_volume": "float64", "taker_buy_quote_volume": "float64",
        },
    )


def fetch_daily_klines(client: JsonHttpClient, symbol: str, start: date, end: date) -> pd.DataFrame:
    start_ms = pd.Timestamp(start).value // 1_000_000
    completed_end = pd.Timestamp(end + timedelta(days=1)) - pd.Timedelta(milliseconds=1)
    end_ms = completed_end.value // 1_000_000
    cursor = start_ms
    rows: list[list[Any]] = []
    while cursor <= end_ms:
        page = client.get_json(BINANCE_KLINES_URL, params={"symbol": symbol, "interval": "1d", "startTime": cursor, "endTime": end_ms, "limit": 1500})
        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1][0]) + DAY_MS
        if next_cursor <= cursor:
            raise ValueError("Binance kline cursor did not advance")
        cursor = next_cursor
    out = _normalize_klines(rows, symbol)
    if not out.empty:
        out = out[(out["close_time"] <= completed_end) & (out["date"] >= pd.Timestamp(start))]
        out = out.drop_duplicates(["date", "symbol"]).sort_values("date").reset_index(drop=True)
    return out


def fetch_funding_events(client: JsonHttpClient, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    cursor = start_ms
    rows: list[dict[str, Any]] = []
    while cursor <= end_ms:
        page = client.get_json(BINANCE_FUNDING_URL, params={"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000})
        if not page:
            break
        rows.extend({"symbol": item.get("symbol", symbol), "funding_time": _datetime_from_ms(item["fundingTime"]), "funding_rate": float(item["fundingRate"]), "mark_price": float(item["markPrice"]) if item.get("markPrice") not in (None, "") else float("nan"), "rate_type": item.get("rateType") or "Regular"} for item in page)
        if len(page) < 1000:
            break
        next_cursor = int(page[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise ValueError("Binance funding cursor did not advance")
        cursor = next_cursor
    columns = ["symbol", "funding_time", "funding_rate", "mark_price", "rate_type"]
    out = pd.DataFrame(rows, columns=columns)
    out = _cast_columns(
        out,
        {"funding_time": "datetime64[ns]", "funding_rate": "float64", "mark_price": "float64"},
    )
    if out.empty:
        return out
    return out.drop_duplicates(["funding_time", "symbol", "rate_type"]).sort_values("funding_time").reset_index(drop=True)


def has_completed_daily_kline(client: JsonHttpClient, symbol: str, decision_date: date) -> bool:
    target = decision_date - timedelta(days=1)
    rows = fetch_daily_klines(client, symbol, target, target)
    return len(rows) == 1 and rows.iloc[0]["date"] == pd.Timestamp(target)
