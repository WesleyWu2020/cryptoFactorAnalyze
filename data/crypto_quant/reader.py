"""Factor-safe readers for the crypto-quant HDF5 store."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd

from .schemas import TABLE_SPECS
from .store import CryptoQuantStore


_MARKET_COLUMNS = [column for column in TABLE_SPECS["klines_daily"].columns if column != "symbol"]
_MARKET_OUTPUT_COLUMNS = ["date", "instrument", *[column for column in _MARKET_COLUMNS if column != "date"]]


def _date_range(start: date, end: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        raise ValueError("start must be on or before end")
    return start_ts, end_ts


def _normalized_dates(values: pd.Series) -> pd.Series:
    parsed = values.map(pd.Timestamp)
    return pd.to_datetime(parsed, errors="raise", utc=True).dt.tz_localize(None).dt.normalize()


def _require_columns(frame: pd.DataFrame, columns: set[str], context: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{context} missing columns: {sorted(missing)}")


def _membership_by_date(universe: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[pd.Timestamp, set[str]]:
    required = {"effective_date", "effective_end_date", "binance_symbol"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"universe_monthly missing columns: {sorted(missing)}")
    data = universe.copy()
    data["_start"] = _normalized_dates(data["effective_date"])
    data["_end"] = _normalized_dates(data["effective_end_date"])
    data["_symbol"] = data["binance_symbol"].astype(str)
    result: dict[pd.Timestamp, set[str]] = {}
    for day in pd.date_range(start, end, freq="D"):
        active = data[(data["_start"] <= day) & (data["_end"].isna() | (data["_end"] >= day))]
        result[day] = set(active["_symbol"])
    return result


def load_market_history(
    path: str | Path,
    start: date,
    end: date,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """Load market history for symbols in memberships overlapping the request."""
    start_ts, end_ts = _date_range(start, end)
    if not isinstance(lookback_days, int) or isinstance(lookback_days, bool) or lookback_days < 0:
        raise ValueError("lookback_days must be a non-negative integer")
    store = CryptoQuantStore(Path(path))
    universe = store.read("universe_monthly")
    memberships = _membership_by_date(universe, start_ts, end_ts)
    symbols = set().union(*memberships.values()) if memberships else set()
    if not symbols:
        return pd.DataFrame(columns=_MARKET_OUTPUT_COLUMNS)

    kline_start = start_ts - pd.Timedelta(days=lookback_days)
    market = store.read(
        "klines_daily",
        where=f"date >= '{kline_start}' & date <= '{end_ts}'",
    )
    _require_columns(market, set(TABLE_SPECS["klines_daily"].columns), "klines_daily")
    market["date"] = _normalized_dates(market["date"])
    market = market[market["symbol"].astype(str).isin(symbols)]
    market = market.rename(columns={"symbol": "instrument"})
    return market.loc[:, _MARKET_OUTPUT_COLUMNS].sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)


def load_daily_universe(
    path: str | Path,
    start: date,
    end: date,
    require_complete: bool = True,
) -> dict[pd.Timestamp, set[str]]:
    """Load exact-date memberships, optionally requiring a complete research panel."""
    start_ts, end_ts = _date_range(start, end)
    store = CryptoQuantStore(Path(path))
    expected = _membership_by_date(store.read("universe_monthly"), start_ts, end_ts)
    panel = store.read(
        "research_panel_daily",
        where=f"date >= '{start_ts}' & date <= '{end_ts}'",
    )
    _require_columns(panel, {"date", "binance_symbol", "has_complete_kline", "has_complete_funding"}, "research_panel_daily")
    if panel.empty:
        return {}
    panel["date"] = _normalized_dates(panel["date"])
    panel = panel[panel["date"].between(start_ts, end_ts)].copy()
    panel["binance_symbol"] = panel["binance_symbol"].astype(str)
    result: dict[pd.Timestamp, set[str]] = {}
    for day, expected_symbols in expected.items():
        rows = panel[panel["date"] == day]
        actual = set(rows["binance_symbol"]) & expected_symbols
        rows = rows[rows["binance_symbol"].isin(actual)]
        complete = actual == expected_symbols
        complete &= rows["has_complete_kline"].fillna(False).all()
        complete &= rows["has_complete_funding"].fillna(False).all()
        if (not require_complete or complete) and actual:
            result[day] = actual
    return result


def filter_factor_output(factors: pd.DataFrame, universe_by_date: Mapping[pd.Timestamp, set[str]]) -> pd.DataFrame:
    """Keep factor rows whose normalized date and instrument are exact same-day members."""
    required = {"date", "instrument", "factor"}
    missing = required - set(factors.columns)
    if missing:
        raise ValueError("factors must contain date, instrument, factor")
    data = factors.loc[:, ["date", "instrument", "factor"]].copy()
    data["date"] = _normalized_dates(data["date"])
    normalized_universe = {}
    for day, symbols in universe_by_date.items():
        key = pd.Timestamp(day)
        if key.tzinfo is not None:
            key = key.tz_convert("UTC").tz_localize(None)
        normalized_universe[key.normalize()] = {str(symbol) for symbol in symbols}
    keep = [instrument in normalized_universe.get(day, set()) for day, instrument in zip(data["date"], data["instrument"].astype(str))]
    return data.loc[keep].reset_index(drop=True)


__all__ = ["filter_factor_output", "load_daily_universe", "load_market_history"]
