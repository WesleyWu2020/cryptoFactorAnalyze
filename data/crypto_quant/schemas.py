"""Schemas and normalization for the research HDF5 store."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TableSpec:
    columns: tuple[str, ...]
    key: tuple[str, ...]
    data_columns: tuple[str, ...]
    min_itemsize: dict[str, int]
    schema_version: int = 1


_STRING_SIZES = {
    "symbol": 32, "binance_symbol": 32, "cmc_symbol": 32, "base_asset": 32,
    "quote_asset": 16, "name": 256, "status": 32, "mapping_source": 64,
    "contract_type": 32,
    "rate_type": 32,
}
_STRING_COLUMNS = set(_STRING_SIZES) | {"rate_type"}
_INTEGER_COLUMNS = {"cmc_id"}
_BOOLEAN_COLUMNS = {"has_complete_kline", "has_complete_funding"}
_DATE_COLUMNS = {
    "date", "decision_date", "effective_date", "effective_end_date",
    "onboard_date", "valid_from", "valid_to", "universe_effective_date",
}


def _spec(columns: tuple[str, ...], key: tuple[str, ...]) -> TableSpec:
    query = tuple(c for c in columns if c in {"date", "decision_date", "effective_date", "funding_time", "valid_from", "valid_to", "symbol", "binance_symbol"})
    sizes = {c: _STRING_SIZES[c] for c in columns if c in _STRING_SIZES}
    return TableSpec(columns, key, query, sizes)


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    if value in (0, 1) and not isinstance(value, (str, bool)):
        return bool(value)
    raise ValueError(f"invalid boolean value: {value!r}")


def _strict_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid integer value: {value!r}")
    converted = pd.to_numeric(value, errors="raise")
    if pd.isna(converted) or int(converted) != converted:
        raise ValueError(f"invalid integer value: {value!r}")
    return int(converted)


TABLE_SPECS = {
    "cmc100_daily": _spec(("date", "index_value", "source_update_time", "fetched_at_utc"), ("date",)),
    "cmc100_constituents": _spec(("date", "cmc_id", "symbol", "name", "weight"), ("date", "cmc_id")),
    "futures_contracts": _spec(("cmc_id", "cmc_symbol", "binance_symbol", "base_asset", "quote_asset", "contract_type", "onboard_date", "status", "mapping_source", "valid_from", "valid_to"), ("cmc_id", "valid_from")),
    "klines_daily": _spec(("date", "symbol", "open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume"), ("date", "symbol")),
    "funding_events": _spec(("funding_time", "symbol", "funding_rate", "mark_price", "rate_type"), ("funding_time", "symbol", "rate_type")),
    "universe_monthly": _spec(("decision_date", "effective_date", "effective_end_date", "cmc_id", "cmc_symbol", "binance_symbol", "market_cap_rank", "cmc_weight"), ("effective_date", "binance_symbol")),
    "research_panel_daily": _spec(("date", "binance_symbol", "open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume", "decision_date", "universe_effective_date", "market_cap_rank", "cmc_weight_at_decision", "funding_rate_sum", "funding_rate_mean", "funding_rate_last", "funding_event_count", "has_complete_kline", "has_complete_funding"), ("date", "binance_symbol")),
}


def normalize_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if name not in TABLE_SPECS:
        raise KeyError(f"unknown table: {name}")
    spec = TABLE_SPECS[name]
    missing = [c for c in spec.columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns for {name}: {missing}")
    result = frame.loc[:, list(spec.columns)].copy()
    for column in result.columns:
        if column in spec.key and result[column].isna().any():
            raise ValueError(f"null primary-key value in {name}.{column}")
        if column in _DATE_COLUMNS:
            result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
        elif column.endswith("_time") or column == "fetched_at_utc" or column == "source_update_time":
            result[column] = pd.to_datetime(result[column], errors="raise", utc=True)
        elif column in _STRING_COLUMNS:
            result[column] = result[column].where(result[column].isna(), result[column].astype(str))
        elif column in _INTEGER_COLUMNS:
            result[column] = result[column].map(_strict_int).astype("int64")
        elif column in _BOOLEAN_COLUMNS:
            result[column] = result[column].map(_strict_bool).astype("bool")
        elif column in {"market_cap_rank", "trade_count", "funding_event_count"}:
            result[column] = pd.to_numeric(result[column], errors="raise")
        elif column not in spec.min_itemsize and result[column].dtype == object:
            if column not in {"rate_type"}:
                result[column] = pd.to_numeric(result[column], errors="ignore")
    if result.duplicated(list(spec.key), keep=False).any():
        conflicts = result[result.duplicated(list(spec.key), keep=False)]
        for _, group in conflicts.groupby(list(spec.key), sort=False, dropna=False):
            if len(group.drop_duplicates()) > 1:
                raise ValueError(f"conflicting duplicate primary keys in {name}")
        result = result.drop_duplicates(list(spec.key), keep="first")
    return result.sort_values(list(spec.key), kind="mergesort").reset_index(drop=True)
