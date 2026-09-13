"""Deterministic causal terminal features."""

from __future__ import annotations

import pandas as pd

from .operators import safe_div


RAW_FIELDS = frozenset({
    "open", "high", "low", "close", "volume", "quote_volume", "trade_count",
    "taker_buy_base_volume", "taker_buy_quote_volume",
})

DERIVED_FIELDS = frozenset({
    "return_1d", "range_relative", "body_relative", "volume_relative_20",
    "quote_volume_relative_20", "taker_base_ratio", "taker_quote_ratio",
    "quote_per_trade",
})

TERMINAL_FIELDS = RAW_FIELDS | DERIVED_FIELDS

# Structural labels are conservative exposure hints, not economic attribution.
FEATURE_FAMILIES = {
    "price": frozenset({"open", "high", "low", "close", "return_1d"}),
    "volatility": frozenset({"range_relative", "body_relative"}),
    "activity": frozenset({"volume", "quote_volume", "trade_count", "volume_relative_20",
                           "quote_volume_relative_20", "quote_per_trade"}),
    "buy_flow": frozenset({"taker_buy_base_volume", "taker_buy_quote_volume", "taker_base_ratio", "taker_quote_ratio"}),
}


def expression_families(tree):
    """Count every referenced family; mixed expressions cannot evade caps."""
    if not tree.children:
        field = tree.field if tree.op == "field" else tree.op
        return frozenset(name for name, fields in FEATURE_FAMILIES.items() if field in fields)
    families = frozenset().union(*(expression_families(child) for child in tree.children))
    if tree.op == "rolling_std" and families == {"price"}:
        return families | {"volatility"}
    return families

TERMINAL_HISTORY = {
    "return_1d": 1,
    "range_relative": 0,
    "body_relative": 0,
    "volume_relative_20": 19,
    "quote_volume_relative_20": 19,
    "taker_base_ratio": 0,
    "taker_quote_ratio": 0,
    "quote_per_trade": 0,
}

TERMINAL_DEPENDENCIES = {
    "return_1d": {"close"},
    "range_relative": {"high", "low", "close"},
    "body_relative": {"close", "open"},
    "volume_relative_20": {"volume"},
    "quote_volume_relative_20": {"quote_volume"},
    "taker_base_ratio": {"taker_buy_base_volume", "volume"},
    "taker_quote_ratio": {"taker_buy_quote_volume", "quote_volume"},
    "quote_per_trade": {"quote_volume", "trade_count"},
}


def evaluate_terminal(field: str, data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if field in RAW_FIELDS:
        return data_ctx[field].astype("float64")
    if field == "return_1d":
        close = data_ctx["close"]
        return safe_div(close, close.shift(1)) - 1.0
    if field == "range_relative":
        close = data_ctx["close"]
        return safe_div(data_ctx["high"] - data_ctx["low"], close)
    if field == "body_relative":
        close = data_ctx["close"]
        return safe_div(close - data_ctx["open"], data_ctx["open"])
    if field == "volume_relative_20":
        return safe_div(data_ctx["volume"], data_ctx["volume"].rolling(20, min_periods=20).mean())
    if field == "quote_volume_relative_20":
        return safe_div(
            data_ctx["quote_volume"],
            data_ctx["quote_volume"].rolling(20, min_periods=20).mean(),
        )
    if field == "taker_base_ratio":
        return safe_div(data_ctx["taker_buy_base_volume"], data_ctx["volume"])
    if field == "taker_quote_ratio":
        return safe_div(data_ctx["taker_buy_quote_volume"], data_ctx["quote_volume"])
    if field == "quote_per_trade":
        return safe_div(data_ctx["quote_volume"], data_ctx["trade_count"])
    raise ValueError(f"unknown field: {field}")


__all__ = [
    "RAW_FIELDS", "DERIVED_FIELDS", "TERMINAL_FIELDS", "TERMINAL_HISTORY",
    "TERMINAL_DEPENDENCIES", "evaluate_terminal",
]
