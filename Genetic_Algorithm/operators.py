"""Numerical, causal operators for expression trees."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a.div(b.where(b.abs() > 1e-12)).replace([np.inf, -np.inf], np.nan)


def add(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a + b


def subtract(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a - b


def multiply(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a * b


def negate(a: pd.DataFrame) -> pd.DataFrame:
    return -a


def absolute(a: pd.DataFrame) -> pd.DataFrame:
    return a.abs()


def lag(a: pd.DataFrame, window: int) -> pd.DataFrame:
    return a.shift(window)


def delta(a: pd.DataFrame, window: int) -> pd.DataFrame:
    return a - a.shift(window)


def rolling_mean(a: pd.DataFrame, window: int) -> pd.DataFrame:
    return a.rolling(window=window, min_periods=window, center=False).mean()


def rolling_std(a: pd.DataFrame, window: int) -> pd.DataFrame:
    return a.rolling(window=window, min_periods=window, center=False).std(ddof=1)


def rolling_min(a: pd.DataFrame, window: int) -> pd.DataFrame:
    return a.rolling(window=window, min_periods=window, center=False).min()


def rolling_max(a: pd.DataFrame, window: int) -> pd.DataFrame:
    return a.rolling(window=window, min_periods=window, center=False).max()


def rolling_correlation(a: pd.DataFrame, b: pd.DataFrame, window: int) -> pd.DataFrame:
    return a.rolling(window=window, min_periods=window, center=False).corr(b)


def cross_sectional_rank(
    a: pd.DataFrame, eligible: pd.DataFrame
) -> pd.DataFrame:
    masked = a.where(eligible.astype(bool))
    counts = masked.notna().sum(axis=1)
    ranked = masked.rank(axis=1, method="average", pct=True)
    return ranked.where(counts.gt(1), np.nan)


masked_rank = cross_sectional_rank
rolling_corr = rolling_correlation


OPERATORS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "negate": negate,
    "abs": absolute,
    "lag": lag,
    "delta": delta,
    "rolling_mean": rolling_mean,
    "rolling_std": rolling_std,
    "rolling_min": rolling_min,
    "rolling_max": rolling_max,
    "rolling_corr": rolling_correlation,
    "rank": cross_sectional_rank,
}
OPERATOR_REGISTRY = OPERATORS

OPERATOR_ARITY = {
    "add": 2,
    "subtract": 2,
    "multiply": 2,
    "negate": 1,
    "abs": 1,
    "lag": 1,
    "delta": 1,
    "rolling_mean": 1,
    "rolling_std": 1,
    "rolling_min": 1,
    "rolling_max": 1,
    "rolling_corr": 2,
    "rank": 1,
}

WINDOW_OPERATORS = frozenset({
    "lag", "delta", "rolling_mean", "rolling_std", "rolling_min",
    "rolling_max", "rolling_corr",
})
ROLLING_OPERATORS = frozenset({
    "rolling_mean", "rolling_std", "rolling_min", "rolling_max", "rolling_corr",
})
COMMUTATIVE_OPERATORS = frozenset({"add", "multiply"})


__all__ = [
    "OPERATORS", "OPERATOR_ARITY", "WINDOW_OPERATORS", "ROLLING_OPERATORS",
    "COMMUTATIVE_OPERATORS", "OPERATOR_REGISTRY", "safe_div", "add", "subtract", "multiply",
    "negate", "absolute", "lag", "delta", "rolling_mean", "rolling_std",
    "rolling_min", "rolling_max", "rolling_correlation", "rolling_corr",
    "cross_sectional_rank", "masked_rank",
]
