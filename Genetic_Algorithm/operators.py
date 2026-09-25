"""Numerical, causal operators for expression trees.

Also hosts the dimension (量纲) system: terminal fields and operator outputs
carry physical-unit labels, and ``combine_dimensions`` rejects unit-incoherent
binary combinations. Enforcement happens in ``expression.validate_tree``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GP_SEMANTICS_VERSION = 2


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


def _validate_window(window: int) -> None:
    if type(window) is not int or window <= 0:
        raise ValueError("window must be positive")


def lag(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.shift(window)


def delta(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a - a.shift(window)


def rolling_mean(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).mean()


def rolling_std(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).std(ddof=1)


def rolling_min(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).min()


def rolling_max(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).max()


def rolling_correlation(a: pd.DataFrame, b: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).corr(b)


def cross_sectional_rank(
    a: pd.DataFrame, eligible: pd.DataFrame
) -> pd.DataFrame:
    masked = a.where(eligible.fillna(False).astype(bool))
    counts = masked.notna().sum(axis=1)
    ranked = masked.rank(axis=1, method="average", pct=True)
    return ranked.where(counts.gt(1), np.nan)


def cross_sectional_zscore(
    a: pd.DataFrame, eligible: pd.DataFrame
) -> pd.DataFrame:
    masked = a.where(eligible.fillna(False).astype(bool))
    counts = masked.notna().sum(axis=1)
    # Reduce each day's finite observations in a fixed order. Pandas block
    # layout can change with history length and perturb axis=1 reductions;
    # subsequent high-order rolling moments amplify those rounding errors.
    moments = []
    for row in masked.to_numpy(dtype=float):
        finite = row[np.isfinite(row)]
        moments.append((finite.mean(), finite.std(ddof=0)) if len(finite) else (np.nan, np.nan))
    mean = pd.Series([m[0] for m in moments], index=masked.index, dtype=float)
    std = pd.Series([m[1] for m in moments], index=masked.index, dtype=float)
    z = masked.sub(mean, axis=0).div(std.where(std.abs() > 1e-12), axis=0)
    return z.where(counts.gt(1), np.nan)


def rolling_median(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).median()


def _window_mad(x: np.ndarray) -> float:
    median = np.median(x)
    return float(np.median(np.abs(x - median)))


def rolling_mad(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).apply(_window_mad, raw=True)


def rolling_iqr(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    rolled = a.rolling(window=window, min_periods=window, center=False)
    return rolled.quantile(0.75) - rolled.quantile(0.25)


def rolling_skew(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).skew()


def rolling_kurt(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).kurt()


def _window_downside_std(x: np.ndarray) -> float:
    negatives = x[x < 0]
    if negatives.size < 2:
        return np.nan
    return float(negatives.std(ddof=1))


def rolling_downside_std(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).apply(_window_downside_std, raw=True)


def _window_ts_rank(x: np.ndarray) -> float:
    return float((x.argsort().argsort()[-1] + 1) / x.size)


def ts_rank(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).apply(_window_ts_rank, raw=True)


def ts_zscore(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return safe_div(a - rolling_mean(a, window), rolling_std(a, window))


def _window_argmax_norm(x: np.ndarray) -> float:
    return float((x.size - 1 - x.argmax()) / x.size)


def ts_argmax(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).apply(_window_argmax_norm, raw=True)


def _window_argmin_norm(x: np.ndarray) -> float:
    return float((x.size - 1 - x.argmin()) / x.size)


def ts_argmin(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).apply(_window_argmin_norm, raw=True)


def _window_slope_norm(x: np.ndarray) -> float:
    mean = x.mean()
    if abs(mean) < 1e-12:
        return np.nan
    slope = np.polyfit(np.arange(x.size, dtype=float), x, 1)[0]
    return float(slope / mean)


def rolling_slope_norm(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).apply(_window_slope_norm, raw=True)


def _window_r2(x: np.ndarray) -> float:
    if x.std() <= 0.0:
        return np.nan
    corr = np.corrcoef(x, np.arange(x.size, dtype=float))[0, 1]
    return float(corr * corr)


def rolling_r2(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).apply(_window_r2, raw=True)


def efficiency_ratio(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    numerator = delta(a, window).abs()
    denominator = a.diff().abs().rolling(window=window, min_periods=window, center=False).sum()
    return safe_div(numerator, denominator)


def rolling_hit_rate(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    positive = a.gt(0).astype("float64").where(np.isfinite(a))
    return positive.rolling(window=window, min_periods=window, center=False).mean()


def _window_autocorr(x: np.ndarray) -> float:
    if x.size < 2 or x.std() <= 0.0:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def rolling_autocorr(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).apply(_window_autocorr, raw=True)


def _window_max_drawdown(x: np.ndarray) -> float:
    running = np.maximum.accumulate(x)
    if running.max() <= 0.0:
        return np.nan
    drawdown = np.where(running > 0.0, x / np.where(running > 0.0, running, 1.0) - 1.0, np.nan)
    if np.isnan(drawdown).all():
        return np.nan
    return float(-np.nanmin(drawdown))


def rolling_max_drawdown(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return a.rolling(window=window, min_periods=window, center=False).apply(_window_max_drawdown, raw=True)


def rolling_cv(a: pd.DataFrame, window: int) -> pd.DataFrame:
    _validate_window(window)
    return safe_div(rolling_std(a, window), rolling_mean(a, window).abs())


def signed_sqrt(a: pd.DataFrame) -> pd.DataFrame:
    return np.sign(a) * np.sqrt(a.abs())


def safe_log(a: pd.DataFrame) -> pd.DataFrame:
    return np.sign(a) * np.log1p(a.abs())


masked_rank = cross_sectional_rank
rolling_corr = rolling_correlation


OPERATORS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "negate": negate,
    "abs": absolute,
    "safe_div": safe_div,
    "lag": lag,
    "delta": delta,
    "rolling_mean": rolling_mean,
    "rolling_std": rolling_std,
    "rolling_min": rolling_min,
    "rolling_max": rolling_max,
    "rolling_corr": rolling_correlation,
    "rank": cross_sectional_rank,
    "cross_sectional_zscore": cross_sectional_zscore,
    "rolling_median": rolling_median,
    "rolling_mad": rolling_mad,
    "rolling_iqr": rolling_iqr,
    "rolling_skew": rolling_skew,
    "rolling_kurt": rolling_kurt,
    "rolling_downside_std": rolling_downside_std,
    "ts_rank": ts_rank,
    "ts_zscore": ts_zscore,
    "ts_argmax": ts_argmax,
    "ts_argmin": ts_argmin,
    "rolling_slope_norm": rolling_slope_norm,
    "rolling_r2": rolling_r2,
    "efficiency_ratio": efficiency_ratio,
    "rolling_hit_rate": rolling_hit_rate,
    "rolling_autocorr": rolling_autocorr,
    "rolling_max_drawdown": rolling_max_drawdown,
    "rolling_cv": rolling_cv,
    "signed_sqrt": signed_sqrt,
    "safe_log": safe_log,
}
OPERATOR_REGISTRY = OPERATORS

OPERATOR_ARITY = {
    "add": 2,
    "subtract": 2,
    "multiply": 2,
    "safe_div": 2,
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
    "cross_sectional_zscore": 1,
    "rolling_median": 1,
    "rolling_mad": 1,
    "rolling_iqr": 1,
    "rolling_skew": 1,
    "rolling_kurt": 1,
    "rolling_downside_std": 1,
    "ts_rank": 1,
    "ts_zscore": 1,
    "ts_argmax": 1,
    "ts_argmin": 1,
    "rolling_slope_norm": 1,
    "rolling_r2": 1,
    "efficiency_ratio": 1,
    "rolling_hit_rate": 1,
    "rolling_autocorr": 1,
    "rolling_max_drawdown": 1,
    "rolling_cv": 1,
    "signed_sqrt": 1,
    "safe_log": 1,
}

WINDOW_OPERATORS = frozenset({
    "lag", "delta", "rolling_mean", "rolling_std", "rolling_min",
    "rolling_max", "rolling_corr", "rolling_median", "rolling_mad",
    "rolling_iqr", "rolling_skew", "rolling_kurt", "rolling_downside_std",
    "ts_rank", "ts_zscore", "ts_argmax", "ts_argmin", "rolling_slope_norm",
    "rolling_r2", "efficiency_ratio", "rolling_hit_rate", "rolling_autocorr",
    "rolling_max_drawdown", "rolling_cv",
})
ROLLING_OPERATORS = frozenset({
    "rolling_mean", "rolling_std", "rolling_min", "rolling_max", "rolling_corr",
    "rolling_median", "rolling_mad", "rolling_iqr", "rolling_skew",
    "rolling_kurt", "rolling_downside_std", "ts_rank", "ts_zscore",
    "ts_argmax", "ts_argmin", "rolling_slope_norm", "rolling_r2",
    "rolling_hit_rate", "rolling_autocorr", "rolling_max_drawdown",
    "rolling_cv",
})
# Lag-style operators shift by the full window; rolling operators need window-1
# extra history. efficiency_ratio internally uses shift(window), so it belongs
# to WINDOW_OPERATORS but not ROLLING_OPERATORS.
LAG_STYLE_OPERATORS = frozenset({"lag", "delta", "efficiency_ratio"})
COMMUTATIVE_OPERATORS = frozenset({"add", "multiply"})
MIN_OPERATOR_WINDOW = {
    "rolling_skew": 20,
    "rolling_kurt": 20,
    "rolling_autocorr": 20,
}


# ---------------------------------------------------------------------------
# Dimension system (量纲)
# ---------------------------------------------------------------------------

TERMINAL_DIMENSIONS = {
    "open": "price",
    "high": "price",
    "low": "price",
    "close": "price",
    "volume": "base_volume",
    "taker_buy_base_volume": "base_volume",
    "quote_volume": "quote_volume",
    "taker_buy_quote_volume": "quote_volume",
    "trade_count": "count",
    "quote_per_trade": "money_per_trade",
    "return_1d": "ratio",
    "range_relative": "ratio",
    "body_relative": "ratio",
    "volume_relative_20": "ratio",
    "quote_volume_relative_20": "ratio",
    "taker_base_ratio": "ratio",
    "taker_quote_ratio": "ratio",
}

PRESERVE_DIMENSION_OPERATORS = frozenset({
    "negate", "abs", "lag", "delta",
    "rolling_mean", "rolling_std", "rolling_min", "rolling_max",
    "rolling_median", "rolling_mad", "rolling_iqr", "rolling_downside_std",
})
RATIO_DIMENSION_OPERATORS = frozenset({
    "rank", "cross_sectional_zscore", "rolling_corr",
    "ts_rank", "ts_zscore", "ts_argmax", "ts_argmin", "rolling_slope_norm",
    "rolling_r2", "efficiency_ratio", "rolling_hit_rate", "rolling_autocorr",
    "rolling_max_drawdown", "rolling_cv", "rolling_skew", "rolling_kurt",
})
MIXED_DIMENSION_OPERATORS = frozenset({"signed_sqrt", "safe_log"})

_DIVISION_DIMENSIONS = {
    ("quote_volume", "base_volume"): "price",
    ("quote_volume", "count"): "money_per_trade",
    ("base_volume", "count"): "base_per_trade",
}


def combine_dimensions(op: str, dim_a: str, dim_b: str) -> str:
    """Dimension algebra for binary operators; illegal combos raise ValueError."""
    if "mixed" in (dim_a, dim_b):
        raise ValueError(
            f"operator {op} cannot combine mixed-dimension operand "
            f"({dim_a}, {dim_b}); mixed outputs only feed unary preserve/ratio ops"
        )
    if op in {"add", "subtract"}:
        if dim_a != dim_b:
            raise ValueError(
                f"operator {op} requires matching dimensions, got {dim_a} and {dim_b}"
            )
        return dim_a
    if op == "multiply":
        if dim_a == "ratio":
            return dim_b
        if dim_b == "ratio":
            return dim_a
        return "mixed"
    if op == "safe_div":
        if dim_a == dim_b:
            return "ratio"
        if dim_b == "ratio":
            return dim_a
        return _DIVISION_DIMENSIONS.get((dim_a, dim_b), "mixed")
    raise ValueError(f"no dimension rule for operator: {op}")


__all__ = [
    "OPERATORS", "OPERATOR_ARITY", "WINDOW_OPERATORS", "ROLLING_OPERATORS",
    "LAG_STYLE_OPERATORS", "COMMUTATIVE_OPERATORS", "MIN_OPERATOR_WINDOW",
    "OPERATOR_REGISTRY", "safe_div", "add", "subtract", "multiply",
    "negate", "absolute", "lag", "delta", "rolling_mean", "rolling_std",
    "rolling_min", "rolling_max", "rolling_correlation", "rolling_corr",
    "cross_sectional_rank", "cross_sectional_zscore", "masked_rank",
    "rolling_median", "rolling_mad", "rolling_iqr", "rolling_skew",
    "rolling_kurt", "rolling_downside_std", "ts_rank", "ts_zscore",
    "ts_argmax", "ts_argmin", "rolling_slope_norm", "rolling_r2",
    "efficiency_ratio", "rolling_hit_rate", "rolling_autocorr",
    "rolling_max_drawdown", "rolling_cv", "signed_sqrt", "safe_log",
    "TERMINAL_DIMENSIONS", "PRESERVE_DIMENSION_OPERATORS",
    "RATIO_DIMENSION_OPERATORS", "MIXED_DIMENSION_OPERATORS",
    "combine_dimensions",
]
