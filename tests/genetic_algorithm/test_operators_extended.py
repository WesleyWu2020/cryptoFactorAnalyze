"""Tests for the 20 extended GP operators: correctness, registration, causality."""

import numpy as np
import pandas as pd
import pytest

from Genetic_Algorithm import operators
from Genetic_Algorithm.operators import (
    LAG_STYLE_OPERATORS,
    MIN_OPERATOR_WINDOW,
    OPERATOR_ARITY,
    OPERATOR_REGISTRY,
    ROLLING_OPERATORS,
    WINDOW_OPERATORS,
    cross_sectional_zscore,
    efficiency_ratio,
    rolling_autocorr,
    rolling_cv,
    rolling_downside_std,
    rolling_hit_rate,
    rolling_iqr,
    rolling_kurt,
    rolling_mad,
    rolling_max_drawdown,
    rolling_median,
    rolling_r2,
    rolling_skew,
    rolling_slope_norm,
    safe_log,
    signed_sqrt,
    ts_argmax,
    ts_argmin,
    ts_rank,
    ts_zscore,
)

NEW_WINDOW_OPERATORS = {
    "rolling_median", "rolling_mad", "rolling_iqr", "rolling_skew",
    "rolling_kurt", "rolling_downside_std", "ts_rank", "ts_zscore",
    "ts_argmax", "ts_argmin", "rolling_slope_norm", "rolling_r2",
    "efficiency_ratio", "rolling_hit_rate", "rolling_autocorr",
    "rolling_max_drawdown", "rolling_cv",
}


def test_zscore_identical_across_memory_layout_and_future_only_columns():
    values = np.random.default_rng(123).normal(size=(258, 130))
    values[values < -0.5] = np.nan
    left = pd.DataFrame(np.array(values, order="C"))
    right = pd.DataFrame(np.array(values, order="F"))
    right[130] = np.nan
    actual = cross_sectional_zscore(left, left.notna())
    replay = cross_sectional_zscore(right, right.notna()).iloc[:, :130]
    pd.testing.assert_frame_equal(actual, replay, check_exact=True)
NEW_ELEMENTWISE_OPERATORS = {"signed_sqrt", "safe_log"}
NEW_CROSS_SECTIONAL_OPERATORS = {"cross_sectional_zscore"}
NEW_OPERATORS = (
    NEW_WINDOW_OPERATORS | NEW_ELEMENTWISE_OPERATORS | NEW_CROSS_SECTIONAL_OPERATORS
)


def _single(values):
    return pd.DataFrame({"A": [float(v) for v in values]})


def test_new_operators_are_registered_with_arity_and_window_sets():
    assert NEW_OPERATORS <= set(OPERATOR_REGISTRY)
    for name in NEW_OPERATORS:
        assert OPERATOR_ARITY[name] == 1
    assert NEW_WINDOW_OPERATORS <= WINDOW_OPERATORS
    assert NEW_WINDOW_OPERATORS - {"efficiency_ratio"} <= ROLLING_OPERATORS
    assert "efficiency_ratio" in WINDOW_OPERATORS
    assert "efficiency_ratio" not in ROLLING_OPERATORS
    assert "efficiency_ratio" in LAG_STYLE_OPERATORS
    assert not NEW_ELEMENTWISE_OPERATORS & WINDOW_OPERATORS
    assert "cross_sectional_zscore" not in WINDOW_OPERATORS
    assert MIN_OPERATOR_WINDOW == {
        "rolling_skew": 20, "rolling_kurt": 20, "rolling_autocorr": 20,
    }


def test_rolling_median_matches_trailing_median():
    result = rolling_median(_single([1.0, 2.0, 4.0]), 2).iloc[:, 0]
    assert np.isnan(result.iloc[0])
    assert result.iloc[1:].tolist() == [1.5, 3.0]


def test_rolling_mad_is_median_absolute_deviation():
    result = rolling_mad(_single([1.0, 2.0, 10.0]), 3).iloc[-1, 0]
    assert result == 1.0  # median 2, |x-2| = {1, 0, 8}, median 1


def test_rolling_iqr_is_quantile_spread():
    result = rolling_iqr(_single([1.0, 2.0, 3.0, 4.0]), 4).iloc[-1, 0]
    assert result == pytest.approx(1.5)  # 3.25 - 1.75


def test_rolling_skew_is_zero_for_symmetric_windows():
    result = rolling_skew(_single([1.0, 2.0, 3.0, 4.0, 5.0]), 5).iloc[-1, 0]
    assert result == pytest.approx(0.0)


def test_rolling_kurt_matches_fisher_kurtosis():
    result = rolling_kurt(_single([1.0, 2.0, 3.0, 4.0] * 5), 20)
    assert not np.isnan(result.iloc[-1, 0])
    short = rolling_kurt(_single([1.0, 2.0, 3.0, 4.0]), 4).iloc[-1, 0]
    assert short == pytest.approx(-1.2)


def test_rolling_downside_std_uses_only_negative_observations():
    result = rolling_downside_std(_single([1.0, -1.0, -3.0, 2.0, -2.0]), 5).iloc[-1, 0]
    assert result == pytest.approx(1.0)  # std([-1, -3, -2], ddof=1)
    assert np.isnan(rolling_downside_std(_single([1.0, 2.0, -1.0]), 3).iloc[-1, 0])


def test_ts_rank_is_percentile_of_last_value():
    result = ts_rank(_single([3.0, 1.0, 2.0]), 3).iloc[-1, 0]
    assert result == pytest.approx(2 / 3)
    assert ts_rank(_single([1.0, 2.0, 3.0]), 3).iloc[-1, 0] == 1.0


def test_ts_zscore_uses_safe_div_against_window_stats():
    result = ts_zscore(_single([1.0, 2.0, 3.0]), 2).iloc[-1, 0]
    assert result == pytest.approx(0.5 / np.sqrt(0.5))
    constant = ts_zscore(_single([2.0, 2.0, 2.0]), 2).iloc[-1, 0]
    assert np.isnan(constant)


def test_ts_argmax_and_argmin_are_normalized_recency():
    values = _single([1.0, 3.0, 2.0])
    assert ts_argmax(values, 3).iloc[-1, 0] == pytest.approx(1 / 3)
    assert ts_argmin(values, 3).iloc[-1, 0] == pytest.approx(2 / 3)
    assert ts_argmax(_single([1.0, 2.0, 3.0]), 3).iloc[-1, 0] == 0.0


def test_rolling_slope_norm_scales_slope_by_window_mean():
    result = rolling_slope_norm(_single([2.0, 4.0, 6.0]), 3).iloc[-1, 0]
    assert result == pytest.approx(0.5)
    assert np.isnan(rolling_slope_norm(_single([-1.0, 0.0, 1.0]), 3).iloc[-1, 0])


def test_rolling_r2_is_squared_trend_correlation():
    assert rolling_r2(_single([1.0, 2.0, 3.0]), 3).iloc[-1, 0] == pytest.approx(1.0)
    assert np.isnan(rolling_r2(_single([2.0, 2.0, 2.0]), 3).iloc[-1, 0])


def test_efficiency_ratio_is_net_move_over_path_length():
    assert efficiency_ratio(_single([1.0, 2.0, 3.0, 4.0]), 3).iloc[-1, 0] == pytest.approx(1.0)
    assert efficiency_ratio(_single([1.0, 3.0, 1.0]), 2).iloc[-1, 0] == pytest.approx(0.0)
    flat = efficiency_ratio(_single([1.0, 1.0, 1.0]), 2).iloc[-1, 0]
    assert np.isnan(flat)  # 0/0 rejected by safe_div


def test_rolling_hit_rate_is_fraction_of_positive_observations():
    result = rolling_hit_rate(_single([1.0, -1.0, 1.0]), 2).iloc[:, 0]
    assert result.iloc[1:].tolist() == [0.5, 0.5]
    assert rolling_hit_rate(_single([1.0, 2.0]), 2).iloc[-1, 0] == 1.0


def test_rolling_autocorr_is_first_order_lag_correlation():
    assert rolling_autocorr(_single([1.0, 2.0, 3.0]), 3).iloc[-1, 0] == pytest.approx(1.0)
    assert np.isnan(rolling_autocorr(_single([2.0, 2.0, 2.0]), 3).iloc[-1, 0])


def test_rolling_max_drawdown_is_peak_to_trough_loss():
    result = rolling_max_drawdown(_single([10.0, 8.0, 10.0]), 3).iloc[-1, 0]
    assert result == pytest.approx(0.2)
    monotonic = rolling_max_drawdown(_single([1.0, 2.0, 3.0]), 3).iloc[-1, 0]
    assert monotonic == pytest.approx(0.0)
    negative = rolling_max_drawdown(_single([-1.0, -2.0, -3.0]), 3).iloc[-1, 0]
    assert np.isnan(negative)


def test_rolling_cv_is_std_over_abs_mean():
    result = rolling_cv(_single([2.0, 4.0]), 2).iloc[-1, 0]
    assert result == pytest.approx(np.sqrt(2.0) / 3.0)


def test_signed_sqrt_and_safe_log_are_sign_preserving():
    values = pd.DataFrame({"A": [-4.0, 4.0]})
    assert signed_sqrt(values)["A"].tolist() == [-2.0, 2.0]
    logged = safe_log(pd.DataFrame({"A": [-1.0, 1.0]}))["A"].tolist()
    assert logged == pytest.approx([-np.log(2.0), np.log(2.0)])


def test_cross_sectional_zscore_masks_like_rank():
    values = pd.DataFrame([[1.0, 2.0, 3.0, np.nan]])
    eligible = pd.DataFrame([[True, True, True, True]])
    result = cross_sectional_zscore(values, eligible)
    assert result.iloc[0, :3].tolist() == pytest.approx(
        [-1.224744871391589, 0.0, 1.224744871391589]
    )
    singleton = cross_sectional_zscore(values.iloc[:, :1], eligible.iloc[:, :1])
    assert singleton.isna().all().all()
    hidden = pd.DataFrame([[True, False, True, True]])
    masked = cross_sectional_zscore(values, hidden)
    assert pd.isna(masked.iloc[0, 1])
    assert masked.iloc[0, 0] == pytest.approx(-1.0)


@pytest.mark.parametrize("name", sorted(NEW_WINDOW_OPERATORS))
def test_new_window_operators_require_complete_trailing_windows(name):
    window = MIN_OPERATOR_WINDOW.get(name, 3)
    values = np.arange(1.0, window + 2.0)
    if name == "rolling_downside_std":
        values = -values  # all-negative windows always yield enough negatives
    values = _single(values)
    result = OPERATOR_REGISTRY[name](values, window).iloc[:, 0]
    # efficiency_ratio shifts by the full window internally (lag-style history).
    warmup = window if name == "efficiency_ratio" else window - 1
    assert result.iloc[:warmup].isna().all()
    assert result.iloc[warmup:].notna().all()


@pytest.mark.parametrize("window", [0, -1, 1.5])
def test_new_window_operators_reject_invalid_windows(window):
    values = _single([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="positive"):
        rolling_median(values, window)


def _signed_panel(rows=45, cols=4, seed=7):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, cols))
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(values, index=index, columns=[f"S{i}" for i in range(cols)])


@pytest.mark.parametrize("name", sorted(NEW_WINDOW_OPERATORS))
def test_new_window_operators_ignore_future_data(name):
    window = MIN_OPERATOR_WINDOW.get(name, 5)
    panel = _signed_panel()
    cutoff = 30
    full = OPERATOR_REGISTRY[name](panel, window)
    truncated = OPERATOR_REGISTRY[name](panel.iloc[:cutoff], window)
    pd.testing.assert_frame_equal(full.iloc[:cutoff], truncated)


@pytest.mark.parametrize("func", [signed_sqrt, safe_log])
def test_elementwise_operators_ignore_future_data(func):
    panel = _signed_panel()
    pd.testing.assert_frame_equal(func(panel).iloc[:30], func(panel.iloc[:30]))


def test_cross_sectional_zscore_ignores_future_data():
    panel = _signed_panel()
    eligible = panel.notna()
    cutoff = 30
    full = cross_sectional_zscore(panel, eligible)
    truncated = cross_sectional_zscore(panel.iloc[:cutoff], eligible.iloc[:cutoff])
    pd.testing.assert_frame_equal(full.iloc[:cutoff], truncated)
