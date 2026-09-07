import numpy as np
import pandas as pd
import pytest

from Genetic_Algorithm.operators import safe_div
from Genetic_Algorithm.operators import (
    cross_sectional_rank,
    rolling_mean,
    rolling_min,
    rolling_max,
    rolling_correlation,
    rolling_std,
)
from Genetic_Algorithm.operators import delta, lag


def test_safe_div_preserves_missing():
    a = pd.DataFrame([[2.0, 2.0, np.nan]])
    b = pd.DataFrame([[2.0, 0.0, 1.0]])
    result = safe_div(a, b)
    assert result.iloc[0, 0] == 1.0
    assert result.iloc[0, 1:].isna().all()


def test_rolling_operators_require_complete_trailing_windows():
    values = pd.DataFrame([[1.0], [2.0], [4.0]])
    mean = rolling_mean(values, 2).iloc[:, 0]
    assert np.isnan(mean.iloc[0])
    assert mean.iloc[1:].tolist() == [1.5, 3.0]
    assert np.isnan(rolling_std(values, 2).iloc[0, 0])
    assert rolling_std(values, 2).iloc[1, 0] == np.sqrt(0.5)


def test_rank_uses_average_percentiles_and_singletons_are_missing():
    values = pd.DataFrame([[1.0, 1.0, 3.0, np.nan]])
    eligible = pd.DataFrame([[True, True, True, True]])
    result = cross_sectional_rank(values, eligible)
    assert result.iloc[0, :3].tolist() == [0.5, 0.5, 1.0]
    singleton = cross_sectional_rank(values.iloc[:, :1], eligible.iloc[:, :1])
    assert singleton.isna().all().all()


def test_rank_treats_unknown_eligibility_as_ineligible():
    values = pd.DataFrame([[1.0, 2.0, 3.0]])
    eligible = pd.DataFrame([[True, np.nan, True]])

    result = cross_sectional_rank(values, eligible)

    assert result.iloc[0, 0] == 0.5
    assert pd.isna(result.iloc[0, 1])
    assert result.iloc[0, 2] == 1.0


@pytest.mark.parametrize("operator", [lag, delta])
@pytest.mark.parametrize("window", [0, -1, -7])
def test_direct_lag_and_delta_reject_non_positive_windows_before_shift(
    operator, window, monkeypatch
):
    values = pd.DataFrame({"close": [1.0, 2.0, 3.0]})

    def unexpected_shift(*args, **kwargs):
        raise AssertionError("invalid windows must be rejected before shift")

    monkeypatch.setattr(pd.DataFrame, "shift", unexpected_shift)

    with pytest.raises(ValueError, match="positive"):
        operator(values, window)


@pytest.mark.parametrize(
    "operator,args",
    [
        (rolling_mean, ()),
        (rolling_std, ()),
        (rolling_min, ()),
        (rolling_max, ()),
        (rolling_correlation, (pd.DataFrame({"open": [1.0, 2.0, 3.0]}),)),
    ],
)
@pytest.mark.parametrize("window", [0, -1, 1.5])
def test_direct_rolling_operators_reject_invalid_windows_before_rolling(
    operator, args, window, monkeypatch
):
    values = pd.DataFrame({"close": [1.0, 2.0, 3.0]})

    def unexpected_rolling(*args, **kwargs):
        raise AssertionError("invalid windows must be rejected before rolling")

    monkeypatch.setattr(pd.DataFrame, "rolling", unexpected_rolling)

    with pytest.raises(ValueError, match="positive"):
        operator(values, *args, window)
