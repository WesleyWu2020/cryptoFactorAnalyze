import numpy as np
import pandas as pd

from Genetic_Algorithm.operators import safe_div
from Genetic_Algorithm.operators import (
    cross_sectional_rank,
    rolling_mean,
    rolling_std,
)


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
