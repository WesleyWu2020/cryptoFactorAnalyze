import numpy as np
import pandas as pd
import pytest

from portfolio.fixed_config import FixedConfig
from portfolio.portfolio_builder.fixed_weights import combine_targets, member_targets


def _config(**overrides):
    values = dict(gross_limit=1.0, long_limit=1.0, short_limit=1.0,
                  single_limit=1.0, net_limit=1.0)
    values.update(overrides)
    return type("Config", (), values)()


def _members():
    index = pd.date_range("2024-01-01", periods=2, name="date")
    columns = pd.Index(["A", "B", "C"], name="instrument")
    return index, columns


def test_combination_allows_exact_cancellation_without_releveraging():
    index, columns = _members()
    a = pd.DataFrame([[1, -1, 0], [0, 0, 0]], index=index, columns=columns)
    b = -a
    combined, risk = combine_targets({"a": a, "b": b}, {"a": .5, "b": .5}, _config())
    assert (combined == 0).all().all()
    assert risk.loc[index[0], ["pre_gross", "post_gross"]].eq(0).all()
    assert risk.loc[index[0], "scale"] == 1


def test_combination_keeps_fixed_allocations_and_does_not_releverage():
    index, columns = _members()
    a = pd.DataFrame([[1, 0, 0], [1, 0, 0]], index=index, columns=columns)
    b = pd.DataFrame([[0, 1, 0], [0, 1, 0]], index=index, columns=columns)
    combined, risk = combine_targets({"a": a, "b": b}, {"a": .25, "b": .75}, _config())
    assert combined.iloc[0].to_dict() == {"A": .25, "B": .75, "C": 0.0}
    assert (risk["scale"] == 1).all()


def test_combination_applies_risk_caps_with_single_common_scale():
    index, columns = _members()
    a = pd.DataFrame([[2, -2, 0], [0, 0, 0]], index=index, columns=columns)
    combined, risk = combine_targets({"a": a}, {"a": 1.0}, _config(gross_limit=1.0,
                                                                     long_limit=1.0,
                                                                     short_limit=1.0,
                                                                     single_limit=.3,
                                                                     net_limit=1.0))
    assert risk.loc[index[0], "scale"] == pytest.approx(.15)
    assert combined.iloc[0].tolist() == pytest.approx([.3, -.3, 0])
    assert risk.loc[index[0], "binding_constraints"] == "single"


def test_combination_rejects_mismatched_axes_and_allocation_sum():
    index, columns = _members()
    a = pd.DataFrame([[1, 0, 0], [0, 0, 0]], index=index, columns=columns)
    b = a.rename(columns={"C": "D"})
    with pytest.raises(ValueError, match="identical axes"):
        combine_targets({"a": a, "b": b}, {"a": .5, "b": .5}, _config())
    with pytest.raises(ValueError, match="sum to 1"):
        combine_targets({"a": a}, {"a": .5}, _config())
    with pytest.raises(ValueError, match="positive"):
        combine_targets({"a": a}, {"a": 0.0}, _config())


def test_combination_preserves_invalid_member_cells_as_cash():
    index, columns = _members()
    a = pd.DataFrame([[np.nan, np.inf, 2], [0, 0, 0]], index=index, columns=columns)
    combined, _ = combine_targets({"a": a}, {"a": 1.0}, _config())
    assert combined.iloc[0].to_dict() == {"A": 0.0, "B": 0.0, "C": 1.0}


def test_combination_reports_invalid_cells_per_member_and_total():
    index, columns = _members()
    a = pd.DataFrame([[np.nan, np.inf, 2], [0, 0, 0]], index=index, columns=columns)
    b = pd.DataFrame([[1, 1, 1], [np.nan, 0, 0]], index=index, columns=columns)
    _, risk = combine_targets({"a": a, "b": b}, {"a": .5, "b": .5}, _config())
    assert risk.loc[index[0], "invalid_cells_a"] == 2
    assert risk.loc[index[0], "invalid_cells_b"] == 0
    assert risk.loc[index[0], "invalid_cells"] == 2
    assert "pre_abs_net" in risk and "post_abs_net" in risk
    assert "binding_constraint" in risk
