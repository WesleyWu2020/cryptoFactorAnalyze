import numpy as np
import pandas as pd
import pytest

from factor_common.grouping import assign_groups, target_weights
from factor_common.profiles import resolve_profile


def frame(rows, start="2024-01-01"):
    return pd.DataFrame(rows, index=pd.date_range(start, periods=len(rows), name="date"))


def test_stable_tie_ordering_breaks_by_instrument_name():
    values = frame([{"B": 1.0, "A": 1.0, "D": 2.0, "C": 2.0}])
    groups, diagnostics = assign_groups(values, 2)
    day = groups.iloc[0]
    assert day["A"] == 1 and day["B"] == 1
    assert day["C"] == 2 and day["D"] == 2
    assert diagnostics["insufficient_dates"] == []


def test_all_equal_values_group_by_instrument_order():
    values = frame([{"D": 5.0, "C": 5.0, "B": 5.0, "A": 5.0}])
    groups, _ = assign_groups(values, 2)
    day = groups.iloc[0]
    assert day["A"] == 1 and day["B"] == 1
    assert day["C"] == 2 and day["D"] == 2


def test_group_assignment_uses_floor_position_formula():
    # count=5, n_groups=2: floor(p * 2 / 5) -> [0, 0, 0, 1, 1] -> groups [1, 1, 1, 2, 2]
    values = frame([{"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}])
    groups, _ = assign_groups(values, 2)
    day = groups.iloc[0]
    assert (day[["A", "B", "C"]] == 1).all()
    assert (day[["D", "E"]] == 2).all()


def test_too_small_universe_returns_diagnostic_without_reducing_groups():
    rows = [
        {"A": 1.0, "B": 2.0, "C": 3.0},  # only 3 valid names for n_groups=4
        {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0},
    ]
    values = frame(rows)
    groups, diagnostics = assign_groups(values, 4)
    assert groups.iloc[0].isna().all()
    assert diagnostics["insufficient_dates"] == [values.index[0]]
    # The valid date still spans every configured group; count is never reduced.
    assert set(groups.iloc[1].dropna().astype(int)) == {1, 2, 3, 4}


def test_daily_calendar_gaps_are_rejected():
    values = frame([{"A": 1.0}, {"A": 2.0}, {"A": 3.0}]).drop(pd.Timestamp("2024-01-02"))
    with pytest.raises(ValueError, match="calendar day"):
        assign_groups(values, 2)
    with pytest.raises(ValueError, match="calendar day"):
        target_weights(values, resolve_profile("perp_1d", {}))


def test_target_weights_keys_are_documented_and_complete():
    values = frame([{"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}])
    weights = target_weights(values, resolve_profile("perp_1d", {"n_groups": 2}))
    assert set(weights) == {"group_1", "group_2", "directional_long", "long_short"}
    for matrix in weights.values():
        assert matrix.index.equals(values.index)
        assert matrix.columns.equals(values.columns)


def test_group_targets_are_equal_weight_and_sum_to_gross_exposure():
    values = frame([{"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}])
    weights = target_weights(values, resolve_profile("perp_1d", {"n_groups": 2}))
    group_1 = weights["group_1"].iloc[0]
    assert group_1["A"] == pytest.approx(0.5)
    assert group_1["B"] == pytest.approx(0.5)
    assert group_1["C"] == 0.0 and group_1["D"] == 0.0
    assert group_1.sum() == pytest.approx(1.0)


def test_direction_reversal_selects_top_or_bottom_group():
    values = frame([{"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}])
    up = target_weights(values, resolve_profile("perp_1d", {"n_groups": 2, "factor_direction": 1}))
    down = target_weights(values, resolve_profile("perp_1d", {"n_groups": 2, "factor_direction": -1}))
    pd.testing.assert_frame_equal(up["directional_long"], up["group_2"])
    pd.testing.assert_frame_equal(down["directional_long"], down["group_1"])


def test_long_short_is_fifty_fifty_top_vs_bottom_group():
    values = frame([{"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}])
    weights = target_weights(values, resolve_profile("perp_1d", {"n_groups": 2}))
    long_short = weights["long_short"].iloc[0]
    assert long_short["C"] == pytest.approx(0.25)
    assert long_short["D"] == pytest.approx(0.25)
    assert long_short["A"] == pytest.approx(-0.25)
    assert long_short["B"] == pytest.approx(-0.25)
    assert long_short.sum() == pytest.approx(0.0)
    assert long_short.abs().sum() == pytest.approx(1.0)


def test_gross_exposure_scales_every_portfolio():
    values = frame([{"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}])
    profile = resolve_profile("perp_1d", {"n_groups": 2, "gross_exposure": 0.5})
    weights = target_weights(values, profile)
    assert weights["group_1"].iloc[0].sum() == pytest.approx(0.5)
    assert weights["directional_long"].iloc[0].sum() == pytest.approx(0.5)
    assert weights["long_short"].iloc[0].abs().sum() == pytest.approx(0.5)


def test_insufficient_dates_have_nan_targets_not_silent_reduction():
    rows = [
        {"A": 1.0, "B": 2.0, "C": 3.0},
        {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},
    ]
    values = frame(rows)
    weights = target_weights(values, resolve_profile("perp_1d", {"n_groups": 4}))
    for matrix in weights.values():
        assert matrix.iloc[0].isna().all()
    assert weights["group_4"].iloc[1].sum() == pytest.approx(1.0)
