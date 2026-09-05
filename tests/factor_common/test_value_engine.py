from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factor_common.definitions import FactorSpec
from factor_common.value_engine import compute_factor


class MemoryProvider:
    """Calendar-preserving provider with no labels, files, or network access."""

    def __init__(self, close, eligible=None):
        self.close = close
        self.eligible = eligible if eligible is not None else close.notna()
        self.calls = []

    def get_single_data(self, field, *, start, end):
        assert field in {"close", "open"}
        self.calls.append((field, start, end))
        return self.close.reindex(pd.date_range(start, end, name="date"))

    def get_universe(self, *, start, end):
        self.calls.append(("universe", start, end))
        return self.eligible.reindex(pd.date_range(start, end, name="date"), fill_value=False)


def spec(calc=lambda ctx: ctx["close"], *, preprocessing="none", warmup=0, **extra):
    return FactorSpec("test", {}, {
        "data_needed": ["close"], "warmup_bars": warmup,
        "preprocessing": preprocessing, **extra,
    }, calc, "test")


def frame(values):
    return pd.DataFrame(values, index=pd.date_range("2024-01-01", periods=len(values), name="date"))


def test_history_before_entry_and_tail_without_labels():
    close = frame([[100.0], [110.0], [121.0]])
    dp = MemoryProvider(close, frame([[False], [True], [True]]))
    momentum = spec(lambda ctx: np.log(ctx["close"] / ctx["close"].shift(1)), warmup=1)
    values, diagnostics = compute_factor(momentum, dp, start="2024-01-02", end="2024-01-03")
    np.testing.assert_allclose(values[0], np.log(1.1))
    pd.testing.assert_index_equal(values.index, close.index[1:])
    assert dp.calls == [(field, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03"))
                        for field in ("close", "universe")]
    assert diagnostics["valid_count"] == 2


def test_missing_calendar_days_are_not_compressed_or_filled():
    close = frame([[100.0], [110.0], [121.0], [133.1]]).drop(pd.Timestamp("2024-01-02"))
    eligible = pd.DataFrame(True, index=pd.date_range("2024-01-01", periods=4, name="date"), columns=[0])
    momentum = spec(lambda ctx: np.log(ctx["close"] / ctx["close"].shift(1)), warmup=1)
    values, diagnostics = compute_factor(momentum, MemoryProvider(close, eligible), start="2024-01-02", end="2024-01-04")
    assert values.index.equals(eligible.index[1:])
    assert values.iloc[:2].isna().all().all()
    assert values.iloc[2, 0] == pytest.approx(np.log(1.1))
    assert diagnostics["missing_count"] == 2
    assert diagnostics["valid_count"] == 1


@pytest.mark.parametrize("preprocessing", ["none", "mad_rank"])
def test_same_day_eligible_finite_only_and_cutoff_invariance(preprocessing):
    close = frame([[1., 2., 3., 1e15], [3., 2., 1., -1e15], [4., 5., 6., 1e20]])
    eligible = frame([[True, True, True, False], [True, True, True, False], [True]*4])
    factor = spec(preprocessing=preprocessing)
    full, _ = compute_factor(factor, MemoryProvider(close, eligible), start=close.index[0], end=close.index[-1])
    cutoff, _ = compute_factor(factor, MemoryProvider(close.iloc[:2], eligible.iloc[:2]), start=close.index[0], end=close.index[1])
    pd.testing.assert_frame_equal(full.iloc[:2], cutoff)
    assert np.nanmax(np.abs(full.iloc[:2] - cutoff)) == 0
    without_entrant, _ = compute_factor(factor, MemoryProvider(close.iloc[:, :3], eligible.iloc[:, :3]), start=close.index[0], end=close.index[-1])
    pd.testing.assert_frame_equal(full.iloc[:2, :3], without_entrant.iloc[:2])
    assert full.iloc[:2, 3].isna().all()
    if preprocessing == "mad_rank":
        np.testing.assert_allclose(full.iloc[:2, :3], [[-1, 0, 1], [1, 0, -1]])


def test_diagnostics_distinguish_missing_from_ineligible_and_keep_nan():
    close = frame([[5., np.inf, -np.inf, np.nan, 99.], [np.nan]*5])
    eligible = frame([[True, True, True, True, False], [True]*5])
    values, diagnostics = compute_factor(spec(preprocessing="mad_rank"), MemoryProvider(close, eligible), start=close.index[0], end=close.index[-1])
    assert values.iloc[0, 0] == 0
    assert values.iloc[0, 1:].isna().all()
    assert values.iloc[1].isna().all()
    assert diagnostics["eligible_count"] == 9
    assert diagnostics["missing_count"] == 8
    assert diagnostics["valid_count"] == 1
    assert diagnostics["ineligible_count"] == 1


@pytest.mark.parametrize("setting", [{"preprocessing": "zscore"}, {"pasteurization": True}])
def test_reject_ambiguous_or_unsupported_settings(setting):
    dp = MemoryProvider(frame([[1.]]))
    with pytest.raises(ValueError, match="preprocessing|pasteurization"):
        compute_factor(spec(**setting), dp, start="2024-01-01", end="2024-01-01")
    assert not dp.calls


def corrupt_axes(df, kind):
    if kind == "drop_date":
        return df.iloc[1:]
    if kind == "drop_instrument":
        return df.iloc[:, 1:]
    if kind == "reorder":
        return df.iloc[:, ::-1]
    result = df.copy()
    if kind == "duplicate_date":
        result.index = [df.index[0]] * len(df)
    elif kind == "duplicate_instrument":
        result.columns = ["same"] * len(df.columns)
    elif kind == "intraday":
        result.index = df.index + pd.Timedelta(hours=1)
    elif kind == "timezone":
        result.index = df.index.tz_localize("UTC")
    elif kind == "string_date":
        result.index = df.index.astype(str)
    elif kind == "nat":
        result.index = pd.DatetimeIndex([df.index[0], pd.NaT])
    return result


@pytest.mark.parametrize("kind", ["drop_date", "drop_instrument", "reorder", "duplicate_date", "duplicate_instrument", "intraday", "timezone", "string_date", "nat"])
@pytest.mark.parametrize("source", ["context", "output", "universe"])
def test_reject_axes_before_alignment(source, kind):
    close = frame([[1., 2.], [3., 4.]])
    dp = MemoryProvider(close)
    factor = spec()
    if source == "output":
        factor = spec(lambda ctx: corrupt_axes(ctx["close"], kind))
    elif source == "context":
        original = dp.get_single_data
        dp.get_single_data = lambda field, **kw: corrupt_axes(original(field, **kw), kind) if field == "open" else original(field, **kw)
        factor = replace(factor, setting={**factor.setting, "data_needed": ["close", "open"]})
    else:
        original = dp.get_universe
        dp.get_universe = lambda **kw: corrupt_axes(original(**kw), kind)
    with pytest.raises((ValueError, TypeError), match="axis|axes|daily|DatetimeIndex|UTC-naive"):
        compute_factor(factor, dp, start=close.index[0], end=close.index[-1])


def test_reusable_preprocessing_preserves_legacy_math():
    from factor_common.preprocessing import rank_to_unit_by_date, winsorize_by_date
    from factor_analyse.factor_mining import util_factor

    df = pd.DataFrame({"date": ["a"]*5 + ["b"]*3, "raw": [0., 1., 2., 3., 100., 7., 7., np.nan]})
    clipped = winsorize_by_date(df, "raw")
    assert clipped.loc[4, "raw"] == pytest.approx(2 + 3 * 1.4826)
    assert clipped.loc[5, "raw"] == 7
    ranked = rank_to_unit_by_date(clipped, "raw")
    np.testing.assert_allclose(ranked.factor[:5], [-1, -.5, 0, .5, 1])
    np.testing.assert_allclose(ranked.factor[5:7], [0, 0])
    assert np.isnan(ranked.factor[7])
    pd.testing.assert_frame_equal(clipped, util_factor.winsorize_by_date(df, "raw"))
    pd.testing.assert_frame_equal(ranked, util_factor.rank_to_unit_by_date(clipped, "raw"))
    assert winsorize_by_date(df.iloc[:0], "raw").empty
    assert rank_to_unit_by_date(df.iloc[:0], "raw").empty
