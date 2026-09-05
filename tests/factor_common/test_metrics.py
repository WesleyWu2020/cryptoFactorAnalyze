import math

import numpy as np
import pandas as pd
import pytest

from factor_common.metrics import (
    _half_life,
    _rank_ic_autocorr,
    evaluate_metrics,
    summarize_returns,
)
from factor_common.profiles import resolve_profile


INSTRUMENTS = ["A", "B", "C", "D"]


def _values(dates, seed=0, instruments=INSTRUMENTS):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(len(dates), len(instruments))), index=dates, columns=instruments
    )


def _ledger_frame(returns, notionals=None, fees=None, start="2024-01-01"):
    n = len(returns)
    dates = pd.date_range(start, periods=n, freq="D")
    equity = (1 + pd.Series(returns, dtype="float64")).cumprod().to_numpy()
    zeros = np.zeros(n)
    return pd.DataFrame(
        {
            "equity": equity,
            "return": np.asarray(returns, dtype="float64"),
            "price_pnl": zeros,
            "funding_cashflow": zeros,
            "fee": np.asarray(fees, dtype="float64") if fees is not None else zeros,
            "trade_notional": (
                np.asarray(notionals, dtype="float64") if notionals is not None else zeros
            ),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _scenario(ledger, status="complete"):
    return {"status": status, "ledger": ledger, "diagnostics": {}}


def _accounting(ledger, *, all_costs=None):
    return {
        "status": "complete",
        "portfolio": "long_short",
        "scenarios": {
            "gross": _scenario(ledger),
            "trading_net": _scenario(ledger),
            "all_costs": all_costs if all_costs is not None else _scenario(ledger),
        },
        "diagnostics": {},
    }


def _assert_no_nan(obj, path="root"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            _assert_no_nan(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _assert_no_nan(value, f"{path}[{index}]")
    elif isinstance(obj, float):
        assert not (math.isnan(obj) or math.isinf(obj)), f"non-finite float at {path}"


# ---------------------------------------------------------------------------
# summarize_returns oracles
# ---------------------------------------------------------------------------


def test_compounded_total_return_oracle():
    summary = summarize_returns(pd.Series([0.1, -0.1]))
    assert summary["total_return"] == pytest.approx(-0.01)


def test_max_drawdown_includes_initial_nav():
    # NAV path 1 -> 1.1 -> 0.99: peak 1.1, drawdown (1.1 - 0.99) / 1.1 = 0.1.
    summary = summarize_returns(pd.Series([0.1, -0.1]))
    assert summary["max_drawdown"] == pytest.approx(0.1)
    # A series that only falls below the initial NAV still counts the initial 1.
    falling = summarize_returns(pd.Series([-0.2]))
    assert falling["max_drawdown"] == pytest.approx(0.2)


def test_zero_variance_sharpe_is_missing():
    summary = summarize_returns(pd.Series([0.01, 0.01, 0.01]))
    assert summary["sharpe"] is None
    assert summary["volatility"] == pytest.approx(0.0)
    assert summary["annual_volatility"] == pytest.approx(0.0)


def test_annualized_return_compounds_over_actual_daily_count():
    n = 180
    summary = summarize_returns(pd.Series([0.01] * n), periods_per_year=365)
    total = 1.01**n - 1
    assert summary["total_return"] == pytest.approx(total)
    assert summary["annual_return"] == pytest.approx((1 + total) ** (365 / n) - 1)
    assert summary["n_periods"] == n


def test_win_rate_and_profit_loss_ratio_on_nonmissing_returns():
    summary = summarize_returns(pd.Series([0.02, -0.01, np.nan, 0.0]))
    assert summary["n_periods"] == 3
    assert summary["win_rate"] == pytest.approx(1 / 3)
    assert summary["profit_loss_ratio"] == pytest.approx(2.0)
    only_gains = summarize_returns(pd.Series([0.01, 0.02]))
    assert only_gains["win_rate"] == pytest.approx(1.0)
    assert only_gains["profit_loss_ratio"] is None
    only_losses = summarize_returns(pd.Series([-0.01, -0.02]))
    assert only_losses["profit_loss_ratio"] is None


def test_empty_returns_are_all_missing():
    summary = summarize_returns(pd.Series([], dtype="float64"))
    assert summary["n_periods"] == 0
    for key in (
        "total_return",
        "annual_return",
        "volatility",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "profit_loss_ratio",
    ):
        assert summary[key] is None, key


# ---------------------------------------------------------------------------
# evaluate_metrics structure and IC oracles
# ---------------------------------------------------------------------------


def test_result_structure_nests_samples_and_scenarios():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    values = _values(dates)
    profile = resolve_profile("perp_1d", {"split_date": "2024-01-06"})
    result = evaluate_metrics(values, values, _accounting(_ledger_frame([0.0] * 10)), profile)
    assert set(result["samples"]) == {"full", "in_sample", "out_of_sample"}
    assert set(result["scenarios"]) == {"gross", "trading_net", "all_costs"}
    for scenario in result["scenarios"].values():
        assert set(scenario) >= {"status", "full", "in_sample", "out_of_sample"}
    for sample in result["samples"].values():
        assert set(sample) >= {
            "ic",
            "ic_decay",
            "rank_ic_autocorr",
            "rank_ic_half_life",
            "coverage",
        }


def test_perfect_daily_rank_ic_is_one():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    values = _values(dates)
    profile = resolve_profile("perp_1d", {"split_date": "2024-01-05"})
    result = evaluate_metrics(values, values, _accounting(_ledger_frame([0.0] * 8)), profile)
    ic = result["samples"]["full"]["ic"]
    assert ic["rank_ic_mean"] == pytest.approx(1.0)
    assert ic["ic_mean"] == pytest.approx(1.0)
    assert ic["n_dates"] == 8
    # A constant IC series has zero variance: ICIR and t/p stay missing.
    assert ic["icir"] is None
    assert ic["annualized_icir"] is None
    assert ic["t_stat"] is None
    assert ic["p_value"] is None


def test_ic_requires_minimum_nonconstant_cross_section():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    values = _values(dates, instruments=["A", "B"])
    profile = resolve_profile("perp_1d", {"split_date": "2024-01-03"})
    result = evaluate_metrics(values, values, _accounting(_ledger_frame([0.0] * 5)), profile)
    ic = result["samples"]["full"]["ic"]
    assert ic["n_dates"] == 0
    assert ic["ic_mean"] is None
    assert ic["rank_ic_mean"] is None


def test_ic_t_stat_with_sufficient_dates():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    values = _values(dates, seed=3)
    labels = values + _values(dates, seed=4) * 0.3
    profile = resolve_profile("perp_1d", {"split_date": "2024-01-20"})
    result = evaluate_metrics(values, labels, _accounting(_ledger_frame([0.0] * 30)), profile)
    ic = result["samples"]["full"]["ic"]
    assert ic["n_dates"] == 30
    assert ic["icir"] is not None
    assert ic["annualized_icir"] == pytest.approx(ic["icir"] * math.sqrt(365))
    assert ic["t_stat"] is not None
    assert ic["p_value"] is not None
    assert 0.0 <= ic["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# sample boundaries
# ---------------------------------------------------------------------------


def test_sample_purge_excludes_labels_crossing_split():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    values = _values(dates)
    # hold_days = rebalance_days = 2: entry t+1, exit t+3. With split on
    # 2024-01-15, signals on 01-13/01-14 enter before the split but exit after.
    profile = resolve_profile("perp_1d", {"rebalance_days": 2, "split_date": "2024-01-15"})
    result = evaluate_metrics(values, values, _accounting(_ledger_frame([0.0] * 20)), profile)
    assert result["split"]["purged_signal_dates"] == ["2024-01-13", "2024-01-14"]

    full_dates = [row["date"] for row in result["samples"]["full"]["ic"]["daily"]]
    in_dates = [row["date"] for row in result["samples"]["in_sample"]["ic"]["daily"]]
    out_dates = [row["date"] for row in result["samples"]["out_of_sample"]["ic"]["daily"]]
    assert "2024-01-13" in full_dates and "2024-01-14" in full_dates
    assert max(in_dates) == "2024-01-12"
    assert min(out_dates) == "2024-01-15"
    assert "2024-01-13" not in in_dates and "2024-01-13" not in out_dates
    assert "2024-01-14" not in in_dates and "2024-01-14" not in out_dates
    # In-sample keeps historical observations whose labels resolve after the
    # split only if their exit is inside the sample; nothing else is dropped.
    assert len(in_dates) == 12
    assert len(out_dates) == 6


def test_default_split_uses_180_natural_days():
    dates = pd.date_range("2023-07-01", periods=400, freq="D")
    values = _values(dates)
    profile = resolve_profile("perp_1d", {})
    result = evaluate_metrics(values, values, _accounting(_ledger_frame([0.0] * 400)), profile)
    expected = (dates[-1] - pd.Timedelta(days=180)).date().isoformat()
    assert result["split"]["split_date"] == expected
    assert result["split"]["out_of_sample_days"] == 180


# ---------------------------------------------------------------------------
# scenario accounting slices
# ---------------------------------------------------------------------------


def test_scenario_slices_share_single_accounting_without_reset():
    returns = [0.05, 0.02, -0.01, 0.03, 0.01, -0.04, 0.02, 0.01, -0.02, 0.03]
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    values = _values(dates)
    profile = resolve_profile("perp_1d", {"split_date": "2024-01-05"})
    result = evaluate_metrics(
        values, values, _accounting(_ledger_frame(returns)), profile
    )
    gross = result["scenarios"]["gross"]
    assert gross["status"] == "complete"

    def compounded(slice_returns):
        return float(np.prod([1 + r for r in slice_returns]) - 1)

    assert gross["full"]["total_return"] == pytest.approx(compounded(returns))
    assert gross["in_sample"]["total_return"] == pytest.approx(compounded(returns[:5]))
    # Out-of-sample continues the same ledger: the first sliced return is the
    # realized return relative to the last in-sample equity, not a fresh start.
    assert gross["out_of_sample"]["total_return"] == pytest.approx(compounded(returns[5:]))
    assert gross["out_of_sample"]["n_periods"] == 5
    assert gross["full"]["n_periods"] == 10


def test_turnover_is_traded_notional_over_pretrade_equity_without_halving():
    # Pretrade equity is the recorded (post-fee) equity plus the day's fee.
    ledger = _ledger_frame([0.0, 0.0], notionals=[0.55, 0.40], fees=[0.10, 0.0])
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    values = _values(dates)
    profile = resolve_profile("perp_1d", {"split_date": "2024-01-01"})
    result = evaluate_metrics(values, values, _accounting(ledger), profile)
    gross = result["scenarios"]["gross"]
    # day1: 0.55 / (1.0 + 0.10) = 0.5 ; day2: 0.40 / 1.0 = 0.4 ; mean = 0.45.
    assert gross["full"]["turnover"] == pytest.approx(0.45)


def test_incomplete_all_costs_metrics_are_null_with_known_segment():
    certified = _ledger_frame([0.05, 0.02, -0.01, 0.03])
    all_costs = _scenario(certified, status="incomplete")
    ledger = _ledger_frame([0.05, 0.02, -0.01, 0.03, 0.01, -0.04])
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    values = _values(dates)
    profile = resolve_profile("perp_1d", {"split_date": "2024-01-03"})
    result = evaluate_metrics(values, values, _accounting(ledger, all_costs=all_costs), profile)

    incomplete = result["scenarios"]["all_costs"]
    assert incomplete["status"] == "incomplete"
    assert incomplete["full"] is None
    assert incomplete["in_sample"] is None
    assert incomplete["out_of_sample"] is None
    known = incomplete["known_segment"]
    assert known["through"] == "2024-01-04"
    expected = float(np.prod([1 + r for r in [0.05, 0.02, -0.01, 0.03]]) - 1)
    assert known["total_return"] == pytest.approx(expected)

    assert result["scenarios"]["gross"]["full"] is not None
    assert result["scenarios"]["trading_net"]["full"] is not None


def test_result_contains_no_nan_literals():
    dates = pd.date_range("2024-01-01", periods=15, freq="D")
    values = _values(dates, seed=7)
    labels = values.shift(-1)
    certified = _ledger_frame([0.01, -0.02, 0.03])
    all_costs = _scenario(certified, status="incomplete")
    profile = resolve_profile("perp_1d", {"split_date": "2024-01-10"})
    result = evaluate_metrics(
        values, labels, _accounting(_ledger_frame([0.01] * 15), all_costs=all_costs), profile
    )
    _assert_no_nan(result)


# ---------------------------------------------------------------------------
# half-life helper
# ---------------------------------------------------------------------------


def test_half_life_is_first_crossing_of_half_lag1_magnitude():
    decay = [
        {"horizon": 1, "rank_ic": 0.8},
        {"horizon": 2, "rank_ic": 0.5},
        {"horizon": 3, "rank_ic": 0.39},
    ]
    assert _half_life(decay) == 3
    negative = [
        {"horizon": 1, "rank_ic": -0.8},
        {"horizon": 2, "rank_ic": -0.41},
        {"horizon": 3, "rank_ic": -0.4},
    ]
    assert _half_life(negative) == 3


def test_half_life_missing_without_crossing_or_base():
    no_crossing = [{"horizon": h, "rank_ic": 0.9} for h in range(1, 11)]
    assert _half_life(no_crossing) is None
    assert _half_life([{"horizon": 1, "rank_ic": None}]) is None


# ---------------------------------------------------------------------------
# RankIC autocorrelation: exact calendar lags
# ---------------------------------------------------------------------------


def test_rank_ic_autocorr_uses_calendar_not_positional_lags():
    calendar = pd.date_range("2024-01-01", periods=5, freq="D")
    # 2024-01-03 has no valid cross-section: the RankIC series has a gap.
    daily = pd.DataFrame(
        {
            "date": [calendar[0], calendar[1], calendar[3], calendar[4]],
            "rank_ic": [0.0, 0.0, 1.0, 1.0],
        }
    )
    by_lag = {row["lag"]: row["autocorr"] for row in _rank_ic_autocorr(daily, calendar)}
    # Calendar lag 1 pairs (d1,d2)=(0,0) and (d4,d5)=(1,1); pairs spanning the
    # gap (d2,d3) and (d3,d4) are dropped, giving corr == 1.0.
    assert by_lag[1] == pytest.approx(1.0)
    # Positional lag 1 over the compacted series [0,0,1,1] would give ~0.577.
    positional = pd.Series([0.0, 0.0, 1.0, 1.0]).autocorr(lag=1)
    assert abs(by_lag[1] - positional) > 0.1
    # Calendar lag 2 leaves a single usable pair across the gap -> missing.
    assert by_lag[2] is None


def test_evaluate_metrics_autocorr_respects_missing_cross_section_date():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    values = _values(dates, seed=5)
    # Constant cross-section on day 4: no valid RankIC that date.
    values.loc[dates[3]] = 0.5
    labels = _values(dates, seed=9)
    profile = resolve_profile("perp_1d", {"split_date": "2024-01-06"})
    result = evaluate_metrics(values, labels, _accounting(_ledger_frame([0.0] * 8)), profile)

    daily_rows = result["samples"]["full"]["ic"]["daily"]
    assert dates[3].date().isoformat() not in [row["date"] for row in daily_rows]

    # Independent calendar-lag expectation: reindex onto the full calendar and
    # correlate pairwise against the shifted series.
    series = pd.Series(
        {pd.Timestamp(row["date"]): row["rank_ic"] for row in daily_rows}
    ).reindex(dates)
    for lag in (1, 2, 3):
        expected = series.corr(series.shift(lag))
        got = result["samples"]["full"]["rank_ic_autocorr"][lag - 1]["autocorr"]
        if pd.isna(expected):
            assert got is None, f"lag {lag}"
        else:
            assert got == pytest.approx(expected), f"lag {lag}"
