from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_common.backtest import run_backtest, run_target_backtest
from factor_common.profiles import resolve_profile


def _fixture(n=5, *, opens=None, complete=True, placeholder=False):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    cols = ["A", "B"]
    if opens is None:
        opens = pd.DataFrame(100.0, index=dates, columns=cols)
    quality = pd.DataFrame(
        {"has_complete_kline": complete, "has_placeholder_kline": placeholder},
        index=pd.MultiIndex.from_product([dates, cols], names=["date", "instrument"]),
    )
    events = pd.DataFrame(columns=["funding_time", "instrument", "funding_rate", "mark_price"])
    profile = resolve_profile("perp_1d", {"fee_rate": 0.001, "slippage": 0.0, "include_funding": False})
    return dates, opens, quality, events, profile


def _targets(dates, first=0.5):
    targets = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    targets.loc[dates[0]] = [first, -first]
    targets.loc[dates[1]] = [0.0, 0.0]
    return targets


def test_target_executes_signal_at_next_open_and_charges_exact_fee():
    dates, opens, quality, events, profile = _fixture()
    result = run_target_backtest(
        _targets(dates), opens, events, quality, profile,
        signal_start=dates[0], signal_end=dates[1],
    )
    scenario = result["scenarios"]["trading_net"]
    entry = scenario["orders"].query("status == 'filled'").iloc[0]
    assert entry["date"] == dates[1]
    assert entry["notional"] == pytest.approx(0.5)
    assert entry["fee"] == pytest.approx(0.0005)
    assert scenario["ledger"].iloc[0]["equity"] == pytest.approx(0.999)


def test_target_strict_missing_open_is_incomplete_and_preserves_position():
    dates, opens, quality, events, profile = _fixture()
    opens.loc[dates[2], "A"] = np.nan
    targets = _targets(dates)
    targets.loc[dates[2]] = [0.5, -0.5]
    result = run_target_backtest(targets, opens, events, quality, profile,
                                 signal_start=dates[0], signal_end=dates[2])
    scenario = result["scenarios"]["gross"]
    assert scenario["status"] == "incomplete"
    assert scenario["diagnostics"]["halt_reason"] == "unpriceable_held_position"
    assert scenario["diagnostics"]["final_quantities"]["A"] != 0
    assert scenario["diagnostics"]["unpriceable_forced_exits"] == []


def test_target_contract_settlement_closes_position_at_terminal_price():
    dates, opens, quality, events, profile = _fixture(6)
    targets = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    targets.loc[dates[0:3]] = [0.5, -0.5]
    targets.loc[dates[3]] = [0.0, 0.0]
    settlement_prices = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    settlement_prices.loc[dates[2], "A"] = 90.0

    result = run_target_backtest(
        targets,
        opens,
        events,
        quality,
        profile,
        signal_start=dates[0],
        signal_end=dates[2],
        settlement_prices=settlement_prices,
    )

    gross = result["scenarios"]["gross"]
    settlement = gross["orders"].query("reason == 'contract_settlement'")
    assert settlement[["date", "instrument", "price", "order_quantity"]].to_dict("records") == [
        {
            "date": dates[2],
            "instrument": "A",
            "price": 90.0,
            "order_quantity": pytest.approx(-0.005),
        }
    ]
    assert gross["positions"].loc[dates[2], "A"] == 0.0
    assert not ((gross["orders"]["date"] > dates[2]) & (gross["orders"]["instrument"] == "A")).any()
    assert gross["ledger"].loc[dates[2], "price_pnl"] == pytest.approx(-0.05)
    assert gross["diagnostics"]["contract_settlements"] == [
        {"date": "2024-01-03", "instrument": "A", "price": 90.0}
    ]


def test_quality_blocks_increase_but_allows_reduction():
    dates, opens, quality, events, profile = _fixture(6)
    quality.loc[(dates[1], "A"), "has_complete_kline"] = False
    targets = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    targets.loc[dates[0]] = [0.2, -0.2]
    targets.loc[dates[1]] = [0.5, -0.5]
    targets.loc[dates[2]] = [0.0, 0.0]
    result = run_target_backtest(targets, opens, events, quality, profile,
                                 signal_start=dates[0], signal_end=dates[2])
    orders = result["scenarios"]["gross"]["orders"]
    blocked = orders[(orders["date"] == dates[2]) & (orders["instrument"] == "A")]
    assert blocked.iloc[0]["status"] == "failed"
    assert blocked.iloc[0]["reason"] == "prior_bar_incomplete_kline"
    # The following target is zero, so the position is allowed to close.
    assert ((orders["date"] == dates[3]) & (orders["instrument"] == "A") &
            (orders["target_weight"] == 0)).any()


def test_no_evaluable_liquidation_tail_is_incomplete():
    dates, opens, quality, events, profile = _fixture(2)
    targets = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    targets.loc[dates[0]] = [0.5, -0.5]
    result = run_target_backtest(targets, opens, events, quality, profile,
                                 signal_start=dates[0], signal_end=dates[0])
    assert result["status"] == "incomplete"
    assert result["scenarios"]["gross"]["diagnostics"]["halt_reason"] == "missing_tail"


def test_explicit_zero_target_after_signal_end_is_liquidation_boundary():
    dates, opens, quality, events, profile = _fixture(4)
    targets = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    targets.loc[dates[0]] = [0.5, -0.5]
    targets.loc[dates[1]] = [0.0, 0.0]

    result = run_target_backtest(
        targets, opens, events, quality, profile,
        signal_start=dates[0], signal_end=dates[0],
    )

    gross = result["scenarios"]["gross"]
    assert result["status"] == "complete"
    assert gross["diagnostics"]["liquidation_date"] == "2024-01-03"
    assert gross["diagnostics"]["liquidation_reached"] is True
    assert gross["diagnostics"]["final_quantities"] == {}
    assert (gross["orders"]["date"] == dates[2]).any()
    assert (gross["orders"]["target_weight"] == 0).any()


def test_explicit_targets_only_trade_on_profile_rebalance_dates():
    dates, opens, quality, events, _ = _fixture(10)
    profile = resolve_profile(
        "perp_1d",
        {
            "rebalance_days": 3,
            "anchor_date": "2024-01-02",
            "fee_rate": 0.0,
            "slippage": 0.0,
            "include_funding": False,
        },
    )
    targets = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    for position, day in enumerate(dates[:7]):
        weight = 0.1 + position * 0.01
        targets.loc[day] = [weight, -weight]
    # Explicit post-window cash target remains an allowed liquidation event,
    # even though it does not need to align with the recurring schedule.
    targets.loc[dates[7]] = [0.0, 0.0]

    result = run_target_backtest(
        targets, opens, events, quality, profile,
        signal_start=dates[0], signal_end=dates[6],
    )

    filled_dates = set(
        result["scenarios"]["gross"]["orders"].query("status == 'filled'")["date"]
    )
    assert result["status"] == "complete"
    assert filled_dates == {dates[1], dates[4], dates[7], dates[8]}
    assert result["diagnostics"]["scheduled_dates"] == [
        "2024-01-02", "2024-01-05", "2024-01-08", "2024-01-09"
    ]


def test_legacy_api_remains_available():
    dates, opens, quality, events, profile = _fixture()
    values = pd.DataFrame([[2.0, 1.0]] + [[np.nan, np.nan]] * (len(dates) - 1),
                          index=dates, columns=["A", "B"])
    legacy = run_backtest(values, opens, events, quality, profile,
                          signal_start=dates[0], signal_end=dates[-1])
    assert legacy["portfolio"] == "long_short"
    assert set(legacy["scenarios"]) == {"gross", "trading_net", "all_costs"}
