"""Hand-calculated tests for fixed-quantity daily portfolio accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_common.backtest import run_backtest
from factor_common.profiles import resolve_profile

DAY0 = pd.Timestamp("2024-01-01")
EMPTY_EVENTS = pd.DataFrame(columns=["funding_time", "instrument", "funding_rate", "mark_price"])


def _profile(**overrides):
    base = {
        "n_groups": 2,
        "anchor_date": "2024-01-01",
        "rebalance_days": 1,
        "fee_rate": 0.0,
        "include_funding": False,
    }
    base.update(overrides)
    return resolve_profile("perp_1d", base)


def _matrix(n_days: int, rows: dict[int, dict[str, float]]) -> pd.DataFrame:
    dates = pd.date_range(DAY0, periods=n_days, freq="D", name="date")
    instruments = sorted({inst for row in rows.values() for inst in row})
    frame = pd.DataFrame(np.nan, index=dates, columns=instruments)
    for offset, row in rows.items():
        for instrument, value in row.items():
            frame.loc[dates[offset], instrument] = value
    return frame


def _frames(n_days, values_rows, opens_rows):
    values = _matrix(n_days, values_rows)
    opens = _matrix(n_days, opens_rows).reindex(columns=values.columns)
    return values, opens


def _quality(n_days, instruments, *, placeholder=(), coverage_overrides=None):
    dates = pd.date_range(DAY0, periods=n_days, freq="D")
    index = pd.MultiIndex.from_product([dates, list(instruments)], names=["date", "instrument"])
    frame = pd.DataFrame(
        {
            "has_placeholder_kline": False,
            "funding_coverage_status": "complete",
        },
        index=index,
    )
    for day, instrument in placeholder:
        frame.loc[(pd.Timestamp(day), instrument), "has_placeholder_kline"] = True
    for (day, instrument), status in (coverage_overrides or {}).items():
        frame.loc[(pd.Timestamp(day), instrument), "funding_coverage_status"] = status
    return frame


def _events(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["funding_time", "instrument", "funding_rate", "mark_price"])
    frame["funding_time"] = pd.to_datetime(frame["funding_time"], utc=True)
    return frame


def _run(values, opens, events=EMPTY_EVENTS, quality=None, profile=None, *,
         signal_start="2024-01-01", signal_end="2024-12-31"):
    if quality is None:
        quality = _quality(len(values.index), values.columns)
    if profile is None:
        profile = _profile()
    return run_backtest(
        values, opens, events, quality, profile,
        signal_start=signal_start, signal_end=signal_end,
    )


def test_long_short_ten_percent_return_container_shape():
    # Entry opens 100/100, exit opens A=110/B=90, zero costs:
    # long 0.5 in A (+10%) and short 0.5 in B (+10%) gives +10% overall.
    values, opens = _frames(
        3,
        {0: {"A": 2.0, "B": 1.0}},
        {1: {"A": 100.0, "B": 100.0}, 2: {"A": 110.0, "B": 90.0}},
    )
    result = _run(values, opens)

    assert set(result) == {"status", "portfolio", "profile_id", "scenarios", "diagnostics"}
    assert result["status"] == "complete"
    assert result["portfolio"] == "long_short"
    assert set(result["scenarios"]) == {"gross", "trading_net", "all_costs"}
    scenario = result["scenarios"]["gross"]
    assert set(scenario) == {
        "status", "ledger", "orders", "positions", "valuation_prices",
        "funding", "funding_coverage", "diagnostics",
    }
    assert scenario["status"] == "complete"
    assert list(scenario["ledger"].columns) == [
        "equity", "return", "price_pnl", "funding_cashflow", "fee", "trade_notional",
    ]
    assert scenario["ledger"].index.name == "date"

    ledger = scenario["ledger"]
    assert ledger["equity"].tolist() == pytest.approx([1.0, 1.1])
    assert ledger["return"].tolist() == pytest.approx([0.0, 0.1])
    assert ledger["price_pnl"].tolist() == pytest.approx([0.0, 0.1])

    positions = scenario["positions"]
    assert positions.loc[DAY0 + pd.Timedelta(days=1), "A"] == pytest.approx(0.005)
    assert positions.loc[DAY0 + pd.Timedelta(days=1), "B"] == pytest.approx(-0.005)
    assert positions.loc[DAY0 + pd.Timedelta(days=2)].abs().sum() == pytest.approx(0.0)

    orders = scenario["orders"]
    filled = orders[orders["status"] == "filled"]
    assert len(filled) == 4
    assert filled["fee"].abs().sum() == pytest.approx(0.0)

    # With zero fees and funding disabled all scenarios coincide.
    for name in ("trading_net", "all_costs"):
        assert result["scenarios"][name]["ledger"]["equity"].tolist() == pytest.approx([1.0, 1.1])


def test_repeating_daily_signals_do_not_trade_daily():
    # rebalance_days=3: anchor-aligned dates only, never daily.
    values, opens = _frames(
        13,
        {offset: {"A": 2.0, "B": 1.0} for offset in range(10)},
        {offset: {"A": 100.0 + offset, "B": 100.0} for offset in range(13)},
    )
    result = _run(values, opens, profile=_profile(rebalance_days=3))
    gross = result["scenarios"]["gross"]
    order_dates = sorted(gross["orders"]["date"].unique())
    expected = [pd.Timestamp(d) for d in ("2024-01-04", "2024-01-07", "2024-01-10", "2024-01-13")]
    assert list(order_dates) == expected
    assert result["diagnostics"]["scheduled_dates"] == [d.date().isoformat() for d in expected]
    assert result["status"] == "complete"


def test_fee_fixture_equity_path():
    # Constant price 100: entry turnover 1 -> fee .001 -> equity .999;
    # liquidation turnover 1 -> another .001 -> equity .998.
    values, opens = _frames(
        3,
        {0: {"A": 2.0, "B": 1.0}},
        {offset: {"A": 100.0, "B": 100.0} for offset in range(3)},
    )
    profile = _profile(fee_rate=0.001, include_funding=True)
    result = _run(values, opens, profile=profile)

    trading = result["scenarios"]["trading_net"]["ledger"]
    assert trading["equity"].tolist() == pytest.approx([0.999, 0.998])
    # First daily return includes the inception fee relative to initial equity 1.
    assert trading["return"].iloc[0] == pytest.approx(-0.001)
    assert trading["fee"].tolist() == pytest.approx([0.001, 0.001])
    assert trading["trade_notional"].tolist() == pytest.approx([1.0, 1.0])

    gross = result["scenarios"]["gross"]["ledger"]
    assert gross["equity"].tolist() == pytest.approx([1.0, 1.0])

    # No funding events and complete coverage: all_costs matches trading_net.
    all_costs = result["scenarios"]["all_costs"]["ledger"]
    pd.testing.assert_frame_equal(all_costs, trading)
    assert result["status"] == "complete"


def test_boundary_funding_settles_on_pre_trade_quantities():
    # Rebalance day boundary event settles on old quantities; intraday event
    # settles on the quantities established by that boundary's trades.
    values, opens = _frames(
        4,
        {0: {"A": 2.0, "B": 1.0}, 1: {"A": 2.0, "B": 1.0}},
        {offset: {"A": 100.0, "B": 100.0} for offset in range(4)},
    )
    events = _events([
        {"funding_time": "2024-01-03 00:00:00", "instrument": "A",
         "funding_rate": 0.001, "mark_price": 100.0},
        {"funding_time": "2024-01-03 08:00:00", "instrument": "A",
         "funding_rate": 0.001, "mark_price": 100.0},
    ])
    profile = _profile(include_funding=True)
    result = _run(values, opens, events=events, profile=profile)

    all_costs = result["scenarios"]["all_costs"]
    funding = all_costs["funding"]
    assert len(funding) == 2
    # Boundary event: pre-trade quantity 0.005 -> cashflow -0.0005.
    assert funding.iloc[0]["quantity"] == pytest.approx(0.005)
    assert funding.iloc[0]["cashflow"] == pytest.approx(-0.0005)
    # Intraday event: post-rebalance quantity 0.5 * 0.9995 / 100 = 0.0049975.
    assert funding.iloc[1]["quantity"] == pytest.approx(0.0049975)
    assert funding.iloc[1]["cashflow"] == pytest.approx(-0.00049975)
    # Final equity: 1 - 0.0005 - 0.00049975 = 0.99900025.
    assert all_costs["ledger"]["equity"].iloc[-1] == pytest.approx(0.99900025)
    # Funding already-due at the boundary feeds the sizing equity.
    assert all_costs["positions"].loc[pd.Timestamp("2024-01-03"), "A"] == pytest.approx(0.0049975)
    # Gross is unaffected by funding and does not rebalance (target unchanged).
    gross = result["scenarios"]["gross"]
    assert gross["positions"].loc[pd.Timestamp("2024-01-03"), "A"] == pytest.approx(0.005)
    assert gross["ledger"]["equity"].iloc[-1] == pytest.approx(1.0)


def test_first_entry_owns_no_boundary_funding_exit_settles():
    values, opens = _frames(
        3,
        {0: {"A": 2.0, "B": 1.0}},
        {offset: {"A": 100.0, "B": 100.0} for offset in range(3)},
    )
    events = _events([
        {"funding_time": "2024-01-02 00:00:00", "instrument": "A",
         "funding_rate": 0.001, "mark_price": 100.0},
        {"funding_time": "2024-01-03 00:00:00", "instrument": "A",
         "funding_rate": 0.001, "mark_price": 100.0},
    ])
    result = _run(values, opens, events=events, profile=_profile(include_funding=True))
    all_costs = result["scenarios"]["all_costs"]
    funding = all_costs["funding"]
    # Entry-boundary event is not ours (pre-trade quantity zero); exit-boundary
    # event settles on 0.005 before liquidation.
    assert len(funding) == 1
    assert funding.iloc[0]["funding_time"] == pd.Timestamp("2024-01-03 00:00:00", tz="UTC")
    assert funding.iloc[0]["quantity"] == pytest.approx(0.005)
    assert funding.iloc[0]["cashflow"] == pytest.approx(-0.0005)
    ledger = all_costs["ledger"]
    assert ledger["funding_cashflow"].tolist() == pytest.approx([0.0, -0.0005])
    assert ledger["equity"].iloc[-1] == pytest.approx(0.9995)


def test_negative_funding_pays_short_and_receives_long():
    values, opens = _frames(
        3,
        {0: {"A": 2.0, "B": 1.0}},
        {offset: {"A": 100.0, "B": 100.0} for offset in range(3)},
    )
    events = _events([
        {"funding_time": "2024-01-03 00:00:00", "instrument": "A",
         "funding_rate": -0.002, "mark_price": 100.0},
        {"funding_time": "2024-01-03 00:00:00", "instrument": "B",
         "funding_rate": -0.002, "mark_price": 100.0},
    ])
    result = _run(values, opens, events=events, profile=_profile(include_funding=True))
    funding = result["scenarios"]["all_costs"]["funding"]
    by_instrument = funding.set_index("instrument")
    # Long A receives: -(0.005 * 100 * -0.002) = +0.001.
    assert by_instrument.loc["A", "cashflow"] == pytest.approx(0.001)
    # Short B pays: -(-0.005 * 100 * -0.002) = -0.001.
    assert by_instrument.loc["B", "cashflow"] == pytest.approx(-0.001)
    # Net zero across the balanced book.
    assert result["scenarios"]["all_costs"]["ledger"]["equity"].iloc[-1] == pytest.approx(1.0)


def test_unresolved_funding_halts_all_costs_only():
    values, opens = _frames(
        3,
        {0: {"A": 2.0, "B": 1.0}},
        {offset: {"A": 100.0, "B": 100.0} for offset in range(3)},
    )
    events = _events([
        {"funding_time": "2024-01-03 00:00:00", "instrument": "A",
         "funding_rate": 0.001, "mark_price": np.nan},
    ])
    result = _run(values, opens, events=events, profile=_profile(include_funding=True))

    all_costs = result["scenarios"]["all_costs"]
    assert all_costs["status"] == "incomplete"
    assert all_costs["diagnostics"]["halt_reason"] == "unresolved_funding"
    assert all_costs["diagnostics"]["halt_date"] == "2024-01-03"
    # Certified ledger stops at the last fully-known boundary.
    assert len(all_costs["ledger"]) == 1
    # Quantities are retained, never silently zeroed, and no later sizes are
    # computed from assumed-zero funding.
    assert all_costs["diagnostics"]["final_quantities"] == {"A": 0.005, "B": -0.005}
    assert set(all_costs["orders"]["date"].unique()) == {pd.Timestamp("2024-01-02")}

    for name in ("gross", "trading_net"):
        scenario = result["scenarios"][name]
        assert scenario["status"] == "complete"
        assert len(scenario["ledger"]) == 2
        assert pd.Timestamp("2024-01-03") in set(scenario["orders"]["date"])
    assert result["status"] == "incomplete"


def test_unknown_coverage_halts_all_costs():
    values, opens = _frames(
        3,
        {0: {"A": 2.0, "B": 1.0}},
        {offset: {"A": 100.0, "B": 100.0} for offset in range(3)},
    )
    quality = _quality(3, ["A", "B"], coverage_overrides={
        ("2024-01-02", "A"): "unknown",
    })
    result = _run(values, opens, quality=quality, profile=_profile(include_funding=True))

    all_costs = result["scenarios"]["all_costs"]
    assert all_costs["status"] == "incomplete"
    assert all_costs["diagnostics"]["halt_reason"] == "unresolved_funding_coverage"
    assert all_costs["diagnostics"]["halt_date"] == "2024-01-02"
    # The entry day itself is not certified: possible missing events that day.
    assert len(all_costs["ledger"]) == 0
    assert all_costs["diagnostics"]["final_quantities"] == {"A": 0.005, "B": -0.005}
    assert result["scenarios"]["gross"]["status"] == "complete"
    assert result["status"] == "incomplete"


def test_missing_tail_records_evaluable_segment():
    # Last usable signal drives a trade on 2024-01-03; the liquidation boundary
    # 2024-01-04 is beyond the data.
    values, opens = _frames(
        3,
        {0: {"A": 2.0, "B": 1.0}, 1: {"A": 2.0, "B": 1.0}},
        {offset: {"A": 100.0, "B": 100.0} for offset in range(3)},
    )
    result = _run(values, opens, profile=_profile(include_funding=True))
    gross = result["scenarios"]["gross"]
    diagnostics = gross["diagnostics"]
    assert gross["status"] == "incomplete"
    assert diagnostics["missing_tail"] is True
    assert diagnostics["liquidation_date"] == "2024-01-04"
    assert diagnostics["liquidation_reached"] is False
    assert diagnostics["tail_evaluated_through"] == "2024-01-03"
    assert len(gross["ledger"]) == 2
    assert diagnostics["final_quantities"] == {"A": 0.005, "B": -0.005}
    assert result["status"] == "incomplete"


def test_membership_exit_while_held_liquidates_next_rebalance():
    # The held name drops out of the signal (membership exit masks its value);
    # the next scheduled boundary exits both legs at available prices.
    values, opens = _frames(
        3,
        {0: {"A": 2.0, "B": 1.0}, 1: {"B": 1.0}},
        {1: {"A": 100.0, "B": 100.0}, 2: {"A": 105.0, "B": 95.0}},
    )
    result = _run(values, opens)
    gross = result["scenarios"]["gross"]
    exit_orders = gross["orders"][gross["orders"]["date"] == pd.Timestamp("2024-01-03")]
    by_instrument = exit_orders.set_index("instrument")
    assert by_instrument.loc["A", "side"] == "sell"
    assert by_instrument.loc["A", "price"] == pytest.approx(105.0)
    assert by_instrument.loc["B", "side"] == "buy"
    assert by_instrument.loc["B", "price"] == pytest.approx(95.0)
    assert gross["positions"].loc[pd.Timestamp("2024-01-03")].abs().sum() == pytest.approx(0.0)
    # +0.025 from the long and +0.025 from the short.
    assert gross["ledger"]["equity"].iloc[-1] == pytest.approx(1.05)
    assert result["status"] == "complete"


def test_failed_entry_price_not_redistributed():
    values, opens = _frames(
        3,
        {0: {"A": 2.0, "B": 1.0}},
        {1: {"A": 100.0, "B": np.nan}, 2: {"A": 110.0, "B": 95.0}},
    )
    result = _run(values, opens)
    gross = result["scenarios"]["gross"]
    orders = gross["orders"]
    failed = orders[orders["status"] == "failed"]
    assert len(failed) == 1
    assert failed.iloc[0]["instrument"] == "B"
    assert failed.iloc[0]["reason"] == "invalid_price"
    assert gross["diagnostics"]["failed_orders"] == 1
    # A keeps its own 50% weight; rejected weight is never redistributed.
    assert gross["positions"].loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(0.005)
    # Only the long leg earns: +10% on half the equity.
    assert gross["ledger"]["equity"].iloc[-1] == pytest.approx(1.05)
    assert gross["status"] == "complete"


def test_empty_signals_no_trades():
    values, opens = _frames(
        3,
        {},
        {offset: {"A": 100.0, "B": 100.0} for offset in range(3)},
    )
    result = _run(values, opens)
    assert result["status"] == "complete"
    assert result["diagnostics"]["no_usable_signals"] is True
    for scenario in result["scenarios"].values():
        assert scenario["status"] == "complete"
        assert len(scenario["ledger"]) == 0
        assert len(scenario["orders"]) == 0
        assert len(scenario["positions"]) == 0


def test_nonpositive_equity_halts_accounting():
    # Long A 0.5 at 100 -> 1 and short B 0.5 at 100 -> 210:
    # equity = 1 - 0.495 - 0.55 = -0.045; accounting halts at that boundary.
    values, opens = _frames(
        3,
        {0: {"A": 2.0, "B": 1.0}},
        {1: {"A": 100.0, "B": 100.0}, 2: {"A": 1.0, "B": 210.0}},
    )
    result = _run(values, opens)
    assert result["status"] == "incomplete"
    for scenario in result["scenarios"].values():
        assert scenario["status"] == "incomplete"
        assert scenario["diagnostics"]["halt_reason"] == "nonpositive_equity"
        assert scenario["diagnostics"]["halt_date"] == "2024-01-03"
        assert len(scenario["ledger"]) == 1
        assert scenario["ledger"]["equity"].iloc[-1] == pytest.approx(1.0)
        assert scenario["diagnostics"]["final_quantities"] == {"A": 0.005, "B": -0.005}


def test_scenarios_size_from_their_own_equity():
    # Fees make trading_net poorer than gross after entry; the next scheduled
    # rebalance sizes each scenario from its own equity.
    values, opens = _frames(
        4,
        {0: {"A": 2.0, "B": 1.0}, 1: {"A": 2.0, "B": 1.0}},
        {offset: {"A": 100.0, "B": 100.0} for offset in range(4)},
    )
    result = _run(values, opens, profile=_profile(fee_rate=0.001))
    day2 = pd.Timestamp("2024-01-03")
    gross_qty = result["scenarios"]["gross"]["positions"].loc[day2, "A"]
    trading_qty = result["scenarios"]["trading_net"]["positions"].loc[day2, "A"]
    assert gross_qty == pytest.approx(0.005)
    # 0.5 * 0.999 / 100 = 0.004995
    assert trading_qty == pytest.approx(0.004995)
    assert gross_qty != pytest.approx(trading_qty)
    assert result["scenarios"]["gross"]["ledger"]["equity"].iloc[-1] == pytest.approx(1.0)
    assert result["scenarios"]["trading_net"]["ledger"]["equity"].iloc[-1] < 1.0
    assert result["status"] == "complete"


def test_execution_calendar_independent_of_plot_window():
    values, opens = _frames(
        4,
        {0: {"A": 2.0, "B": 1.0}, 1: {"A": 2.0, "B": 1.0}},
        {offset: {"A": 100.0 + offset, "B": 100.0} for offset in range(4)},
    )
    run1 = _run(values, opens, signal_start="2024-01-01", signal_end="2024-01-02")
    run2 = _run(values, opens, signal_start="2024-01-01", signal_end="2024-01-04")
    # A wider plot window over the same signals changes nothing.
    pd.testing.assert_frame_equal(
        run1["scenarios"]["gross"]["ledger"], run2["scenarios"]["gross"]["ledger"]
    )
    pd.testing.assert_frame_equal(
        run1["scenarios"]["gross"]["orders"], run2["scenarios"]["gross"]["orders"]
    )
    # Excluding the first signal shifts entries but not the anchor calendar.
    run3 = _run(values, opens, signal_start="2024-01-02", signal_end="2024-01-04")
    dates1 = set(run1["scenarios"]["gross"]["orders"]["date"])
    dates3 = set(run3["scenarios"]["gross"]["orders"]["date"])
    assert pd.Timestamp("2024-01-02") in dates1
    assert pd.Timestamp("2024-01-02") not in dates3
    assert pd.Timestamp("2024-01-03") in dates3


def test_prior_bar_placeholder_eligibility_and_retrospective_evidence():
    # A's prior completed bar (2024-01-01) is a zero-volume placeholder: the
    # 2024-01-02 opening entry is blocked. B's own trade day is a placeholder,
    # known only at that day's close: the trade stands and is reported
    # retrospectively, without being excluded by same-day knowledge.
    values, opens = _frames(
        3,
        {0: {"A": 2.0, "B": 1.0}},
        {offset: {"A": 100.0, "B": 100.0} for offset in range(3)},
    )
    quality = _quality(
        3, ["A", "B"],
        placeholder=[("2024-01-01", "A"), ("2024-01-02", "B")],
    )
    result = _run(values, opens, quality=quality)
    gross = result["scenarios"]["gross"]
    orders = gross["orders"]
    failed = orders[orders["status"] == "failed"]
    assert len(failed) == 1
    assert failed.iloc[0]["instrument"] == "A"
    assert failed.iloc[0]["reason"] == "prior_bar_ineligible"
    assert {"date": "2024-01-02", "instrument": "A"} in gross["diagnostics"]["blocked_orders"]
    assert {"date": "2024-01-02", "instrument": "B"} in gross["diagnostics"]["retrospective_nonexecution"]

    entry = orders[(orders["status"] == "filled") & (orders["date"] == pd.Timestamp("2024-01-02"))]
    assert list(entry["instrument"]) == ["B"]
    # B's exit is never blocked by ineligibility (exits always allowed).
    assert gross["status"] == "complete"
    assert gross["positions"].loc[pd.Timestamp("2024-01-03")].abs().sum() == pytest.approx(0.0)


def test_unpriceable_held_position_halts_valuation():
    # A is held into 2024-01-03 but has no valid open there: certified
    # valuation halts for every scenario and the position never disappears.
    values, opens = _frames(
        3,
        {0: {"A": 2.0, "B": 1.0}},
        {1: {"A": 100.0, "B": 100.0}, 2: {"A": np.nan, "B": 100.0}},
    )
    result = _run(values, opens)
    assert result["status"] == "incomplete"
    for scenario in result["scenarios"].values():
        assert scenario["status"] == "incomplete"
        assert scenario["diagnostics"]["halt_reason"] == "unpriceable_position"
        assert scenario["diagnostics"]["halt_date"] == "2024-01-03"
        assert len(scenario["ledger"]) == 1
        assert scenario["diagnostics"]["final_quantities"] == {"A": 0.005, "B": -0.005}
