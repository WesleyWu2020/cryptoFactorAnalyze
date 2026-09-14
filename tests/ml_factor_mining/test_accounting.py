"""Integration certification for cost-adjusted portfolio accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_common.backtest import run_backtest
from ML_factor_mining.config import Config
from ML_factor_mining.scoring import backtest_profile


def test_three_day_holding_fees_slippage_and_funding_reconcile():
    """A next-open three-day holding has independently reconciled cash flows."""
    dates = pd.date_range("2025-01-01", periods=5, name="date")
    symbols = list("ABCDE")
    values = pd.DataFrame(np.nan, index=dates, columns=symbols)
    values.loc[dates[0]] = np.arange(5)
    opens = pd.DataFrame(100.0, index=dates, columns=symbols)
    opens.loc[dates[-1], "E"] = 110.0
    opens.loc[dates[-1], "A"] = 90.0
    events = pd.DataFrame(
        {
            "funding_time": [pd.Timestamp("2025-01-03 08:00", tz="UTC")],
            "instrument": ["E"],
            "funding_rate": [0.001],
            "mark_price": [100.0],
        }
    )
    index = pd.MultiIndex.from_product(
        [dates, symbols], names=["date", "instrument"]
    )
    quality = pd.DataFrame(
        {
            "has_placeholder_kline": False,
            "funding_coverage_status": "complete",
        },
        index=index,
    )
    config = Config(anchor_date="2025-01-02", fee_rate=0.0005, slippage=0.001)

    result = run_backtest(
        values,
        opens,
        events,
        quality,
        backtest_profile(config),
        signal_start=dates[0],
        signal_end=dates[0],
    )
    net = result["scenarios"]["all_costs"]

    assert net["status"] == "complete"
    assert net["diagnostics"]["first_order_date"] == "2025-01-02"
    assert net["diagnostics"]["liquidation_date"] == "2025-01-05"
    # Fixed quantities: long 0.005 E and short 0.005 A. Price PnL is 0.10.
    # Entry+exit notional is 2.0; fee+slippage is 0.003; funding is -0.0005.
    assert net["ledger"]["equity"].iloc[-1] == pytest.approx(1.0965)
    assert net["ledger"]["fee"].sum() == pytest.approx(0.001)
    assert net["ledger"]["slippage"].sum() == pytest.approx(0.002)
    assert net["funding"]["cashflow"].sum() == pytest.approx(-0.0005)
    assert net["positions"].iloc[-1].abs().sum() == 0

