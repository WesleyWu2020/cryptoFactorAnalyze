"""Tests for Path B backtest engine: run_btc_core_short_backtest."""

import pandas as pd
import numpy as np
from portfolio.backtester.engine import run_btc_core_short_backtest


def _prices():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "BTCUSDT": [100.0, 110.0, 105.0],
        "AUSDT": [10.0, 12.0, 9.0],
        "BUSDT": [5.0, 4.0, 4.5],
    }).set_index("date")


def test_smoke_3day_2coin_no_funding_no_fees():
    prices = _prices()
    btc_w = pd.Series({pd.Timestamp("2024-01-01"): 0.5,
                       pd.Timestamp("2024-01-02"): 0.6,
                       pd.Timestamp("2024-01-03"): 0.5})
    shorts = {"2024-01-01": ["AUSDT", "BUSDT"]}  # set at day0 end -> held day1
    funding = pd.DataFrame(columns=["date", "instrument", "funding_daily"])
    nav = run_btc_core_short_backtest(prices=prices, btc_weights=btc_w,
                                       shorts_by_date=shorts, funding=funding,
                                       alt_short_exposure=-0.2, fee_rate=0.0)
    assert len(nav) == 3
    assert nav.iloc[0] == 1.0
    # Day1: BTC +10%, w_btc(day0)=0.5 -> +5%. Shorts: A +20%, B -20%, w_each=-0.1 each.
    # r_short = -0.1*0.2 + -0.1*(-0.2) = -0.02 + 0.02 = 0. NAV = 1.05
    assert abs(nav.iloc[1] - 1.05) < 1e-9


def test_btc_only_when_no_shorts():
    prices = _prices()
    btc_w = pd.Series({d: 1.0 for d in prices.index})
    nav = run_btc_core_short_backtest(prices=prices, btc_weights=btc_w,
                                       shorts_by_date={}, funding=pd.DataFrame(columns=["date", "instrument", "funding_daily"]),
                                       alt_short_exposure=-0.0, fee_rate=0.0)
    # Day1 ret = 10%, day2 = -5/110 ≈ -4.545%
    assert abs(nav.iloc[1] - 1.10) < 1e-9
    assert abs(nav.iloc[2] - 1.10 * (1 + (105/110 - 1))) < 1e-9


def test_funding_benefits_shorts():
    prices = _prices()
    btc_w = pd.Series({d: 0.0 for d in prices.index})  # no BTC
    shorts = {"2024-01-01": ["AUSDT"]}
    # A flat in prices: set same close so r_short = 0. Only funding PnL remains.
    prices2 = prices.copy()
    prices2["AUSDT"] = [10.0, 10.0, 10.0]
    funding = pd.DataFrame({"date": ["2024-01-02"], "instrument": ["AUSDT"], "funding_daily": [0.001]})
    nav = run_btc_core_short_backtest(prices=prices2, btc_weights=btc_w,
                                       shorts_by_date=shorts, funding=funding,
                                       alt_short_exposure=-0.3, fee_rate=0.0)
    # w_each=-0.3; funding_pnl = -(-0.3) * 0.001 = 0.0003
    assert abs(nav.iloc[1] - (1.0 + 0.0003)) < 1e-9


def test_turnover_fee_on_change():
    prices = _prices()
    btc_w = pd.Series({d: 0.0 for d in prices.index})
    # Change shorts from A to B on day 2 (rebalance at day1 close -> held day2)
    shorts = {"2024-01-01": ["AUSDT"], "2024-01-02": ["BUSDT"]}
    funding = pd.DataFrame(columns=["date", "instrument", "funding_daily"])
    nav = run_btc_core_short_backtest(prices=prices, btc_weights=btc_w,
                                       shorts_by_date=shorts, funding=funding,
                                       alt_short_exposure=-0.2, fee_rate=0.01)
    # Day2: turnover_syms = {A, B} (size 2), new_shorts has 1 -> fee = 0.01 * 2 * 0.2 / 1 = 0.004
    # Day2 BUSDT r = 4.5/4 - 1 = 0.125; w_each=-0.2; r_short = -0.2*0.125 = -0.025
    # r_total = 0 + -0.025 + 0 - 0.004 = -0.029; nav[2] = nav[1] * (1 - 0.029)
    # Day1: shorts=A (turnover from empty to {A}, size 1 diff, fee = 0.01*1*0.2/1 = 0.002)
    # Day1 AUSDT r=0.2; w_each=-0.2; r_short=-0.04; r_total = -0.04 - 0.002 = -0.042
    assert abs(nav.iloc[1] - (1.0 * (1 - 0.042))) < 1e-9
    assert abs(nav.iloc[2] - (nav.iloc[1] * (1 - 0.029))) < 1e-9


def test_missing_price_skipped_gracefully():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "BTCUSDT": [100.0, 110.0],
        "AUSDT": [np.nan, 12.0],  # prev NaN -> skip
    }).set_index("date")
    btc_w = pd.Series({d: 0.5 for d in prices.index})
    shorts = {"2024-01-01": ["AUSDT"]}
    nav = run_btc_core_short_backtest(prices=prices, btc_weights=btc_w,
                                       shorts_by_date=shorts,
                                       funding=pd.DataFrame(columns=["date", "instrument", "funding_daily"]),
                                       alt_short_exposure=-0.3, fee_rate=0.0)
    # A skipped; only BTC contributes: 0.5*0.1 = 0.05
    assert abs(nav.iloc[1] - 1.05) < 1e-9
