import numpy as np
import pandas as pd


def _make_kline(days=20):
    """BTC flat, ETH +1%/day (provides pure long-side profit)."""
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    for d in dates:
        rows += [
            {"symbol": "BTCUSDT", "date": d, "close": 50000.0},
            {"symbol": "ETHUSDT", "date": d, "close": 1500 * (1.01 ** ((d - dates[0]).days))},
        ]
    return pd.DataFrame(rows)


def test_hedged_engine_no_hedge_same_as_long_only():
    from portfolio.backtester.engine import run_hedged_backtest
    kline = _make_kline()
    dates = sorted(kline["date"].unique())
    weights = {dates[0]: {"ETHUSDT": 1.0}}
    # 全程不对冲
    alt_exp = pd.Series(1.0, index=dates)
    hedge = pd.Series(0.0, index=dates)
    ret = run_hedged_backtest(kline, weights, alt_exp, hedge,
                               alt_fee=0.0, perp_fee=0.0, funding_annual=0.0)
    # 首日扣 0 手续费；此后 +1%/day
    assert abs(ret.iloc[1] - 0.01) < 1e-6


def test_hedged_engine_full_hedge_kills_btc_exposure():
    """BTC空头对冲完全对抵BTC涨幅；ETH=BTC涨跌时，hedge=beta=1 should net 0."""
    from portfolio.backtester.engine import run_hedged_backtest
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for i, d in enumerate(dates):
        price = 50000 * 1.01 ** i  # 都涨1%
        rows += [
            {"symbol": "BTCUSDT", "date": d, "close": price},
            {"symbol": "ETHUSDT", "date": d, "close": price * 30},
        ]
    kline = pd.DataFrame(rows)
    weights = {dates[0]: {"ETHUSDT": 1.0}}
    alt_exp = pd.Series(1.0, index=dates)
    hedge = pd.Series(1.0, index=dates)  # 1:1 对冲
    ret = run_hedged_backtest(kline, weights, alt_exp, hedge,
                               alt_fee=0.0, perp_fee=0.0, funding_annual=0.0)
    # 每日都应 ≈ 0（ETH +1% * 1 - BTC +1% * 1 = 0）
    for r in ret.iloc[1:]:
        assert abs(r) < 1e-6


def test_hedged_engine_funding_cost_applied():
    from portfolio.backtester.engine import run_hedged_backtest
    kline = _make_kline(days=5)
    dates = sorted(kline["date"].unique())
    weights = {dates[0]: {"ETHUSDT": 1.0}}
    alt_exp = pd.Series(1.0, index=dates)
    hedge = pd.Series(0.6, index=dates)
    ret = run_hedged_backtest(kline, weights, alt_exp, hedge,
                               alt_fee=0.0, perp_fee=0.0,
                               funding_annual=0.365)  # 极端: 0.1%/day
    # 第2天 ETH +1%, BTC 0, 资金费 0.6*0.365/365=0.0006
    # return = 1.0*0.01 - 0.6*0 - 0.0006 = 0.0094
    assert abs(ret.iloc[1] - 0.0094) < 1e-5


def test_hedged_engine_returns_empty_when_no_weights():
    from portfolio.backtester.engine import run_hedged_backtest
    kline = _make_kline(days=5)
    dates = sorted(kline["date"].unique())
    alt_exp = pd.Series(1.0, index=dates)
    hedge = pd.Series(0.0, index=dates)
    ret = run_hedged_backtest(kline, {}, alt_exp, hedge)
    assert ret.empty
