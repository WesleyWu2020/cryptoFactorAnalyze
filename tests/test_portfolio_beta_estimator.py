import numpy as np
import pandas as pd


def _make_kline(days=200):
    np.random.seed(7)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    btc_ret = np.random.randn(days) * 0.03
    btc_close = 20000 * (1 + pd.Series(btc_ret)).cumprod().values
    # ETH beta=1.5
    eth_ret = 1.5 * btc_ret + np.random.randn(days) * 0.01
    eth_close = 1500 * (1 + pd.Series(eth_ret)).cumprod().values
    # SOL beta=2.0
    sol_ret = 2.0 * btc_ret + np.random.randn(days) * 0.01
    sol_close = 100 * (1 + pd.Series(sol_ret)).cumprod().values
    rows = []
    for i, d in enumerate(dates):
        rows += [
            {"symbol": "BTCUSDT", "date": d, "close": btc_close[i]},
            {"symbol": "ETHUSDT", "date": d, "close": eth_close[i]},
            {"symbol": "SOLUSDT", "date": d, "close": sol_close[i]},
        ]
    return pd.DataFrame(rows)


def test_coin_beta_matches_true_value():
    from portfolio.portfolio_builder.beta_estimator import compute_coin_betas
    kline = _make_kline()
    betas = compute_coin_betas(kline, window=60, btc_symbol="BTCUSDT")
    eth_beta = betas[betas["instrument"] == "ETHUSDT"]["beta"].iloc[-1]
    sol_beta = betas[betas["instrument"] == "SOLUSDT"]["beta"].iloc[-1]
    assert 1.2 < eth_beta < 1.8, f"ETH beta {eth_beta} off"
    assert 1.7 < sol_beta < 2.3, f"SOL beta {sol_beta} off"


def test_portfolio_beta_equal_weighted_average():
    from portfolio.portfolio_builder.beta_estimator import (
        compute_coin_betas, compute_portfolio_beta
    )
    kline = _make_kline()
    betas = compute_coin_betas(kline, window=60)
    date = pd.Timestamp(kline["date"].max())
    holdings = {"ETHUSDT": 0.5, "SOLUSDT": 0.5}
    port_beta = compute_portfolio_beta(betas, holdings, date, beta_prior=1.3)
    # Expected ~ (1.5 + 2.0)/2 = 1.75
    assert 1.5 < port_beta < 2.0


def test_portfolio_beta_uses_prior_when_no_history():
    from portfolio.portfolio_builder.beta_estimator import compute_portfolio_beta
    betas_empty = pd.DataFrame(columns=["date", "instrument", "beta"])
    date = pd.Timestamp("2023-01-01")
    holdings = {"NEWCOIN": 1.0}
    port_beta = compute_portfolio_beta(betas_empty, holdings, date, beta_prior=1.3)
    assert port_beta == 1.3


def test_beta_no_future_leak():
    from portfolio.portfolio_builder.beta_estimator import compute_coin_betas
    kline = _make_kline(days=200)
    cutoff = kline["date"].unique()[150]
    b_full = compute_coin_betas(kline, window=60)
    b_cut = compute_coin_betas(kline[kline["date"] <= cutoff], window=60)
    m = b_full.merge(b_cut, on=["date", "instrument"], suffixes=("_full", "_cut"))
    m = m[m["date"] <= cutoff]
    diff = (m["beta_full"] - m["beta_cut"]).abs().max()
    assert diff < 1e-9, f"Beta future leak: {diff}"
