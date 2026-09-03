import pandas as pd
import numpy as np


def _make_btc_kline(days=400, trend="up"):
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    if trend == "up":
        close = 20000 * np.exp(np.linspace(0, 1.5, days))
    elif trend == "down":
        close = 60000 * np.exp(np.linspace(0, -1.0, days))
    else:  # flat
        close = 30000 + 1000*np.sin(np.arange(days)/10)
    return pd.DataFrame({"symbol": "BTCUSDT", "date": dates, "close": close})


def test_trend_strong_up_gives_positive_T():
    from portfolio.regime.btc_trend import compute_btc_trend_score
    kline = _make_btc_kline(trend="up")
    T = compute_btc_trend_score(kline)
    # 长期上升，最近期 T 应接近 +1
    last_T = T.iloc[-1]
    assert last_T > 0.7, f"Expected strong positive T, got {last_T}"


def test_trend_strong_down_gives_negative_T():
    from portfolio.regime.btc_trend import compute_btc_trend_score
    kline = _make_btc_kline(trend="down")
    T = compute_btc_trend_score(kline)
    last_T = T.iloc[-1]
    assert last_T < -0.5, f"Expected negative T, got {last_T}"


def test_trend_output_bounded_in_minus_one_plus_one():
    from portfolio.regime.btc_trend import compute_btc_trend_score
    for trend in ["up", "down", "flat"]:
        kline = _make_btc_kline(trend=trend)
        T = compute_btc_trend_score(kline)
        assert (T.dropna() >= -1.0).all()
        assert (T.dropna() <= 1.0).all()


def test_trend_no_future_leak():
    """对比全量计算与截断计算，cutoff 前的值必须一致。"""
    from portfolio.regime.btc_trend import compute_btc_trend_score
    kline = _make_btc_kline(days=500, trend="up")
    cutoff = kline["date"].iloc[300]
    T_full = compute_btc_trend_score(kline)
    T_cut = compute_btc_trend_score(kline[kline["date"] <= cutoff])
    aligned = T_full.reindex(T_cut.index)
    diff = (aligned - T_cut).abs().max()
    assert diff < 1e-9, f"Future leak detected: max diff {diff}"


def test_trend_requires_btc_symbol():
    from portfolio.regime.btc_trend import compute_btc_trend_score
    kline = pd.DataFrame({"symbol": ["ETHUSDT"]*10,
                           "date": pd.date_range("2023-01-01", periods=10),
                           "close": range(10)})
    # 无 BTCUSDT 应抛错或返回空
    import pytest
    with pytest.raises((ValueError, KeyError)):
        compute_btc_trend_score(kline)
