import numpy as np
import pandas as pd
import pytest
import tempfile
import pathlib


@pytest.fixture
def synthetic_data(tmp_path):
    """生成 300 天、10 个币、2 个因子的合成数据。"""
    np.random.seed(123)
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    symbols = [f"C{i}USDT" for i in range(10)] + ["BTCUSDT"]

    # Kline
    rows = []
    for s in symbols:
        prices = [100.0]
        for _ in range(len(dates) - 1):
            prices.append(prices[-1] * (1 + np.random.randn()*0.02))
        for d, p in zip(dates, prices):
            rows.append({"symbol": s, "date": d, "close": p})
    kline = pd.DataFrame(rows)
    kline_path = tmp_path / "kline.csv"
    kline.to_csv(kline_path, index=False)

    # Factors
    f1 = []
    for d in dates:
        for s in symbols:
            if s == "BTCUSDT":
                continue
            f1.append({"date": d.strftime("%Y-%m-%d"), "instrument": s,
                       "factor": np.random.randn()})
    f1_df = pd.DataFrame(f1)
    f1_path = tmp_path / "f1.csv"
    f1_df.to_csv(f1_path, index=False)

    # Universe CSV
    uni_rows = []
    for d in dates:
        for s in symbols:
            if s == "BTCUSDT":
                continue
            uni_rows.append({
                "Decision_Window_Start": d.strftime("%Y-%m-%d"),
                "Decision_Window_End": d.strftime("%Y-%m-%d"),
                "CoinGecko_ID": s.lower(),
                "Binance_Symbol": s,
                "Trading_Pairs": s,
            })
    uni_path = tmp_path / "universe.csv"
    pd.DataFrame(uni_rows).to_csv(uni_path, index=False)

    return {
        "kline": str(kline_path),
        "factor_paths": {"f1": str(f1_path)},
        "universe": str(uni_path),
    }


def test_hedged_pipeline_runs_end_to_end(synthetic_data):
    from portfolio.pipeline import run_hedged_pipeline
    from portfolio.config import PortfolioConfig

    cfg = PortfolioConfig(
        top_n=3,
        rebalance_period=5,
        is_end_date="2023-06-01",
        min_ic_ir=0.0,  # 接受所有因子
        ema_short=10,   # 缩短让 warmup 合理
        ema_long=40,
        zscore_window=30,
        t_smoothing_span=3,
        beta_window=20,
    )
    ret = run_hedged_pipeline(
        factor_paths=synthetic_data["factor_paths"],
        kline_csv=synthetic_data["kline"],
        universe_csv=synthetic_data["universe"],
        config=cfg,
    )
    assert isinstance(ret, pd.Series)
    assert not ret.empty
    assert not ret.isna().all()


def test_hedged_pipeline_produces_report(synthetic_data, tmp_path):
    from portfolio.pipeline import run_hedged_pipeline
    from portfolio.config import PortfolioConfig

    out = tmp_path / "report.html"
    cfg = PortfolioConfig(
        top_n=3, rebalance_period=5, is_end_date="2023-06-01",
        min_ic_ir=0.0, ema_short=10, ema_long=40,
        zscore_window=30, t_smoothing_span=3, beta_window=20,
    )
    run_hedged_pipeline(
        factor_paths=synthetic_data["factor_paths"],
        kline_csv=synthetic_data["kline"],
        universe_csv=synthetic_data["universe"],
        config=cfg,
        output_path=str(out),
    )
    assert out.exists()
    assert out.stat().st_size > 1000
