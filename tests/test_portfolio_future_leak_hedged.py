import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _generate_synthetic_bundle(tmp_path, days=500):
    """与 test_portfolio_pipeline_hedged 相同的合成数据，但时间更长。"""
    np.random.seed(77)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    symbols = [f"C{i}USDT" for i in range(8)] + ["BTCUSDT"]

    rows = []
    for s in symbols:
        prices = [100.0]
        for _ in range(len(dates) - 1):
            prices.append(prices[-1] * (1 + np.random.randn() * 0.02))
        for d, p in zip(dates, prices):
            rows.append({"symbol": s, "date": d, "close": p})
    kline = pd.DataFrame(rows)
    kline_path = tmp_path / "kline.csv"
    kline.to_csv(kline_path, index=False)

    f_rows = []
    for d in dates:
        for s in symbols:
            if s == "BTCUSDT":
                continue
            f_rows.append({"date": d.strftime("%Y-%m-%d"), "instrument": s,
                           "factor": np.random.randn()})
    f_path = tmp_path / "f1.csv"
    pd.DataFrame(f_rows).to_csv(f_path, index=False)

    uni_rows = []
    trading_pairs = ",".join([s for s in symbols if s != "BTCUSDT"])
    for d in dates:
        uni_rows.append({
            "Decision_Window_Start": d.strftime("%Y-%m-%d"),
            "Decision_Window_End": d.strftime("%Y-%m-%d"),
            "Trading_Pairs": trading_pairs,
        })
    uni_path = tmp_path / "universe.csv"
    pd.DataFrame(uni_rows).to_csv(uni_path, index=False)
    return str(kline_path), {"f1": str(f_path)}, str(uni_path)


def test_hedged_pipeline_no_future_leak(tmp_path):
    """对比全量回测 vs 截断到 cutoff 的回测；cutoff 前的收益必须一致。"""
    from portfolio.pipeline import run_hedged_pipeline
    from portfolio.config import PortfolioConfig

    kline_path, factor_paths, uni_path = _generate_synthetic_bundle(tmp_path)

    cfg = PortfolioConfig(
        top_n=3, rebalance_period=5, is_end_date="2022-08-01",
        min_ic_ir=0.0, ema_short=10, ema_long=40,
        zscore_window=30, t_smoothing_span=3, beta_window=20,
    )

    # Full run
    ret_full = run_hedged_pipeline(factor_paths, kline_path, uni_path, config=cfg)

    # Truncate at cutoff
    cutoff = pd.Timestamp("2023-01-15")
    kline = pd.read_csv(kline_path)
    kline["date"] = pd.to_datetime(kline["date"])
    kline_cut = kline[kline["date"] <= cutoff]
    kline_cut_path = tmp_path / "kline_cut.csv"
    kline_cut.to_csv(kline_cut_path, index=False)

    # Truncate factor too
    f_df = pd.read_csv(factor_paths["f1"])
    f_df["date"] = pd.to_datetime(f_df["date"])
    f_cut = f_df[f_df["date"] <= cutoff]
    f_cut_path = tmp_path / "f1_cut.csv"
    f_cut.to_csv(f_cut_path, index=False)

    ret_cut = run_hedged_pipeline({"f1": str(f_cut_path)}, str(kline_cut_path), uni_path, config=cfg)

    # Compare dates <= cutoff
    common = ret_full.index.intersection(ret_cut.index)
    common = common[common <= cutoff]
    if len(common) == 0:
        pytest.skip("No common dates before cutoff")
    diff = (ret_full.reindex(common) - ret_cut.reindex(common)).abs().max()
    assert diff < 1e-6, f"Future leak detected: max_abs_diff = {diff}"
