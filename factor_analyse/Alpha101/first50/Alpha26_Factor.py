"""Alpha101 Alpha#26 因子（factor_common 契约版）。

原始定义:
    Alpha#26 = -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    vol_rank  = ts_rank(volume, ts_rank_window)   # 窗口内 pct 秩（rank().iloc[-1] / window）
    high_rank = ts_rank(high, ts_rank_window)
    corr      = correlation(vol_rank, high_rank, ts_rank_window)
    factor    = -1 * ts_max(corr, ts_max_window)
参数取原脚本 __main__ 实际调用值 ts_rank_window=10, ts_max_window=5
（rebalance_period 仅用于旧版 future_ret，丢弃）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha26_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #26: -ts_max(correlation(ts_rank(volume,10), ts_rank(high,10), 10), 5)",
}

SETTING = {
    "data_needed": ["volume", "high"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"ts_rank_window": 10, "ts_max_window": 5},
    "factor_direction": 1,
}


def _ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series pct rank of the latest value within the window, causal."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: pd.Series(arr).rank(pct=True).iloc[-1], raw=False
    )


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#26 matrix (date x instrument)."""

    ts_rank_window = SETTING["params"]["ts_rank_window"]
    ts_max_window = SETTING["params"]["ts_max_window"]
    for name, value in (("ts_rank_window", ts_rank_window), ("ts_max_window", ts_max_window)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    volume = data_ctx["volume"].astype("float64")
    high = data_ctx["high"].astype("float64")

    # 1-2. 量与高价各自的时间序列 pct 秩
    vol_rank = _ts_rank(volume, ts_rank_window)
    high_rank = _ts_rank(high, ts_rank_window)

    # 3. 两者在 ts_rank_window 窗口内的滚动相关
    corr = vol_rank.rolling(window=ts_rank_window, min_periods=ts_rank_window).corr(high_rank)

    # 4-5. 取 ts_max_window 日内相关性最大值并加负号（相关性越高越看跌）
    return -1.0 * corr.rolling(window=ts_max_window, min_periods=ts_max_window).max()
