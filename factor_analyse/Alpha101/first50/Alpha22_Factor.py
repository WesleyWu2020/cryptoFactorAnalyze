"""Alpha101 Alpha#22 因子（factor_common 契约版）。

原始定义:
    Alpha#22 = -1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准，与经典公式有偏离）:
    原脚本三个窗口全部使用 10（而非经典公式的 5/5/20）:
    corr       = correlation(high, volume, 10)
    delta_corr = delta(corr, 10)
    std        = ts_std(close, 10)
    factor     = -1 * (delta_corr * cross_sectional_rank(std))

参数取原脚本 __main__ 实际调用值: 窗口固定为 10（写入 params）。
rebalance_period 只用于旧版 future_ret，不属于因子值计算，丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
公式内部的横截面 rank 保留；最终去极值与归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha22_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #22: -1 * delta(corr(high, volume, 10), 10) * rank(ts_std(close, 10))",
}

SETTING = {
    "data_needed": ["high", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 25,
    "preprocessing": "mad_rank",
    "params": {"corr_window": 10, "delta_window": 10, "std_window": 10},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#22 matrix (date x instrument)."""

    corr_window = SETTING["params"]["corr_window"]
    delta_window = SETTING["params"]["delta_window"]
    std_window = SETTING["params"]["std_window"]

    high = data_ctx["high"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 价量相关性及其变化
    corr_high_vol = high.rolling(window=corr_window, min_periods=corr_window).corr(volume)
    delta_corr = corr_high_vol - corr_high_vol.shift(delta_window)

    # 收盘价波动性，公式内部横截面 rank 保留
    std_close = close.rolling(window=std_window, min_periods=std_window).std()
    std_rank = std_close.rank(axis=1, pct=True)

    return -1.0 * (delta_corr * std_rank)
