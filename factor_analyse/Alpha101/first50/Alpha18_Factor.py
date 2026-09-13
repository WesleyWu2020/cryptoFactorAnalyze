"""Alpha101 Alpha#18 因子（factor_common 契约版）。

原始定义:
    Alpha#18 = -1 * rank(((stddev(abs((close - open)), 5) + (close - open))
                          + correlation(close, open, 10)))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    std_abs_co = ts_std(abs(close - open), std_window)
    co_diff    = close - open
    corr_co    = correlation(close, open, corr_window)
    factor     = -1 * cross_sectional_rank((std_abs_co + co_diff) + corr_co)

参数取原脚本 __main__ 实际调用值: std_window=5, corr_window=5。
原脚本的 normalization_method 仅生成未被因子使用的 normalized_* 列，丢弃；
rebalance_period 只用于旧版 future_ret，不属于因子值计算，丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
公式内部的横截面 rank 保留；最终去极值与归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha18_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #18: -1 * rank(ts_std(abs(close-open), 5) + (close-open) + corr(close, open, 5))",
}

SETTING = {
    "data_needed": ["open", "close"],
    "universe": "historical_top50",
    "warmup_bars": 15,
    "preprocessing": "mad_rank",
    "params": {"std_window": 5, "corr_window": 5},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#18 matrix (date x instrument)."""

    std_window = SETTING["params"]["std_window"]
    corr_window = SETTING["params"]["corr_window"]

    open_ = data_ctx["open"].astype("float64")
    close = data_ctx["close"].astype("float64")

    # 日内波动性与日内收益
    abs_co = (close - open_).abs()
    std_abs_co = abs_co.rolling(window=std_window, min_periods=std_window).std()
    co_diff = close - open_

    # 收盘-开盘时序相关性
    corr_co = close.rolling(window=corr_window, min_periods=corr_window).corr(open_)

    # 复合信号 -> 横截面 rank -> 取反（公式内部 rank 保留）
    raw = (std_abs_co + co_diff) + corr_co
    return -1.0 * raw.rank(axis=1, pct=True)
