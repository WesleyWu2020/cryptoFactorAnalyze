"""Alpha101 Alpha#23 因子（factor_common 契约版）。

原始定义:
    Alpha#23 = if (high > 20日最高价均值): -1 * delta(high, 2)
               else: 0

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    high_ma   = ts_mean(high, 20)
    delta_high = delta(high, 2)
    factor    = (high > high_ma) ? -1 * delta_high : 0

参数取原脚本 __main__ 实际调用值: ma_window=20, delta_window=2（函数内固定值，写入 params）。
rebalance_period 只用于旧版 future_ret，不属于因子值计算，丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha23_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #23: (high > ts_mean(high, 20)) ? -1 * delta(high, 2) : 0",
}

SETTING = {
    "data_needed": ["high"],
    "universe": "historical_top50",
    "warmup_bars": 25,
    "preprocessing": "mad_rank",
    "params": {"ma_window": 20, "delta_window": 2},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#23 matrix (date x instrument)."""

    ma_window = SETTING["params"]["ma_window"]
    delta_window = SETTING["params"]["delta_window"]

    high = data_ctx["high"].astype("float64")

    high_ma = high.rolling(window=ma_window, min_periods=ma_window).mean()
    delta_high = high - high.shift(delta_window)

    # 只在突破 20 日最高价均值时给出反转信号，否则为 0
    factor = (-1.0 * delta_high).where(high > high_ma, 0.0)

    # 数据未就绪时输出 NaN（与旧脚本行级 isna 检查等价）
    valid = high_ma.notna() & delta_high.notna()
    return factor.where(valid)
