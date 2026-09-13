"""Alpha101 Alpha#53 因子（factor_common 契约版）。

原始定义:
    (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    1. 价格归一化: price_base = mean(close, 20)（min_periods=1），
       norm_close/norm_high/norm_low = 原值 / price_base
    2. 日内位置: intraday_position =
       ((norm_close - norm_low) - (norm_high - norm_close)) / ((norm_close - norm_low) + 1e-8)
    3. 因子值: -1 * delta(intraday_position, 9)

参数取原脚本 __main__ 实际调用值（delta_window=9）。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口；
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha53_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #53: -1 * delta(((close-low)-(high-close))/(close-low), 9)",
}

SETTING = {
    "data_needed": ["close", "high", "low"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"delta_window": 9, "norm_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#53 matrix (date x instrument)."""

    delta_window = SETTING["params"]["delta_window"]
    norm_window = SETTING["params"]["norm_window"]

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")

    # 1. 价格归一化（与原脚本一致：min_periods=1 的 20 日均价基准）
    price_base = close.rolling(window=norm_window, min_periods=1).mean()
    norm_close = close / (price_base + _EPSILON)
    norm_high = high / (price_base + _EPSILON)
    norm_low = low / (price_base + _EPSILON)

    # 2. 日内价格位置指标
    close_low_diff = norm_close - norm_low
    high_close_diff = norm_high - norm_close
    intraday_position = (close_low_diff - high_close_diff) / (close_low_diff + _EPSILON)

    # 3. -1 * 位置变化（delta_window 日差分）
    return -1.0 * intraday_position.diff(delta_window)
