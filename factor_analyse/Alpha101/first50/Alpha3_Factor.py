"""Alpha101 Alpha#3 因子（factor_common 契约版，币圈优化版）。

原始定义:
    Alpha#3 = -1 * correlation(rank(open), rank(volume), 10)

本实现保留原脚本“币圈优化版”的实际计算逻辑（``__main__`` 实际调用
price_normalization='relative_price'，correlation_window=20）:
    relative_price = open / mean(open, 20)        # 相对自身 20 日均值的归一化价格
    factor = -1 * correlation(rank(relative_price), rank(volume), correlation_window)
其中 rank 为按日横截面百分比秩（公式内部项，保留）。币圈价格量级差异巨大，
直接 rank(open) 近似常数列，故原脚本用相对价格替代；'market_cap' / 'price_change'
两个未启用分支未迁移。

原脚本的“历史市值可用性池”过滤由框架 universe="historical_top50" 承担；
rebalance_period 仅用于旧版 future_ret，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha3_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #3: -1 * correlation(rank(open/mean(open,20)), rank(volume), 20)",
}

SETTING = {
    "data_needed": ["open", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 45,
    "preprocessing": "mad_rank",
    "params": {"correlation_window": 20, "price_base_window": 20},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#3 matrix (date x instrument)."""

    correlation_window = SETTING["params"]["correlation_window"]
    price_base_window = SETTING["params"]["price_base_window"]
    for name, value in (
        ("correlation_window", correlation_window),
        ("price_base_window", price_base_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    open_ = data_ctx["open"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 步骤1: 相对价格归一化（相对自身历史均值；与原脚本一致 min_periods=1，
    # 上市初期即产出信号，由框架 warmup 控制有效起点）
    price_mean = open_.rolling(window=price_base_window, min_periods=1).mean()
    relative_price = open_ / price_mean

    # 步骤2: 按日横截面百分比秩（公式内部 rank 项）
    price_rank = relative_price.rank(axis=1, pct=True)
    volume_rank = volume.rank(axis=1, pct=True)

    # 步骤3: 过去 correlation_window 日两秩序列的时序相关性，方向反转
    correlation = price_rank.rolling(
        window=correlation_window, min_periods=correlation_window
    ).corr(volume_rank)
    return -1.0 * correlation
