"""Alpha101 Alpha#4 因子（factor_common 契约版，币圈优化版）。

原始定义:
    Alpha#4 = -1 * Ts_Rank(rank(low), 9)

本实现保留原脚本“币圈优化版”的实际计算逻辑（``__main__`` 实际调用
ts_rank_window=9, price_normalization='relative_price'）:
    relative_low = low / mean(low, 20)            # 相对自身 20 日均值的归一化最低价
    factor = -1 * ts_rank(rank(relative_low), ts_rank_window)
其中 rank 为按日横截面百分比秩、ts_rank 为当前值在过去 ts_rank_window 日窗口内
的百分比秩（公式内部项，保留）。币圈价格量级差异巨大，故用相对价格替代原始 low；
'market_cap' / 'price_change' / 'log' 三个未启用分支未迁移。
ts_rank 采用 pandas rank(pct=True) 约定（ ties 取平均、分母为窗口长度），与原脚本
严格小于计数 / (window-1) 的写法在 ties 处理上略有差异，排序语义一致。

rebalance_period 仅用于旧版 future_ret，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha4_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #4: -1 * ts_rank(rank(low/mean(low,20)), 9)",
}

SETTING = {
    "data_needed": ["low"],
    "universe": "historical_top50",
    "warmup_bars": 35,
    "preprocessing": "mad_rank",
    "params": {"ts_rank_window": 9, "price_base_window": 20},
    "factor_direction": 1,
}


def _ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series pct rank of the latest value within the window, causal."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: arr.rank(pct=True).iloc[-1], raw=False
    )


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#4 matrix (date x instrument)."""

    ts_rank_window = SETTING["params"]["ts_rank_window"]
    price_base_window = SETTING["params"]["price_base_window"]
    for name, value in (
        ("ts_rank_window", ts_rank_window),
        ("price_base_window", price_base_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    low = data_ctx["low"].astype("float64")

    # 步骤1: 相对价格归一化（与原脚本一致 min_periods=1）
    price_mean = low.rolling(window=price_base_window, min_periods=1).mean()
    relative_low = low / price_mean

    # 步骤2: 按日横截面百分比秩（公式内部 rank 项）
    low_rank = relative_low.rank(axis=1, pct=True)

    # 步骤3: 过去 ts_rank_window 日的时序排名，方向反转
    return -1.0 * _ts_rank(low_rank, ts_rank_window)
