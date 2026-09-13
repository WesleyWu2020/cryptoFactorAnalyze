"""Alpha101 Alpha#33 因子（factor_common 契约版）。

原始定义:
    Alpha#33 = rank((-1 * ((1 - (open / close))^1))) == rank((open - close) / close)

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    factor = (open - close) / close
日内反转：日内下跌越多因子值越大，预期短期反弹。旧脚本的 rebalance_period
仅用于旧版 future_ret，与因子值无关，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面 rank、去极值与秩归一化由框架 preprocessing="mad_rank" 完成，
这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha33_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #33: (open - close) / close 日内反转",
}

SETTING = {
    "data_needed": ["open", "close"],
    "universe": "historical_top50",
    "warmup_bars": 5,
    "preprocessing": "mad_rank",
    "params": {},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#33 matrix (date x instrument)."""

    open_ = data_ctx["open"].astype("float64")
    close = data_ctx["close"].astype("float64")

    # (open - close) / close 等价于 -(1 - open/close)
    return (open_ - close) / close
