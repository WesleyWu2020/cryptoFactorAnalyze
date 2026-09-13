"""均线多头排列强度因子 (MA Cross Trend)。

公式：factor = MA_fast / MA_slow - 1

- > 0: 快线在慢线上方（多头排列），值越大趋势越强
- < 0: 空头排列，值越小下跌趋势越强

经典趋势跟踪信号：20MA/60MA 是最常用的趋势判断工具。
计算只使用当日及历史数据（rolling 均向历史方向滑动），无未来函数。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "MA_Cross_Trend",
    "author": "local",
    "level": "daily",
    "category": "momentum",
    "description": "Moving-average cross trend strength: MA_fast / MA_slow - 1",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 60,
    "preprocessing": "mad_rank",
    "params": {"fast": 20, "slow": 60},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始均线多头排列强度矩阵（date x instrument）。"""

    fast = SETTING["params"]["fast"]
    slow = SETTING["params"]["slow"]

    close = data_ctx["close"].astype("float64")
    # 与旧版 compute_one 逐点等价：rolling 未指定 min_periods，默认等于 window
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    return ma_fast / ma_slow - 1
