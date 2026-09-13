"""价格动量因子 60日（Price Momentum 60D）。

公式：factor = log(close_t / close_{t-window})，window 默认 60。
捕捉中期趋势延续效应（加密市场 2 个月动量显著）。

无未来函数：shift(window) 为向历史方向平移，因子值仅依赖当日及历史收盘价。
新框架入口是 ``calc_factor(data_ctx)``，截面 MAD 去极值与按日 rank 由框架
preprocessing("mad_rank") 统一处理，此处返回原始因子值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Price_Momentum_60d",
    "author": "local",
    "level": "daily",
    "category": "momentum",
    "description": "60-day log price momentum",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 60,
    "preprocessing": "mad_rank",
    "params": {"window": 60, "rebalance_period": 10, "top_n": 50},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始 60 日对数价格动量矩阵（date × instrument）。"""

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be an integer >= 1")

    close = data_ctx["close"].astype("float64")
    return np.log(close / close.shift(window))
