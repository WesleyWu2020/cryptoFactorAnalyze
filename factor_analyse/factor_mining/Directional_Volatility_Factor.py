"""定向波动率因子（Directional Volatility Factor）。

公式（与旧版 compute_one 逐点等价，window 来自 SETTING["params"]）：

    up_volatility   = (high - open) / open          （0 替换为 eps=1e-10）
    down_volatility = (open - low) / open           （0 替换为 eps=1e-10）
    up_roc   = (high - open.shift(window)) / open.shift(window)
    down_roc = (open.shift(window) - low) / open.shift(window)
    factor   = up_roc / MA(up_volatility, window)
               - down_roc / MA(down_volatility, window)

含义：上行 ROC 相对上行波动率的效率，减去下行 ROC 相对下行波动率的效率。
rolling 默认 min_periods=window，与旧版一致；只使用当日及历史数据，无未来函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TYPE = "regular"

META = {
    "factor_name": "Directional_Volatility_Factor",
    "author": "local",
    "level": "daily",
    "category": "volatility",
    "description": "定向波动率效率因子：上行ROC效率减去下行ROC效率",
}

SETTING = {
    "data_needed": ["open", "high", "low"],
    "universe": "historical_top50",
    # shift(window) 与 rolling(window).mean() 叠加，保守取 2 * window
    "warmup_bars": 40,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "rebalance_period": 10},
    "factor_direction": 1,
}

_EPSILON = 1e-10


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw directional-volatility factor matrix.

    ``data_ctx`` contains date-by-instrument matrices. ``shift`` and
    ``rolling`` operate along the daily index, so each value only depends
    on the current and historical observations.
    """

    window = SETTING["params"]["window"]

    open_ = data_ctx["open"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")

    up_volatility = ((high - open_) / open_).replace(0, _EPSILON)
    down_volatility = ((open_ - low) / open_).replace(0, _EPSILON)

    open_lag = open_.shift(window)
    up_roc = (high - open_lag) / open_lag
    down_roc = (open_lag - low) / open_lag

    up_eff = up_roc / up_volatility.rolling(window).mean()
    down_eff = down_roc / down_volatility.rolling(window).mean()

    factor = up_eff - down_eff
    return factor.replace([np.inf, -np.inf], np.nan)
