"""N 天动量因子 (N-Day Momentum)。

核心公式（与旧版 compute_one 逐点等价，window=1、不平滑、不做波动率调整）：

    momentum = log(close / close.shift(window))

即 window 天对数收益率。旧版研究结论：1 天滞后收益率的预测能力最强，
故默认 window=1。计算只使用当日及历史数据（shift 为正向滞后），无未来函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "N_Day_Momentum",
    "author": "local",
    "level": "daily",
    "category": "momentum",
    "description": "N-day log-return momentum (default 1-day lagged log return)",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 2,
    "preprocessing": "mad_rank",
    "params": {"window": 1},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw N-day log-momentum matrix.

    ``close.shift(window)`` looks strictly backward along the daily index,
    so each value only depends on data at ``t - window`` and earlier.
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be an integer >= 1")

    close = data_ctx["close"].astype("float64")
    return np.log(close / close.shift(window))
