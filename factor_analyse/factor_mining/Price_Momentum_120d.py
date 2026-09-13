"""价格动量因子 120 日 (Price Momentum 120D)。

公式：factor = log(close_t / close_{t-window})，window 默认为 120。
捕捉加密市场约 4 个月的长期趋势动量，是最经典的牛熊择时信号之一。

计算只使用当日及历史数据（shift(window) 向后取历史价格），无未来函数。
FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Price_Momentum_120d",
    "author": "local",
    "level": "daily",
    "category": "momentum",
    "description": "120-day log price momentum",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 120,
    "preprocessing": "mad_rank",
    "params": {"window": 120},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始 N 日对数动量矩阵。

    ``data_ctx`` 是 date（升序）× instrument 的矩阵字典；
    ``shift`` 沿时间轴取历史价格，每个因子值只依赖当日及过去 ``window`` 天的数据。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be an integer >= 1")

    close = data_ctx["close"].astype("float64")
    return np.log(close / close.shift(window))
