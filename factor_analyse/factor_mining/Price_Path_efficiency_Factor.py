"""价格路径效率因子（Price Path Efficiency，方向性动量）。

公式：

    net_ret  = close / close.shift(window) - 1
    path_len = sum_{t-window+1..t} |pct_change(close)|
    factor   = net_ret / (path_len + 1e-8)

含义：同期净涨跌相对“走过的路径”越高，路径效率越高，噪声少、趋势更干净。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Price_Path_efficiency_Factor",
    "author": "local",
    "level": "daily",
    "category": "momentum",
    "description": "N-day net return over total price path length (trend efficiency)",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 11,
    "preprocessing": "mad_rank",
    "params": {"window": 10},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily price-path-efficiency matrix.

    ``rolling`` / ``shift`` operate along the daily index, so each value only
    depends on the current and preceding ``window`` observations. The rolling
    sum uses the default ``min_periods=window`` semantics of the legacy script.
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be an integer >= 1")

    close = data_ctx["close"].astype("float64")
    net_ret = close / close.shift(window) - 1
    path_len = close.pct_change().abs().rolling(window).sum()
    return net_ret / (path_len + _EPSILON)
