"""波动效率因子（Volatility Efficiency）。

公式（与旧版 CSV 管线脚本逐点等价）：

    roc       = close / close.shift(window) - 1
    range_pct = rolling_mean( (high - low) / (close.shift(1) + eps), window )
    factor    = roc / (range_pct + eps)

其中 ``rolling`` 未指定 ``min_periods``，沿用 pandas 默认（等于 window），
每个因子值只依赖当日及历史数据，无未来函数。新框架入口为
``calc_factor(data_ctx)``，去极值与按日 rank 由框架 preprocessing 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Volatility_Efficiency",
    "author": "local",
    "level": "daily",
    "category": "volatility",
    "description": "N-day return on close scaled by average daily range (volatility efficiency)",
}

SETTING = {
    "data_needed": ["high", "low", "close"],
    "universe": "historical_top50",
    "warmup_bars": 21,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "rebalance_period": 10},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回波动效率原始因子矩阵（date × instrument）。

    roc 需要 ``window`` 期历史，range_pct 在 close.shift(1) 上再滚
    ``window`` 期，因此首个有效值出现在第 ``window + 1`` 根 bar。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be an integer >= 1")

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")

    roc = close / close.shift(window) - 1
    range_pct = ((high - low) / (close.shift(1) + _EPSILON)).rolling(window=window).mean()
    return roc / (range_pct + _EPSILON)
