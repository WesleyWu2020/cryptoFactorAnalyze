"""方向动量因子（Directional Momentum）。

公式（与旧版 CSV 管线 compute_one 逐点等价）：

    daily_return  = close.pct_change()
    up_momentum   = rolling_sum(max(0, daily_return), window)
    down_momentum = rolling_sum(max(0, -daily_return), window)
    factor        = up_momentum / (down_momentum + 1e-10)

因子衡量过去 window 日内上涨动量相对下跌动量的强度，值越大表示上涨动能越占优。
计算只使用当日及历史数据（pct_change 向后看 1 日，rolling 向后看 window 日），
无未来函数。框架入口为 ``calc_factor(data_ctx)``，截面 MAD 去极值与按日 rank
归一化由框架 preprocessing="mad_rank" 完成，这里返回原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Directional_Momentum_Factor",
    "author": "local",
    "level": "daily",
    "category": "momentum",
    "description": "上涨动量之和 / 下跌动量之和的方向动量比值",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 21,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "rebalance_period": 10, "top_n": 50},
    "factor_direction": 1,
}

_EPSILON = 1e-10


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回方向动量原始因子矩阵（date × instrument）。

    ``rolling`` 沿时间轴（axis 0）计算，每个因子值仅依赖当日及之前
    ``window`` 个日收益，不使用任何未来数据。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be an integer >= 1")

    close = data_ctx["close"].astype("float64")
    daily_return = close.pct_change()

    up_momentum = daily_return.clip(lower=0).rolling(window).sum()
    down_momentum = daily_return.clip(upper=0).abs().rolling(window).sum()

    return up_momentum / (down_momentum + _EPSILON)
