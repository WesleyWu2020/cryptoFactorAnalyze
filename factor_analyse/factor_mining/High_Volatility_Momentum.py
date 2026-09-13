"""高波动性动量因子（High Volatility Momentum）。

公式（ratio 模式，与旧版默认一致）：

    roc       = close.pct_change(window)                    # N 日动量
    tr        = max(high - low, |high - close_prev|, |low - close_prev|)
    atr       = rolling_mean(tr, atr_window)
    range_pct = (high - low) / close
    factor    = roc / range_pct          （ratio 模式）
    factor    = roc * atr                （product 模式）

ratio 模式刻画单位日内振幅所承载的动量强度；product 模式为 ATR 加权的动量。
计算只使用当日及历史数据（shift(1)、pct_change、trailing rolling），无未来函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "High_Volatility_Momentum",
    "author": "local",
    "level": "daily",
    "category": "momentum",
    "description": "Momentum scaled by volatility: roc / range_pct (ratio) or roc * ATR (product)",
}

SETTING = {
    "data_needed": ["high", "low", "close"],
    "universe": "historical_top50",
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {
        "window": 20,
        "atr_window": 14,
        "momentum_type": "ratio",
        "rebalance_period": 10,
    },
    # 动量类因子：值越大动量越强（ratio 为单位振幅动量），方向取 +1。
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始高波动性动量矩阵（date × instrument）。"""

    window = SETTING["params"]["window"]
    atr_window = SETTING["params"]["atr_window"]
    momentum_type = SETTING["params"]["momentum_type"]

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")

    roc = close.pct_change(periods=window)

    close_prev = close.shift(1)
    tr = pd.concat(
        [high - low, (high - close_prev).abs(), (low - close_prev).abs()],
        axis=0,
    ).groupby(level=0).max()

    if momentum_type == "product":
        atr = tr.rolling(window=atr_window).mean()
        return roc * atr

    range_pct = (high - low) / close
    return roc / range_pct.replace(0, np.nan)
