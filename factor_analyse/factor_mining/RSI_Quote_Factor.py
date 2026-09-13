"""RSI 截面因子（RSI Quote Factor）。

公式（与旧版 compute_one 逐点等价）：

    price_change = close.diff()
    gain = max(price_change, 0)，loss = max(-price_change, 0)
    avg_gain = rolling_mean(gain, window)，avg_loss = rolling_mean(loss, window)
    rs  = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)，并 clip 到 [0, 100]

rolling 沿时间轴（min_periods=window），只依赖当日及历史数据，无未来函数。
截面去极值与 rank 归一化由框架 preprocessing="mad_rank" 完成，这里返回原始 RSI 值。
"""

from __future__ import annotations

import pandas as pd

TYPE = "regular"

META = {
    "factor_name": "RSI_Quote_Factor",
    "author": "local",
    "level": "daily",
    "category": "momentum",
    "description": "N-day RSI (rolling mean of gains/losses), contrarian overbought/oversold signal",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 15,
    "preprocessing": "mad_rank",
    "params": {"window": 14},
    "factor_direction": -1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily RSI matrix (0-100).

    ``data_ctx["close"]`` is a date-by-instrument matrix; ``diff`` and
    ``rolling`` operate along the daily index, so each value only depends on
    the current and preceding ``window`` observations.
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.window must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    price_change = close.diff()
    gain = price_change.clip(lower=0.0)
    loss = (-price_change).clip(lower=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + _EPSILON)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.clip(0.0, 100.0)
