"""散户追涨 FOMO 情绪指数因子 (Retail FOMO Ratio)。

新框架入口是 ``calc_factor(data_ctx)``：

    up_day_trades = trade_count if close > close.shift(1) else 0
    factor = rolling_sum(up_day_trades, window) / (rolling_sum(trade_count, window) + eps)

即最近 window 天内"上涨日成交笔数"占全部成交笔数的比例，刻画散户追涨情绪
的拥挤程度。计算只使用当日及历史数据，无未来函数。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Retail_FOMO_Ratio_Factor",
    "author": "local",
    "level": "daily",
    "category": "volume",
    "description": "Share of trade count occurring on up days over a trailing window",
}

SETTING = {
    "data_needed": ["close", "trade_count"],
    "universe": "historical_top50",
    "warmup_bars": 21,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "eps": 1e-5},
    "factor_direction": -1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily retail FOMO ratio matrix.

    ``data_ctx`` contains date-by-instrument matrices. ``rolling``/``shift``
    operate along the daily index, so each value only depends on the current
    and preceding observations.
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be an integer >= 1")
    eps = float(SETTING["params"]["eps"])

    close = data_ctx["close"].astype("float64")
    trade_count = data_ctx["trade_count"].astype("float64")

    up_day = close > close.shift(1)
    up_day_trades = trade_count.where(up_day, 0.0)
    up_day_trades_sum = up_day_trades.rolling(window=window, min_periods=window).sum()
    trades_sum = trade_count.rolling(window=window, min_periods=window).sum()
    return up_day_trades_sum / (trades_sum + eps)
