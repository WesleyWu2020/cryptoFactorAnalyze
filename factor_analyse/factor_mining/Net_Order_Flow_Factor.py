"""净订单流因子（Net Order Flow）。

公式（与旧版 compute_one 逐点等价）：

    daily_flow = taker_buy_quote_volume - taker_sell_quote_volume
    其中 taker_sell_quote_volume 用 quote_volume - taker_buy_quote_volume 近似，
    即 daily_flow = 2 * taker_buy_quote_volume - quote_volume
    net_flow = rolling_sum(daily_flow, window) / rolling_sum(quote_volume, window)
    （分母为 0 时置 NaN）

因子衡量过去 window 天内主动买入净流量占总成交额的比例，值越大表示
净主动买入压力越强。计算只使用当日及历史数据，无未来函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Net_Order_Flow_Factor",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "N-day net taker buy order flow over quote volume",
}

SETTING = {
    "data_needed": ["quote_volume", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {"window": 20},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily net-order-flow matrix.

    ``rolling`` operates along the daily index with default
    ``min_periods == window`` (same as the legacy script), so each value
    only depends on the current and preceding ``window - 1`` observations.
    """

    window = SETTING["params"]["window"]

    quote_volume = data_ctx["quote_volume"].astype("float64")
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # taker_sell_quote 近似为 quote_volume - taker_buy_quote（与旧版一致）
    taker_sell_quote = quote_volume - taker_buy_quote
    daily_flow = taker_buy_quote - taker_sell_quote

    numerator = daily_flow.rolling(window).sum()
    denominator = quote_volume.rolling(window).sum().replace(0, np.nan)
    return numerator / denominator
