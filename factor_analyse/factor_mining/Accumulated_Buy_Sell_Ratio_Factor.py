"""累积主买/累积主卖比率因子（Accumulated Buy/Sell Ratio）。

公式（对旧版 compute_one 逐点等价）：

    daily_return       = close / close.shift(1) - 1
    taker_sell_quote   = quote_volume - taker_buy_quote_volume   # 主动卖盘
    accumulated_buy    = rolling_sum( taker_buy_quote_volume * (daily_return > 0), window )
    accumulated_sell   = rolling_sum( taker_sell_quote        * (daily_return < 0), window )
    factor             = accumulated_buy / (accumulated_sell + 1e-8)

即过去 window 个交易日内，上涨日的主动买盘累计额与下跌日的主动卖盘累计额之比。
比值越高，表明上涨趋势获得的主动资金支撑越强，为正向因子。

计算只使用当日及历史数据（shift(1) 与 trailing rolling），无未来函数。
输出为原始因子值，截面去极值与 rank 归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Accumulated_Buy_Sell_Ratio_Factor",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "上涨日累计主动买盘 / 下跌日累计主动卖盘（N 日窗口）",
}

SETTING = {
    "data_needed": ["close", "quote_volume", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 31,  # shift(1) 的日收益 + window=30 的 rolling(min_periods=window)
    "preprocessing": "mad_rank",
    "params": {"window": 30},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始累积主买/主卖比率矩阵（date × instrument）。

    rolling/shift 均沿时间轴（axis 0），每个因子值只依赖当日及历史数据。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be an integer >= 1")

    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # 日收益率（首行为 NaN，NaN > 0 / NaN < 0 均为 False，与旧版一致）
    daily_return = close / close.shift(1) - 1

    # 主动卖盘 = 总成交额 - 主动买盘
    taker_sell_quote = quote_volume - taker_buy_quote

    # 上涨日累积主动买盘、下跌日累积主动卖盘（条件不满足时记 0；
    # 保持旧版布尔乘法语义：False * NaN = NaN，会沿 rolling 传播）
    buy_conditional = (daily_return > 0) * taker_buy_quote
    sell_conditional = (daily_return < 0) * taker_sell_quote

    accumulated_buy = buy_conditional.rolling(window=window, min_periods=window).sum()
    accumulated_sell = sell_conditional.rolling(window=window, min_periods=window).sum()

    return accumulated_buy / (accumulated_sell + _EPSILON)
