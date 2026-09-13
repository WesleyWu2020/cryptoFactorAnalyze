"""波动率-订单流因子（Volatility Order Flow Factor）。

公式（与旧版 CSV 管线 compute_one 逐点等价）：

    roc                = close / close.shift(window) - 1
    range_pct          = ((high - low) / close.shift(1)).rolling(window).mean()
    vol_eff            = roc / (range_pct + 1e-8)
    avg_trade_size     = quote_volume / (trade_count + 1e-8)
    trades_ratio       = trade_count / (trade_count.rolling(window).mean() + 1e-8)
    avg_trade_size_ratio = avg_trade_size / (avg_trade_size.rolling(window).mean() + 1e-8)
    order_flow         = trades_ratio / (avg_trade_size_ratio + 1e-8)
    factor             = vol_eff * order_flow

所有 rolling/shift 均沿时间轴向后回看，只使用当日及历史数据，无未来函数。
注意旧数据字段 ``trades_count`` 在新数据中名为 ``trade_count``。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Volatility_Order_Flow_Factor",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "波动率效率与订单流强度复合因子：vol_eff * order_flow",
}

SETTING = {
    "data_needed": ["high", "low", "close", "quote_volume", "trade_count"],
    "universe": "historical_top50",
    # 最长回看：range_pct 需要 close.shift(1) 再 rolling(window)，共 window + 1 根
    "warmup_bars": 21,
    "preprocessing": "mad_rank",
    # rebalance_period 在旧版仅用于 future_ret 标签（已删除），此处保留默认参数记录
    "params": {"window": 20, "rebalance_period": 8},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始波动率-订单流因子矩阵（date × instrument）。

    rolling 默认 min_periods=window，与旧版行为一致；此处显式写出。
    """

    window = SETTING["params"]["window"]

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")
    trade_count = data_ctx["trade_count"].astype("float64")

    roc = close / close.shift(window) - 1
    range_pct = ((high - low) / close.shift(1)).rolling(
        window=window, min_periods=window
    ).mean()
    vol_eff = roc / (range_pct + _EPSILON)

    avg_trade_size = quote_volume / (trade_count + _EPSILON)
    trades_ratio = trade_count / (
        trade_count.rolling(window=window, min_periods=window).mean() + _EPSILON
    )
    avg_trade_size_ratio = avg_trade_size / (
        avg_trade_size.rolling(window=window, min_periods=window).mean() + _EPSILON
    )
    order_flow = trades_ratio / (avg_trade_size_ratio + _EPSILON)

    return vol_eff * order_flow
