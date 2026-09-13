"""价格-散户相关性背离因子 (Price-Retail Correlation Divergence)。

公式（与旧版 CSV 管线 compute_one 逐点等价）：

    close_ret_1d  = PCT_CHANGE(close, 1)        = close / close.shift(1) - 1
    trades_ret_1d = PCT_CHANGE(trade_count, 1)  （旧字段 trades_count，新数据为 trade_count）
    qv_ret_1d     = PCT_CHANGE(quote_volume, 1)
    corr_close_trades = CORRELATION(close_ret_1d, trades_ret_1d, window)
    corr_close_qv     = CORRELATION(close_ret_1d, qv_ret_1d, window)
    factor = corr_close_trades - corr_close_qv

其中 CORRELATION 为 rolling(window).corr，min_periods 等于 window（旧版
operator_utils 默认行为）。因子含义：收盘价变动与散户活跃度（成交笔数）
变动的相关性，减去其与成交额变动的相关性；相关性差越小（越负）越好，
故 factor_direction = -1。

计算只使用当日及历史数据，无未来函数。返回原始因子矩阵，截面
MAD 去极值与按日 rank 归一化由框架 preprocessing="mad_rank" 处理。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Price_Retail_Correlation_Divergence_Factor",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "Price-retail correlation divergence: corr(close_ret, trades_ret) - corr(close_ret, quote_volume_ret)",
}

SETTING = {
    "data_needed": ["close", "quote_volume", "trade_count"],
    "universe": "historical_top50",
    "warmup_bars": 21,
    "preprocessing": "mad_rank",
    "params": {"window": 20},
    "factor_direction": -1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw price-retail correlation divergence matrix.

    ``data_ctx`` contains date-by-instrument matrices. ``pct_change`` and
    ``rolling`` operate along the daily index (axis 0), so each value only
    depends on the current and preceding observations. ``min_periods`` equals
    ``window``, matching the old operator_utils default.
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.window must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")
    trade_count = data_ctx["trade_count"].astype("float64")

    close_ret_1d = close.pct_change(1)
    trades_ret_1d = trade_count.pct_change(1)
    qv_ret_1d = quote_volume.pct_change(1)

    corr_close_trades = close_ret_1d.rolling(window=window, min_periods=window).corr(trades_ret_1d)
    corr_close_qv = close_ret_1d.rolling(window=window, min_periods=window).corr(qv_ret_1d)

    return corr_close_trades - corr_close_qv
