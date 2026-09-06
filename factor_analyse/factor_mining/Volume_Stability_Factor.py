"""成交量稳定性因子。

新框架入口是 ``calc_factor(data_ctx)``：

    average_trade_size = quote_volume / trade_count
    factor = rolling_mean(average_trade_size) / rolling_std(average_trade_size)

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Volume_Stability_Factor",
    "author": "local",
    "level": "daily",
    "category": "volume",
    "description": "N-day average trade-size stability",
}

SETTING = {
    "data_needed": ["quote_volume", "trade_count"],
    "universe": "historical_top50",
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {"window": 20},
    "factor_direction": -1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily volume-stability matrix.

    ``data_ctx`` contains date-by-instrument matrices. ``rolling`` operates
    along the daily index, so each value only depends on the current and
    preceding ``window - 1`` observations.
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.window must be an integer >= 2")

    quote_volume = data_ctx["quote_volume"].astype("float64")
    trade_count = data_ctx["trade_count"].astype("float64")
    # No-trade bars have no observed average trade size. Keep the calendar
    # gap so the full trailing window must recover before producing a signal.
    observed = (quote_volume > 0) & (trade_count > 0)
    average_trade_size = (quote_volume / (trade_count + _EPSILON)).where(observed)
    rolling_mean = average_trade_size.rolling(
        window=window,
        min_periods=window,
    ).mean()
    rolling_std = average_trade_size.rolling(
        window=window,
        min_periods=window,
    ).std()
    return rolling_mean / (rolling_std + _EPSILON)
