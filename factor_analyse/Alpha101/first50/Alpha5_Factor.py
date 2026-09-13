"""Alpha101 Alpha#5 因子（factor_common 契约版）。

原始定义:
    Alpha#5 = rank(open - sum(vwap, 10) / 10) * (-1 * abs(rank(close - vwap)))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    vwap = taker_buy_quote_volume / taker_buy_base_volume
    factor_raw = (open - mean(vwap, window)) * (close - vwap)
effective.txt 推荐参数 vwap_window=20。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha5_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #5: (open - mean(vwap, N)) * (close - vwap)",
}

SETTING = {
    "data_needed": ["open", "close", "taker_buy_base_volume", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 25,
    "preprocessing": "mad_rank",
    "params": {"vwap_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#5 matrix (date x instrument)."""

    window = SETTING["params"]["vwap_window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.vwap_window must be an integer >= 2")

    open_ = data_ctx["open"].astype("float64")
    close = data_ctx["close"].astype("float64")
    taker_base = data_ctx["taker_buy_base_volume"].astype("float64")
    taker_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # VWAP 仅在有主动买入成交的日子可观测
    observed = (taker_base > 0) & (taker_quote > 0)
    vwap = (taker_quote / (taker_base + _EPSILON)).where(observed)
    vwap_ma = vwap.rolling(window=window, min_periods=window).mean()

    trend_deviation = open_ - vwap_ma
    intraday_deviation = close - vwap
    return trend_deviation * intraday_deviation
