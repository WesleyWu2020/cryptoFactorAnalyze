"""平均价格偏差因子 (Average Price Bias, APB)。

逻辑来源：东方证券 - 因子选股系列研究六十：基于量价关系度量股票的买卖压力。

公式：
    daily_vwap  = quote_volume / (volume + eps)
    window_twap = rolling_mean(daily_vwap, window)            # 分子：T 天日均价算术平均
    window_vwap = rolling_sum(volume * daily_vwap, window)
                  / (rolling_sum(volume, window) + eps)       # 分母：T 天成交量加权均价
    APB = ln(window_twap / (window_vwap + eps))

含义：APB > 0 表示买压（低位放量）；APB < 0 表示卖压（高位放量）。
计算只使用当日及历史数据，无未来函数；窗口统计量沿用旧版 rolling 默认
min_periods=window 语义。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "APB_Factor",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "Average Price Bias: ln(window TWAP / window VWAP), 度量买卖压力",
}

SETTING = {
    "data_needed": ["quote_volume", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 3,
    "preprocessing": "mad_rank",
    "params": {"window": 3},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回 date(升序索引) × instrument(列) 的原始 APB 因子矩阵。"""

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be an integer >= 1")

    quote_volume = data_ctx["quote_volume"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 1. 单日均价 (Daily VWAP)，eps 防零
    daily_vwap = quote_volume / (volume + _EPSILON)

    # 2. 窗口 TWAP（分子）：过去 window 天日均价的算术平均值
    #    rolling 未指定 min_periods，默认等于 window，与旧版一致
    window_twap = daily_vwap.rolling(window=window).mean()

    # 3. 窗口 VWAP（分母）：过去 window 天成交量加权的 vwap 均值
    vwap_weighted_sum = (volume * daily_vwap).rolling(window=window).sum()
    window_vol = volume.rolling(window=window).sum()
    window_vwap = vwap_weighted_sum / (window_vol + _EPSILON)

    # 4. APB = ln(TWAP / VWAP)
    return np.log(window_twap / (window_vwap + _EPSILON))
