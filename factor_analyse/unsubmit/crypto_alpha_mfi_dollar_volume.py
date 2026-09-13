"""crypto_alpha_mfi_dollar_volume 因子（factor_common 日频契约版）。

原始定义（分钟版语义，Money Flow Index 以成交额计价，120d 窗口）：
    typical_price  = (high + low + close) / 3
    raw_money_flow = typical_price * dollar_volume
    positive_flow  = raw_money_flow where typical_price.diff() > 0 else 0
    negative_flow  = raw_money_flow where typical_price.diff() < 0 else 0
    factor = 100 * sum(positive_flow, 120d) / (sum(positive) + sum(negative))

日频改写说明：
    - data_ctx 已是日频矩阵（high=max/low=min/close=last、dollar_volume 日求和 ==
      日频 quote_volume），分钟版的 resample("1440min") 日聚合恒等，直接使用
      quote_volume。
    - 删除分钟版末尾的 .shift(1)（执行延迟由框架次日开盘成交处理）与分钟 index
      重广播。
    - 窗口单位由分钟 bar 改写为日历日，语义不变。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_mfi_dollar_volume",
    "author": "wesleywu",
    "level": "daily",
    "category": "volume_price",
    "description": "以成交额计价的资金流量指标 MFI：120d 正资金流占正负资金流之和的百分比",
}

SETTING = {
    "data_needed": ["high", "low", "close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 130,
    "preprocessing": "mad_rank",
    "params": {"window_days": 120},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily dollar-volume MFI matrix (date x instrument)."""

    window_days = SETTING["params"]["window_days"]

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].clip(lower=0).astype("float64")

    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * quote_volume
    typ_diff = typical_price.diff()

    positive_flow = raw_money_flow.where(typ_diff > 0, 0.0).where(typ_diff.notna())
    negative_flow = raw_money_flow.where(typ_diff < 0, 0.0).where(typ_diff.notna())

    positive_sum = positive_flow.rolling(window_days, min_periods=window_days).sum()
    negative_sum = negative_flow.rolling(window_days, min_periods=window_days).sum()
    total_sum = positive_sum + negative_sum

    factor = 100.0 * positive_sum / total_sum.replace(0.0, np.nan)
    return factor.replace([np.inf, -np.inf], np.nan)
