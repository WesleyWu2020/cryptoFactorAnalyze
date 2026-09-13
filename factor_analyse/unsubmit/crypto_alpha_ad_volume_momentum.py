"""crypto_alpha_ad_volume_momentum 因子（factor_common 日频契约版）。

原始定义（Chaikin A/D 线动量）：
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    daily_ad = money_flow_multiplier * volume
    factor = rolling_sum(daily_ad, 5d)

转换判断说明：
- 旧分钟版 resample("D") 的 OHLCV 日聚合在日频输入下为恒等，直接删除。
- 删除旧版 rolling_sum(...).shift(1) 中的 shift(1)（仅为分钟级执行延迟）；
  t 日因子只用 <= t 数据。
- 删除分块列循环，直接整帧运算。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_ad_volume_momentum",
    "author": "wesleywu",
    "level": "daily",
    "category": "volume_price",
    "description": "rolling_sum(money_flow_multiplier * volume, 5d) (Chaikin A/D momentum)",
}

SETTING = {
    "data_needed": ["high", "low", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 12,
    "preprocessing": "mad_rank",
    "params": {"window_days": 5},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily A/D-volume-momentum matrix (date x instrument)."""

    window_days = SETTING["params"]["window_days"]

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64").clip(lower=0)

    spread = high - low
    money_flow_multiplier = ((close - low) - (high - close)).div(spread.replace(0.0, np.nan))
    daily_ad = money_flow_multiplier * volume

    factor = daily_ad.rolling(window_days, min_periods=window_days).sum()
    return factor.replace([np.inf, -np.inf], np.nan)
