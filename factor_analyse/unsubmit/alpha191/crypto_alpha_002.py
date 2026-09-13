"""crypto_alpha_002 因子（factor_common 日频契约版，GTJA Alpha2）。

原始公式（国泰君安 Alpha191 #2，日频）：
    (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))

转换说明：
- 旧版分钟实现先 resample("D") 聚合再重采样回分钟索引；日频输入下直接
  使用日 OHLC 字段，聚合步骤消失，返回日频矩阵。
- 删除旧版末尾 ``factor_daily.shift(1)`` 执行延迟（框架已按次日开盘执行）。
- DELTA(x,1) 即 ``x.diff(1)``，均为因果时序运算。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_002",
    "author": "wesleywu",
    "level": "daily",
    "category": "intraday_reversal",
    "description": "-1 * DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW), 1)",
}

SETTING = {
    "data_needed": ["high", "low", "close"],
    "universe": "historical_top50",
    "warmup_bars": 7,
    "preprocessing": "mad_rank",
    "params": {"delta_window": 1},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha2 matrix (date x instrument)."""

    delta_window = SETTING["params"]["delta_window"]

    daily_high = data_ctx["high"].astype("float64")
    daily_low = data_ctx["low"].astype("float64")
    daily_close = data_ctx["close"].astype("float64")

    daily_range = daily_high - daily_low
    close_location = (
        (daily_close - daily_low) - (daily_high - daily_close)
    ) / daily_range.where(daily_range.abs() > _EPS)
    factor = -close_location.diff(delta_window)
    return factor.replace([np.inf, -np.inf], np.nan)
