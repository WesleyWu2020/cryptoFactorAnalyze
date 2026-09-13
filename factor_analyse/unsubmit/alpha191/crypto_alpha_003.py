"""crypto_alpha_003 因子（factor_common 日频契约版，GTJA Alpha3）。

原始公式（国泰君安 Alpha191 #3，日频）：
    SUM((CLOSE=DELAY(CLOSE,1) ? 0
         : CLOSE - (CLOSE>DELAY(CLOSE,1) ? MIN(LOW,DELAY(CLOSE,1))
                                          : MAX(HIGH,DELAY(CLOSE,1)))), 6)

转换说明：
- 旧版分钟实现先 resample("D") 聚合再重采样回分钟索引；日频输入下直接
  使用日 OHLC 字段，聚合步骤消失，返回日频矩阵。
- 删除旧版末尾 ``factor_daily.shift(1)`` 执行延迟（框架已按次日开盘执行）。
- 旧版 ``calc_factor`` 外层对结果取负（即实现 ``-SUM(...)``）。本次按
  alpha191_factor_prompts.md 的原始日频公式实现（不带额外取负），并通过
  ``factor_direction = -1`` 保留旧版的做多偏好语义（旧实现高值=做多
  ``-SUM``，等价于低 SUM 做多）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_003",
    "author": "wesleywu",
    "level": "daily",
    "category": "intraday_strength",
    "description": "SUM(CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1))), 6)",
}

SETTING = {
    "data_needed": ["high", "low", "close"],
    "universe": "historical_top50",
    "warmup_bars": 13,
    "preprocessing": "mad_rank",
    "params": {"sum_window": 6},
    "factor_direction": -1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha3 matrix (date x instrument)."""

    sum_window = SETTING["params"]["sum_window"]

    daily_high = data_ctx["high"].astype("float64")
    daily_low = data_ctx["low"].astype("float64")
    daily_close = data_ctx["close"].astype("float64")
    prev_close = daily_close.shift(1)

    low_or_prev = np.minimum(daily_low, prev_close)
    high_or_prev = np.maximum(daily_high, prev_close)

    contrib = pd.DataFrame(
        np.where(
            daily_close == prev_close,
            0.0,
            np.where(daily_close > prev_close, daily_close - low_or_prev, daily_close - high_or_prev),
        ),
        index=daily_close.index,
        columns=daily_close.columns,
    ).where(daily_close.notna() & prev_close.notna())

    factor = contrib.rolling(sum_window, min_periods=sum_window).sum()
    return factor.replace([np.inf, -np.inf], np.nan)
