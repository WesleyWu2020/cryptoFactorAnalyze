"""crypto_alpha_006 因子（factor_common 日频契约版，GTJA Alpha6）。

原始公式（国泰君安 Alpha191 #6，日频）：
    (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4))) * -1)

转换说明：
- 旧版分钟实现先 resample("D") 聚合再重采样回分钟索引；日频输入下直接
  使用日开盘价与日最高价，聚合步骤消失，返回日频矩阵。
- 删除旧版末尾 ``factor_daily.shift(1)`` 执行延迟（框架已按次日开盘执行）。
- DELTA 为因果时序差分，RANK 为当日截面百分比排名。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_006",
    "author": "wesleywu",
    "level": "daily",
    "category": "price_volume",
    "description": "-1 * RANK(SIGN(DELTA(OPEN*0.85 + HIGH*0.15, 4)))",
}

SETTING = {
    "data_needed": ["open", "high"],
    "universe": "historical_top50",
    "warmup_bars": 10,
    "preprocessing": "mad_rank",
    "params": {"delta_window": 4, "open_weight": 0.85, "high_weight": 0.15},
    "factor_direction": 1,
}


def _cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="average", pct=True)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha6 matrix (date x instrument)."""

    delta_window = SETTING["params"]["delta_window"]
    open_weight = SETTING["params"]["open_weight"]
    high_weight = SETTING["params"]["high_weight"]

    daily_open = data_ctx["open"].astype("float64")
    daily_high = data_ctx["high"].astype("float64")

    weighted_price = daily_open * open_weight + daily_high * high_weight
    factor = -_cross_sectional_rank(np.sign(weighted_price.diff(delta_window)))
    return factor.replace([np.inf, -np.inf], np.nan)
