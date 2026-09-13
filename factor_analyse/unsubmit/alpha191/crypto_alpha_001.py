"""crypto_alpha_001 因子（factor_common 日频契约版，GTJA Alpha1）。

原始公式（国泰君安 Alpha191 #1，日频）：
    (-1 * CORR(RANK(DELTA(LOG(VOLUME),1)), RANK(((CLOSE - OPEN) / OPEN)), 6))

转换说明：
- 旧版分钟实现先 resample("D") 聚合再重采样回分钟索引；日频输入下直接
  使用日 OHLCV 字段，聚合步骤消失，返回日频矩阵。
- 删除旧版末尾 ``factor_daily.shift(1)`` 执行延迟（框架已按次日开盘执行）。
- CORR 为按标的滚动 6 日的时序相关，RANK 为当日截面百分比排名。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_001",
    "author": "wesleywu",
    "level": "daily",
    "category": "price_volume",
    "description": "-1 * CORR(RANK(DELTA(LOG(VOLUME),1)), RANK((CLOSE-OPEN)/OPEN), 6)",
}

SETTING = {
    "data_needed": ["open", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 13,
    "preprocessing": "mad_rank",
    "params": {"corr_window": 6},
    "factor_direction": 1,
}

_EPS = 1e-12


def _cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="average", pct=True)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha1 matrix (date x instrument)."""

    corr_window = SETTING["params"]["corr_window"]

    daily_open = data_ctx["open"].astype("float64")
    daily_close = data_ctx["close"].astype("float64")
    daily_volume = data_ctx["volume"].astype("float64")

    intraday_return = (daily_close - daily_open) / daily_open.where(daily_open.abs() > _EPS)
    log_volume_delta = np.log(daily_volume.where(daily_volume > 0)).diff()

    ranked_volume = _cross_sectional_rank(log_volume_delta)
    ranked_return = _cross_sectional_rank(intraday_return)

    factor = -ranked_volume.rolling(corr_window, min_periods=corr_window).corr(ranked_return)
    return factor.replace([np.inf, -np.inf], np.nan)
