"""crypto_alpha_007 因子（factor_common 日频契约版，GTJA Alpha7）。

原始公式（国泰君安 Alpha191 #7，日频）：
    ((RANK(MAX((VWAP - CLOSE),3)) + RANK(MIN((VWAP - CLOSE),3))) * RANK(DELTA(VOLUME,3)))

转换说明：
- 旧版分钟实现用 dollar_volume/volume 重构日 VWAP 再 resample("D") 聚合、
  重采样回分钟索引；日频输入下日 VWAP = quote_volume / volume（
  ``dollar_volume -> quote_volume`` 字段映射），聚合步骤消失，返回日频矩阵。
- 删除旧版末尾 ``factor_daily.shift(1)`` 执行延迟（框架已按次日开盘执行）。
- MAX/MIN(x,3)/DELTA 为因果时序运算，RANK 为当日截面百分比排名。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_007",
    "author": "wesleywu",
    "level": "daily",
    "category": "price_volume",
    "description": "(RANK(MAX(VWAP-CLOSE,3)) + RANK(MIN(VWAP-CLOSE,3))) * RANK(DELTA(VOLUME,3))",
}

SETTING = {
    "data_needed": ["close", "volume", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 10,
    "preprocessing": "mad_rank",
    "params": {"roll_window": 3, "delta_window": 3},
    "factor_direction": 1,
}

_EPS = 1e-12


def _cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="average", pct=True)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha7 matrix (date x instrument)."""

    roll_window = SETTING["params"]["roll_window"]
    delta_window = SETTING["params"]["delta_window"]

    daily_close = data_ctx["close"].astype("float64")
    daily_volume = data_ctx["volume"].astype("float64")
    daily_quote_volume = data_ctx["quote_volume"].astype("float64")
    daily_vwap = daily_quote_volume / daily_volume.where(daily_volume.abs() > _EPS)

    spread = daily_vwap - daily_close
    spread_max = spread.rolling(roll_window, min_periods=roll_window).max()
    spread_min = spread.rolling(roll_window, min_periods=roll_window).min()
    volume_delta = daily_volume.diff(delta_window)

    factor = (
        _cross_sectional_rank(spread_max)
        + _cross_sectional_rank(spread_min)
    ) * _cross_sectional_rank(volume_delta)
    return factor.replace([np.inf, -np.inf], np.nan)
