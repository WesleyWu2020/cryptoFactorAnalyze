"""crypto_alpha_008 因子（factor_common 日频契约版，GTJA Alpha8）。

原始公式（国泰君安 Alpha191 #8，日频）：
    RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)

转换说明：
- 旧版分钟实现用 dollar_volume/volume 重构日 VWAP 再 resample("D") 聚合、
  重采样回分钟索引；日频输入下日 VWAP = quote_volume / volume（
  ``dollar_volume -> quote_volume`` 字段映射），聚合步骤消失，返回日频矩阵。
- 删除旧版末尾 ``factor_daily.shift(1)`` 执行延迟（框架已按次日开盘执行）。
- DELTA 为因果时序差分，RANK 为当日截面百分比排名。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_008",
    "author": "wesleywu",
    "level": "daily",
    "category": "price_volume",
    "description": "RANK(-1 * DELTA(((HIGH+LOW)/2)*0.2 + VWAP*0.8, 4))",
}

SETTING = {
    "data_needed": ["high", "low", "volume", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 10,
    "preprocessing": "mad_rank",
    "params": {"delta_window": 4, "mid_weight": 0.2, "vwap_weight": 0.8},
    "factor_direction": 1,
}

_EPS = 1e-12


def _cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="average", pct=True)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha8 matrix (date x instrument)."""

    delta_window = SETTING["params"]["delta_window"]
    mid_weight = SETTING["params"]["mid_weight"]
    vwap_weight = SETTING["params"]["vwap_weight"]

    daily_high = data_ctx["high"].astype("float64")
    daily_low = data_ctx["low"].astype("float64")
    daily_volume = data_ctx["volume"].astype("float64")
    daily_quote_volume = data_ctx["quote_volume"].astype("float64")
    daily_vwap = daily_quote_volume / daily_volume.where(daily_volume.abs() > _EPS)

    price_mix = ((daily_high + daily_low) / 2.0) * mid_weight + daily_vwap * vwap_weight
    factor = _cross_sectional_rank(price_mix.diff(delta_window) * -1.0)
    return factor.replace([np.inf, -np.inf], np.nan)
