"""crypto_alpha_009 因子（factor_common 日频契约版）。

GTJA Alpha9 日频原始公式（见 alpha191_factor_prompts.md）：
    SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME, 7, 2)

即：日频中点的一日差分，乘以日内振幅、除以日成交量，再做 GTJA/Tongdaxin
风格 SMA(X, 7, 2)（等价 ewm(alpha=2/7, adjust=False)）。

转换判断说明：
- 旧分钟版先 resample("D") 聚合再算同一公式，日频输入下聚合步骤为恒等，直接删除。
- 删除旧版 factor_daily.shift(1)（仅为分钟级执行延迟）；t 日因子只用 <= t 数据。
- 删除分块列循环，直接整帧运算。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_009",
    "author": "wesleywu",
    "level": "daily",
    "category": "price_volume",
    "description": "SMA(((HIGH+LOW)/2-DELAY((HIGH+LOW)/2,1))*(HIGH-LOW)/VOLUME, 7, 2)",
}

SETTING = {
    "data_needed": ["high", "low", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 15,
    "preprocessing": "mad_rank",
    "params": {"sma_window": 7, "sma_weight": 2},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha9 matrix (date x instrument)."""

    sma_window = SETTING["params"]["sma_window"]
    sma_weight = SETTING["params"]["sma_weight"]

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    midpoint = (high + low) / 2.0
    midpoint_delta = midpoint - midpoint.shift(1)
    intraday_range = high - low

    raw = midpoint_delta * intraday_range / volume.where(volume.abs() > _EPS)

    # GTJA/Tongdaxin SMA(X, N, M): Y_t = (M * X_t + (N - M) * Y_{t-1}) / N
    factor = raw.ewm(
        alpha=sma_weight / sma_window,
        adjust=False,
        min_periods=sma_window,
    ).mean()
    return factor.replace([np.inf, -np.inf], np.nan)
