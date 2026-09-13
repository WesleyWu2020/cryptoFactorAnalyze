"""crypto_alpha_volume_short_mean_anomaly 因子（factor_common 日频契约版）。

原始定义（分钟版）：
    ref_mean = rolling_mean(daily_volume.shift(1), 3d)   # 前 3 日（不含当日）均量
    factor = log1p(daily_volume) - log1p(ref_mean)        # 当日量相对短期基准的异常

分钟->日频改写说明：
    - 原实现对分钟 volume 取正后 resample 求和得到日成交量；日频契约下 volume
      直接为日频总量，聚合为恒等。
    - ref_mean 内部的 .shift(1) 是定义的一部分（基准不含当日），保留；
      删除最外层对整个因子序列的 .shift(1)（执行延迟）与 reindex 回分钟索引步骤；
      框架按次日开盘执行，factor(t) 允许使用 t 及以前数据。
    - 删除 CHUNK_SIZE 分块循环，整表向量化计算。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_volume_short_mean_anomaly",
    "author": "wesleywu",
    "level": "daily",
    "category": "liquidity",
    "description": "log1p(daily volume) minus log1p(mean volume of the previous 3 days): "
                   "short-term volume anomaly",
}

SETTING = {
    "data_needed": ["volume"],
    "universe": "historical_top50",
    # 1 (shift) + 3 (ref mean) + buffer
    "warmup_bars": 10,
    "preprocessing": "mad_rank",
    "params": {"window_days": 3},
    # 无 ITER_NOTE 方向记录；高值=当日量能异常放大，默认 1
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily volume-anomaly matrix (date x instrument)."""

    window_days = SETTING["params"]["window_days"]

    daily_volume = data_ctx["volume"].astype("float64").clip(lower=0)

    # shift(1) 是定义的一部分：基准为前 window_days 日（不含当日）均量
    ref_mean = daily_volume.shift(1).rolling(
        window_days,
        min_periods=window_days,
    ).mean()
    factor = np.log1p(daily_volume) - np.log1p(ref_mean)
    return factor.replace([np.inf, -np.inf], np.nan)
