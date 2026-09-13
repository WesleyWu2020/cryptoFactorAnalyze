"""crypto_alpha_volume_convergence_factor 因子（factor_common 日频契约版）。

原始定义（分钟版）：
    mas = [rolling_mean(daily_volume, w) for w in (1, 5, 10, 20, 30, 60)]
    factor = -log1p(std across the 6 MAs)，仅当 6 条均线全部有效时输出

分钟->日频改写说明：
    - 原实现对分钟 volume 取正后 resample 求和得到日成交量；日频契约下 volume
      直接为日频总量，聚合为恒等。
    - 删除原实现的 .shift(1)（执行延迟）与 reindex 回分钟索引步骤；
      框架按次日开盘执行，factor(t) 允许使用 t 及以前数据。
    - 删除 CHUNK_SIZE 分块循环，整表向量化计算（沿用 pandas rolling + numpy stack）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_volume_convergence_factor",
    "author": "wesleywu",
    "level": "daily",
    "category": "liquidity",
    "description": "-log1p(std of volume MAs over 1/5/10/20/30/60d): "
                   "multi-horizon volume convergence",
}

SETTING = {
    "data_needed": ["volume"],
    "universe": "historical_top50",
    # 60 (longest MA) + buffer
    "warmup_bars": 70,
    "preprocessing": "mad_rank",
    "params": {"windows": (1, 5, 10, 20, 30, 60)},
    # 无 ITER_NOTE 方向记录；factor=-log1p(std)，高值=多周期量能收敛，默认 1
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily volume-convergence matrix (date x instrument)."""

    windows = tuple(SETTING["params"]["windows"])

    daily_volume = data_ctx["volume"].astype("float64").clip(lower=0)

    mas = [
        daily_volume.rolling(window, min_periods=window).mean()
        for window in windows
    ]
    values = np.stack(
        [ma.to_numpy(dtype=np.float64, copy=False) for ma in mas],
        axis=0,
    )
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    safe_values = np.where(valid, values, 0.0)
    mean = np.divide(
        safe_values.sum(axis=0),
        count,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    var = np.divide(
        np.where(valid, (values - mean) ** 2, 0.0).sum(axis=0),
        count,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    std = np.sqrt(var)
    raw = np.where(count == len(windows), -np.log1p(std), np.nan)
    return pd.DataFrame(raw, index=daily_volume.index, columns=daily_volume.columns)
