"""crypto_alpha_004 因子（factor_common 日频契约版，GTJA Alpha4）。

原始公式（国泰君安 Alpha191 #4，日频）：
    ((SUM(CLOSE,8)/8 + STD(CLOSE,8)) < (SUM(CLOSE,2)/2)) ? -1
      : ((SUM(CLOSE,2)/2 < (SUM(CLOSE,8)/8 - STD(CLOSE,8))) ? 1
         : ((1 <= VOLUME/MEAN(VOLUME,20)) ? 1 : -1))

转换说明：
- 旧版分钟实现先 resample("D") 聚合再重采样回分钟索引；日频输入下直接
  使用日收盘价与日成交量，聚合步骤消失，返回日频矩阵。
- 删除旧版末尾 ``factor_daily.shift(1)`` 执行延迟（框架已按次日开盘执行）。
- 因子值为 ±1 离散信号；趋势破位向下做空（-1），向上突破或放量做多（+1）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_004",
    "author": "wesleywu",
    "level": "daily",
    "category": "trend_regime",
    "description": "(MEAN(CLOSE,8)+STD(CLOSE,8)<MEAN(CLOSE,2)) ? -1 : (MEAN(CLOSE,2)<MEAN(CLOSE,8)-STD(CLOSE,8) ? 1 : (VOLUME>=MEAN(VOLUME,20) ? 1 : -1))",
}

SETTING = {
    "data_needed": ["close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 26,
    "preprocessing": "mad_rank",
    "params": {"band_window": 8, "short_window": 2, "volume_window": 20, "volume_threshold": 1.0},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha4 matrix (date x instrument)."""

    band_window = SETTING["params"]["band_window"]
    short_window = SETTING["params"]["short_window"]
    volume_window = SETTING["params"]["volume_window"]
    volume_threshold = SETTING["params"]["volume_threshold"]

    daily_close = data_ctx["close"].astype("float64")
    daily_volume = data_ctx["volume"].astype("float64")

    mean_long = daily_close.rolling(band_window, min_periods=band_window).mean()
    std_long = daily_close.rolling(band_window, min_periods=band_window).std()
    mean_short = daily_close.rolling(short_window, min_periods=short_window).mean()
    volume_mean = daily_volume.rolling(volume_window, min_periods=volume_window).mean()
    volume_ratio = daily_volume / volume_mean.where(volume_mean.abs() > _EPS)

    cond_down = (mean_long + std_long) < mean_short
    cond_up = mean_short < (mean_long - std_long)
    valid = mean_long.notna() & std_long.notna() & mean_short.notna() & volume_ratio.notna()

    raw = np.where(
        cond_down.to_numpy(),
        -1.0,
        np.where(
            cond_up.to_numpy(),
            1.0,
            np.where(volume_ratio.to_numpy() >= volume_threshold, 1.0, -1.0),
        ),
    )
    factor = pd.DataFrame(
        np.where(valid.to_numpy(), raw, np.nan),
        index=daily_close.index,
        columns=daily_close.columns,
    )
    return factor.replace([np.inf, -np.inf], np.nan)
