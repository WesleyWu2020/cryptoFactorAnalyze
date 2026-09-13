"""crypto_alpha_005 因子（factor_common 日频契约版，GTJA Alpha5）。

原始公式（国泰君安 Alpha191 #5，日频）：
    (-1 * TSMAX(CORR(TSRANK(VOLUME,5), TSRANK(HIGH,5), 5), 3))

转换说明：
- 旧版分钟实现先 resample("D") 聚合再重采样回分钟索引；日频输入下直接
  使用日最高价与日成交量，聚合步骤消失，返回日频矩阵。
- 删除旧版末尾 ``factor_daily.shift(1)`` 执行延迟（框架已按次日开盘执行）。
- TSRANK/CORR/TSMAX 均为按标的的因果时序运算。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_005",
    "author": "wesleywu",
    "level": "daily",
    "category": "volume_price",
    "description": "-1 * TSMAX(CORR(TSRANK(VOLUME,5), TSRANK(HIGH,5), 5), 3)",
}

SETTING = {
    "data_needed": ["high", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {"tsrank_window": 5, "corr_window": 5, "tsmax_window": 3},
    "factor_direction": 1,
}


def _ts_rank_last(values: np.ndarray) -> float:
    if not np.isfinite(values).all():
        return np.nan
    last = values[-1]
    less = np.sum(values < last)
    equal = np.sum(values == last)
    return (less + 0.5 * (equal + 1.0)) / values.size


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha5 matrix (date x instrument)."""

    tsrank_window = SETTING["params"]["tsrank_window"]
    corr_window = SETTING["params"]["corr_window"]
    tsmax_window = SETTING["params"]["tsmax_window"]

    daily_high = data_ctx["high"].astype("float64")
    daily_volume = data_ctx["volume"].astype("float64")

    volume_rank = daily_volume.rolling(tsrank_window, min_periods=tsrank_window).apply(
        _ts_rank_last, raw=True
    )
    high_rank = daily_high.rolling(tsrank_window, min_periods=tsrank_window).apply(
        _ts_rank_last, raw=True
    )

    rolling_corr = volume_rank.rolling(corr_window, min_periods=corr_window).corr(high_rank)
    factor = -rolling_corr.rolling(tsmax_window, min_periods=tsmax_window).max()
    return factor.replace([np.inf, -np.inf], np.nan)
