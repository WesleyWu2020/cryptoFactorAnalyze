"""crypto_alpha_price_convergence_factor 因子（factor_common 日频契约版）。

原始定义：
    对日收盘价取窗口 (1, 5, 10, 20, 60) 的多周期均线，
    factor = -log1p(std(MA_1, MA_5, MA_10, MA_20, MA_60))
    均线越收敛（std 越小）因子值越高。

分钟->日频改写判断（记录在案）：
  - 原实现对分钟 close resample("1440min").last() 得日收盘；日频契约下直接用 daily close。
  - 原实现 factor_daily.shift(1) 后广播回分钟索引，属于执行延迟；框架以次开盘价执行，
    删除该 shift，date t 因子可直接使用 t 日完整数据。
  - CHUNK_SIZE 分块循环纯为内存控制，日频数据量小，删除。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_price_convergence_factor",
    "author": "wesleywu",
    "level": "daily",
    "category": "trend",
    "description": "-log1p(std of MA(1/5/10/20/60d) of close)：多周期均线收敛度",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 70,
    "preprocessing": "mad_rank",
    "params": {"windows": [1, 5, 10, 20, 60]},
    "factor_direction": 1,  # 默认：均线收敛（高因子值）优先，原文档未给方向先验
}


def _convergence_factor(daily_value: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    mas = [
        daily_value.rolling(window, min_periods=window).mean()
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
    return pd.DataFrame(raw, index=daily_value.index, columns=daily_value.columns)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily price-convergence matrix (date x instrument)."""

    windows = tuple(int(w) for w in SETTING["params"]["windows"])
    close = data_ctx["close"].astype("float64")
    factor = _convergence_factor(close, windows)
    return factor.replace([np.inf, -np.inf], np.nan)
