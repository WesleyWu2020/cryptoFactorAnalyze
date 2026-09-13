"""crypto_alpha_acd_accumulation_distribution 因子（factor_common 日频契约版）。

原始定义（A/D 累积派发线变体）：
    true_low  = min(low_d, prev_close_d); true_high = max(high_d, prev_close_d)
    dif = close - true_low   (close > prev_close)
          close - true_high  (close < prev_close)
          0                  (close == prev_close 或 prev_close 缺失)
    factor = -(rolling_sum(dif, 20d) / 20)

旧分钟版顶层 calc_factor 对结果取负号（`return -factor`），负号是因子语义的一部分，
在此保留：factor_direction=1 表示因子值越高越偏多。
转换判断说明：
- 旧分钟版 resample("D") 的 OHLC 日聚合在日频输入下为恒等，直接删除。
- 删除旧版 factor_daily.shift(1)（仅为分钟级执行延迟）；t 日因子只用 <= t 数据。
- 删除分块列循环，直接整帧运算。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_acd_accumulation_distribution",
    "author": "wesleywu",
    "level": "daily",
    "category": "momentum",
    "description": "-(rolling_sum(A/D accumulation-distribution dif, 20d) / 20)",
}

SETTING = {
    "data_needed": ["high", "low", "close"],
    "universe": "historical_top50",
    "warmup_bars": 26,
    "preprocessing": "mad_rank",
    "params": {"long_window": 20},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily (negated) ACD matrix (date x instrument)."""

    long_window = SETTING["params"]["long_window"]

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")

    last_close = close.shift(1)
    true_low = pd.DataFrame(
        np.minimum(low.to_numpy(), last_close.to_numpy()),
        index=low.index,
        columns=low.columns,
    )
    true_high = pd.DataFrame(
        np.maximum(high.to_numpy(), last_close.to_numpy()),
        index=high.index,
        columns=high.columns,
    )

    close_arr = close.to_numpy()
    last_close_arr = last_close.to_numpy()
    dif = np.select(
        [close_arr > last_close_arr, close_arr < last_close_arr],
        [close_arr - true_low.to_numpy(), close_arr - true_high.to_numpy()],
        default=0.0,
    )
    dif = pd.DataFrame(dif, index=close.index, columns=close.columns)
    dif = dif.where(last_close.notna())

    acd = dif.rolling(long_window, min_periods=long_window).sum() / long_window
    factor = -acd
    return factor.replace([np.inf, -np.inf], np.nan)
