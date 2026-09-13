"""crypto_alpha_amplitude_sorted_momentum_ratio 因子（factor_common 日频契约版）。

原始定义（分钟版 numba 实现的日聚合语义）：
    ret_d = close_d / close_{d-1} - 1
    amplitude_d = (high_d - low_d) / close_d
    对每个 20d 滚动窗口：按 amplitude 升序排序，
    factor = sum(ret | 振幅最高的 10 天) / sum(ret | 振幅最低的 10 天)

即“高振幅日动量 / 低振幅日动量”的比值，衡量动量在高低波动日之间的分布。
转换判断说明：
- 旧分钟版在 numba 内做分钟->日 OHLC 聚合，日频输入下为恒等，直接删除 numba 内核，
  改用 numpy sliding_window_view 向量化（语义逐条对齐：整窗 20 天全部有效才输出、
  low_sum == 0 或 ratio 非有限时置 NaN）。
- 旧版把窗口结果写到 end_day + 1（分钟级执行延迟），此处删去该 +1 日延迟；
  t 日因子只用 <= t 数据。
- 删除分块列循环，直接整帧运算。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_amplitude_sorted_momentum_ratio",
    "author": "wesleywu",
    "level": "daily",
    "category": "momentum",
    "description": "sum(ret of 10 highest-amplitude days) / sum(ret of 10 lowest) over 20d",
}

SETTING = {
    "data_needed": ["high", "low", "close"],
    "universe": "historical_top50",
    "warmup_bars": 26,
    "preprocessing": "mad_rank",
    "params": {"window_days": 20, "tail_days": 10},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily amplitude-sorted-momentum-ratio matrix (date x instrument)."""

    window_days = SETTING["params"]["window_days"]
    tail_days = SETTING["params"]["tail_days"]

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")

    prev_close = close.shift(1)
    ret = (close / prev_close.where(prev_close != 0.0) - 1.0)
    amplitude = ((high - low) / close.where(close != 0.0)).replace([np.inf, -np.inf], np.nan)

    ret_np = ret.to_numpy(dtype=np.float64)
    amp_np = amplitude.to_numpy(dtype=np.float64)
    n_rows, n_cols = ret_np.shape
    factor_np = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    if n_rows >= window_days:
        # (n_rows - window_days + 1, n_cols, window_days)
        ret_win = sliding_window_view(ret_np, window_days, axis=0)
        amp_win = sliding_window_view(amp_np, window_days, axis=0)

        full_valid = (np.isfinite(ret_win) & np.isfinite(amp_win)).all(axis=-1)

        order = np.argsort(amp_win, axis=-1, kind="stable")
        sorted_ret = np.take_along_axis(ret_win, order, axis=-1)
        low_sum = sorted_ret[..., :tail_days].sum(axis=-1)
        high_sum = sorted_ret[..., window_days - tail_days:].sum(axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = high_sum / low_sum
        ok = full_valid & (low_sum != 0.0) & np.isfinite(ratio)
        factor_np[window_days - 1:, :] = np.where(ok, ratio, np.nan)

    return pd.DataFrame(factor_np, index=close.index, columns=close.columns)
