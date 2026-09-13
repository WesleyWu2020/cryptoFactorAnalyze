"""crypto_alpha_high_amplitude_decisive_momentum 因子（factor_common 日频契约版）。

原始定义（分钟版语义）：
    daily_return    = close / prev_close - 1（日频）
    daily_amplitude = high / low - 1（日频）
    factor = 最近 60 个有效交易日中，振幅最高的 top 30%（18 天）的日收益之和
    （要求窗口内 60 天收益与振幅全部有效）

日频改写说明：
    - data_ctx 已是日频矩阵（close=last/high=max/low=min），日聚合恒等，直接使用。
    - 删除分钟版把结果写入 end_day+1 的执行延迟位移与分钟 index 重广播；
      factor 在 t 日只用 <= t 的数据，由框架次日开盘成交。
    - 窗口单位由分钟 bar 改写为日历日，语义不变；numba 分块循环替换为
      sliding_window + argpartition 的向量化实现（同语义：振幅降序 top-K 收益求和）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_high_amplitude_decisive_momentum",
    "author": "wesleywu",
    "level": "daily",
    "category": "momentum",
    "description": "最近 60d 中振幅最高 top 30% 交易日的日收益之和（高振幅日的决断性动量）",
}

SETTING = {
    "data_needed": ["high", "low", "close"],
    "universe": "historical_top50",
    "warmup_bars": 65,
    "preprocessing": "mad_rank",
    "params": {"window_days": 60, "high_amp_ratio": 0.3},
    "factor_direction": 1,
}


def _selective_top_amp_return_sum(
    ret: pd.DataFrame,
    amp: pd.DataFrame,
    window: int,
    select: int,
) -> pd.DataFrame:
    """Sum of daily returns over the top-``select`` amplitude days in each
    trailing ``window``-day window; NaN unless all ``window`` days are valid."""

    ret_np = ret.to_numpy(dtype=np.float64)
    amp_np = amp.to_numpy(dtype=np.float64)
    n_rows, n_cols = ret_np.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    if n_rows < window:
        return pd.DataFrame(out, index=ret.index, columns=ret.columns)

    ret_w = np.lib.stride_tricks.sliding_window_view(ret_np, window, axis=0)
    amp_w = np.lib.stride_tricks.sliding_window_view(amp_np, window, axis=0)
    # shapes: (n_rows - window + 1, n_cols, window)

    valid = np.isfinite(ret_w).all(axis=2) & np.isfinite(amp_w).all(axis=2)

    top_idx = np.argpartition(amp_w, window - select, axis=2)[:, :, window - select:]
    selected_ret = np.take_along_axis(ret_w, top_idx, axis=2)
    selected_sum = selected_ret.sum(axis=2)

    out[window - 1:] = np.where(valid, selected_sum, np.nan)
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily high-amplitude decisive momentum matrix
    (date x instrument)."""

    window_days = SETTING["params"]["window_days"]
    high_amp_ratio = SETTING["params"]["high_amp_ratio"]
    select_days = max(1, int(window_days * high_amp_ratio))

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")

    prev_close = close.shift(1).replace(0.0, np.nan)
    daily_return = close.where(close > 0) / prev_close - 1.0
    daily_amplitude = high / low.where(low > 0) - 1.0

    factor = _selective_top_amp_return_sum(
        daily_return.replace([np.inf, -np.inf], np.nan),
        daily_amplitude.replace([np.inf, -np.inf], np.nan),
        window_days,
        select_days,
    )
    return factor.replace([np.inf, -np.inf], np.nan)
