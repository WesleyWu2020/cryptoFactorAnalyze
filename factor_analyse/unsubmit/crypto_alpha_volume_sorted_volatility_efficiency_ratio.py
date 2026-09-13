"""crypto_alpha_volume_sorted_volatility_efficiency_ratio 因子（factor_common 日频契约版）。

原始定义（分钟版语义）：
    每日 signal = daily_ret / daily_amplitude，
    其中 daily_ret = close/prev_close - 1，daily_amplitude = (high - low)/prev_close。
    在 20d 窗口内按成交量升序排序，取量最低 10 日与量最高 10 日的 signal 之和：
        factor = high_volume_signal_sum / low_volume_signal_sum
    要求窗口内 20 天 signal/volume 全部有效，low_sum == 0 时输出 NaN。

分钟 -> 日频改写说明：
    - 分钟聚合（close=last, high=max, low=min, volume=sum）在日频输入下为恒等，直接使用日频字段。
    - 原实现将 t 日窗口结果写入 t+1 日（factor_by_day[end_day + 1]），属于执行延迟 shift，
      按日频契约删除——框架已按次日开盘执行，t 日因子可用 <= t 数据。
    - numba 分块内核改写为整体 DataFrame + 逐日滚动排序循环（窗口内按列 argsort）。
    - 窗口语义不变：20d 全有效、10/10 尾部、量比排序均在日频刻度上与原逻辑一一对应。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_volume_sorted_volatility_efficiency_ratio",
    "author": "wesleywu",
    "level": "daily",
    "category": "volatility",
    "description": "20d 窗口内按量排序：高量 10 日 return/amplitude 之和 / 低量 10 日之和",
}

SETTING = {
    "data_needed": ["high", "low", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"window_days": 20, "tail_days": 10},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily volume-sorted efficiency-ratio matrix (date x instrument)."""

    window_days = SETTING["params"]["window_days"]
    tail_days = SETTING["params"]["tail_days"]

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    volume = data_ctx["volume"].astype("float64").clip(lower=0.0)

    prev_close = close.shift(1)
    daily_ret = close / prev_close - 1.0
    daily_amplitude = (high - low) / (prev_close + _EPS)
    signal = (daily_ret / (daily_amplitude + _EPS)).replace([np.inf, -np.inf], np.nan)
    # 原实现要求 close/prev_close/high/low/volume 全部有效才计入窗口
    signal = signal.where(high.notna() & low.notna() & volume.notna())

    sig = signal.to_numpy(dtype=np.float64)
    vol = volume.to_numpy(dtype=np.float64)
    n_days, n_cols = sig.shape
    out = np.full((n_days, n_cols), np.nan)

    for end_day in range(window_days - 1, n_days):
        start_day = end_day - window_days + 1
        sig_win = sig[start_day:end_day + 1]
        vol_win = vol[start_day:end_day + 1]
        # 与原 numba 内核一致：窗口内 20 天必须全部有效
        full = (~np.isnan(sig_win) & ~np.isnan(vol_win)).all(axis=0)
        if not full.any():
            continue
        order = np.argsort(vol_win, axis=0)
        sorted_sig = np.take_along_axis(sig_win, order, axis=0)
        low_sum = sorted_sig[:tail_days].sum(axis=0)
        high_sum = sorted_sig[window_days - tail_days:].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = high_sum / low_sum
        mask = full & (low_sum != 0.0) & np.isfinite(ratio)
        out[end_day, mask] = ratio[mask]

    return pd.DataFrame(out, index=close.index, columns=close.columns)
