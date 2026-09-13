"""crypto_alpha_acceleration_short_reversal 因子（factor_common 日频契约版）。

原始定义（分钟版日聚合语义）：
    recent_return   = log(close_d) - log(close_d).shift(accel_days)
    previous_return = log(close_d).shift(accel_days) - log(close_d).shift(2*accel_days)
    acceleration    = recent_return - previous_return
    normalized      = acceleration / rolling_std(daily_log_return, vol_window_days)
    volume_ratio    = quote_volume_d / rolling_mean(quote_volume_d, volume_window_days)
    raw = -normalized（仅在 min_volume_ratio <= volume_ratio <= max_volume_ratio 时保留）
    factor = ewm_span10(rolling_mean5(raw))

动量加速度的短反转：加速度过高（相对自身波动）做空、加速度过低做多。
转换判断说明：
- dollar_volume -> quote_volume（分钟成交额日求和 == 日 quote_volume）。
- 旧分钟版 groupby(day) 日聚合在日频输入下为恒等，直接删除。
- 删除旧版末尾 factor_daily.shift(1)（仅为分钟级执行延迟）；t 日因子只用 <= t 数据。
- 删除旧版 TAIL_Q 截面尾部过滤（组合构造层的极端分位掩码，框架分组已承担该职能；
  保留会把 80% 截面置 NaN 导致覆盖率不达标）。
- 删除分块列循环，直接整帧运算。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_acceleration_short_reversal",
    "author": "wesleywu",
    "level": "daily",
    "category": "reversal",
    "description": "-ewm10(mean5(vol-normalized return acceleration)), liquidity-band filtered",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 75,
    "preprocessing": "mad_rank",
    "params": {
        "accel_days": 3,
        "vol_window_days": 45,
        "volume_window_days": 45,
        "raw_smooth_days": 5,
        "smooth_days": 10,
        "min_volume_ratio": 0.2,
        "max_volume_ratio": 5.0,
    },
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily acceleration-short-reversal matrix (date x instrument)."""

    params = SETTING["params"]
    accel_days = params["accel_days"]
    vol_window_days = params["vol_window_days"]
    volume_window_days = params["volume_window_days"]
    raw_smooth_days = params["raw_smooth_days"]
    smooth_days = params["smooth_days"]
    min_volume_ratio = params["min_volume_ratio"]
    max_volume_ratio = params["max_volume_ratio"]

    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64").clip(lower=0)

    log_close = np.log(close.where(close > 0))

    recent_return = log_close - log_close.shift(accel_days)
    previous_return = log_close.shift(accel_days) - log_close.shift(accel_days * 2)
    acceleration = recent_return - previous_return

    daily_return_1d = log_close.diff()
    realized_vol = daily_return_1d.rolling(
        vol_window_days,
        min_periods=max(5, vol_window_days // 4),
    ).std()

    normalized_acceleration = acceleration / realized_vol.replace(0, np.nan)

    volume_base = quote_volume.rolling(
        volume_window_days,
        min_periods=max(5, volume_window_days // 4),
    ).mean()
    volume_ratio = quote_volume / volume_base.replace(0, np.nan)
    liquid_mask = volume_ratio.ge(min_volume_ratio) & volume_ratio.le(max_volume_ratio)

    raw = -normalized_acceleration.where(liquid_mask)
    raw = raw.replace([np.inf, -np.inf], np.nan)

    if raw_smooth_days > 1:
        raw = raw.rolling(
            raw_smooth_days,
            min_periods=max(2, raw_smooth_days // 2),
        ).mean()

    factor = raw.ewm(
        span=smooth_days,
        min_periods=max(2, smooth_days // 2),
        adjust=False,
    ).mean()

    return factor.replace([np.inf, -np.inf], np.nan)
