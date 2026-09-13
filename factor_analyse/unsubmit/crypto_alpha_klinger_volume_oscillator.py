"""crypto_alpha_klinger_volume_oscillator 因子（factor_common 日频契约版）。

原始定义（分钟版语义，标准 Klinger Volume Oscillator 日频版）：
    typical_price = (high + low + close) / 3
    signed_volume = volume * sign(typical_price.diff())
    kvo = ema(signed_volume, span=5) - ema(signed_volume, span=20)
    factor = ema(kvo, span=3)   （signal 线）

日频改写说明：
    - data_ctx 已是日频矩阵（high=max/low=min/close=last/volume=sum），
      分钟版的 resample("1440min") 日聚合恒等，直接使用日频字段。
    - 删除分钟版末尾的 .shift(1)（执行延迟由框架次日开盘成交处理）与分钟 index
      重广播。
    - EWM span 单位由分钟 bar 改写为日历日，语义不变。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_klinger_volume_oscillator",
    "author": "wesleywu",
    "level": "daily",
    "category": "volume_price",
    "description": "Klinger 成交量振荡器：ema5(量价方向签名量) - ema20 后再 ema3 平滑",
}

SETTING = {
    "data_needed": ["high", "low", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"fast_span": 5, "slow_span": 20, "signal_span": 3},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Klinger volume oscillator matrix
    (date x instrument)."""

    fast_span = SETTING["params"]["fast_span"]
    slow_span = SETTING["params"]["slow_span"]
    signal_span = SETTING["params"]["signal_span"]

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].clip(lower=0).astype("float64")

    typical_price = (close + high + low) / 3.0
    direction = np.sign(typical_price.diff())
    signed_volume = (volume * direction).where(direction.notna())

    fast_ema = signed_volume.ewm(span=fast_span, min_periods=fast_span, adjust=False).mean()
    slow_ema = signed_volume.ewm(span=slow_span, min_periods=slow_span, adjust=False).mean()
    kvo_diff = fast_ema - slow_ema
    factor = kvo_diff.ewm(span=signal_span, min_periods=signal_span, adjust=False).mean()

    return factor.replace([np.inf, -np.inf], np.nan)
