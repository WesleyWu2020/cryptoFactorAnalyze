"""crypto_alpha_vmacd_volume 因子（factor_common 日频契约版）。

原始定义（分钟版语义）：
    daily_volume = sum(volume, 1d)
    DIF = EMA(daily_volume, span=5) - EMA(daily_volume, span=60)
    DEA = EMA(DIF, span=5)
    factor = DIF - DEA  （成交量 MACD 柱）

无 ITER_NOTE/方向实测记录，factor_direction 默认 1（高因子值=量能 MACD 柱走强为优选）。

日频改写判断：
- 分钟级 resample("1440min").sum() 聚合直接由日频 volume 字段替代。
- 删除末尾 shift(1)（原仅用于执行延迟，框架按次日开盘执行，不人为延迟）。
- 删除广播回分钟 index 与 CHUNK_SIZE 分块逻辑，直接返回日频矩阵。
- EMA span 语义不变（窗口单位由分钟棒改为日棒，与原日聚合后再 EMA 的口径一致）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_vmacd_volume",
    "author": "wesleywu",
    "level": "daily",
    "category": "volume_price",
    "description": "成交量 MACD 柱：(EMA5-EMA60) DIF 减 EMA5 DEA（日成交量）",
}

SETTING = {
    "data_needed": ["volume"],
    "universe": "historical_top50",
    "warmup_bars": 75,
    "preprocessing": "mad_rank",
    "params": {"fast_span": 5, "slow_span": 60, "signal_span": 5},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily volume-MACD histogram matrix."""

    fast_span = SETTING["params"]["fast_span"]
    slow_span = SETTING["params"]["slow_span"]
    signal_span = SETTING["params"]["signal_span"]

    daily_volume = data_ctx["volume"].astype("float64").clip(lower=0)

    fast_ema = daily_volume.ewm(span=fast_span, min_periods=fast_span, adjust=False).mean()
    slow_ema = daily_volume.ewm(span=slow_span, min_periods=slow_span, adjust=False).mean()
    diff = fast_ema - slow_ema
    dea = diff.ewm(span=signal_span, min_periods=signal_span, adjust=False).mean()
    factor = diff - dea
    return factor.replace([np.inf, -np.inf], np.nan)
