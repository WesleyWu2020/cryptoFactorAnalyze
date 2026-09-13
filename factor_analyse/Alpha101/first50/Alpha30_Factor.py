"""Alpha101 Alpha#30 因子（factor_common 契约版）。

原始定义:
    Alpha#30 = ((1.0 - rank(sign(close - delay(close,1)) + sign(delay(close,1) - delay(close,2))
                 + sign(delay(close,2) - delay(close,3)))) * sum(volume,5)) / sum(volume,20)

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    price_continuity = sign(close - delay(close,1)) + sign(delay(close,1) - delay(close,2))
                       + sign(delay(close,2) - delay(close,3))   # 连续3日价格方向之和
    volume_ratio     = sum(volume, volume_short_window) / sum(volume, volume_long_window)
    reversal_weight  = 1.0 - cross_section_rank_pct(price_continuity)
    factor           = reversal_weight * volume_ratio
参数取原脚本 __main__ 实际调用值 volume_short_window=10, volume_long_window=60
（rebalance_period 仅用于旧版 future_ret，丢弃）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成（公式内部的 rank 保留）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha30_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #30: (1 - cs_rank(sum of 3 daily sign(close diff))) * sum(volume,10)/sum(volume,60)",
}

SETTING = {
    "data_needed": ["close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 70,
    "preprocessing": "mad_rank",
    "params": {"volume_short_window": 10, "volume_long_window": 60},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#30 matrix (date x instrument)."""

    volume_short_window = SETTING["params"]["volume_short_window"]
    volume_long_window = SETTING["params"]["volume_long_window"]
    for name, value in (
        ("volume_short_window", volume_short_window),
        ("volume_long_window", volume_long_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 1-2. 连续3日价格变化方向之和（取值 -3 ~ +3）
    price_continuity = (
        np.sign(close - close.shift(1))
        + np.sign(close.shift(1) - close.shift(2))
        + np.sign(close.shift(2) - close.shift(3))
    )

    # 3-5. 短期/长期交易量比率
    volume_short = volume.rolling(window=volume_short_window, min_periods=volume_short_window).sum()
    volume_long = volume.rolling(window=volume_long_window, min_periods=volume_long_window).sum()
    volume_ratio = volume_short / (volume_long + _EPSILON)

    # 6. 反向权重：价格连续性越强权重越低（公式内部的横截面 rank 保留）
    reversal_weight = 1.0 - price_continuity.rank(axis=1, pct=True)

    return reversal_weight * volume_ratio
