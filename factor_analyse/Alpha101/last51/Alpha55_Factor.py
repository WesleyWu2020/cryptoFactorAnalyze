"""Alpha101 Alpha#55 因子（factor_common 契约版）。

原始定义:
    (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))),
                      rank(volume), 6))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    1. 12 日区间位置: pos = (close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12) + 1e-8)
    2. 横截面 rank: pos_rank = rank(pos, pct)，vol_rank = rank(volume, pct)（按日期）
    3. 因子值: -1 * correlation(pos_rank, vol_rank, 6)（每个 symbol 内 6 日滚动相关）

参数取原脚本 __main__ 实际调用值（pos_window=12, corr_window=6）。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口；
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha55_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #55: -1 * corr(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))), rank(volume), 6)",
}

SETTING = {
    "data_needed": ["close", "high", "low", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 25,
    "preprocessing": "mad_rank",
    "params": {"pos_window": 12, "corr_window": 6},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#55 matrix (date x instrument)."""

    pos_window = SETTING["params"]["pos_window"]
    corr_window = SETTING["params"]["corr_window"]

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 1. 12 日区间位置
    min_low = low.rolling(window=pos_window, min_periods=pos_window).min()
    max_high = high.rolling(window=pos_window, min_periods=pos_window).max()
    pos = (close - min_low) / (max_high - min_low + _EPSILON)

    # 2. 横截面 rank（按日期，pct=True）
    pos_rank = pos.rank(axis=1, pct=True)
    vol_rank = volume.rank(axis=1, pct=True)

    # 3. 每个 symbol 内 corr_window 日滚动相关并取负
    corr = pos_rank.rolling(window=corr_window, min_periods=corr_window).corr(vol_rank)
    return -1.0 * corr
