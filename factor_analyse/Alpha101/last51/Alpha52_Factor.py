"""Alpha101 Alpha#52 因子（factor_common 契约版）。

原始定义:
    ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) *
     ts_rank(volume, 5))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准，与经典公式存在偏离）:
    1. 价格归一化: price_base = mean(close, 20)（min_periods=1），norm_low = low / price_base，
       norm_close = close / price_base（横截面 rank 由框架完成，归一化主要影响 ts_min 与 returns）
    2. 技术面反转信号: (-1 * ts_min(norm_low, 7)) + delay(ts_min(norm_low, 7), 7)
    3. 基本面信号: (sum(returns, 140) - sum(returns, 21)) / (140 - 21)
       （经典公式为 240/20 再取截面 rank；原脚本用 140/21 且未做截面 rank，保留原实现）
    4. 交易量确认: ts_rank(volume, 14)
    5. 因子值: 三部分相乘

参数取原脚本 __main__ 实际调用值。计算只使用当日及历史数据，无未来函数。
FactorManager 只调用下面的标准模块接口；横截面去极值与秩归一化由框架
preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha52_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": (
        "Alpha101 #52: ((-ts_min(low,7) + delay(ts_min(low,7),7)) "
        "* (sum(returns,140)-sum(returns,21))/119 * ts_rank(volume,14))"
    ),
}

SETTING = {
    "data_needed": ["low", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 150,
    "preprocessing": "mad_rank",
    "params": {
        "ts_min_window": 7,
        "delay_days": 7,
        "long_window": 140,
        "short_window": 21,
        "volume_ts_rank_window": 14,
        "norm_window": 20,
    },
    "factor_direction": 1,
}

_EPSILON = 1e-8


def _ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series rank of the last observation (pct), causal."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: pd.Series(arr).rank(pct=True).iloc[-1], raw=False
    )


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#52 matrix (date x instrument)."""

    ts_min_window = SETTING["params"]["ts_min_window"]
    delay_days = SETTING["params"]["delay_days"]
    long_window = SETTING["params"]["long_window"]
    short_window = SETTING["params"]["short_window"]
    volume_ts_rank_window = SETTING["params"]["volume_ts_rank_window"]
    norm_window = SETTING["params"]["norm_window"]

    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 1. 价格归一化（与原脚本一致：min_periods=1 的 20 日均价基准）
    price_base = close.rolling(window=norm_window, min_periods=1).mean()
    norm_low = low / (price_base + _EPSILON)
    norm_close = close / (price_base + _EPSILON)

    # 2. 技术面反转信号: (-1 * ts_min(norm_low, W)) + delay(ts_min(norm_low, W), D)
    ts_min_low = norm_low.rolling(window=ts_min_window, min_periods=ts_min_window).min()
    technical_signal = (-1.0 * ts_min_low) + ts_min_low.shift(delay_days)

    # 3. 基本面信号: (sum(returns, L) - sum(returns, S)) / (L - S)
    returns = norm_close.pct_change()
    long_sum_returns = returns.rolling(window=long_window, min_periods=long_window).sum()
    short_sum_returns = returns.rolling(window=short_window, min_periods=short_window).sum()
    fundamental_signal = (long_sum_returns - short_sum_returns) / (long_window - short_window)

    # 4. 交易量时间序列排名
    volume_ts_rank = _ts_rank(volume, volume_ts_rank_window)

    # 5. 三部分相乘
    return technical_signal * fundamental_signal * volume_ts_rank
