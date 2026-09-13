"""Alpha101 Alpha#40 因子（factor_common 契约版）。

原始定义:
    Alpha#40 = (-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    price_base = mean(close, 20)（min_periods=1）
    high_normalized = high / price_base
    volume_normalized = volume / mean(volume, 20)（min_periods=1）
    high_std = std(high_normalized, std_window)
    high_vol_corr = correlation(high_normalized, volume_normalized, corr_window)
    factor = -rank(high_std) * high_vol_corr

偏离说明: 旧脚本对 high_std 的 rank 是在单 symbol 全历史时间序列上做的
（依赖未来数据，存在前视），本实现改为横截面 pct rank（axis=1），
与经典 Alpha101 rank() 语义一致且无前视。
参数取原脚本 __main__ 实际调用值: std_window=10, corr_window=20。
旧脚本 rebalance_period 仅用于旧版 future_ret，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha40_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #40: -rank(stddev(high, N)) * correlation(high, volume, N)",
}

SETTING = {
    "data_needed": ["high", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 25,
    "preprocessing": "mad_rank",
    "params": {"std_window": 10, "corr_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#40 matrix (date x instrument)."""

    std_window = SETTING["params"]["std_window"]
    corr_window = SETTING["params"]["corr_window"]

    high = data_ctx["high"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 币圈价格/成交量归一化（与原脚本一致 min_periods=1）
    price_base = close.rolling(window=20, min_periods=1).mean()
    high_normalized = high / (price_base + _EPSILON)
    volume_normalized = volume / (volume.rolling(window=20, min_periods=1).mean() + _EPSILON)

    # 最高价波动率 与 价量相关性
    high_std = high_normalized.rolling(window=std_window, min_periods=std_window).std()
    high_vol_corr = high_normalized.rolling(window=corr_window, min_periods=corr_window).corr(
        volume_normalized
    )

    # 低波动性得高分（负号），乘以价量相关性；rank 为横截面 pct rank
    high_std_rank = high_std.rank(axis=1, pct=True)
    return -1.0 * high_std_rank * high_vol_corr
