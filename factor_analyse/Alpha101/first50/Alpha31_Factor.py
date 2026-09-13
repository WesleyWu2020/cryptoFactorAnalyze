"""Alpha101 Alpha#31 因子（factor_common 契约版）。

原始定义:
    Alpha#31 = rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10))))
               + rank((-1 * delta(close, 3)))
               + sign(scale(correlation(adv20, low, 12)))

本实现保留原脚本“币圈7×24h优化版”的实际计算结构（长期趋势 + 短期动量 + 流动性/支撑）:
    项1: rank(rank(rank(decay_linear(-1 * rank(rank(delta(close, delta_long_window))), decay_window))))
         （所有 rank 均为当日横截面 pct 秩；decay_linear 为时间序列加权衰减，权重 1..d）
    项2: rank(-1 * delta(close, delta_short_window))（横截面 pct 秩）
    项3: sign(scale(correlation(mean(volume, adv_window), low, corr_window)))
         （scale 实现为当日横截面 z-score；std 为 0/NaN 时退化为中心化，sign 后为 0）
    factor = 项1 + 项2 + 项3
参数取原脚本 __main__ 实际调用值 delta_long_window=20, decay_window=20,
delta_short_window=5, adv_window=20, corr_window=18（rebalance_period 仅用于旧版 future_ret，丢弃）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成（公式内部的 rank 保留）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha31_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #31: rank^3(decay_linear(-rank^2(delta(close,20)),20)) + rank(-delta(close,5)) + sign(scale(corr(adv20,low,18)))",
}

SETTING = {
    "data_needed": ["close", "low", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 45,
    "preprocessing": "mad_rank",
    "params": {
        "delta_long_window": 20,
        "decay_window": 20,
        "delta_short_window": 5,
        "adv_window": 20,
        "corr_window": 18,
    },
    "factor_direction": 1,
}


def _decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling linearly-decayed weighted average (weights 1..d, recent heaviest), causal."""
    weights = np.arange(1, window + 1, dtype=float)
    weight_sum = weights.sum()
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: float(np.dot(arr, weights) / weight_sum), raw=True
    )


def _cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional (per-date) z-score; degenerate std degenerates to centering."""
    mean = x.mean(axis=1)
    std = x.std(axis=1).replace(0.0, np.nan)
    return x.sub(mean, axis=0).div(std, axis=0)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#31 matrix (date x instrument)."""

    params = SETTING["params"]
    delta_long_window = params["delta_long_window"]
    decay_window = params["decay_window"]
    delta_short_window = params["delta_short_window"]
    adv_window = params["adv_window"]
    corr_window = params["corr_window"]
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    low = data_ctx["low"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # === 项1：长期趋势的横截面秩 + 时间序列衰减 + 多层横截面秩 ===
    delta_long = close.diff(delta_long_window)
    neg_rank2_delta_long = -1.0 * delta_long.rank(axis=1, pct=True).rank(axis=1, pct=True)
    decay = _decay_linear(neg_rank2_delta_long, decay_window)
    rank_decay = decay.rank(axis=1, pct=True).rank(axis=1, pct=True).rank(axis=1, pct=True)

    # === 项2：短期动量反转的横截面秩 ===
    delta_short = close.diff(delta_short_window)
    rank_neg_delta_short = (-1.0 * delta_short).rank(axis=1, pct=True)

    # === 项3：流动性与最低价相关性的横截面 scale 后取符号 ===
    adv = volume.rolling(window=adv_window, min_periods=adv_window).mean()
    corr_adv_low = adv.rolling(window=corr_window, min_periods=corr_window).corr(low)
    corr_sign = np.sign(_cs_zscore(corr_adv_low).fillna(0.0))

    # 合成（任一项为 NaN 则合成结果为 NaN，与旧脚本 dropna 行为一致）
    return rank_decay + rank_neg_delta_short + corr_sign
