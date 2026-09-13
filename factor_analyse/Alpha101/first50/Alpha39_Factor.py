"""Alpha101 Alpha#39 因子（factor_common 契约版）。

原始定义:
    Alpha#39 = ((-1 * rank(delta(close, 7) * (1 - rank(decay_linear(volume/adv20, 9)))))
                * (1 + rank(sum(returns, 250))))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    price_base = mean(close, 20)（min_periods=1）
    norm_close = close / price_base
    delta_short = delta(norm_close, delta_window)
    vol_ratio = volume / mean(volume, adv_window)
    vol_decay = decay_linear(vol_ratio, decay_window)（权重 1..N，最近权重最大，窗口含 NaN 则 NaN）
    sum_ret_long = sum(pct_change(norm_close), long_ret_window)
    factor = (-rank(delta_short * (1 - rank(vol_decay)))) * (1 + rank(sum_ret_long))
其中 rank 为横截面 pct rank（axis=1）。
参数取原脚本 __main__ 实际调用值: delta=10, adv=10, decay=10, long_ret=90。
旧脚本 rebalance_period 仅用于旧版 future_ret，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha39_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #39: -rank(delta(close,N)*(1-rank(decay_linear(vol/adv,N)))) * (1+rank(sum(ret,N)))",
}

SETTING = {
    "data_needed": ["close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 100,
    "preprocessing": "mad_rank",
    "params": {
        "delta_window": 10,
        "adv_window": 10,
        "decay_window": 10,
        "long_ret_window": 90,
    },
    "factor_direction": 1,
}

_EPSILON = 1e-8


def _decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Linearly weighted rolling mean (weights 1..window, newest heaviest), causal.

    NaN anywhere inside the window propagates to NaN, matching the legacy
    implementation which returned NaN when the window contained NaN.
    """
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: float(np.dot(arr, weights)), raw=True
    )


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#39 matrix (date x instrument)."""

    delta_window = SETTING["params"]["delta_window"]
    adv_window = SETTING["params"]["adv_window"]
    decay_window = SETTING["params"]["decay_window"]
    long_ret_window = SETTING["params"]["long_ret_window"]

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 相对价格基准（缓解不同币种量纲差异），与原脚本一致 min_periods=1
    price_base = close.rolling(window=20, min_periods=1).mean()
    norm_close = close / (price_base + _EPSILON)

    # 短期动量: delta(norm_close, delta_window)
    delta_short = norm_close - norm_close.shift(delta_window)

    # 交易量确认: volume / adv -> 线性衰减均值
    adv = volume.rolling(window=adv_window, min_periods=adv_window).mean()
    vol_ratio = volume / (adv + _EPSILON)
    vol_decay = _decay_linear(vol_ratio, decay_window)

    # 长期收益: sum(returns, long_ret_window)
    returns = norm_close.pct_change()
    sum_ret_long = returns.rolling(window=long_ret_window, min_periods=long_ret_window).sum()

    # 横截面组合（公式内部 rank 保留，最终归一化由 preprocessing 完成）
    decay_rank = vol_decay.rank(axis=1, pct=True)
    sumret_rank = sum_ret_long.rank(axis=1, pct=True)
    inner = delta_short * (1.0 - decay_rank)
    short_component = -1.0 * inner.rank(axis=1, pct=True)
    long_component = 1.0 + sumret_rank
    return short_component * long_component
