"""Alpha101 Alpha#43 因子（factor_common 契约版）。

原始定义:
    Alpha#43 = (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    price_base = mean(close, 20)（min_periods=1）
    norm_close = close / price_base
    vol_ratio = volume / mean(volume, adv_window)
    neg_delta_close = -delta(norm_close, delta_window)
    factor = ts_rank(vol_ratio, vol_ts_rank_window) * ts_rank(neg_delta_close, price_ts_rank_window)
ts_rank 为窗口内最后一个值的百分比排名，窗口含 NaN 则结果为 NaN（与原脚本一致）。
参数取原脚本 __main__ 实际调用值: adv=20, vol_ts_rank=20, price_ts_rank=8, delta=7。
旧脚本 rebalance_period 仅用于旧版 future_ret，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha43_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #43: ts_rank(volume/adv, N) * ts_rank(-delta(close, N), N)",
}

SETTING = {
    "data_needed": ["close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 45,
    "preprocessing": "mad_rank",
    "params": {
        "adv_window": 20,
        "vol_ts_rank_window": 20,
        "price_ts_rank_window": 8,
        "delta_window": 7,
    },
    "factor_direction": 1,
}

_EPSILON = 1e-8


def _ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling pct-rank of the newest value within the window, causal.

    Returns NaN when the window contains NaN, matching the legacy implementation.
    """

    def rank_last(arr: pd.Series) -> float:
        if arr.isna().any():
            return np.nan
        return float(arr.rank(pct=True).iloc[-1])

    return x.rolling(window=window, min_periods=window).apply(rank_last, raw=False)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#43 matrix (date x instrument)."""

    adv_window = SETTING["params"]["adv_window"]
    vol_ts_rank_window = SETTING["params"]["vol_ts_rank_window"]
    price_ts_rank_window = SETTING["params"]["price_ts_rank_window"]
    delta_window = SETTING["params"]["delta_window"]

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 价格归一化（适应币圈价格差异），与原脚本一致 min_periods=1
    price_base = close.rolling(window=20, min_periods=1).mean()
    norm_close = close / (price_base + _EPSILON)

    # 交易量比率 volume / adv
    adv = volume.rolling(window=adv_window, min_periods=adv_window).mean()
    vol_ratio = volume / (adv + _EPSILON)

    # 价格变化负值 -1 * delta(norm_close, delta_window)
    neg_delta_close = -1.0 * (norm_close - norm_close.shift(delta_window))

    # 时间序列排名相乘
    vol_ts_rank = _ts_rank(vol_ratio, vol_ts_rank_window)
    price_ts_rank = _ts_rank(neg_delta_close, price_ts_rank_window)
    return vol_ts_rank * price_ts_rank
