"""Alpha101 Alpha#17 因子（factor_common 契约版）。

原始定义:
    Alpha#17 = ((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)))
               * rank(ts_rank((volume / adv20(volume)), 5))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    factor = (-1 * rank(ts_rank(close, 15)))
             * rank(delta(delta(close, 2), 2))
             * rank(ts_rank(volume / ts_mean(volume, 30), 7))
参数取原脚本 __main__ 实际调用值（策略3为唯一启用项）:
    ts_rank_window=15, delta_window=2, volume_rank_window=7, adv_window=30,
    normalization_method='market_cap'
（rebalance_period 仅用于旧版 future_ret，已丢弃）。
与经典公式的差异: 窗口由 (10,1,5,20) 改为 (15,2,7,30)。
注意: 原脚本虽计算 normalized_close/normalized_volume，但因子三条腿全部使用原始
close/volume，归一化结果未参与因子值，故本实现不再计算归一化（normalization_method
参数对因子无影响，仅保留在 params 中以记录原调用）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
公式内的横截面 rank 保留；最终去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha17_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #17: -rank(ts_rank(close,15)) * rank(ddelta(close,2)) * rank(ts_rank(volume/adv30,7))",
}

SETTING = {
    "data_needed": ["close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 42,
    "preprocessing": "mad_rank",
    "params": {
        "ts_rank_window": 15,
        "delta_window": 2,
        "volume_rank_window": 7,
        "adv_window": 30,
        "normalization_method": "market_cap",
    },
    "factor_direction": 1,
}

_EPSILON = 1e-8


def _ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series rank of the last value in each window (pct), causal."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: pd.Series(arr).rank(pct=True).iloc[-1], raw=False
    )


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#17 matrix (date x instrument)."""

    ts_rank_window = SETTING["params"]["ts_rank_window"]
    delta_window = SETTING["params"]["delta_window"]
    volume_rank_window = SETTING["params"]["volume_rank_window"]
    adv_window = SETTING["params"]["adv_window"]
    for name, value in (
        ("ts_rank_window", ts_rank_window),
        ("delta_window", delta_window),
        ("volume_rank_window", volume_rank_window),
        ("adv_window", adv_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 腿1: 价格强度反转信号 -1 * rank(ts_rank(close, ts_rank_window))
    ts_rank_close = _ts_rank(close, ts_rank_window)
    leg_price_strength = -1.0 * ts_rank_close.rank(axis=1, pct=True)

    # 腿2: 价格加速度排名 rank(delta(delta(close, delta_window), delta_window))
    delta_delta_close = close.diff(delta_window).diff(delta_window)
    leg_price_accel = delta_delta_close.rank(axis=1, pct=True)

    # 腿3: 成交量相对强度排名 rank(ts_rank(volume / adv, volume_rank_window))
    adv = volume.rolling(window=adv_window, min_periods=adv_window).mean()
    volume_adv_ratio = volume / (adv + _EPSILON)
    ts_rank_volume_adv = _ts_rank(volume_adv_ratio, volume_rank_window)
    leg_volume_strength = ts_rank_volume_adv.rank(axis=1, pct=True)

    # 复合因子: 三条腿相乘
    return leg_price_strength * leg_price_accel * leg_volume_strength
