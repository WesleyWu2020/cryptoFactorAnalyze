"""Alpha101 Alpha#11 因子（factor_common 契约版）。

原始定义:
    Alpha#11 = (rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    vwap = (high + low + close) / 3   # 原脚本无 vwap 字段，用典型价近似
    price_deviation = vwap - close
    factor = (rank(ts_max(price_deviation, 20)) + rank(ts_min(price_deviation, 20)))
             * rank(delta(volume, 20))
参数取原脚本 __main__ 实际调用值: ts_window=20, volume_delta_window=20
（rebalance_period 仅用于旧版 future_ret，已丢弃）。
与经典公式的差异: 窗口由 3 改为 20，且 vwap 用 (high+low+close)/3 近似。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
公式内部的横截面 rank 保留（axis=1 pct rank）；最终去极值与秩归一化由框架
preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha11_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #11: (rank(ts_max(vwap-close,N)) + rank(ts_min(vwap-close,N))) * rank(delta(volume,N))",
}

SETTING = {
    "data_needed": ["high", "low", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 25,
    "preprocessing": "mad_rank",
    "params": {"ts_window": 20, "volume_delta_window": 20},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#11 matrix (date x instrument)."""

    ts_window = SETTING["params"]["ts_window"]
    volume_delta_window = SETTING["params"]["volume_delta_window"]
    for name, value in (("ts_window", ts_window), ("volume_delta_window", volume_delta_window)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 1")

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 步骤1: 典型价近似 vwap，计算价格偏离
    vwap = (high + low + close) / 3.0
    price_deviation = vwap - close

    # 步骤2: 时序极值
    ts_max_dev = price_deviation.rolling(window=ts_window, min_periods=ts_window).max()
    ts_min_dev = price_deviation.rolling(window=ts_window, min_periods=ts_window).min()

    # 步骤3: 交易量变化 delta(volume, volume_delta_window)
    volume_delta = volume - volume.shift(volume_delta_window)

    # 步骤4: 公式内横截面 rank（保留），复合后输出原始值
    return (
        ts_max_dev.rank(axis=1, pct=True) + ts_min_dev.rank(axis=1, pct=True)
    ) * volume_delta.rank(axis=1, pct=True)
