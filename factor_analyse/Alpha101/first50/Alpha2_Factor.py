"""Alpha101 Alpha#2 因子（factor_common 契约版）。

原始定义:
    Alpha#2 = -1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)

本实现保留原脚本的实际计算逻辑:
    volume_delta    = log(volume).diff(delta_window)          # 交易量对数变化
    intraday_return = (close - open) / open                   # 日内收益率
    factor          = -1 * correlation(rank(volume_delta), rank(intraday_return), correlation_window)
其中 rank 为按日横截面百分比秩（公式内部项，保留）；log(volume) 对 volume<=0 的
交易日掩码为 NaN（原脚本 log(0)=-inf 会污染相关性窗口，此处视为不可观测）。

参数取原脚本 ``__main__`` 实际调用值: delta_window=3, correlation_window=7
（rebalance_period 仅用于旧版 future_ret，已丢弃）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha2_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #2: -1 * correlation(rank(delta(log(volume),3)), rank((close-open)/open), 7)",
}

SETTING = {
    "data_needed": ["open", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 15,
    "preprocessing": "mad_rank",
    "params": {"delta_window": 3, "correlation_window": 7},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#2 matrix (date x instrument)."""

    delta_window = SETTING["params"]["delta_window"]
    correlation_window = SETTING["params"]["correlation_window"]
    for name, value in (("delta_window", delta_window), ("correlation_window", correlation_window)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 1")

    open_ = data_ctx["open"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 步骤1: 交易量对数变化 delta(log(volume), delta_window)；无成交日不可观测
    log_volume = np.log(volume.where(volume > 0))
    volume_delta = log_volume - log_volume.shift(delta_window)

    # 步骤2: 日内收益率 (close - open) / open
    intraday_return = (close - open_) / open_

    # 步骤3: 按日横截面百分比秩（公式内部 rank 项）
    volume_delta_rank = volume_delta.rank(axis=1, pct=True)
    intraday_return_rank = intraday_return.rank(axis=1, pct=True)

    # 步骤4: 过去 correlation_window 日两秩序列的时序相关性，方向反转
    correlation = volume_delta_rank.rolling(
        window=correlation_window, min_periods=correlation_window
    ).corr(intraday_return_rank)
    return -1.0 * correlation
