"""Alpha101 Alpha#14 因子（factor_common 契约版）。

原始定义:
    Alpha#14 = (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    returns 用对数收益率；open/volume 先做 20 日相对归一化（relative_price）:
    returns = log(close / delay(close, 1))
    returns_delta = delta(returns, 5)
    normalized_open = open / ts_mean(open, 20)
    normalized_volume = volume / ts_mean(volume, 20)
    factor = (-1 * rank(returns_delta)) * correlation(normalized_open, normalized_volume, 10)
参数取原脚本 __main__ 实际调用值（策略5为唯一启用项）:
    delta_window=5, correlation_window=10, normalization_method='relative_price'
（rebalance_period 仅用于旧版 future_ret，已丢弃）。
与经典公式的差异: returns 用对数收益率；delta 窗口由 3 改为 5；相关性输入先相对归一化。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
公式内的横截面 rank 保留；最终去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha14_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #14: -rank(delta(log_returns,5)) * corr(norm_open, norm_volume, 10)",
}

SETTING = {
    "data_needed": ["open", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 35,
    "preprocessing": "mad_rank",
    "params": {"delta_window": 5, "correlation_window": 10, "norm_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#14 matrix (date x instrument)."""

    delta_window = SETTING["params"]["delta_window"]
    correlation_window = SETTING["params"]["correlation_window"]
    norm_window = SETTING["params"]["norm_window"]
    for name, value in (
        ("delta_window", delta_window),
        ("correlation_window", correlation_window),
        ("norm_window", norm_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    open_ = data_ctx["open"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 步骤1: 对数收益率及其变化 delta(returns, delta_window)
    returns = np.log(close / close.shift(1))
    returns_delta = returns - returns.shift(delta_window)

    # 步骤2: 20 日相对归一化
    normalized_open = open_ / (open_.rolling(window=norm_window, min_periods=norm_window).mean() + _EPSILON)
    normalized_volume = volume / (volume.rolling(window=norm_window, min_periods=norm_window).mean() + _EPSILON)

    # 步骤3: 价量时序相关性
    correlation = normalized_open.rolling(
        window=correlation_window, min_periods=correlation_window
    ).corr(normalized_volume)

    # 步骤4: 复合因子（rank 属公式本身故保留）
    return (-1.0 * returns_delta.rank(axis=1, pct=True)) * correlation
