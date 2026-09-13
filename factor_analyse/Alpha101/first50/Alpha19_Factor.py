"""Alpha101 Alpha#19 因子（factor_common 契约版）。

原始定义:
    Alpha#19 = ((-1 * sign(((close - delay(close, 7)) + delta(close, 7))))
                * (1 + rank((1 + sum(returns, 250)))))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    reversal        = -1 * sign((close - delay(close, d)) + delta(close, d))
    long_strength   = 1 + cross_sectional_rank(1 + ts_sum(returns, w))
    factor          = reversal * long_strength

参数取原脚本 __main__ 实际调用值: delay_window=5, returns_sum_window=60。
原脚本的 normalization_method 仅生成未被因子使用的 normalized_close 列，丢弃；
rebalance_period 只用于旧版 future_ret，不属于因子值计算，丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
公式内部的横截面 rank 保留；最终去极值与归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha19_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #19: -sign(delta(close,5)*2) * (1 + rank(1 + ts_sum(returns, 60)))",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 70,
    "preprocessing": "mad_rank",
    "params": {"delay_window": 5, "returns_sum_window": 60},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#19 matrix (date x instrument)."""

    delay_window = SETTING["params"]["delay_window"]
    returns_sum_window = SETTING["params"]["returns_sum_window"]

    close = data_ctx["close"].astype("float64")
    returns = close.pct_change()

    # 短期价格方向反转信号: (close - delay(close,d)) 与 delta(close,d) 等价，求和后取 sign 取反
    composite_change = (close - close.shift(delay_window)) + (close - close.shift(delay_window))
    reversal_signal = -1.0 * np.sign(composite_change)

    # 长期累积收益强度调整: 1 + rank(1 + ts_sum(returns, w))（公式内部 rank 保留）
    cumulative_returns = returns.rolling(
        window=returns_sum_window, min_periods=returns_sum_window
    ).sum()
    long_term_strength = 1.0 + (1.0 + cumulative_returns).rank(axis=1, pct=True)

    return reversal_signal * long_term_strength
