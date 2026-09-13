"""Alpha101 Alpha#16 因子（factor_common 契约版）。

原始定义:
    Alpha#16 = -1 * rank(covariance(rank(high), rank(volume), 5))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    原脚本 __main__ 启用 normalization_method='market_cap'（用成交额相对规模近似市值加权），
    且未对归一化序列再做公式中的内层 rank:
    normalized_high = high * volume / ts_mean(volume, 20)
    normalized_volume = volume / ts_mean(volume, 20)
    factor = -1 * rank(covariance(normalized_high, normalized_volume, 10))
参数取原脚本 __main__ 实际调用值: covariance_window=10, normalization_method='market_cap'
（rebalance_period 仅用于旧版 future_ret，已丢弃）。
与经典公式的差异: 协方差窗口由 5 改为 10；输入先市值加权归一化；省略内层 rank(high)/rank(volume)。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
外层 rank 是公式的一部分故保留；最终去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha16_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #16: -rank(cov(high*volume/ts_mean(volume,20), volume/ts_mean(volume,20), 10))",
}

SETTING = {
    "data_needed": ["high", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 35,
    "preprocessing": "mad_rank",
    "params": {"covariance_window": 10, "norm_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#16 matrix (date x instrument)."""

    covariance_window = SETTING["params"]["covariance_window"]
    norm_window = SETTING["params"]["norm_window"]
    for name, value in (
        ("covariance_window", covariance_window),
        ("norm_window", norm_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    high = data_ctx["high"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 步骤1: market_cap 归一化（成交额相对规模近似市值加权）
    volume_base = volume.rolling(window=norm_window, min_periods=norm_window).mean() + _EPSILON
    normalized_high = high * volume / volume_base
    normalized_volume = volume / volume_base

    # 步骤2: 归一化最高价与交易量的时序协方差
    covariance = normalized_high.rolling(
        window=covariance_window, min_periods=covariance_window
    ).cov(normalized_volume)

    # 步骤3: 信号反转，外层 rank 属公式本身故保留
    return -1.0 * covariance.rank(axis=1, pct=True)
