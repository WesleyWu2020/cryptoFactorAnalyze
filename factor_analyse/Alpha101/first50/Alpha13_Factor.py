"""Alpha101 Alpha#13 因子（factor_common 契约版）。

原始定义:
    Alpha#13 = -1 * rank(covariance(rank(close), rank(volume), 5))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    原脚本对价格和交易量先做 20 日相对归一化（normalization_method='relative_price'），
    且未对归一化序列再做公式中的内层 rank:
    normalized_close = close / ts_mean(close, 20)
    normalized_volume = volume / ts_mean(volume, 20)
    factor = -1 * rank(covariance(normalized_close, normalized_volume, 20))
参数取原脚本 __main__ 实际调用值: covariance_window=20, normalization_method='relative_price'
（rebalance_period 仅用于旧版 future_ret，已丢弃）。
与经典公式的差异: 协方差窗口由 5 改为 20；输入先相对归一化；省略内层 rank(close)/rank(volume)。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
外层 rank 是公式的一部分故保留；最终去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha13_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #13: -rank(cov(close/ts_mean(close,20), volume/ts_mean(volume,20), N))",
}

SETTING = {
    "data_needed": ["close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 45,
    "preprocessing": "mad_rank",
    "params": {"covariance_window": 20, "norm_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#13 matrix (date x instrument)."""

    covariance_window = SETTING["params"]["covariance_window"]
    norm_window = SETTING["params"]["norm_window"]
    for name, value in (
        ("covariance_window", covariance_window),
        ("norm_window", norm_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 步骤1: 20 日相对归一化（relative_price）
    normalized_close = close / (close.rolling(window=norm_window, min_periods=norm_window).mean() + _EPSILON)
    normalized_volume = volume / (volume.rolling(window=norm_window, min_periods=norm_window).mean() + _EPSILON)

    # 步骤2: 归一化价格与交易量的时序协方差
    covariance = normalized_close.rolling(
        window=covariance_window, min_periods=covariance_window
    ).cov(normalized_volume)

    # 步骤3: 信号反转，外层 rank 属公式本身故保留
    return -1.0 * covariance.rank(axis=1, pct=True)
