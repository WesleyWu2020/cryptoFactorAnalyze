"""Alpha101 Alpha#15 因子（factor_common 契约版）。

原始定义:
    Alpha#15 = -1 * rank(ts_sum(delta(close, 1), 3)) * rank(correlation(high, volume, 3))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    原脚本将 high 替换为 close，且 close/volume 先做 20 日相对归一化
    （normalization_method='relative_price'，基准均值 min_periods=1）:
    normalized_close = close / ts_mean(close, 20)
    normalized_volume = volume / ts_mean(volume, 20)
    price_delta_sum = ts_sum(delta(normalized_close, 1), 10)
    factor = -1 * (rank(price_delta_sum) * rank(correlation(normalized_volume, normalized_close, 10)))
参数取原脚本 __main__ 实际调用值: correlation_window=10, sum_window=10,
normalization_method='relative_price'（rebalance_period 仅用于旧版 future_ret，已丢弃）。
与经典公式的差异: high -> close；两个窗口由 3 改为 10；输入先相对归一化。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
公式内的横截面 rank 保留；最终去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha15_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #15: -rank(ts_sum(delta(norm_close,1),10)) * rank(corr(norm_volume, norm_close, 10))",
}

SETTING = {
    "data_needed": ["close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 35,
    "preprocessing": "mad_rank",
    "params": {"correlation_window": 10, "sum_window": 10, "norm_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#15 matrix (date x instrument)."""

    correlation_window = SETTING["params"]["correlation_window"]
    sum_window = SETTING["params"]["sum_window"]
    norm_window = SETTING["params"]["norm_window"]
    for name, value in (
        ("correlation_window", correlation_window),
        ("sum_window", sum_window),
        ("norm_window", norm_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 步骤1: 20 日相对归一化（原脚本基准均值用 min_periods=1，保留）
    normalized_close = close / (
        close.rolling(window=norm_window, min_periods=1).mean() + _EPSILON
    )
    normalized_volume = volume / (
        volume.rolling(window=norm_window, min_periods=1).mean() + _EPSILON
    )

    # 步骤2: 价格变化的滚动求和 ts_sum(delta(close, 1), sum_window)
    price_delta_sum = normalized_close.diff().rolling(
        window=sum_window, min_periods=sum_window
    ).sum()

    # 步骤3: 归一化量价时序相关性
    correlation = normalized_volume.rolling(
        window=correlation_window, min_periods=correlation_window
    ).corr(normalized_close)

    # 步骤4: 复合因子取反（rank 属公式本身故保留）
    return -1.0 * (price_delta_sum.rank(axis=1, pct=True) * correlation.rank(axis=1, pct=True))
