"""Alpha101 Alpha#48 因子（factor_common 契约版）。

原始定义:
    Alpha#48 = (indneutralize(((correlation(delta(close, 1),
                delta(delay(close, 1), 1), 250) * delta(close, 1)) / close),
                IndClass.subindustry)
                / sum(((delta(close, 1) / delay(close, 1))^2), 250))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（移除行业中性化）:
    price_base = mean(close, 20)（min_periods=1，滚动归一化基准）
    norm_close = close / price_base
    delta_close = diff(norm_close, 1)
    delta_delay_close = diff(delay(norm_close, 1), 1)（等价于 delay(delta_close, 1)）
    pct_change = delta_close / delay(norm_close, 1)
    corr = correlation(delta_close, delta_delay_close, corr_window)（收益一阶自相关）
    factor = (corr * delta_close / norm_close) / sum(pct_change^2, corr_window)

参数取原脚本 __main__ 实际调用值 corr_window=20（注意 __main__ 覆盖了函数
签名默认的 250）；旧版 rebalance_period 仅用于 future_ret，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha48_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #48: return autocorrelation * return / volatility-sum normalization",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"corr_window": 20, "norm_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#48 matrix (date x instrument)."""

    corr_window = SETTING["params"]["corr_window"]
    norm_window = SETTING["params"]["norm_window"]
    for name, value in (("corr_window", corr_window), ("norm_window", norm_window)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    close = data_ctx["close"].astype("float64")

    # 价格归一化（适应币圈价格量级差异），与原脚本一致 min_periods=1
    price_base = close.rolling(window=norm_window, min_periods=1).mean()
    norm_close = close / (price_base + _EPSILON)

    # 今日/昨日归一化收益与百分比收益率
    delta_close = norm_close.diff()
    delay_close = norm_close.shift(1)
    delta_delay_close = delay_close.diff()
    pct_change = delta_close / (delay_close + _EPSILON)

    # 收益率平方和（波动率标准化项）
    sum_pct_change_squared = (pct_change ** 2).rolling(
        window=corr_window, min_periods=corr_window
    ).sum()

    # 收益序列一阶自相关（沿时间轴 rolling corr）
    return_correlation = delta_close.rolling(
        window=corr_window, min_periods=corr_window
    ).corr(delta_delay_close)

    numerator = return_correlation * delta_close / (norm_close + _EPSILON)
    return numerator / (sum_pct_change_squared + _EPSILON)
