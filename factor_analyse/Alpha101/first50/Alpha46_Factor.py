"""Alpha101 Alpha#46 因子（factor_common 契约版）。

原始定义:
    Alpha#46 = ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10)
                - ((delay(close, 10) - close) / 10))) ? (-1 * 1)
                : (((((delay(close, 20) - delay(close, 10)) / 10)
                - ((delay(close, 10) - close) / 10)) < 0) ? 1
                : ((-1 * 1) * (close - delay(close, 1)))))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    price_base = mean(close, 20)（min_periods=1，滚动归一化基准）
    norm_close = close / price_base
    far_change_rate    = (delay(norm_close, 20) - delay(norm_close, 10)) / 10
    recent_change_rate = (delay(norm_close, 10) - norm_close) / 10
    trend_change = far_change_rate - recent_change_rate
    factor = -1                if trend_change > 0.25   （加速下跌）
             1                 if trend_change < 0      （加速上涨）
             -delta(norm_close, 1)  otherwise             （昨日归一化收益取负）
参数取原脚本 __main__ 实际调用值 long_delay=20, mid_delay=10；旧版
rebalance_period 仅用于 future_ret，不属于因子值计算，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha46_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #46: trend-acceleration switch on delayed normalized close",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"long_delay": 20, "mid_delay": 10, "norm_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#46 matrix (date x instrument)."""

    long_delay = SETTING["params"]["long_delay"]
    mid_delay = SETTING["params"]["mid_delay"]
    norm_window = SETTING["params"]["norm_window"]
    for name, value in (
        ("long_delay", long_delay),
        ("mid_delay", mid_delay),
        ("norm_window", norm_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 1")
    if long_delay <= mid_delay:
        raise ValueError("SETTING.params.long_delay must be greater than mid_delay")

    close = data_ctx["close"].astype("float64")

    # 价格归一化（适应币圈价格量级差异），与原脚本一致 min_periods=1
    price_base = close.rolling(window=norm_window, min_periods=1).mean()
    norm_close = close / (price_base + _EPSILON)

    # 远期变化率: (long_delay 天前到 mid_delay 天前的变化) / (long_delay - mid_delay)
    delay_close_long = norm_close.shift(long_delay)
    delay_close_mid = norm_close.shift(mid_delay)
    far_change_rate = (delay_close_long - delay_close_mid) / (long_delay - mid_delay)

    # 近期变化率: (mid_delay 天前到现在的变化) / mid_delay
    recent_change_rate = (delay_close_mid - norm_close) / mid_delay

    # 趋势变化 = 远期变化率 - 近期变化率
    trend_change = far_change_rate - recent_change_rate

    # 昨日归一化收益
    daily_return = norm_close - norm_close.shift(1)

    # 三层条件判断（trend_change 为 NaN 的行保持 NaN）
    factor = (-1.0 * daily_return).where(trend_change.notna())
    factor = factor.where(~(trend_change > 0.25), -1.0)
    factor = factor.where(~(trend_change < 0), 1.0)
    return factor
