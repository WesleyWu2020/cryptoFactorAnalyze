"""Alpha101 Alpha#8 因子（factor_common 契约版，币圈7×24h优化版）。

原始定义:
    Alpha#8 = -1 * rank((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（``__main__`` 实际调用
sum_window=3, delay_window=7, price_normalization='relative_price'）:
    returns         = close.pct_change()
    normalized_open = open / mean(close, 20)   # 注意: 原脚本基准为 close 的 20 日均值
    signal          = sum(normalized_open, sum_window) * sum(returns, sum_window)
    factor          = -1 * rank(signal - delay(signal, delay_window))
其中外层 rank 为按日横截面百分比秩（公式内部项，保留）。价格归一化用于适应币圈
巨大的价格量级差异；'log_price' / 'z_score' 等未启用分支未迁移。
relative_price 基准与原脚本一致采用 min_periods=1。

rebalance_period 仅用于旧版 future_ret，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha8_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": (
        "Alpha101 #8 (crypto): -1 * rank(delta(sum(open/mean(close,20),3) * sum(returns,3), 7))"
    ),
}

SETTING = {
    "data_needed": ["open", "close"],
    "universe": "historical_top50",
    "warmup_bars": 16,
    "preprocessing": "mad_rank",
    "params": {"sum_window": 3, "delay_window": 7, "price_base_window": 20},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#8 matrix (date x instrument)."""

    sum_window = SETTING["params"]["sum_window"]
    delay_window = SETTING["params"]["delay_window"]
    price_base_window = SETTING["params"]["price_base_window"]
    for name, value in (
        ("sum_window", sum_window),
        ("delay_window", delay_window),
        ("price_base_window", price_base_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 1")

    open_ = data_ctx["open"].astype("float64")
    close = data_ctx["close"].astype("float64")

    # 步骤1: 日收益率
    returns = close.pct_change()

    # 步骤2: 相对价格归一化 —— 原脚本基准为 close 的 20 日均值（min_periods=1）
    price_base = close.rolling(window=price_base_window, min_periods=1).mean()
    normalized_open = open_ / price_base

    # 步骤3: 价格-动量组合信号及其与 delay_window 天前的差异
    sum_open = normalized_open.rolling(window=sum_window, min_periods=sum_window).sum()
    sum_returns = returns.rolling(window=sum_window, min_periods=sum_window).sum()
    signal = sum_open * sum_returns
    signal_diff = signal - signal.shift(delay_window)

    # 步骤4: 横截面 rank 取反（公式内部 rank 项；最终归一化由 preprocessing 完成）
    return -1.0 * signal_diff.rank(axis=1, pct=True)
