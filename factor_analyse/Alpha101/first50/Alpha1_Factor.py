"""Alpha101 Alpha#1 因子（factor_common 契约版）。

原始定义:
    Alpha#1 = rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2), 5)) - 0.5

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面 rank 与去极值由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha1_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #1: rank(ts_argmax(signed_power(cond(std, close)), 5)) - 0.5",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"stddev_window": 20, "argmax_window": 5},
    "factor_direction": 1,
}


def _ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling window argmax position (0 = oldest in window), causal."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: float(np.argmax(arr)), raw=True
    )


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#1 matrix (date x instrument)."""

    stddev_window = SETTING["params"]["stddev_window"]
    argmax_window = SETTING["params"]["argmax_window"]

    close = data_ctx["close"].astype("float64")
    returns = close.pct_change()

    # 步骤1: 负收益取过去 stddev_window 日收益率标准差，否则取收盘价
    returns_std = returns.rolling(window=stddev_window, min_periods=stddev_window).std()
    conditional = returns_std.where(returns < 0, close)

    # 步骤2: SignedPower(x, 2) = sign(x) * |x|^2
    signed_power = np.sign(conditional) * conditional.abs() ** 2

    # 步骤3: 过去 argmax_window 日内最大值出现位置（0 = 窗口最旧的一天）
    ts_argmax = _ts_argmax(signed_power, argmax_window)

    # 步骤4: 横截面 rank - 0.5（最终归一化由 preprocessing 完成）
    return ts_argmax.rank(axis=1, pct=True) - 0.5
