"""Alpha101 Alpha#10 因子（factor_common 契约版，币圈7×24h优化版）。

原始定义:
    Alpha#10 = rank((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
                    ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : -1 * delta(close, 1)))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    returns = close.pct_change()               # 原脚本用收益率实现 delta(close, 1)
    factor  = rank( ts_min(returns, N) > 0 ? returns :
                   (ts_max(returns, N) < 0 ? returns : -returns) )
即过去 N 天收益全正（强上涨）或全负（强下跌）时跟随动量，混合/横盘时做反转；
外层 rank 为按日横截面百分比秩（公式内部项，保留）。

参数取原脚本 ``__main__`` 实际调用值: trend_window=7
（rebalance_period 仅用于旧版 future_ret，已丢弃）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha10_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": (
        "Alpha101 #10 (crypto): rank(ts_min(returns,7)>0 or ts_max(returns,7)<0 "
        "? returns : -returns)"
    ),
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 13,
    "preprocessing": "mad_rank",
    "params": {"trend_window": 7},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#10 matrix (date x instrument)."""

    trend_window = SETTING["params"]["trend_window"]
    if isinstance(trend_window, bool) or not isinstance(trend_window, int) or trend_window < 2:
        raise ValueError("SETTING.params.trend_window must be an integer >= 2")

    close = data_ctx["close"].astype("float64")

    # 步骤1: 日收益率（原脚本以 returns 实现 delta(close, 1)）
    returns = close.pct_change()

    # 步骤2: 过去 trend_window 日收益率的最小/最大值，判断趋势状态
    ts_min_returns = returns.rolling(window=trend_window, min_periods=trend_window).min()
    ts_max_returns = returns.rolling(window=trend_window, min_periods=trend_window).max()

    # 步骤3: 自适应策略 —— 强趋势跟动量，混合/横盘做反转；
    # 趋势窗口未形成时输出 NaN（与原脚本一致）
    trending = (ts_min_returns > 0) | (ts_max_returns < 0)
    adaptive = returns.where(trending, -1.0 * returns)
    adaptive = adaptive.where(ts_min_returns.notna() & ts_max_returns.notna())

    # 步骤4: 横截面 rank（公式内部 rank 项；最终归一化由 preprocessing 完成）
    return adaptive.rank(axis=1, pct=True)
