"""Alpha101 Alpha#37 因子（factor_common 契约版）。

原始定义:
    Alpha#37 = (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（参数取旧脚本 __main__
策略1 实际调用值 corr_window=50, delay_lag=5，而非签名默认值 200/1）:
    price_base        = mean(close, 20)            # min_periods=1，与旧脚本一致
    normalized_open   = open / price_base
    normalized_close  = close / price_base
    intraday          = normalized_open - normalized_close
    factor = cs_rank(corr(delay(intraday, delay_lag), normalized_close, corr_window))
             + cs_rank(intraday)
公式内部的横截面 rank 予以保留（axis=1 pct rank）；最终去极值与秩归一化由
框架 preprocessing="mad_rank" 完成。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha37_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #37 (7x24 优化版): rank(corr(delay(open-close,5), close, 50)) + rank(open-close)，价格经 20 日均线归一化",
}

SETTING = {
    "data_needed": ["open", "close"],
    "universe": "historical_top50",
    "warmup_bars": 65,
    "preprocessing": "mad_rank",
    "params": {"corr_window": 50, "delay_lag": 5},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#37 matrix (date x instrument)."""

    corr_window = SETTING["params"]["corr_window"]
    delay_lag = SETTING["params"]["delay_lag"]

    open_ = data_ctx["open"].astype("float64")
    close = data_ctx["close"].astype("float64")

    # 价格归一化（适应币圈价格差异），20 日均线基准，min_periods=1 与旧脚本一致
    price_base = close.rolling(window=20, min_periods=1).mean()
    normalized_open = open_ / (price_base + _EPSILON)
    normalized_close = close / (price_base + _EPSILON)

    # 组件1: 历史日内模式与价格相关性 corr(delay(open - close, D), close, W)
    intraday = normalized_open - normalized_close
    intraday_delay = intraday.shift(delay_lag)
    corr_historical = intraday_delay.rolling(
        window=corr_window, min_periods=corr_window
    ).corr(normalized_close)

    # 组件2: 当前日内表现 (open - close)
    current_intraday = intraday

    # 公式内部横截面 rank，两项权重各 1.0
    return corr_historical.rank(axis=1, pct=True) + current_intraday.rank(axis=1, pct=True)
