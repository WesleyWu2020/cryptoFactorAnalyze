"""Alpha101 Alpha#38 因子（factor_common 契约版）。

原始定义:
    Alpha#38 = ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（参数取旧脚本 __main__
实际调用值 ts_rank_window=10）:
    price_base       = mean(close, 20)          # min_periods=1，与旧脚本一致
    normalized_open  = open / price_base
    normalized_close = close / price_base
    ts_rank_close    = Ts_Rank(normalized_close, ts_rank_window)
    intraday_ratio   = normalized_close / normalized_open
    factor = (-1 * cs_rank(ts_rank_close)) * cs_rank(intraday_ratio)
公式内部的横截面 rank 予以保留（axis=1 pct rank）；最终去极值与秩归一化由
框架 preprocessing="mad_rank" 完成。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha38_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #38 (7x24 优化版): (-rank(ts_rank(close,10))) * rank(close/open)，价格经 20 日均线归一化",
}

SETTING = {
    "data_needed": ["open", "close"],
    "universe": "historical_top50",
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {"ts_rank_window": 10},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def _ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series percentile rank of the current value within the window."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: pd.Series(arr).rank(pct=True).iloc[-1], raw=False
    )


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#38 matrix (date x instrument)."""

    ts_rank_window = SETTING["params"]["ts_rank_window"]

    open_ = data_ctx["open"].astype("float64")
    close = data_ctx["close"].astype("float64")

    # 价格归一化（适应币圈价格差异），20 日均线基准，min_periods=1 与旧脚本一致
    price_base = close.rolling(window=20, min_periods=1).mean()
    normalized_open = open_ / (price_base + _EPSILON)
    normalized_close = close / (price_base + _EPSILON)

    # 组件1: 价格强度反转 Ts_Rank(close, ts_rank_window)
    ts_rank_close = _ts_rank(normalized_close, ts_rank_window)

    # 组件2: 日内动量 close / open
    intraday_ratio = normalized_close / (normalized_open + _EPSILON)

    # 公式: (-1 * rank(ts_rank_close)) * rank(intraday_ratio)
    return (-1.0 * ts_rank_close.rank(axis=1, pct=True)) * intraday_ratio.rank(axis=1, pct=True)
