"""crypto_alpha_retail_activity_divergence 因子（factor_common 日频契约版）。

原始定义（分钟版改造）：
    trade_count_ma = rolling_mean(minute trade_count, 20d*1440 bars)
    quote_volume_ma = rolling_mean(minute quote_volume, 20d*1440 bars)
    factor = rank_cs(trade_count_ma) - rank_cs(quote_volume_ma)

日频改写判断：
    - 20d*1440 根分钟 bar 的均值 ≡ (20d 日频总量均值) / 1440，截面 rank 对
      正比例缩放不变，因此日频忠实等价物为 daily sum 字段的 20d rolling mean。
    - 分钟版 numba/分块管线整帧 pandas 化（语义不变）。
    - 无执行延迟 shift，原始因子本身在日聚合后未做 shift。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_retail_activity_divergence",
    "author": "wesleywu",
    "level": "daily",
    "category": "microstructure",
    "description": "rank_cs(mean(trade_count, 20d)) - rank_cs(mean(quote_volume, 20d)): 笔数活跃度相对成交额的背离",
}

SETTING = {
    "data_needed": ["trade_count", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 25,
    "preprocessing": "mad_rank",
    "params": {"window_days": 20},
    "factor_direction": 1,  # 默认：高值=笔数活跃相对成交额偏高（散户关注度）优先做多
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily retail-activity-divergence matrix (date x instrument)."""

    window_days = SETTING["params"]["window_days"]

    trade_count = data_ctx["trade_count"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")

    trade_count = trade_count.where(trade_count >= 0.0)
    quote_volume = quote_volume.where(quote_volume >= 0.0)

    trade_ma = trade_count.rolling(window_days, min_periods=window_days).mean()
    quote_ma = quote_volume.rolling(window_days, min_periods=window_days).mean()

    factor = trade_ma.rank(axis=1, pct=True) - quote_ma.rank(axis=1, pct=True)
    return factor.replace([np.inf, -np.inf], np.nan)
