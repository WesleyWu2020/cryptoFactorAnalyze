"""Alpha101 Alpha#27 因子（factor_common 契约版）。

原始定义:
    Alpha#27 = (0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? -1 : 1

本实现保留原脚本“币圈7×24h优化版”的实际计算结构:
    vwap      = quote_volume / volume（volume>0 掩码）
    corr      = correlation(rank(volume), rank(vwap), corr_window)
    corr_mean = mean(corr, mean_window)
    factor    = where(cross_section_rank_pct(corr_mean) > 0.5, -1, 1)
参数取原脚本 __main__ 实际调用值 corr_window=20, mean_window=10
（rebalance_period 仅用于旧版 future_ret，丢弃）。

与旧脚本的一处必要偏离（修复未来函数）:
    旧脚本中 rank(volume)/rank(vwap) 是“单 symbol 全历史时间序列秩”，
    其 t 时点取值依赖未来数据，属于前视偏差。这里按经典 Alpha#27 公式语义
    改为当日横截面 pct 秩（axis=1），既是公式原意又满足无未来函数约束。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成（对 -1/1 二元信号等价于保序归一）。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha27_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #27: where(cs_rank(mean(corr(cs_rank(volume), cs_rank(vwap), 20), 10)) > 0.5, -1, 1)",
}

SETTING = {
    "data_needed": ["volume", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 35,
    "preprocessing": "mad_rank",
    "params": {"corr_window": 20, "mean_window": 10},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#27 binary signal matrix (date x instrument)."""

    corr_window = SETTING["params"]["corr_window"]
    mean_window = SETTING["params"]["mean_window"]
    for name, value in (("corr_window", corr_window), ("mean_window", mean_window)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    volume = data_ctx["volume"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")

    # 1. VWAP（仅在有成交流的日子可观测）
    vwap = (quote_volume / (volume + _EPSILON)).where(volume > 0)

    # 2. 横截面 pct 秩（经典公式语义；替代旧脚本非因果的全历史 TS 秩）
    vol_rank = volume.rank(axis=1, pct=True)
    vwap_rank = vwap.rank(axis=1, pct=True)

    # 3. corr_window 日滚动相关
    corr = vol_rank.rolling(window=corr_window, min_periods=corr_window).corr(vwap_rank)

    # 4. mean_window 日均值（sum(..., 2) / 2 即 2 日均值）
    corr_mean = corr.rolling(window=mean_window, min_periods=mean_window).mean()

    # 5. 横截面秩 > 0.5 给 -1，否则给 1（二元信号；NaN 保持 NaN）
    cs_rank = corr_mean.rank(axis=1, pct=True)
    signal = 1.0 - 2.0 * cs_rank.gt(0.5).astype(float)
    return signal.where(cs_rank.notna())
