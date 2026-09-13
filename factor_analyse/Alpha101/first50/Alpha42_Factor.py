"""Alpha101 Alpha#42 因子（factor_common 契约版）。

原始定义:
    Alpha#42 = (rank((vwap - close)) / rank((vwap + close)))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    price_base = mean(close, 20)（min_periods=1）
    norm_close = close / price_base
    vwap = sum(close * volume, vwap_window) / sum(volume, vwap_window)
           （原脚本显式定义为成交量加权的收盘价移动平均，予以保留，
            非默认的 quote_volume / volume）
    norm_vwap = vwap / price_base
    factor = rank(norm_vwap - norm_close) / (rank(norm_vwap + norm_close) + 1e-8)
其中 rank 为横截面 pct rank（axis=1）。
参数取原脚本 __main__ 实际调用值: vwap_window=20。
旧脚本 rebalance_period 仅用于旧版 future_ret，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha42_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #42: rank(vwap - close) / rank(vwap + close)",
}

SETTING = {
    "data_needed": ["close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 25,
    "preprocessing": "mad_rank",
    "params": {"vwap_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#42 matrix (date x instrument)."""

    vwap_window = SETTING["params"]["vwap_window"]
    if isinstance(vwap_window, bool) or not isinstance(vwap_window, int) or vwap_window < 2:
        raise ValueError("SETTING.params.vwap_window must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 价格归一化（适应币圈价格差异），与原脚本一致 min_periods=1
    price_base = close.rolling(window=20, min_periods=1).mean()
    norm_close = close / (price_base + _EPSILON)

    # VWAP 近似: 成交量加权的收盘价移动平均（保留原脚本定义）
    vol_price_sum = (close * volume).rolling(window=vwap_window, min_periods=vwap_window).sum()
    vol_sum = volume.rolling(window=vwap_window, min_periods=vwap_window).sum()
    vwap = vol_price_sum / (vol_sum + _EPSILON)
    norm_vwap = vwap / (price_base + _EPSILON)

    # 因子组件: 价格偏离 与 价格水平，横截面 rank 比值
    vwap_minus_close = norm_vwap - norm_close
    vwap_plus_close = norm_vwap + norm_close
    rank_numerator = vwap_minus_close.rank(axis=1, pct=True)
    rank_denominator = vwap_plus_close.rank(axis=1, pct=True)
    return rank_numerator / (rank_denominator + _EPSILON)
