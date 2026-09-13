"""Alpha101 Alpha#41 因子（factor_common 契约版）。

原始定义:
    Alpha#41 = (((high * low)^0.5) - vwap)

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    price_base = mean(close, 20)（min_periods=1）
    geo_mean = sqrt((high / price_base) * (low / price_base))
    taker_buy_price = taker_buy_quote_volume / taker_buy_base_volume（替代 VWAP）
    taker_buy_ma = mean(taker_buy_price, vwap_window)（min_periods=1，与原脚本一致）
    factor = geo_mean - taker_buy_ma / price_base
原脚本显式用 taker_buy_quote/taker_buy_base 定义均价，予以保留（非 quote_volume/volume）。
参数按 effective.txt 推荐: vwap_window=20。
旧脚本 rebalance_period 仅用于旧版 future_ret，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha41_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #41: sqrt(high*low) - vwap (taker-buy-price MA 替代 vwap)",
}

SETTING = {
    "data_needed": [
        "high",
        "low",
        "close",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ],
    "universe": "historical_top50",
    "warmup_bars": 25,
    "preprocessing": "mad_rank",
    "params": {"vwap_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#41 matrix (date x instrument)."""

    vwap_window = SETTING["params"]["vwap_window"]
    if isinstance(vwap_window, bool) or not isinstance(vwap_window, int) or vwap_window < 2:
        raise ValueError("SETTING.params.vwap_window must be an integer >= 2")

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    taker_base = data_ctx["taker_buy_base_volume"].astype("float64")
    taker_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # 价格归一化（适应币圈价格差异），与原脚本一致 min_periods=1
    price_base = close.rolling(window=20, min_periods=1).mean()
    norm_high = high / (price_base + _EPSILON)
    norm_low = low / (price_base + _EPSILON)

    # 几何平均价格 (high * low)^0.5
    geo_mean = (norm_high * norm_low) ** 0.5

    # 主买价及其移动平均（替代 VWAP）；无主动买入成交的日子不可观测
    observed = (taker_base > 0) & (taker_quote > 0)
    taker_buy_price = (taker_quote / (taker_base + _EPSILON)).where(observed)
    taker_buy_ma = taker_buy_price.rolling(window=vwap_window, min_periods=1).mean()
    norm_taker_buy_ma = taker_buy_ma / (price_base + _EPSILON)

    return geo_mean - norm_taker_buy_ma
