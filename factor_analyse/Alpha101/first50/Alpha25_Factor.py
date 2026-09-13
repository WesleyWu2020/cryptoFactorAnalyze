"""Alpha101 Alpha#25 因子（factor_common 契约版）。

原始定义:
    Alpha#25 = rank(((((-1 * returns) * adv20) * vwap) * (high - close)))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准，与经典公式有差异）:
    returns        = close.pct_change()
    adv            = mean(taker_buy_quote_volume, adv_window)   # 原脚本用主动买入金额而非总成交额
    vwap           = quote_volume / volume（volume>0 掩码）
    vwap_norm      = vwap / close
    high_close_diff = (high - close) / close
    factor         = (-1 * returns) * adv * vwap_norm * high_close_diff
原脚本最外层的横截面 rank 由框架 preprocessing="mad_rank" 完成，这里输出原始值。
参数取原脚本 __main__ 实际调用值 adv_window=7（rebalance_period 仅用于旧版 future_ret，丢弃）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha25_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #25: (-returns) * mean(taker_buy_quote, 7) * (vwap/close) * ((high-close)/close)",
}

SETTING = {
    "data_needed": ["close", "high", "volume", "quote_volume", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 15,
    "preprocessing": "mad_rank",
    "params": {"adv_window": 7},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#25 matrix (date x instrument)."""

    adv_window = SETTING["params"]["adv_window"]
    if isinstance(adv_window, bool) or not isinstance(adv_window, int) or adv_window < 2:
        raise ValueError("SETTING.params.adv_window must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    volume = data_ctx["volume"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # 1. 收益率（反转项）
    returns = close.pct_change()

    # 2. adv_window 日平均主动买入金额（原脚本定义，非总成交额 adv）
    adv = taker_buy_quote.rolling(window=adv_window, min_periods=adv_window).mean()

    # 3. VWAP 归一化（仅在有成交流的日子可观测）
    vwap = (quote_volume / (volume + _EPSILON)).where(volume > 0)
    vwap_norm = vwap / close

    # 4. 日内抛压（归一化）
    high_close_diff = (high - close) / close

    # 5. 复合项；最外层横截面 rank 由 preprocessing 完成
    return (-1.0 * returns) * adv * vwap_norm * high_close_diff
