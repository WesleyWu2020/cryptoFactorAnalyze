"""Alpha101 Alpha#32 因子（factor_common 契约版）。

原始定义:
    Alpha#32 = scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(taker_buy_price, delay(close, 5), 30)))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    taker_buy_price = taker_buy_quote_volume / taker_buy_base_volume
    mean_revert = mean(close, short_mean_window) - close
    corr_long   = correlation(taker_buy_price, delay(close, delay_lag), corr_window)
    factor      = cs_scale(mean_revert) + corr_weight * cs_scale(corr_long)
其中 cs_scale 为横截面 scale: x / sum(|x|)（公式内部的横截面操作，予以保留）。
参数取旧脚本 __main__ 实际调用值: short_mean_window=7, corr_window=30, delay_lag=8。
注意: 旧脚本在窗口方差过小时把相关性置 0，矩阵版直接使用 rolling.corr，
退化窗口产生 NaN（由框架覆盖统计处理），语义差异极小。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha32_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #32: cs_scale(mean(close,7)-close) + 20*cs_scale(corr(taker_buy_price, delay(close,8), 30))",
}

SETTING = {
    "data_needed": ["close", "taker_buy_base_volume", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 45,
    "preprocessing": "mad_rank",
    "params": {"short_mean_window": 7, "corr_window": 30, "delay_lag": 8, "corr_weight": 20.0},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def _cs_scale(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional scale: x / sum(|x|) per date (axis=1)."""
    denom = x.abs().sum(axis=1)
    return x.div(denom.where(denom > 0), axis=0)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#32 matrix (date x instrument)."""

    short_mean_window = SETTING["params"]["short_mean_window"]
    corr_window = SETTING["params"]["corr_window"]
    delay_lag = SETTING["params"]["delay_lag"]
    corr_weight = SETTING["params"]["corr_weight"]

    close = data_ctx["close"].astype("float64")
    taker_base = data_ctx["taker_buy_base_volume"].astype("float64")
    taker_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # 1. 主买价 (taker_buy_quote / taker_buy_base)
    taker_buy_price = taker_quote / (taker_base + _EPSILON)

    # 2. 均值回归项: (MA(close, N) - close)
    close_ma = close.rolling(window=short_mean_window, min_periods=short_mean_window).mean()
    mean_revert = close_ma - close

    # 3. 长期相关性: corr(taker_buy_price, delay(close, D), W)
    close_delay = close.shift(delay_lag)
    corr_long = taker_buy_price.rolling(window=corr_window, min_periods=corr_window).corr(close_delay)

    # 4. 横截面 scale 后按权重组合（scale 是公式内部横截面操作，保留）
    return _cs_scale(mean_revert) + corr_weight * _cs_scale(corr_long)
