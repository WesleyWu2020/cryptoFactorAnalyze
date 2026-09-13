"""crypto_alpha_volatility_order_flow_efficiency 因子（factor_common 日频契约版）。

原始定义（分钟版 numba 实现，窗口 = 20*1440 根分钟 bar）：
    range_mean          = mean((minute_high - minute_low) / prev_minute_close, 20d)
    trade_count_mean    = mean(minute_trade_count, 20d)
    avg_trade_size_mean = mean(minute_quote_volume / minute_trade_count, 20d)
    roc                 = close / close.shift(20d) - 1
    vol_eff             = roc / range_mean
    order_flow          = (trade_count / trade_count_mean)
                          / (avg_trade_size / avg_trade_size_mean)
    factor              = vol_eff * order_flow

分钟->日频改写说明（语义判断）：
    - 窗口语义从"20 天的分钟 bar"变为"20 个交易日"：roc 用 close.shift(20)。
    - range_mean / trade_count_mean / avg_trade_size_mean 原来是分钟级均值，
      日频下无法还原分钟分布，按公式意图用日频对应量替代：
      range_pct = (daily_high - daily_low)/prev_daily_close，
      trade_count / avg_trade_size = quote_volume / trade_count 均为日频总量/比值，
      再取 20 日均值；当前值取当日日频值。属忠实的日频等价改写，量级与原
      分钟版不同，但截面排序语义一致。
    - 分钟版要求窗口内观测完整（min_periods = window）且当前 bar 有效，日频版保留。
    - 分钟版无执行延迟 shift，日频版同样不加；框架按次日开盘执行。
    - 删除 numba 内核与 CHUNK_SIZE 分块循环，整表向量化计算。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_volatility_order_flow_efficiency",
    "author": "wesleywu",
    "level": "daily",
    "category": "microstructure",
    "description": "20d volatility efficiency (ROC / mean daily range) multiplied by "
                   "order-flow ratio (trade_count vs 20d mean)/(avg trade size vs 20d mean)",
}

SETTING = {
    "data_needed": ["close", "high", "low", "quote_volume", "trade_count"],
    "universe": "historical_top50",
    # 20 (roc base) + 20 (means) + 1 (prev_close) + buffer
    "warmup_bars": 45,
    "preprocessing": "mad_rank",
    "params": {"window_days": 20},
    # 无 ITER_NOTE 方向记录；高波动效率且订单流走强的标的高值，默认 1
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily order-flow-efficiency matrix (date x instrument)."""

    window_days = SETTING["params"]["window_days"]

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")
    trade_count = data_ctx["trade_count"].astype("float64")

    close = close.where(close > 0)
    prev_close = close.shift(1)

    range_pct = ((high - low) / prev_close).where(prev_close > _EPS)
    valid_trade_count = trade_count.where(trade_count >= 0)
    avg_trade_size = (quote_volume / trade_count).where(
        (quote_volume > 0) & (trade_count > 0)
    )

    range_mean = range_pct.rolling(window_days, min_periods=window_days).mean()
    trade_count_mean = valid_trade_count.rolling(
        window_days, min_periods=window_days
    ).mean()
    avg_trade_size_mean = avg_trade_size.rolling(
        window_days, min_periods=window_days
    ).mean()

    base_close = close.shift(window_days)
    roc = close / base_close - 1.0
    vol_eff = roc / (range_mean + _EPS)

    trades_ratio = valid_trade_count / (trade_count_mean + _EPS)
    avg_trade_size_ratio = avg_trade_size / (avg_trade_size_mean + _EPS)

    order_flow = trades_ratio / (avg_trade_size_ratio + _EPS)
    factor = vol_eff * order_flow

    # 与分钟版一致的有效性约束：均值与当前值必须为正
    valid = (
        (range_mean > _EPS)
        & (trade_count_mean > _EPS)
        & (avg_trade_size_mean > _EPS)
        & (avg_trade_size > _EPS)
    )
    return factor.where(valid).replace([np.inf, -np.inf], np.nan)
