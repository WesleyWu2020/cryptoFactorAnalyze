"""Alpha101 Alpha#60 因子（factor_common 契约版）。

原始定义（原脚本 docstring）:
    (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * taker_buy_quote))))
              - scale(rank(ts_argmax(close, ts_argmax_window))))))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    1. 日内位置: intraday_position =
       ((close - low) - (high - close)) / (high - low + 1e-8)
    2. 乘以主动买入金额: position_taker_buy = intraday_position * taker_buy_quote_volume
    3. ts_argmax(close, 49): 过去 49 日收盘价最大值位置（0 = 窗口最旧）
    4. 横截面 rank + scale: scale(x) = (rank(x, pct) - 0.5) * 2（缩放到 [-1, 1]）
    5. 因子值: -(2 * scale(position_taker_buy) - scale(ts_argmax_close))

参数取原脚本 __main__ 实际调用值（ts_argmax_window=49；函数签名默认 10 但 __main__ 用 49）。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口；
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha60_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": (
        "Alpha101 #60: -(2*scale(rank(intraday_pos * taker_buy_quote)) "
        "- scale(rank(ts_argmax(close, 49))))"
    ),
}

SETTING = {
    "data_needed": ["close", "high", "low", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 55,
    "preprocessing": "mad_rank",
    "params": {"ts_argmax_window": 49},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def _ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling window argmax position (0 = oldest in window), causal."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: float(np.argmax(arr)), raw=True
    )


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#60 matrix (date x instrument)."""

    ts_argmax_window = SETTING["params"]["ts_argmax_window"]

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # 1. 日内位置指标并乘以主动买入金额
    intraday_position = ((close - low) - (high - close)) / (high - low + _EPSILON)
    position_taker_buy = intraday_position * taker_buy_quote

    # 2. 过去 ts_argmax_window 日收盘价最大值位置
    ts_argmax_close = _ts_argmax(close, ts_argmax_window)

    # 3. 横截面 rank 与 scale（(rank - 0.5) * 2 -> [-1, 1]）
    scale_position_taker_buy = (position_taker_buy.rank(axis=1, pct=True) - 0.5) * 2.0
    scale_ts_argmax = (ts_argmax_close.rank(axis=1, pct=True) - 0.5) * 2.0

    # 4. 组合并取负
    return -1.0 * (2.0 * scale_position_taker_buy - scale_ts_argmax)
