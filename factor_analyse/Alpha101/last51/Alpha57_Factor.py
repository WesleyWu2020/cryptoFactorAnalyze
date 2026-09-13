"""Alpha101 Alpha#57 因子（factor_common 契约版）。

原始定义（原脚本 docstring）:
    (0 - (1 * ((close - (taker_buy_quote/taker_buy_base))
              / decay_linear(rank(ts_argmax(close, ts_argmax_window)), decay_window))))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（含一处修正，见下）:
    1. 主动买入价格: taker_buy_price = taker_buy_quote_volume / (taker_buy_base_volume + 1e-8)
    2. 价差: price_diff = close - taker_buy_price
    3. ts_argmax(close, 14): 过去 14 日收盘价最大值位置（0 = 窗口最旧）
    4. 横截面 rank(ts_argmax, pct)
    5. decay_linear(rank, 2): 2 日线性衰减加权（权重 1, 2）
    6. 因子值: -(price_diff / (decay_rank + 1e-8))

与原脚本的偏离说明:
    原脚本在 groupby('date') 内对 rank_ts_argmax 做 rolling(decay_window)，
    实际是沿 symbol 顺序的横截面滚动而非时间序列衰减，属于实现 bug；
    经典公式与脚本 docstring 步骤（"decay_window天线性衰减加权"）均为时间序列
    decay_linear，本实现按时间序列 decay_linear 实现并在 docstring 中注明。

参数取原脚本 __main__ 实际调用值（ts_argmax_window=14, decay_window=2）。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口；
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha57_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": (
        "Alpha101 #57: -((close - taker_buy_quote/taker_buy_base) "
        "/ decay_linear(rank(ts_argmax(close, 14)), 2))"
    ),
}

SETTING = {
    "data_needed": ["close", "taker_buy_base_volume", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 25,
    "preprocessing": "mad_rank",
    "params": {"ts_argmax_window": 14, "decay_window": 2},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def _ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling window argmax position (0 = oldest in window), causal."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: float(np.argmax(arr)), raw=True
    )


def _decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling linear-decay weighted mean with weights 1..window (recent heaviest)."""
    weights = np.arange(1, window + 1, dtype="float64")
    weights = weights / weights.sum()
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: float(np.dot(arr, weights)), raw=True
    )


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#57 matrix (date x instrument)."""

    ts_argmax_window = SETTING["params"]["ts_argmax_window"]
    decay_window = SETTING["params"]["decay_window"]

    close = data_ctx["close"].astype("float64")
    taker_base = data_ctx["taker_buy_base_volume"].astype("float64")
    taker_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # 主动买入价格与价差（与原脚本一致：未对零成交日做掩码）
    taker_buy_price = taker_quote / (taker_base + _EPSILON)
    price_diff = close - taker_buy_price

    # 过去 ts_argmax_window 日收盘价最大值位置 -> 横截面 rank -> 时间序列 decay_linear
    ts_argmax_close = _ts_argmax(close, ts_argmax_window)
    rank_ts_argmax = ts_argmax_close.rank(axis=1, pct=True)
    decay_rank = _decay_linear(rank_ts_argmax, decay_window)

    return -1.0 * (price_diff / (decay_rank + _EPSILON))
