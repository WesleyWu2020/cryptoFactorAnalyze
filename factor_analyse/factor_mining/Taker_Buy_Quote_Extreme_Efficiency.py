"""Taker 主动买入额极端效率因子（Taker Buy Quote Extreme Efficiency）。

公式（与旧版 CSV 管线 compute_one 逐点等价）：

    roc       = close / close.shift(window) - 1
    range_pct = ((high - low) / close.shift(1)).rolling(window).mean()
    vol_eff   = roc / (range_pct + 1e-8)

在截至当日的过去 window 根 K 线内，按 taker_buy_quote_volume 降序排序，
取排序前 10 根的 vol_eff 之和除以后 10 根的 vol_eff 之和：

    factor = sum(vol_eff[主动买入额最高的 10 根]) / sum(vol_eff[主动买入额最低的 10 根])

分母为 0、窗口内 vol_eff 全 NaN 或结果非有限值时输出 NaN（与旧版跳过逻辑一致）。
仅使用当日及历史数据，无未来函数；universe 掩码与 mad_rank 截面预处理由框架负责，
calc_factor 返回原始因子矩阵。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TYPE = "regular"

META = {
    "factor_name": "Taker_Buy_Quote_Extreme_Efficiency",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "Ratio of trend efficiency on extreme taker-buy days vs quiet days within a rolling window",
}

SETTING = {
    "data_needed": ["close", "high", "low", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 40,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "rebalance_period": 10},
    "factor_direction": 1,
}

_EPSILON = 1e-8
_TOP_N = 10


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily TBQ extreme-efficiency matrix.

    All rolling/shift operations run along the daily index (axis 0); each
    value at date t only depends on observations at or before t.
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2 * _TOP_N:
        raise ValueError(f"SETTING.params.window must be an integer >= {2 * _TOP_N}")

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # vol_eff: N 日涨跌幅 / 窗口内平均振幅（分母加 epsilon 防零，同旧版）。
    roc = close / close.shift(window) - 1.0
    range_pct = ((high - low) / close.shift(1)).rolling(window=window).mean()
    vol_eff = roc / (range_pct + _EPSILON)

    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype="float64")
    n = len(close)
    if n < window:
        return out

    # 对每个 (t, symbol)：取截至 t 的 window 根 K 线，按 taker_buy_quote 降序排序，
    # 前 _TOP_N 根与后 _TOP_N 根的 vol_eff 分别求和（NaN 跳过，同 pandas sum 默认行为）。
    ve = vol_eff.to_numpy(dtype="float64")
    tq = taker_buy_quote.to_numpy(dtype="float64")
    ve_win = np.lib.stride_tricks.sliding_window_view(ve, window, axis=0)
    tq_win = np.lib.stride_tricks.sliding_window_view(tq, window, axis=0)

    # NaN 的 taker_buy_quote 在降序排序中排最后，与 pandas sort_values 一致。
    order = np.argsort(-tq_win, axis=2, kind="stable")
    ve_sorted = np.take_along_axis(ve_win, order, axis=2)
    top_sum = np.nansum(ve_sorted[..., :_TOP_N], axis=2)
    bot_sum = np.nansum(ve_sorted[..., -_TOP_N:], axis=2)

    valid = ~np.isnan(ve_win).all(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = top_sum / bot_sum
    factor[~valid | (bot_sum == 0) | ~np.isfinite(factor)] = np.nan

    out.iloc[window - 1:] = factor
    return out
