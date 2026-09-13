"""主动买入占比-收益差因子（W-Cutting 逻辑变种）。

公式（与旧版 compute_one 逐点等价）：

    ret        = log(close / close.shift(1))
    buy_ratio  = taker_buy_quote_volume / (quote_volume + 1e-8)

对任意交易日 t，取过去 window 天（不含当日 t，即 [t-window, t)）的
(ret, buy_ratio) 序列，按 buy_ratio 升序排序后对半切：

    m_high = 买入占比最高的一半日子的 ret 之和（多头主导日累计涨幅）
    m_low  = 买入占比最低的一半日子的 ret 之和（空头主导日累计涨幅）
    factor = m_high - m_low

含义：因子值越大，说明该币在"多头主动进攻"的日子里涨得好、在"空头砸盘"
的日子里抗跌，用于衡量上涨质量。窗口内存在 NaN、或 m_low == 0、或结果
非有限值时输出 NaN（与旧版 skip 行为一致）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


TYPE = "regular"

META = {
    "factor_name": "Return_Taker_Ratio_Diff",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "Sum of returns on high taker-buy-ratio days minus low-ratio days (W-Cutting)",
}

SETTING = {
    "data_needed": ["close", "quote_volume", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 12,
    "preprocessing": "mad_rank",
    "params": {"window": 10, "rebalance_period": 5},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily taker-ratio return-diff matrix.

    The trailing window of length ``window`` ends at ``t - 1`` (excludes the
    current day), matching the old loop over ``[i - window, i)``. ``rolling``
    style operations run along the daily index only.
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.window must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")
    taker_buy_quote_volume = data_ctx["taker_buy_quote_volume"].astype("float64")

    ret = np.log(close / close.shift(1))
    buy_ratio = taker_buy_quote_volume / (quote_volume + _EPSILON)

    # Trailing window [t - window, t): shift by 1 so the window excludes day t.
    ret_past = ret.shift(1).to_numpy()
    ratio_past = buy_ratio.shift(1).to_numpy()

    n_rows, n_cols = ret_past.shape
    out = np.full((n_rows, n_cols), np.nan)
    if n_rows < window + 1:
        return pd.DataFrame(out, index=close.index, columns=close.columns)

    # Sliding windows over rows; window k covers rows [k, k + window).
    ret_w = sliding_window_view(ret_past, window, axis=0)
    ratio_w = sliding_window_view(ratio_past, window, axis=0)

    valid = ~(np.isnan(ret_w).any(axis=-1) | np.isnan(ratio_w).any(axis=-1))

    # Sort each window by buy_ratio ascending and split in half (W-Cutting).
    sorted_idx = np.argsort(ratio_w, axis=-1)
    sorted_ret = np.take_along_axis(ret_w, sorted_idx, axis=-1)
    n_split = window // 2
    m_low = np.nansum(sorted_ret[..., :n_split], axis=-1)
    m_high = np.nansum(sorted_ret[..., n_split:], axis=-1)

    factor = m_high - m_low
    factor = np.where(
        valid & (m_low != 0) & np.isfinite(m_high) & np.isfinite(m_low),
        factor,
        np.nan,
    )

    # Sliding window k covers rows [k, k + window) of the shifted series and
    # ends at row k + window - 1, so its value belongs to date t = k + window - 1
    # (i.e. ret days [t - window, t)). First valid t = window + 1, since
    # ret[0] is NaN — same as the old loop starting at i = window.
    out[window - 1:] = factor
    return pd.DataFrame(out, index=close.index, columns=close.columns)
