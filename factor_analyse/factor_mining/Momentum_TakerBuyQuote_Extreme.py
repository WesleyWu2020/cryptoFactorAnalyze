"""动量-主动买入极值因子（common 框架迁移版）。

公式（与旧版 CSV 管线 compute_one 逐点等价）：

    momentum = close / close.shift(window) - 1
    对每个交易日 t，取最近 window 天的滚动窗口，按 taker_buy_quote_volume
    降序排序，取主动买入额最高的 Top10 天的 momentum 之和，除以主动买入额
    最低的 Bottom10 天的 momentum 之和：

    factor = sum(momentum[top10 by taker_buy_quote_volume])
             / sum(momentum[bot10 by taker_buy_quote_volume])

    若分母为 0 或结果为 NaN/Inf，则该日因子值为 NaN（与旧版 continue 等价）。

只使用当日及历史数据，无未来函数。输出原始因子值，截面 MAD 去极值与
按日 rank 归一化由框架 preprocessing="mad_rank" 处理。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TYPE = "regular"

META = {
    "factor_name": "Momentum_TakerBuyQuote_Extreme",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "Ratio of momentum sums on top/bottom-10 taker-buy-quote days within an N-day window",
}

SETTING = {
    "data_needed": ["close", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    # momentum 需要 window 天历史，滚动窗口再需 window 天，故首个有效值需 2*window 天
    "warmup_bars": 40,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "rebalance_period": 10},
    # +1：因子值越大，说明主动买入最猛烈的交易日上的动量越强（相对主动买入
    # 最弱的日子），即上涨由激进买盘驱动，通常视为买入驱动的趋势延续信号
    "factor_direction": 1,
}

_TOP_N = 10


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """按日 × 标的矩阵计算动量-主动买入极值因子。

    rolling 沿时间轴（axis 0）进行：每个值只依赖当前及之前
    ``2 * window - 1`` 个观测。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < _TOP_N:
        raise ValueError(f"SETTING.params.window must be an integer >= {_TOP_N}")

    close = data_ctx["close"].astype("float64")
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    momentum = close / close.shift(window) - 1.0

    mom = momentum.to_numpy(dtype="float64")
    tbq = taker_buy_quote.to_numpy(dtype="float64")
    n_dates, n_symbols = mom.shape

    result = np.full((n_dates, n_symbols), np.nan, dtype="float64")
    if n_dates >= window:
        mom_w = sliding_window_view(mom, window, axis=0)  # (T-w+1, N, w)
        tbq_w = sliding_window_view(tbq, window, axis=0)

        # 按窗口内 taker_buy_quote_volume 降序排序（NaN 同旧版 sort_values 一样排在末尾）
        order = np.argsort(-tbq_w, axis=-1, kind="stable")
        sorted_mom = np.take_along_axis(mom_w, order, axis=-1)

        top_sum = sorted_mom[..., :_TOP_N].sum(axis=-1)
        bot_sum = sorted_mom[..., -_TOP_N:].sum(axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            factor = top_sum / bot_sum
        # 与旧版一致：分母为 0 或结果 NaN/Inf 时丢弃该日
        factor = np.where((bot_sum == 0) | ~np.isfinite(factor), np.nan, factor)
        result[window - 1:] = factor

    return pd.DataFrame(result, index=close.index, columns=close.columns)
