"""Top-Bottom Trades Ratio 因子（订单流）。

公式（与旧版 compute_one 逐点等价）：

    buy  = taker_buy_quote_volume
    sell = quote_volume - taker_buy_quote_volume
    ratio = buy / (sell + 1e-8)

    对每个交易日 t，取过去 window 天（含当日）：
      按当日 trade_count 降序排序，
      前 k_top 天的 ratio 之和 / 后 k_bottom 天的 ratio 之和
    factor = top_sum / (bottom_sum + 1e-8)

排序窗口仅使用 t 时点及历史数据，无未来函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "TopBottom_Trades_Ratio_Factor",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "Ratio of taker buy/sell summed on high-trade-count days vs low-trade-count days",
}

SETTING = {
    "data_needed": ["quote_volume", "taker_buy_quote_volume", "trade_count"],
    "universe": "historical_top50",
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "k_top": 10, "k_bottom": 10},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始 Top-Bottom Trades Ratio 因子矩阵（date x instrument）。"""

    window = SETTING["params"]["window"]
    k_top = SETTING["params"]["k_top"]
    k_bottom = SETTING["params"]["k_bottom"]

    quote_volume = data_ctx["quote_volume"].astype("float64")
    taker_buy = data_ctx["taker_buy_quote_volume"].astype("float64")
    trade_count = data_ctx["trade_count"].astype("float64")

    buy = taker_buy
    sell = quote_volume - taker_buy
    ratio = buy / (sell + _EPSILON)

    ratio_arr = ratio.to_numpy()
    tc_arr = trade_count.to_numpy()
    n_rows = ratio_arr.shape[0]
    result = np.full_like(ratio_arr, np.nan)

    # 逐日滚动窗口：按 trade_count 降序，分别累积 top/bottom 天的 ratio。
    # 与旧版一致：trade_count 为 NaN 的天排在最后；ratio 的 NaN 在求和时跳过。
    for i in range(window - 1, n_rows):
        w_ratio = ratio_arr[i - window + 1 : i + 1]
        w_tc = tc_arr[i - window + 1 : i + 1]
        # 排序键：-tc 实现降序；NaN 映射为 +inf 排在最后（同 sort_values 默认行为）
        sort_key = np.where(np.isnan(w_tc), np.inf, -w_tc)
        order = np.argsort(sort_key, axis=0, kind="stable")
        sorted_ratio = np.take_along_axis(w_ratio, order, axis=0)
        top_sum = np.nansum(sorted_ratio[:k_top], axis=0)
        bottom_sum = np.nansum(sorted_ratio[window - k_bottom :], axis=0)
        result[i] = top_sum / (bottom_sum + _EPSILON)

    return pd.DataFrame(result, index=ratio.index, columns=ratio.columns)
