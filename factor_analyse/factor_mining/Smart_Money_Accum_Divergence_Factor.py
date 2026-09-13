"""聪明钱潜伏背离因子 (Smart Money Accumulation Divergence)。

公式:

    average_trade_size = quote_volume / (trade_count + eps)
    ats_roc   = average_trade_size / average_trade_size.shift(window) - 1
    price_roc = close / close.shift(window) - 1
    factor    = rank_cs(ats_roc) - rank_cs(price_roc)   # 按日截面 rank(pct=True)

含义: 平均单笔成交规模 (聪明钱代理) 的动量排名高于价格动量排名时,
表明有大资金在价格未动时潜伏吸筹, 因子值越高越看多。

计算只使用当日及历史数据, 无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Smart_Money_Accum_Divergence_Factor",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "Divergence between average-trade-size momentum and price momentum",
}

SETTING = {
    "data_needed": ["quote_volume", "trade_count", "close"],
    "universe": "historical_top50",
    "warmup_bars": 21,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "eps": 1e-5},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily smart-money accumulation divergence matrix.

    ``data_ctx`` contains date-by-instrument matrices. ``shift`` operates
    along the daily index and the cross-sectional rank along ``axis=1``,
    so each value only depends on current and historical observations.
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be an integer >= 1")
    eps = float(SETTING["params"]["eps"])

    quote_volume = data_ctx["quote_volume"].astype("float64")
    trade_count = data_ctx["trade_count"].astype("float64")
    close = data_ctx["close"].astype("float64")

    average_trade_size = quote_volume / (trade_count + eps)
    ats_roc = average_trade_size / average_trade_size.shift(window) - 1.0
    price_roc = close / close.shift(window) - 1.0

    # 组合子因子前的截面 rank 是公式的一部分, 保留在 calc_factor 内实现
    ats_roc_rank = ats_roc.rank(axis=1, method="average", pct=True)
    price_roc_rank = price_roc.rank(axis=1, method="average", pct=True)
    return ats_roc_rank - price_roc_rank
