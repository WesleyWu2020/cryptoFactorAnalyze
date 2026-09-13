"""日内买卖压强分布因子 (Intraday Accumulation/Distribution, IAD)。

公式（与旧版 compute_one 逐点等价）：

    CLV  = ((close - low) - (high - close)) / (high - low + eps)
    AD   = CLV * quote_volume
    IAD  = rolling_sum(AD, window) / (rolling_sum(quote_volume, window) + eps)

即过去 ``window`` 日的成交额加权平均收盘价位置（Close Location Value）。
rolling 沿时间轴且 min_periods=window（与旧版 SUM 默认一致），只依赖当日
及历史数据，无未来函数。截面 MAD 去极值与 rank 归一化由框架
``preprocessing="mad_rank"`` 处理，``calc_factor`` 返回原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Intraday_Accumulation_Distribution_Factor",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "N-day quote-volume-weighted average close location value (IAD)",
}

SETTING = {
    "data_needed": ["high", "low", "close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "eps": 1e-5},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始 IAD 因子矩阵（date 升序 × instrument）。

    rolling 沿日线索引计算，每个值只依赖当前及之前 ``window - 1`` 根 K 线；
    min_periods=window，与旧版 ``SUM(x, n=window)`` 默认行为一致。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be a positive integer")
    eps = float(SETTING["params"]["eps"])

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")

    clv = ((close - low) - (high - close)) / (high - low + eps)
    ad = clv * quote_volume
    ad_sum = ad.rolling(window=window, min_periods=window).sum()
    quote_volume_sum = quote_volume.rolling(window=window, min_periods=window).sum()
    return ad_sum / (quote_volume_sum + eps)
