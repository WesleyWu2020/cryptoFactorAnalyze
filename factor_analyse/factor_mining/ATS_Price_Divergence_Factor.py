"""ATS-Price 背离因子（Average Trade Size 与价格变动背离）。

公式（与旧版 compute_one 逐点等价，无未来函数）：

    ATS        = quote_volume / (trade_count + eps)
    ATS_ROC    = ATS / ATS.shift(window) - 1        # 等价旧版 PCT_CHANGE(ATS, window)
    PRICE_ROC  = close / close.shift(window) - 1    # 等价旧版 PCT_CHANGE(close, window)
    factor     = ATS_ROC - PRICE_ROC

所有计算只依赖当日及历史数据（shift 为正位移）。新框架入口为
``calc_factor(data_ctx)``，输出原始因子矩阵，截面去极值与归一化由框架的
``preprocessing="mad_rank"`` 完成。
"""

from __future__ import annotations

import pandas as pd

TYPE = "regular"

META = {
    "factor_name": "ATS_Price_Divergence_Factor",
    "author": "local",
    "level": "daily",
    "category": "volume",
    "description": "N-day ROC divergence between average trade size and price",
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
    """返回 date × instrument 的原始 ATS-Price 背离因子矩阵。

    ``data_ctx`` 键为 ``SETTING["data_needed"]`` 中的字段，值为日期升序、
    列为 instrument 的矩阵。shift 沿时间轴（axis 0），因子值只依赖当日
    及历史数据。
    """

    window = SETTING["params"]["window"]
    eps = SETTING["params"]["eps"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be an integer >= 1")

    quote_volume = data_ctx["quote_volume"].astype("float64")
    trade_count = data_ctx["trade_count"].astype("float64")
    close = data_ctx["close"].astype("float64")

    ats = quote_volume / (trade_count + eps)
    ats_roc = ats / ats.shift(window) - 1.0
    price_roc = close / close.shift(window) - 1.0
    return ats_roc - price_roc
