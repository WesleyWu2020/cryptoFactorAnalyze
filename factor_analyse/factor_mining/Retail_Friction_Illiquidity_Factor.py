"""散户拥挤下的流动性耗散因子（Retail Friction / Illiquidity）。

公式（与旧版 CSV 管线 ``compute_one`` 逐点等价）：

    ret_1d_abs      = ABS(PCT_CHANGE(close, 1))
    friction_raw_1d = (ret_1d_abs / (quote_volume + eps)) * trade_count
    factor          = log1p(TS_MEAN(friction_raw_1d, window))

即“单位成交额对应的绝对收益率 × 成交笔数”的 N 日均值，再做 log1p 压缩右尾。
计算只使用当日及历史数据，无未来函数。截面 MAD 去极值与按日 rank 由
factor_common 框架的 ``mad_rank`` 预处理完成，这里返回原始因子值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Retail_Friction_Illiquidity_Factor",
    "author": "local",
    "level": "daily",
    "category": "volume",
    "description": "N-day mean of |ret| / quote_volume * trade_count (log1p)",
}

SETTING = {
    "data_needed": ["close", "quote_volume", "trade_count"],
    "universe": "historical_top50",
    "warmup_bars": 21,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "eps": 1e-5},
    "factor_direction": -1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回逐日 Retail Friction / Illiquidity 原始因子矩阵。

    ``data_ctx`` 为 date(升序) × instrument 的矩阵字典；``rolling`` 与
    ``pct_change`` 沿时间轴（axis 0）进行，每个值只依赖当日及之前的数据。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be a positive integer")
    eps = SETTING["params"]["eps"]

    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")
    trade_count = data_ctx["trade_count"].astype("float64")

    # PCT_CHANGE(close, 1)：旧版 roc 语义 close / close.shift(1) - 1
    ret_1d_abs = close.pct_change(periods=1).abs()
    friction_raw_1d = (ret_1d_abs / (quote_volume + eps)) * trade_count
    # TS_MEAN 旧版默认 min_periods = window
    factor_raw = friction_raw_1d.rolling(window=window, min_periods=window).mean()
    # log1p 变换与旧管线一致（在 winsorize/rank 之前，单调且只用当日值）
    return np.log1p(factor_raw)
