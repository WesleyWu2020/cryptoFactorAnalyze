"""crypto_alpha_cum_return_ratio 因子（factor_common 日频契约版）。

原始定义（GP 字段语义）：
    cum_return = cumprod(1 + daily_return)（不减 1）
    factor = -sum(cum_return, 40d) / sum(sum(cum_return, 5d), 10d)

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_cum_return_ratio",
    "author": "wesleywu",
    "level": "daily",
    "category": "momentum",
    "description": "-sum(cum_return, 40d) / sum(sum(cum_return, 5d), 10d)",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 60,
    "preprocessing": "mad_rank",
    "params": {"num_window": 40, "inner_window": 5, "outer_window": 10},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily cum-return-ratio matrix (date x instrument)."""

    num_window = SETTING["params"]["num_window"]
    inner_window = SETTING["params"]["inner_window"]
    outer_window = SETTING["params"]["outer_window"]

    close = data_ctx["close"].astype("float64")
    ret_d = close.pct_change()

    # GP field semantics: cum_return = cumprod(1 + daily_return), not minus 1.
    cum_return = (1.0 + ret_d.fillna(0.0)).cumprod()

    num = -cum_return.rolling(window=num_window, min_periods=num_window).sum()
    den = cum_return.rolling(window=inner_window, min_periods=inner_window).sum()
    den = den.rolling(window=outer_window, min_periods=outer_window).sum()

    factor = num / den.where(den.abs() > _EPS)
    return factor.replace([np.inf, -np.inf], np.nan)
