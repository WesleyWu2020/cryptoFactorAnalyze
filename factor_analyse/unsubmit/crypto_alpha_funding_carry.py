"""crypto_alpha_funding_carry 因子（factor_common 日频契约版）。

原始定义：
    signal = rolling_mean(daily_mean_funding, 24d)
    factor = ewm_span5(signal)

资金费率信号：高 funding -> 正向信号（数据验证高 funding 资产未来上涨）。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_funding_carry",
    "author": "wesleywu",
    "level": "daily",
    "category": "funding",
    "description": "ewm_span5(rolling_mean(daily funding rate, 24d)); high funding -> long",
}

SETTING = {
    "data_needed": ["funding"],
    "universe": "historical_top50",
    "warmup_bars": 35,
    "preprocessing": "mad_rank",
    "params": {"funding_window": 24, "smooth_span": 5},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily funding-carry matrix (date x instrument).

    ``data_ctx["funding"]`` is the daily mean funding rate
    (research_panel_daily.funding_rate_mean).
    """

    funding_window = SETTING["params"]["funding_window"]
    smooth_span = SETTING["params"]["smooth_span"]

    funding = data_ctx["funding"].astype("float64")
    signal = funding.rolling(window=funding_window, min_periods=funding_window).mean()
    # 时序 EWMA 平滑：提升信号跨日持续性，降低换手
    factor = signal.ewm(span=smooth_span, min_periods=smooth_span).mean()
    return factor.replace([np.inf, -np.inf], np.nan)
