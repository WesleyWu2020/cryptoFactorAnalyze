"""crypto_alpha_funding_deviation_revert_20_60 因子（factor_common 日频契约版）。

原始定义（分钟版语义）：
    f_daily = 日频 funding 均值
    f_level = rolling_mean(f_daily, 20d)
    f_base  = rolling_median(f_daily, 60d)
    factor  = -(f_level - f_base)   # mean_reversion: 做空持续高偏离，做多低偏离

日频改写说明：
    - data_ctx 已是日频矩阵，funding 日聚合（均值）恒等，直接使用。
    - 删除分钟版末尾的 .shift(1)（执行延迟由框架次日开盘成交处理）与分钟 index 重广播。
    - 窗口单位由 1440 分钟 bar 改写为日历日，语义不变。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_funding_deviation_revert_20_60",
    "author": "wesleywu",
    "level": "daily",
    "category": "funding_sentiment",
    "description": "日频 funding 20d 均值 - 60d 滚动中位数，取负（做空高偏离，押拥挤平仓回归均衡）",
}

SETTING = {
    "data_needed": ["funding"],
    "universe": "historical_top50",
    "warmup_bars": 70,
    "preprocessing": "mad_rank",
    "params": {"level_days": 20, "base_days": 60},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily funding-deviation-revert matrix (date x instrument).

    ``data_ctx["funding"]`` is the daily mean funding rate
    (research_panel_daily.funding_rate_mean).
    """

    level_days = SETTING["params"]["level_days"]
    base_days = SETTING["params"]["base_days"]

    funding = data_ctx["funding"].astype("float64")

    f_level = funding.rolling(level_days, min_periods=level_days).mean()
    f_base = funding.rolling(base_days, min_periods=base_days).median()

    # mean_reversion: 做空持续高偏离，做多持续低偏离
    factor = -(f_level - f_base)
    return factor.replace([np.inf, -np.inf], np.nan)
