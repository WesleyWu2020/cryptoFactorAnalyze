"""crypto_alpha_activity_burst_premium_20d 因子（factor_common 日频契约版）。

原始定义（ITER_NOTE 假设）：
    tc_daily = sum(trade_count)  ->  日频 trade_count 直接使用
    burst = rolling_std(tc, 20d) / rolling_mean(tc, 20d)   （交易活跃度 20d 变异系数）
    factor = -(rank(burst) - 0.5)   （空阵发 / 多平稳）

活动阵发性 = 订单流层的彩票特征：高 CV（事件驱动投机脉冲）做空，低 CV（稳定活动）做多。
符号已内嵌在因子公式中，factor_direction=1（因子值越高越偏多）。
转换判断说明：
- xbinance_trade_count -> trade_count（分钟计数日求和 == 日 trade_count）。
- 旧分钟版仅需 close 的分钟索引做广播，日频输入下不再需要，data_needed 只留 trade_count。
- 删除旧版 factor_daily.shift(1)（仅为分钟级执行延迟）；t 日因子只用 <= t 数据。
- 删除分块列循环，直接整帧运算。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_activity_burst_premium_20d",
    "author": "wesleywu",
    "level": "daily",
    "category": "orderflow",
    "description": "-(rank(20d CV of trade_count) - 0.5): short bursty, long steady activity",
}

SETTING = {
    "data_needed": ["trade_count"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"burst_days": 20},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily activity-burst-premium matrix (date x instrument)."""

    burst_days = SETTING["params"]["burst_days"]

    tc_daily = data_ctx["trade_count"].astype("float64").clip(lower=0)

    # event activity_burst: 活跃度 20d 变异系数（尺度无关，慢变）
    tc_mean = tc_daily.rolling(burst_days, min_periods=burst_days).mean()
    tc_std = tc_daily.rolling(burst_days, min_periods=burst_days).std()
    burst = tc_std / (tc_mean + _EPS)

    # direction premium: 空阵发（彩票/事件驱动投机）多平稳（持续吸筹），固定符号
    factor = -(burst.rank(axis=1, pct=True) - 0.5)
    return factor.replace([np.inf, -np.inf], np.nan)
