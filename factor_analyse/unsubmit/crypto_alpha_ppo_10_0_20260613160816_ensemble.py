"""crypto_alpha_ppo_10_0_20260613160816_ensemble 因子（factor_common 日频契约版）。

原始定义（PPO ensemble，4 项加权；前三项代数抵消/权重为零）：
    effective: factor = 0.3333 * Abs(Std($cum_return, 5d))
    其中 cum_return = cumprod(1 + daily_return)

分钟->日频改写判断（记录在案）：
  - 原实现对分钟 close resample("D").last() 得日收盘；日频契约下直接用 daily close。
  - 删除把日因子广播回分钟索引的 reindex 步骤与 tradable_mask 掩码
    （框架按 universe 自动掩码）。
  - 无额外 .shift 执行延迟，未做删除。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_ppo_10_0_20260613160816_ensemble",
    "author": "wesleywu",
    "level": "daily",
    "category": "ppo",
    "description": "0.3333 * Abs(Std(cumprod(1+daily_ret), 5d))：累积收益路径的短期离散度",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 15,
    "preprocessing": "mad_rank",
    "params": {"std_window": 5, "weight": 0.3333},
    "factor_direction": 1,  # 有效项权重 +0.3333，高离散度为高因子值
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily cum-return-std matrix (date x instrument)."""

    std_window = SETTING["params"]["std_window"]
    weight = SETTING["params"]["weight"]

    close = data_ctx["close"].astype("float64")
    daily_ret = close.pct_change(fill_method=None)
    cum_return = (1.0 + daily_ret.fillna(0.0)).cumprod()

    factor = cum_return.rolling(std_window, min_periods=std_window).std(ddof=1).abs()
    factor = factor * weight
    return factor.replace([np.inf, -np.inf], np.nan)
