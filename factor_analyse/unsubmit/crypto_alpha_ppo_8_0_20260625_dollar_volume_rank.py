"""crypto_alpha_ppo_8_0_20260625_dollar_volume_rank 因子（factor_common 日频契约版）。

原始定义（PPO best pool, crypto_8_0_20260625134822）：
    有效表达式：-0.5 * $dollar_volume_rank（另一项权重为 0）
    AlphaGen DOLLAR_VOLUME_RANK：
      smoothed = mean(daily_dollar_volume, 30d, min_periods=1)
      rank = 截面 rank(log1p(smoothed))，最低值 -> 0
      factor = -rank（高因子 = 低成交额排名 = 小市值/低流动性币做多）

分钟->日频改写判断（记录在案）：
  - 原实现把分钟 close*volume 按日求和得到日 dollar_volume；日频契约下直接等价于
    quote_volume 字段（分钟成交额按日求和 == 日 quote_volume），无需再用 close*volume。
  - 原 AlphaGen 定义中 daily_dv.shift(1) 是为了避免使用当日未完成的部分成交额；
    日频契约下 date t 的成交额是完整交易日数据，该 1 日滞后属于执行延迟，予以删除。
  - 截面 rank 原实现对 tradable 集合排序；框架按 universe 自动掩码，
    改用 pandas rank(axis=1, pct=True)（NaN 自动排除）。
  - 常数权重 0.5 对单调打分无影响，保留符号 -rank。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_ppo_8_0_20260625_dollar_volume_rank",
    "author": "wesleywu",
    "level": "daily",
    "category": "liquidity",
    "description": "-rank(log1p(mean(quote_volume, 30d)))：做多低成交额排名（低流动性）币",
}

SETTING = {
    "data_needed": ["quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 40,
    "preprocessing": "mad_rank",
    "params": {"dv_lookback": 30},
    "factor_direction": 1,  # 因子已带负号：高因子值 = 低成交额 = 多头侧
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily dollar-volume-rank matrix (date x instrument)."""

    dv_lookback = SETTING["params"]["dv_lookback"]

    quote_volume = data_ctx["quote_volume"].astype("float64").clip(lower=0.0)

    # 30 日平滑（min_periods=1 与 AlphaGen 一致）
    dv_smoothed = quote_volume.rolling(dv_lookback, min_periods=1).mean()
    log_dv = np.log1p(dv_smoothed.where(dv_smoothed > 0))

    # 截面百分位 rank（最低值 -> 最小 rank）；取负使低成交额为高因子值
    dv_rank = log_dv.rank(axis=1, pct=True)
    factor = -dv_rank
    return factor.replace([np.inf, -np.inf], np.nan)
