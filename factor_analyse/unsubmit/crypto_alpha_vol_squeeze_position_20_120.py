"""crypto_alpha_vol_squeeze_position_20_120 因子（factor_common 日频契约版）。

原始定义（分钟版语义）：
    ret = close.pct_change()（日收益）
    vol20 = rolling_std(ret, 20d)
    position = (vol20 - min(vol20, 120d)) / (max(vol20, 120d) - min(vol20, 120d))
    factor = -position  （0=最压缩 1=最扩张，取负后越压缩值越高）

方向 continuation（低波异象 / lottery 规避）：做多持续压缩、做空持续扩张，
高因子值（最压缩）为优选多腿，factor_direction=1。

日频改写判断：
- 分钟级 groupby(day).last() 聚合直接由日频 close 字段替代；20d 已实现波动率
  由日收益滚动 std 计算（与原"日频 close 收益 20d std"的定义一致，非分钟收益 RV）。
- 删除末尾 shift(1)（原仅用于执行延迟，框架按次日开盘执行，不人为延迟）。
- 删除广播回分钟 index 与 CHUNK_SIZE 分块逻辑，直接返回日频矩阵。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_vol_squeeze_position_20_120",
    "author": "wesleywu",
    "level": "daily",
    "category": "vol",
    "description": "20d 波动率在自身 120d min-max 区间位置取负，做多持续压缩（低波溢价）",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 150,
    "preprocessing": "mad_rank",
    "params": {"vol_days": 20, "range_days": 120},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily vol-squeeze position matrix (higher = more compressed)."""

    vol_days = SETTING["params"]["vol_days"]
    range_days = SETTING["params"]["range_days"]

    close_daily = data_ctx["close"].astype("float64")

    ret = close_daily.pct_change()
    vol20 = ret.rolling(vol_days, min_periods=vol_days).std()

    # 当前 vol 在自身 120d 区间中的位置，0=最压缩，1=最扩张
    lo = vol20.rolling(range_days, min_periods=range_days).min()
    hi = vol20.rolling(range_days, min_periods=range_days).max()
    position = (vol20 - lo) / ((hi - lo) + _EPS)

    # 越压缩值越高
    factor = -position
    return factor.replace([np.inf, -np.inf], np.nan)
