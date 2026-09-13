"""crypto_alpha_illiq_premium_20d 因子（factor_common 日频契约版）。

原始定义（分钟版语义）：
    amihud = mean(|ret1| / dollar_volume, 20d)   非流动性水平（慢变）
    factor = 截面 pct_rank(amihud) - 0.5         多薄市币、空厚市币（非流动性溢价）

日频改写说明：
    - data_ctx 已是日频矩阵，日聚合恒等（close=last、dollar_volume 日求和 ==
      日频 quote_volume），直接使用 quote_volume。
    - 删除分钟版末尾的 .shift(1)（执行延迟由框架次日开盘成交处理）与分钟 index
      重广播。
    - 窗口单位由分钟 bar 改写为日历日，语义不变。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_illiq_premium_20d",
    "author": "wesleywu",
    "level": "daily",
    "category": "liquidity",
    "description": "amihud 非流动性 20d 均值截面 rank：多薄市币空厚市币收割非流动性溢价",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"amihud_days": 20},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily illiquidity-premium matrix (date x instrument)."""

    amihud_days = SETTING["params"]["amihud_days"]

    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].clip(lower=0).astype("float64")

    ret1 = close.pct_change(fill_method=None)

    # event illiquidity_level: amihud 非流动性 20d 均值（慢变）
    amihud = (ret1.abs() / (quote_volume + _EPS)).rolling(
        amihud_days, min_periods=amihud_days
    ).mean()

    # direction premium: 多薄市（高 amihud）空厚市（低 amihud），截面 rank 中心化
    factor = amihud.rank(axis=1, pct=True) - 0.5
    return factor.replace([np.inf, -np.inf], np.nan)
