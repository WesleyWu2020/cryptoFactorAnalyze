"""crypto_alpha_taker_buy_ratio_persist10 因子（factor_common 日频契约版）。

原始定义（分钟版语义）：
    buy_ratio = sum(taker_buy_quote_volume, 1d) / sum(quote_volume, 1d)
    factor = rolling_mean(buy_ratio, 10d)

持续的 taker 主动买入失衡代表知情资金净流入，对未来 1 日收益有正向预测力
（知情交易延续假说；该假说后被 STATISTICAL_REJECT，但构造口径保留）。

日频改写判断：
- 分钟级 groupby(day).sum() 聚合直接由日频 taker_buy_quote_volume/quote_volume 字段替代。
- 删除 rolling 后的 shift(1)（原仅用于执行延迟，框架按次日开盘执行，不人为延迟）。
- 删除广播回分钟 index 与 CHUNK_SIZE 分块逻辑，直接返回日频矩阵。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_taker_buy_ratio_persist10",
    "author": "wesleywu",
    "level": "daily",
    "category": "pvc",
    "description": "日频 taker 主动买入成交额占比 10 日滚动均值（知情交易延续假说）",
}

SETTING = {
    "data_needed": ["taker_buy_quote_volume", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 15,
    "preprocessing": "mad_rank",
    "params": {"window_days": 10},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily persistent taker-buy-ratio matrix."""

    window_days = SETTING["params"]["window_days"]

    taker_daily = data_ctx["taker_buy_quote_volume"].astype("float64").clip(lower=0)
    qv_daily = data_ctx["quote_volume"].astype("float64").clip(lower=0)

    buy_ratio = taker_daily / (qv_daily + _EPS)
    buy_ratio = buy_ratio.where(qv_daily > 0)

    # 10 日滚动均值（persistence）
    factor = buy_ratio.rolling(window_days, min_periods=window_days).mean()
    return factor.replace([np.inf, -np.inf], np.nan)
