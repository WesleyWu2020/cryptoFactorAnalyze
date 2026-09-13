"""crypto_alpha_taker_fomo_reversal_z20 因子（factor_common 日频契约版）。

原始定义（分钟版语义）：
    buy_ratio = sum(taker_buy_quote_volume, 1d) / sum(quote_volume, 1d)
    imbalance_z = zscore_20d(buy_ratio)  （自身 20 日时序 z-score，clip ±5）
    hi_zone = close / rolling_max(close, 20d)  （(0,1]，越接近 1 越处 20 日新高）
    vol_mult = clip(quote_volume / mean(quote_volume, 20d), 0, 2) / 2  （放量确认）
    factor = -tanh(imbalance_z * hi_zone * vol_mult)

方向 reversal：高位放量买爆 = 散户 FOMO 顶部，做空（高 factor 值 = 低 FOMO = 多腿）。

日频改写判断：
- 分钟级 groupby(day).sum()/last() 聚合直接由日频字段替代。
- 删除末尾 shift(1)（原仅用于执行延迟，框架按次日开盘执行，不人为延迟）。
- 删除广播回分钟 index 与 CHUNK_SIZE 分块逻辑，直接返回日频矩阵。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_taker_fomo_reversal_z20",
    "author": "wesleywu",
    "level": "daily",
    "category": "pvc",
    "description": "taker 买占比 20d z-score × 价格高位度 × 放量倍数，取 -tanh 做空 FOMO 顶部",
}

SETTING = {
    "data_needed": ["taker_buy_quote_volume", "quote_volume", "close"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"z_window_days": 20},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily bounded FOMO-reversal score matrix."""

    z_window_days = SETTING["params"]["z_window_days"]

    taker_daily = data_ctx["taker_buy_quote_volume"].astype("float64").clip(lower=0)
    qv_daily = data_ctx["quote_volume"].astype("float64").clip(lower=0)
    close_daily = data_ctx["close"].astype("float64")

    buy_ratio = (taker_daily / (qv_daily + _EPS)).where(qv_daily > 0)

    # event: taker 失衡脉冲（自身 20 日时序 z-score，滚动窗口无前视）
    mu = buy_ratio.rolling(z_window_days, min_periods=z_window_days).mean()
    sd = buy_ratio.rolling(z_window_days, min_periods=z_window_days).std()
    imbalance_z = ((buy_ratio - mu) / (sd + _EPS)).clip(-5, 5)

    # context: 近期高位度，(0, 1]，越接近 1 越处于 20 日新高
    hi_zone = close_daily / close_daily.rolling(z_window_days, min_periods=z_window_days).max()

    # quality: 放量确认，封顶 2 倍
    vol_mult = (
        qv_daily / (qv_daily.rolling(z_window_days, min_periods=z_window_days).mean() + _EPS)
    ).clip(0, 2) / 2.0

    # reversal: 高位放量买爆 -> 做空；bounded_score
    factor = -np.tanh(imbalance_z * hi_zone * vol_mult)
    return factor.replace([np.inf, -np.inf], np.nan)
