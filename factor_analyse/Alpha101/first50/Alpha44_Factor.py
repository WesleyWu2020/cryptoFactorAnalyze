"""Alpha101 Alpha#44 因子（factor_common 契约版）。

原始定义:
    Alpha#44 = (-1 * correlation(high, rank(volume), 5))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    price_base = mean(close, 20)（min_periods=1）
    norm_high = high / price_base
    vol_rank = rolling_rank(volume, 20, min_periods=5)
        （注意: 原脚本用的是相对自身历史的时间序列 rolling rank，而非横截面 rank，
         注释中明确说明关注单 token 内部的价量关系，予以保留）
    factor = -correlation(norm_high, vol_rank, corr_window)
参数取原脚本 __main__ 实际调用值: corr_window=20。
旧脚本 rebalance_period 仅用于旧版 future_ret，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha44_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #44: -correlation(high, ts_rolling_rank(volume), N)",
}

SETTING = {
    "data_needed": ["high", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 45,
    "preprocessing": "mad_rank",
    "params": {"corr_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#44 matrix (date x instrument)."""

    corr_window = SETTING["params"]["corr_window"]
    if isinstance(corr_window, bool) or not isinstance(corr_window, int) or corr_window < 2:
        raise ValueError("SETTING.params.corr_window must be an integer >= 2")

    high = data_ctx["high"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 价格归一化（适应币圈价格差异），与原脚本一致 min_periods=1
    price_base = close.rolling(window=20, min_periods=1).mean()
    norm_high = high / (price_base + _EPSILON)

    # 交易量相对自身历史的 rolling rank（与原脚本一致 window=20, min_periods=5）
    vol_rank = volume.rolling(window=20, min_periods=5).rank(pct=True)

    # 最高价与交易量排名的相关性，取负
    high_vol_corr = norm_high.rolling(window=corr_window, min_periods=corr_window).corr(vol_rank)
    return -1.0 * high_vol_corr
