"""Alpha101 Alpha#28 因子（factor_common 契约版）。

原始定义:
    Alpha#28 = scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准，与经典公式有差异）:
    adv20            = mean(volume, adv_window)
    corr             = correlation(adv20, low, corr_window)
    price_center_norm = ((high + low) / 2) / close   # 价格中心对 close 归一化
    factor           = corr + price_center_norm - 1  # 归一化后 - close 等价于 -1
参数取原脚本 __main__ 实际调用值 corr_window=10，adv_window 沿用代码内硬编码 20
（rebalance_period 仅用于旧版 future_ret，丢弃；经典公式的 scale 由框架 preprocessing 替代）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha28_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #28: correlation(mean(volume,20), low, 10) + (high+low)/2/close - 1",
}

SETTING = {
    "data_needed": ["volume", "high", "low", "close"],
    "universe": "historical_top50",
    "warmup_bars": 35,
    "preprocessing": "mad_rank",
    "params": {"adv_window": 20, "corr_window": 10},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#28 matrix (date x instrument)."""

    adv_window = SETTING["params"]["adv_window"]
    corr_window = SETTING["params"]["corr_window"]
    for name, value in (("adv_window", adv_window), ("corr_window", corr_window)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    volume = data_ctx["volume"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")

    # 1. adv_window 日平均成交量
    adv = volume.rolling(window=adv_window, min_periods=adv_window).mean()

    # 2. correlation(adv, low, corr_window)
    corr = adv.rolling(window=corr_window, min_periods=corr_window).corr(low)

    # 3. 日内价格中心对 close 归一化
    price_center_norm = ((high + low) / 2.0) / close

    # 4. 复合项（归一化后原式的 - close 变为 - 1）
    return corr + price_center_norm - 1.0
