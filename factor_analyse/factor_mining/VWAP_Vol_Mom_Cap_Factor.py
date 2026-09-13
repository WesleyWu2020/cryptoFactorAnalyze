"""VWAP-波动率-动量-市值 融合因子。

四个子因子（window 日窗口）等权合成：

1. VWAP偏离度: (close - vwap) / vwap，其中 vwap = quote_volume / volume
   （收盘价相对当日成交均价的偏离，正值表示尾盘强势）
2. 波动率效率: |ROC| / rolling_mean((high - low) / close, window)
   （净价格变化相对日内振幅的效率，高值表示趋势性强）
3. 动量: ROC = close / close.shift(window) - 1
4. 市值代理: log(quote_volume.rolling(window).mean() + 1)
   （用成交额均值作为市值/流动性代理）

合成方法：每个子因子先按日做截面 rank 归一化到 [-1, 1]，再等权平均。
原始合成值交由框架 mad_rank 预处理（MAD 去极值 + 按日 rank 到 [-1,1]）。
计算只使用当日及历史数据，无未来函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TYPE = "regular"

META = {
    "factor_name": "VWAP_Vol_Mom_Cap_Factor",
    "author": "local",
    "level": "daily",
    "category": "composite",
    "description": "VWAP deviation + volatility efficiency + momentum + cap-proxy composite",
}

SETTING = {
    "data_needed": ["high", "low", "close", "volume", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 21,
    "preprocessing": "mad_rank",
    "params": {"window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-10


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw composite matrix (equal-weight mean of unit-ranked sub-factors).

    All rolling/shift operations run along the daily index (axis 0), so each
    value only depends on the current and preceding ``window`` observations.
    Cross-sectional ranks are computed per date (axis 1).
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.window must be an integer >= 2")

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")

    # 子因子1: VWAP偏离度 = (close - vwap) / vwap
    vwap = quote_volume / (volume + _EPSILON)
    vwap_deviation = (close - vwap) / (vwap + _EPSILON)

    # 子因子2: 波动率效率 = |ROC| / 平均日内振幅
    roc = close / close.shift(window) - 1
    intraday_range = (high - low) / (close + _EPSILON)
    avg_range = intraday_range.rolling(window).mean()
    vol_efficiency = roc.abs() / (avg_range + _EPSILON)

    # 子因子3: 动量 = ROC
    momentum = roc

    # 子因子4: 市值代理 = log(平均成交额 + 1)
    avg_quote_volume = quote_volume.rolling(window).mean()
    cap_proxy = np.log(avg_quote_volume + 1)

    # 每个子因子按日截面 rank 归一化到 [-1, 1]，再等权合成。
    # 旧版在 rank 前做了 winsorize_by_date(n_std=3)，截断只影响极端并列值，
    # 不改变截面排序，故 rank 结果等价，此处直接对原始子因子 rank。
    def _rank_unit(mat: pd.DataFrame) -> pd.DataFrame:
        return mat.rank(axis=1, pct=True) * 2.0 - 1.0

    composite = (
        _rank_unit(vwap_deviation)
        + _rank_unit(vol_efficiency)
        + _rank_unit(momentum)
        + _rank_unit(cap_proxy)
    ) / 4.0

    return composite
