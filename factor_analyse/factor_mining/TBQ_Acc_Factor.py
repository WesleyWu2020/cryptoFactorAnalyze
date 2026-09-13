"""主动买入比率累积量 × 波动效率因子 (Taker Buy Ratio Accumulation × Volatility Efficiency)。

公式:

    taker_buy_ratio = clip(taker_buy_quote_volume / (quote_volume + eps), 0, 1)
    accumulation    = rolling_sum(taker_buy_ratio, accumulation_window)
    roc             = close / close.shift(vol_eff_window) - 1
    range_pct       = rolling_mean((high - low) / (close.shift(1) + eps), vol_eff_window)
    vol_efficiency  = roc / (range_pct + eps)
    factor          = rank_cs(accumulation, pct=True) * vol_efficiency

其中 ``rank_cs`` 为按日截面百分位排名(axis=1)。主动买入比率的高累积
代表持续的主动买盘压力，与波动效率（方向性收益/区间波动）相乘，
衡量“有方向的主动买入动量”。全部计算只依赖当日及历史数据，无未来函数。
"""

from __future__ import annotations

import pandas as pd

TYPE = "regular"

META = {
    "factor_name": "TBQ_Acc_Factor",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "Taker-buy-ratio accumulation times volatility efficiency",
}

SETTING = {
    "data_needed": ["close", "high", "low", "quote_volume", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 11,
    "preprocessing": "mad_rank",
    "params": {"accumulation_window": 10, "vol_eff_window": 10, "rebalance_period": 10},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始因子矩阵(date × instrument)。

    rolling/shift 均沿时间轴,截面 rank 沿 axis=1,
    每个因子值只依赖当日及历史数据。
    """

    accumulation_window = SETTING["params"]["accumulation_window"]
    vol_eff_window = SETTING["params"]["vol_eff_window"]

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # 1. 主动买入比率,裁剪到 [0, 1]
    taker_buy_ratio = (taker_buy_quote / (quote_volume + _EPSILON)).clip(0, 1)

    # 2. 累积量: 对主动买入比率做 accumulation_window 日滚动求和
    accumulation = taker_buy_ratio.rolling(
        window=accumulation_window,
        min_periods=accumulation_window,
    ).sum()

    # 3. 波动效率: roc / range_pct
    roc = close / close.shift(vol_eff_window) - 1
    range_pct = ((high - low) / (close.shift(1) + _EPSILON)).rolling(
        window=vol_eff_window,
        min_periods=vol_eff_window,
    ).mean()
    volatility_efficiency = roc / (range_pct + _EPSILON)

    # 4. 组合: 累积量的按日截面百分位排名 × 波动效率
    accumulation_rank = accumulation.rank(axis=1, pct=True)
    return accumulation_rank * volatility_efficiency
