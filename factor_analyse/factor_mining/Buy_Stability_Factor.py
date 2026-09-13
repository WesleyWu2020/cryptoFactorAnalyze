"""买入稳定因子（主动买入/成交额 的稳定性）。

公式（与旧版 compute_one 逐点等价）：

    trades_density = trade_count / (quote_volume + eps)
    density_ratio  = trades_density / (rolling_mean_N(trades_density) + eps)
    buy_ratio      = taker_buy_quote_volume
                     / (quote_volume - taker_buy_quote_volume + eps)
                     * (density_ratio ** 2 + eps)
    factor         = rolling_mean_N(buy_ratio) / (rolling_std_N(buy_ratio) + eps)

即主动买入相对主动卖出强度（经交易密度相对水平平方放大）的 N 日变异系数倒数。
所有 rolling 均沿时间轴且只用当日及历史数据，无未来函数。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Buy_Stability_Factor",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "N-day stability of the active buy/sell ratio scaled by trade density",
}

SETTING = {
    "data_needed": ["quote_volume", "trade_count", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 40,
    "preprocessing": "mad_rank",
    "params": {"lookback_days": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回买入稳定因子原始矩阵（date × instrument）。

    每列为一个 symbol 的日线序列，rolling/shift 沿时间轴进行，
    因子值只依赖当日及历史数据。
    """

    lookback_days = SETTING["params"]["lookback_days"]
    if (
        isinstance(lookback_days, bool)
        or not isinstance(lookback_days, int)
        or lookback_days < 2
    ):
        raise ValueError("SETTING.params.lookback_days must be an integer >= 2")

    quote_volume = data_ctx["quote_volume"].astype("float64")
    trade_count = data_ctx["trade_count"].astype("float64").fillna(0)
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # 交易密度（每单位成交额的笔数）→ 比例化并做时间序列归一
    trades_density = trade_count / (quote_volume + _EPSILON)
    trades_density_mean = trades_density.rolling(
        window=lookback_days, min_periods=lookback_days
    ).mean()
    trades_density_ratio = trades_density / (trades_density_mean + _EPSILON)

    # 将买/卖比值再乘以交易密度相对强度的平方
    buy_ratio = (
        taker_buy_quote / (quote_volume - taker_buy_quote + _EPSILON)
    ) * (trades_density_ratio**2 + _EPSILON)

    # 稳定性：均值 / 标准差
    buy_ratio_mean = buy_ratio.rolling(
        window=lookback_days, min_periods=lookback_days
    ).mean()
    buy_ratio_std = buy_ratio.rolling(
        window=lookback_days, min_periods=lookback_days
    ).std()
    return buy_ratio_mean / (buy_ratio_std + _EPSILON)
