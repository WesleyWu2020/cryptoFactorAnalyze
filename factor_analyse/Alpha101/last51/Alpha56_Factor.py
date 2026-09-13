"""Alpha101 Alpha#56 因子（factor_common 契约版）。

原始定义（原脚本 docstring）:
    (0 - (1 * (rank((sum(returns, long_window) / sum(sum(returns, short_window), sum_window)))
              * rank((returns * taker_buy_quote)))))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    1. returns = close.pct_change()
    2. returns_ratio = sum(returns, 21) / (sum(sum(returns, 3), 7) + 1e-8)
    3. returns_taker_buy = returns * taker_buy_quote_volume
    4. 因子值: -(rank(returns_ratio, 截面pct) * rank(returns_taker_buy, 截面pct))

参数取原脚本 __main__ 实际调用值（long_window=21, short_window=3, sum_window=7）。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口；
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha56_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": (
        "Alpha101 #56: -(rank(sum(returns,21)/sum(sum(returns,3),7)) "
        "* rank(returns * taker_buy_quote))"
    ),
}

SETTING = {
    "data_needed": ["close", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"long_window": 21, "short_window": 3, "sum_window": 7},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#56 matrix (date x instrument)."""

    long_window = SETTING["params"]["long_window"]
    short_window = SETTING["params"]["short_window"]
    sum_window = SETTING["params"]["sum_window"]

    close = data_ctx["close"].astype("float64")
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    returns = close.pct_change()

    # 长期收益率总和
    sum_returns_long = returns.rolling(window=long_window, min_periods=long_window).sum()
    # 短期收益率总和再累积
    sum_returns_short = returns.rolling(window=short_window, min_periods=short_window).sum()
    sum_sum_returns_short = sum_returns_short.rolling(
        window=sum_window, min_periods=sum_window
    ).sum()

    returns_ratio = sum_returns_long / (sum_sum_returns_short + _EPSILON)
    returns_taker_buy = returns * taker_buy_quote

    # 横截面 rank（按日期，pct=True）相乘后取负
    rank_returns_ratio = returns_ratio.rank(axis=1, pct=True)
    rank_returns_taker_buy = returns_taker_buy.rank(axis=1, pct=True)
    return -1.0 * (rank_returns_ratio * rank_returns_taker_buy)
