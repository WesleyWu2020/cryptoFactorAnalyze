"""Liquidity-adjusted volatility efficiency factor."""

from __future__ import annotations

import numpy as np
import pandas as pd


ITER_NOTE = {
    "op_type": "add_factor",
    "hypothesis": "高流动性环境中的单位波动趋势更可交易，低成交额下的效率更可能是价格失真。",
    "change": "用 20d 平均 log quote volume 的当日截面 rank 作为流动性质量门控。",
    "expected": "净 Sharpe 提升至少 0.1，最大回撤下降，换手不超过 1.0。",
}

SEMANTIC_PLAN = {
    "event": "large_move",
    "context": None,
    "quality": ["liquidity_filter"],
    "direction": "continuation",
    "output": "rank",
    "parent_factor_id": "crypto_alpha_volatility_efficiency",
    "semantic_key": "large_move|-|liquidity_filter|continuation|rank",
}

TYPE = "regular"
META = {
    "factor_name": "crypto_alpha_volatility_efficiency_liquidity_adjusted",
    "author": "wesleywu",
    "level": "daily",
    "category": "volatility_liquidity",
    "description": "20d volatility efficiency weighted by cross-sectional liquidity quality",
}
SETTING = {
    "data_needed": ["close", "high", "low", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 45,
    "preprocessing": "mad_rank",
    "params": {"window_days": 20},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    window = SETTING["params"]["window_days"]
    close = data_ctx["close"].astype("float64").where(lambda x: x > 0)
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64").where(lambda x: x > 0)

    prev_close = close.shift(1)
    roc = close / close.shift(window) - 1.0
    range_mean = ((high - low) / prev_close).rolling(window, min_periods=window).mean()
    base_efficiency = roc / (range_mean + _EPS)

    log_volume = np.log1p(quote_volume)
    volume_level = log_volume.rolling(window, min_periods=window).mean()
    liquidity_rank = volume_level.rank(axis=1, pct=True)
    factor = base_efficiency * (0.5 + 0.5 * liquidity_rank)
    return factor.replace([np.inf, -np.inf], np.nan)
