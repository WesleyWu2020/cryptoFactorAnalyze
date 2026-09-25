"""Funding-crowding-filtered volatility efficiency factor."""

from __future__ import annotations

import numpy as np
import pandas as pd


ITER_NOTE = {
    "op_type": "add_factor",
    "hypothesis": "未被极端 funding 拥挤放大的趋势效率更可能持续，拥挤趋势更容易反转。",
    "change": "用 funding 20d 时序 z-score 构造 0.5~1.0 的拥挤度惩罚门控。",
    "expected": "RankIC 提升至少 0.002，最大回撤下降，净 Sharpe 提升至少 0.1。",
}

SEMANTIC_PLAN = {
    "event": "large_move",
    "context": "funding_crowding",
    "quality": [],
    "direction": "continuation",
    "output": "rank",
    "parent_factor_id": "crypto_alpha_volatility_efficiency",
    "semantic_key": "large_move|funding_crowding|-|continuation|rank",
}

TYPE = "regular"
META = {
    "factor_name": "crypto_alpha_volatility_efficiency_funding_crowding",
    "author": "wesleywu",
    "level": "daily",
    "category": "volatility_funding",
    "description": "20d volatility efficiency penalized by funding-rate crowding",
}
SETTING = {
    "data_needed": ["close", "high", "low", "funding"],
    "universe": "historical_top50",
    "warmup_bars": 45,
    "preprocessing": "mad_rank",
    "params": {"window_days": 20, "crowding_z_cap": 3.0},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    params = SETTING["params"]
    window = params["window_days"]
    close = data_ctx["close"].astype("float64").where(lambda x: x > 0)
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    funding = data_ctx["funding"].astype("float64")

    prev_close = close.shift(1)
    roc = close / close.shift(window) - 1.0
    range_mean = ((high - low) / prev_close).rolling(window, min_periods=window).mean()
    base_efficiency = roc / (range_mean + _EPS)

    funding_mean = funding.rolling(window, min_periods=window).mean()
    funding_std = funding.rolling(window, min_periods=window).std().replace(0.0, np.nan)
    funding_z = (funding - funding_mean) / (funding_std + _EPS)
    crowding = (funding_z.abs() / params["crowding_z_cap"]).clip(0.0, 1.0)
    factor = base_efficiency * (1.0 - 0.5 * crowding)
    return factor.replace([np.inf, -np.inf], np.nan)
