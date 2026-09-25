"""Downside-risk efficiency extension of the volatility-efficiency factor."""

from __future__ import annotations

import numpy as np
import pandas as pd


ITER_NOTE = {
    "op_type": "add_factor",
    "hypothesis": "把趋势收益除以下行半方差而非总日内振幅，可区分健康上涨与脆弱反弹。",
    "change": "分子保留 20d ROC，分母改为 20d downside semideviation。",
    "expected": "RankIC 提升至少 0.003，净 Sharpe 提升至少 0.1，最大回撤不超过 0.25。",
}

SEMANTIC_PLAN = {
    "event": "large_move",
    "context": None,
    "quality": [],
    "direction": "continuation",
    "output": "rank",
    "parent_factor_id": "crypto_alpha_volatility_efficiency",
    "semantic_key": "large_move|-|-|continuation|rank",
}

TYPE = "regular"
META = {
    "factor_name": "crypto_alpha_volatility_efficiency_downside_risk",
    "author": "wesleywu",
    "level": "daily",
    "category": "volatility",
    "description": "20d return scaled by downside semideviation",
}
SETTING = {
    "data_needed": ["close"],
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
    prev_close = close.shift(1)
    daily_ret = close / prev_close - 1.0
    roc = close / close.shift(window) - 1.0
    downside = daily_ret.clip(upper=0.0).pow(2).rolling(
        window, min_periods=window
    ).mean().pow(0.5)
    factor = roc / (downside + _EPS)
    return factor.replace([np.inf, -np.inf], np.nan)
