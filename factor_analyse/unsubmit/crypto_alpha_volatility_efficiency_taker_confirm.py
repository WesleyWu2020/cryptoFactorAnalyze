"""Taker-flow-confirmed volatility efficiency factor."""

from __future__ import annotations

import numpy as np
import pandas as pd


ITER_NOTE = {
    "op_type": "add_factor",
    "hypothesis": "趋势效率若得到主动买卖流持续确认，较少是被动价格噪声，延续概率更高。",
    "change": "在原因子的 20d 波动效率上加入 20d taker quote imbalance 方向确认门控。",
    "expected": "RankIC 提升至少 0.002，净 Sharpe 提升至少 0.1，覆盖率保持 >= 0.6。",
}

SEMANTIC_PLAN = {
    "event": "large_move",
    "context": None,
    "quality": ["taker_confirm"],
    "direction": "continuation",
    "output": "rank",
    "parent_factor_id": "crypto_alpha_volatility_efficiency",
    "semantic_key": "large_move|-|taker_confirm|continuation|rank",
}

TYPE = "regular"
META = {
    "factor_name": "crypto_alpha_volatility_efficiency_taker_confirm",
    "author": "wesleywu",
    "level": "daily",
    "category": "volatility_order_flow",
    "description": "20d volatility efficiency confirmed by directional taker flow",
}
SETTING = {
    "data_needed": ["close", "high", "low", "quote_volume", "taker_buy_quote_volume"],
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
    taker_buy = data_ctx["taker_buy_quote_volume"].astype("float64")

    prev_close = close.shift(1)
    roc = close / close.shift(window) - 1.0
    range_mean = ((high - low) / prev_close).rolling(window, min_periods=window).mean()
    base_efficiency = roc / (range_mean + _EPS)

    flow_imbalance = 2.0 * taker_buy / quote_volume - 1.0
    flow_confirmation = flow_imbalance.rolling(window, min_periods=window).mean()
    confirmation = (flow_confirmation * np.sign(roc)).clip(-1.0, 1.0)
    factor = base_efficiency * (1.0 + 0.5 * confirmation)
    return factor.replace([np.inf, -np.inf], np.nan)
