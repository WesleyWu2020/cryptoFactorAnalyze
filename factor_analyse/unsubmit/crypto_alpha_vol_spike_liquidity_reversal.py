"""Liquidity-conditioned volatility-shock reversal factor."""

from __future__ import annotations

import numpy as np
import pandas as pd


ITER_NOTE = {
    "op_type": "add_factor",
    "hypothesis": "高流动性标的的异常波动冲击更可能是短期流动性失衡，冲击后存在可交易的反转。",
    "change": "用 5d/60d realized-volatility spike 识别冲击，以 20d 成交额流动性分位过滤，并对冲击反转做 5d 衰减。",
    "expected": "IS RankIC <= -0.02、RankICIR <= -0.10、净 Sharpe >= 1、最大回撤 <= 0.25。",
}

SEMANTIC_PLAN = {
    "event": "vol_spike",
    "context": "liquidity_bucket",
    "quality": ["outlier_filter"],
    "direction": "reversal",
    "output": "event_decay",
    "parent_factor_id": None,
    "semantic_key": "vol_spike|liquidity_bucket|outlier_filter|reversal|event_decay",
}

TYPE = "regular"
META = {
    "factor_name": "crypto_alpha_vol_spike_liquidity_reversal",
    "author": "wesleywu",
    "level": "daily",
    "category": "volatility_liquidity",
    "description": "liquidity-conditioned reversal after a realized-volatility spike",
}
SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 75,
    "preprocessing": "mad_rank",
    "params": {"short_days": 5, "long_days": 60, "liquidity_days": 20, "decay_days": 5, "decay_halflife": 2.0},
    "factor_direction": 1,
}

_EPS = 1e-12


def _event_decay(event: pd.DataFrame, days: int, halflife: float) -> pd.DataFrame:
    weights = np.exp(-np.arange(days, dtype=float) / halflife)
    result = pd.DataFrame(0.0, index=event.index, columns=event.columns)
    for lag, weight in enumerate(weights):
        result = result.add(event.shift(lag).fillna(0.0) * weight)
    return result


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    params = SETTING["params"]
    close = data_ctx["close"].astype("float64").where(lambda x: x > 0)
    quote_volume = data_ctx["quote_volume"].astype("float64").where(lambda x: x > 0)

    daily_ret = close / close.shift(1) - 1.0
    short_rv = daily_ret.abs().rolling(
        params["short_days"], min_periods=params["short_days"]
    ).mean()
    long_rv = daily_ret.abs().rolling(
        params["long_days"], min_periods=params["long_days"]
    ).mean()
    vol_spike = (short_rv / (long_rv + _EPS) - 1.0).clip(lower=0.0)
    signed_shock = daily_ret * vol_spike

    liquidity_level = np.log1p(quote_volume).rolling(
        params["liquidity_days"], min_periods=params["liquidity_days"]
    ).mean()
    liquidity_rank = liquidity_level.rank(axis=1, pct=True)
    liquid_weight = (0.5 + 0.5 * ((liquidity_rank - 0.4) / 0.6).clip(0.0, 1.0))

    reversal_event = -signed_shock * liquid_weight
    factor = _event_decay(reversal_event, params["decay_days"], params["decay_halflife"])
    return factor.replace([np.inf, -np.inf], np.nan)
