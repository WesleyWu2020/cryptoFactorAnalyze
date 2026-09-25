"""Volatility-compression breakout release factor."""

from __future__ import annotations

import numpy as np
import pandas as pd


ITER_NOTE = {
    "op_type": "add_factor",
    "hypothesis": "低波动压缩后出现有成交额确认的区间突破，代表新信息释放，趋势更可能延续。",
    "change": "以 20d 区间突破乘 120d 波动压缩分数和成交额确认，并用 5d 指数衰减。",
    "expected": "IS RankIC >= 0.02、RankICIR >= 0.10、净 Sharpe >= 1，且与库内因子相关 < 0.85。",
}

SEMANTIC_PLAN = {
    "event": "breakout+vol_squeeze",
    "context": None,
    "quality": ["volume_confirm"],
    "direction": "continuation",
    "output": "event_decay",
    "parent_factor_id": None,
    "semantic_key": "breakout+vol_squeeze|-|volume_confirm|continuation|event_decay",
}

TYPE = "regular"
META = {
    "factor_name": "crypto_alpha_breakout_vol_squeeze_release",
    "author": "wesleywu",
    "level": "daily",
    "category": "breakout_volatility",
    "description": "volume-confirmed breakout released from a low-volatility squeeze",
}
SETTING = {
    "data_needed": ["close", "high", "low", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 150,
    "preprocessing": "mad_rank",
    "params": {"breakout_days": 20, "squeeze_days": 120, "decay_days": 5, "decay_halflife": 2.0},
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
    breakout_days = params["breakout_days"]
    squeeze_days = params["squeeze_days"]
    close = data_ctx["close"].astype("float64").where(lambda x: x > 0)
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64").where(lambda x: x > 0)

    prev_close = close.shift(1)
    daily_range = ((high - low) / (prev_close + _EPS)).replace([np.inf, -np.inf], np.nan)
    range_level = daily_range.rolling(breakout_days, min_periods=breakout_days).mean()
    range_baseline = range_level.rolling(
        squeeze_days, min_periods=squeeze_days // 2
    ).median()
    squeeze_score = (1.0 - range_level / (range_baseline + _EPS)).clip(0.0, 1.0)

    prior_high = close.rolling(breakout_days, min_periods=breakout_days).max().shift(1)
    prior_low = close.rolling(breakout_days, min_periods=breakout_days).min().shift(1)
    upside_breakout = (close / (prior_high + _EPS) - 1.0).clip(lower=0.0)
    downside_breakout = (prior_low / (close + _EPS) - 1.0).clip(lower=0.0)
    breakout_distance = upside_breakout - downside_breakout

    volume_mean = quote_volume.rolling(breakout_days, min_periods=breakout_days).mean()
    volume_ratio = quote_volume / (volume_mean + _EPS)
    volume_confirm = (0.5 + 0.5 * (volume_ratio / 2.0).clip(0.0, 1.0))

    event = breakout_distance * squeeze_score * volume_confirm
    factor = _event_decay(event, params["decay_days"], params["decay_halflife"])
    return factor.replace([np.inf, -np.inf], np.nan)
