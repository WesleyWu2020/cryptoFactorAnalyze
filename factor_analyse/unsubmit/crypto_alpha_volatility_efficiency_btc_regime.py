"""BTC-regime-conditioned volatility efficiency factor."""

from __future__ import annotations

import numpy as np
import pandas as pd


ITER_NOTE = {
    "op_type": "add_factor",
    "hypothesis": "个币趋势效率在 BTC 明确趋势状态下更容易延续，在市场震荡状态下更易反转。",
    "change": "用 BTC 20d 趋势相对 20d 波动的强度构造市场状态权重。",
    "expected": "趋势状态下 RankIC 提升至少 0.002，净 Sharpe 提升至少 0.1。",
}

SEMANTIC_PLAN = {
    "event": "large_move",
    "context": "btc_trend",
    "quality": [],
    "direction": "continuation",
    "output": "rank",
    "parent_factor_id": "crypto_alpha_volatility_efficiency",
    "semantic_key": "large_move|btc_trend|-|continuation|rank",
}

TYPE = "regular"
META = {
    "factor_name": "crypto_alpha_volatility_efficiency_btc_regime",
    "author": "wesleywu",
    "level": "daily",
    "category": "volatility_regime",
    "description": "20d volatility efficiency weighted by BTC trend regime strength",
}
SETTING = {
    "data_needed": ["close", "high", "low"],
    "universe": "historical_top50",
    "warmup_bars": 45,
    "preprocessing": "mad_rank",
    "params": {"window_days": 20, "trend_strength_cap": 2.0},
    "factor_direction": 1,
}

_EPS = 1e-12
_BTC_SYMBOLS = ("BTCUSDT", "BTC-USD", "BTCUSD")


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    params = SETTING["params"]
    window = params["window_days"]
    close = data_ctx["close"].astype("float64").where(lambda x: x > 0)
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")

    btc = next((name for name in _BTC_SYMBOLS if name in close.columns), None)
    if btc is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    prev_close = close.shift(1)
    roc = close / close.shift(window) - 1.0
    range_mean = ((high - low) / prev_close).rolling(window, min_periods=window).mean()
    base_efficiency = roc / (range_mean + _EPS)

    btc_ret = close[btc] / close[btc].shift(1) - 1.0
    btc_trend = close[btc] / close[btc].shift(window) - 1.0
    btc_vol = btc_ret.rolling(window, min_periods=window).std()
    trend_strength = btc_trend.abs() / (btc_vol * np.sqrt(window) + _EPS)
    regime = (trend_strength / params["trend_strength_cap"]).clip(0.0, 1.0)
    factor = base_efficiency.mul(0.5 + 0.5 * regime, axis=0)
    return factor.replace([np.inf, -np.inf], np.nan)
