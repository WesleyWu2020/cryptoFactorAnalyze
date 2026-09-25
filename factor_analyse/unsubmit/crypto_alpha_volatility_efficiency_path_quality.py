"""Path-quality extension of ``crypto_alpha_volatility_efficiency``."""

from __future__ import annotations

import numpy as np
import pandas as pd


ITER_NOTE = {
    "op_type": "add_factor",
    "hypothesis": (
        "在相同 20d ROC / 平均日内振幅下，路径更平滑的趋势更可能代表持续性"
        "而非反复震荡；用路径效率作为有界质量门控可提升延续信号的可交易性。"
    ),
    "change": (
        "保留原因子的 20d 波动效率，乘以 0.5 + 0.5 * path_efficiency；"
        "path_efficiency = abs(20d ROC) / 20d 累计绝对日收益，并裁剪到 [0, 1]。"
    ),
    "expected": (
        "相对原因子 RankIC 提升至少 0.002，trading_net Sharpe 不下降超过 0.1，"
        "覆盖率保持 >= 0.6；先仅在 IS 评估，未通过则不进行 OOS。"
    ),
}

SEMANTIC_PLAN = {
    "event": "large_move",
    "context": None,
    "quality": ["path_cleanliness"],
    "direction": "continuation",
    "output": "rank",
    "parent_factor_id": "crypto_alpha_volatility_efficiency",
    "semantic_key": "large_move|-|path_cleanliness|continuation|rank",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_volatility_efficiency_path_quality",
    "author": "wesleywu",
    "level": "daily",
    "category": "volatility",
    "description": (
        "20d ROC per average daily range, gated by 20d price-path cleanliness"
    ),
}

SETTING = {
    "data_needed": ["close", "high", "low"],
    "universe": "historical_top50",
    "warmup_bars": 45,
    "preprocessing": "mad_rank",
    "params": {"window_days": 20},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the causal daily volatility-efficiency path-quality matrix."""

    window_days = SETTING["params"]["window_days"]

    close = data_ctx["close"].astype("float64").where(lambda frame: frame > 0)
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")

    prev_close = close.shift(1)
    roc = close / close.shift(window_days) - 1.0
    range_pct = ((high - low) / prev_close).rolling(
        window_days,
        min_periods=window_days,
    ).mean()

    daily_ret = close / prev_close - 1.0
    path_length = daily_ret.abs().rolling(
        window_days,
        min_periods=window_days,
    ).sum()
    path_efficiency = (roc.abs() / (path_length + _EPS)).clip(0.0, 1.0)

    base_efficiency = roc / (range_pct + _EPS)
    factor = base_efficiency * (0.5 + 0.5 * path_efficiency)
    return factor.replace([np.inf, -np.inf], np.nan)
