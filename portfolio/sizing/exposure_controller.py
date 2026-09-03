"""Map (BTC trend T, portfolio beta) -> (alt_exposure, hedge_ratio)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_exposure_and_hedge(
    T: float,
    beta_port: float,
    alt_min: float = 0.5,
    alt_max: float = 1.0,
    hedge_cap_mult: float = 1.2,
) -> tuple[float, float]:
    """Compute target (alt_exposure, hedge_ratio) from trend T and portfolio beta.

    alt_exposure = alt_min + (alt_max - alt_min) * (T + 1) / 2
    hedge_raw    = beta_port * alt_exposure * (1 - T)
    hedge_ratio  = min(hedge_raw, alt_exposure * hedge_cap_mult)
    hedge_ratio  = max(hedge_ratio, 0.0)

    Args:
        T: BTC trend score, expected in [-1, +1] but clipped defensively
        beta_port: Portfolio beta vs BTC
    Returns:
        (alt_exposure, hedge_ratio) both as floats.
    """
    T_clip = max(-1.0, min(1.0, float(T)))
    alt_exp = alt_min + (alt_max - alt_min) * (T_clip + 1.0) / 2.0
    hedge_raw = max(0.0, float(beta_port) * alt_exp * (1.0 - T_clip))
    hedge_cap = alt_exp * hedge_cap_mult
    hedge = min(hedge_raw, hedge_cap)
    return float(alt_exp), float(hedge)


def smooth_exposure_series(raw: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Apply 1-step recursive EMA: x_t = alpha * raw_t + (1-alpha) * x_{t-1}.

    Uses ewm with adjust=False for consistency with backward-looking semantics.
    """
    if raw.empty:
        return raw
    # span s.t. alpha = 2/(s+1) => s = 2/alpha - 1
    span = max(1.0, 2.0 / alpha - 1.0)
    return raw.ewm(span=span, adjust=False, min_periods=1).mean()
