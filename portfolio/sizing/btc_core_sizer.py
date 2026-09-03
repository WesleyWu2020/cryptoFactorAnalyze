"""BTC core weight sizer: maps trend score T to BTC allocation weight."""

import pandas as pd


def btc_core_weight(T: pd.Series, w_min: float = 0.40, w_max: float = 0.70) -> pd.Series:
    """Map BTC trend score T to core weight allocation.
    
    Args:
        T: pd.Series indexed by date with trend scores in [-1, +1].
           -1 = bearish, 0 = neutral, +1 = bullish.
        w_min: Minimum BTC weight (default 0.40 / 40%).
        w_max: Maximum BTC weight (default 0.70 / 70%).
    
    Returns:
        pd.Series of BTC weights w_btc in [w_min, w_max].
        
    Formula:
        w_btc = w_min + (w_max - w_min) * (T + 1) / 2
        - T = -1 → w_btc = w_min
        - T = 0  → w_btc = (w_min + w_max) / 2
        - T = +1 → w_btc = w_max
        - NaN T  → w_btc = w_min (defensive neutral)
    """
    # Handle NaN by filling with w_min (conservative default)
    T_filled = T.fillna(-1.0)
    
    # Linear mapping: [-1, 1] → [w_min, w_max]
    w_btc = w_min + (w_max - w_min) * (T_filled + 1) / 2
    
    # Clip to [w_min, w_max] to handle out-of-range scores
    w_btc = w_btc.clip(w_min, w_max)
    
    return w_btc
