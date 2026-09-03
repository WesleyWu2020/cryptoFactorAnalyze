"""Composite-rank short-side selector: bottom-N by sum of per-factor ranks.

For each factor with direction +1 (higher=better for long), shorts pick the LOWEST rank.
For direction -1 (lower=better for long), shorts pick the HIGHEST rank.
Composite = sum of ranks; bottom-N = smallest composite (worst coins).
"""
from typing import Dict, List, Optional

import pandas as pd


def select_short_bottom_n(
    panel: pd.DataFrame,
    factors: List[str],
    directions: Dict[str, int],
    n: int,
    allowed_col: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Select bottom-N instruments for shorting based on composite rank.

    Args:
        panel: DataFrame with columns [date, instrument, *factors, ...]
        factors: List of factor column names to use
        directions: Dict mapping factor name to direction (+1 or -1)
                   +1: higher=better for long (so short lowest)
                   -1: lower=better for long (so short highest)
        n: Number of bottom (worst) instruments to select per date
        allowed_col: Optional column name for boolean mask (only use if True)

    Returns:
        Dict mapping date (str) to list of selected instruments
    """
    selected = {}
    for date, grp in panel.groupby("date"):
        g = grp.copy()
        # Apply allowed mask if provided
        if allowed_col and allowed_col in g.columns:
            g = g[g[allowed_col].astype(bool)]
        if g.empty:
            selected[str(date)] = []
            continue
        # Compute composite rank by summing per-factor ranks
        composite = pd.Series(0.0, index=g.index)
        for f in factors:
            direction = directions.get(f, +1)
            if direction == -1:
                # direction -1: higher=worse -> rank descending so worst gets lowest rank
                r = g[f].rank(ascending=False, method="average")
            else:
                # direction +1: higher=better -> rank ascending so worst (lowest value) gets lowest rank
                r = g[f].rank(ascending=True, method="average")
            composite = composite + r
        # Sort by composite rank (ascending) and pick bottom N
        g = g.assign(_composite=composite).sort_values("_composite", ascending=True)
        selected[str(date)] = g["instrument"].head(n).tolist()
    return selected
