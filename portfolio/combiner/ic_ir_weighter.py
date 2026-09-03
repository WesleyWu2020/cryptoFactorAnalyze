"""Rolling IC/IC_IR weight computation with double-lag cutoff."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _ic_series(
    panel: pd.DataFrame,
    factor_col: str,
    label_col: str,
) -> pd.Series:
    """Cross-sectional Spearman IC per date."""
    def _ic_one(grp: pd.DataFrame) -> float:
        xy = grp[[factor_col, label_col]].dropna()
        if len(xy) < 3:
            return np.nan
        r, _ = spearmanr(xy[factor_col], xy[label_col])
        return float(r)

    return panel.groupby("date").apply(_ic_one).rename(factor_col)


def compute_ic_ir_weights(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    label_col: str = "future_ret",
    window: int = 20,
) -> pd.DataFrame:
    """Compute rolling IC_IR weights per date.

    For date t, uses IC values from [t-window, t-1] (strict: t not included).

    Returns DataFrame with columns [date, *factor_cols].
    Dates with fewer than `window` valid ICs have NaN weights.
    """
    ic_dict = {col: _ic_series(panel, col, label_col) for col in factor_cols}
    ic_df = pd.DataFrame(ic_dict)  # index = date

    results = []
    dates = ic_df.index.sort_values()

    for i, t in enumerate(dates):
        # window covers [t-window, t-1]: use ic values STRICTLY before t
        window_ic = ic_df[ic_df.index < t].iloc[-window:]
        if len(window_ic) < window:
            results.append({"date": t, **{col: np.nan for col in factor_cols}})
            continue

        weights = {}
        for col in factor_cols:
            vals = window_ic[col].dropna()
            if len(vals) < 2:
                weights[col] = 0.0
            else:
                ic_mean = vals.mean()
                ic_std = vals.std(ddof=1)
                if ic_std < 1e-12 or not np.isfinite(ic_std):
                    weights[col] = ic_mean
                else:
                    weights[col] = ic_mean / ic_std

        # Normalize: sum(|w|) = 1
        total = sum(abs(v) for v in weights.values())
        if total > 1e-12:
            weights = {k: v / total for k, v in weights.items()}
        else:
            n = len(factor_cols)
            weights = {k: 1.0 / n for k in factor_cols}

        results.append({"date": t, **weights})

    return pd.DataFrame(results).reset_index(drop=True)
