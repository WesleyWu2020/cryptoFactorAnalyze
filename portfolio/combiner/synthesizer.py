"""Synthesize normalized factors into a composite score using IC_IR weights."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def synthesize(
    panel: pd.DataFrame,
    weights: pd.DataFrame,
    factor_cols: Sequence[str],
) -> pd.DataFrame:
    """Compute composite_score = weighted sum of normalized factor columns.

    Args:
        panel: DataFrame with [date, instrument, *factor_cols]
        weights: DataFrame with [date, *factor_cols] — one row per date
        factor_cols: list of factor column names

    Returns:
        DataFrame with [date, instrument, composite_score]
    """
    n = len(factor_cols)
    merged = panel[["date", "instrument"] + list(factor_cols)].merge(
        weights[["date"] + list(factor_cols)].rename(
            columns={col: f"_w_{col}" for col in factor_cols}
        ),
        on="date",
        how="left",
    )

    score = pd.Series(0.0, index=merged.index)
    for col in factor_cols:
        w_col = f"_w_{col}"
        w = merged[w_col]
        # If all weights for this date are NaN, fall back to equal weight
        all_nan_mask = merged.groupby("date")[w_col].transform(lambda s: s.isna().all())
        w = w.where(~all_nan_mask, other=1.0 / n)
        score += merged[col] * w

    result = merged[["date", "instrument"]].copy()
    result["composite_score"] = score
    return result.reset_index(drop=True)
