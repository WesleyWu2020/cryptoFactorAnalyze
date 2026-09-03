"""In-Sample factor filter based on IC_IR threshold with direction auto-correction."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _cross_sectional_ic(panel: pd.DataFrame, factor_col: str, label_col: str) -> pd.Series:
    """Spearman rank IC per date."""
    def _one(grp: pd.DataFrame) -> float:
        xy = grp[[factor_col, label_col]].dropna()
        if len(xy) < 3:
            return np.nan
        r, _ = spearmanr(xy[factor_col], xy[label_col])
        return float(r)
    return panel.groupby("date").apply(_one)


def filter_factors_by_is_ic_ir(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    label_col: str = "future_ret",
    is_end_date: pd.Timestamp | str = "2023-12-31",
    min_ic_ir: float = 0.05,
) -> dict[str, int]:
    """Filter factors by IS-period |IC_IR| >= threshold, returning direction.

    Args:
        panel: Long-format DataFrame with [date, instrument, factor_cols..., label_col]
        factor_cols: Candidate factor column names
        label_col: Forward return column
        is_end_date: Cutoff date (inclusive) — only dates <= this are used for IC
        min_ic_ir: |IC_IR| threshold

    Returns:
        {factor_name: direction} where direction in {+1, -1}.
        Factors with |IC_IR| < threshold are excluded.
    """
    is_end = pd.Timestamp(is_end_date).normalize()
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    is_panel = panel[panel["date"] <= is_end]

    selected: dict[str, int] = {}
    for col in factor_cols:
        ic_series = _cross_sectional_ic(is_panel, col, label_col).dropna()
        if len(ic_series) < 20:  # need enough samples
            continue
        ic_mean = ic_series.mean()
        ic_std = ic_series.std(ddof=1)
        if not np.isfinite(ic_std) or ic_std < 1e-12:
            continue
        ic_ir = ic_mean / ic_std
        if abs(ic_ir) >= min_ic_ir:
            selected[col] = 1 if ic_ir > 0 else -1
    return selected
