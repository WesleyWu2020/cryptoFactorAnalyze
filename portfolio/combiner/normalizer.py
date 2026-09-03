"""Cross-sectional winsorize + rank + z-score normalization."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    if s.dropna().empty:
        return s
    low = s.quantile(lo, interpolation='lower')
    high = s.quantile(hi, interpolation='lower')
    return s.clip(lower=low, upper=high)


def _zscore(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return s * 0.0
    return (s - m) / sd


def normalize_cross_section(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    directions: Mapping[str, int],
    winsorize_pct: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    lo, hi = winsorize_pct
    out = panel.copy()
    for col in factor_cols:
        sign = directions.get(col, 1)
        out[col] = out[col] * sign
        out[col] = out.groupby("date")[col].transform(
            lambda s: _winsorize(s, lo, hi)
        )
        out[col] = out.groupby("date")[col].rank(method="average", pct=True) - 0.5
        out[col] = out.groupby("date")[col].transform(_zscore)
    return out
