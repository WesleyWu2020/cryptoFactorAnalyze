"""Composite-score (先加权后分组) portfolio construction.

Each factor value matrix is standardized per day (cross-sectional z-score),
multiplied by its factor_direction, then allocation-weighted into a single
composite score. Quintile long/short targets are built on that score.

All operations use same-day cross-sections only; no future data enters.
A day on which a factor's valid values are all identical carries no ranking
signal: that factor abstains (z = 0) instead of poisoning the composite,
mirroring member_targets' handling of no-cross-sectional-spread days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_common.grouping import target_weights


def zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-day cross-sectional z-score; zero-variance days abstain (z = 0)."""
    clean = frame.replace([np.inf, -np.inf], np.nan)
    mean = clean.mean(axis=1)
    std = clean.std(axis=1, ddof=0)
    z = clean.sub(mean, axis=0).div(std.where(std > 0.0), axis=0)
    degenerate = (std <= 0.0) & clean.notna().any(axis=1)
    z.loc[degenerate] = clean.loc[degenerate].notna().astype(float) * 0.0
    return z


def composite_score(values: dict[str, pd.DataFrame], directions: dict[str, int],
                    allocations: dict[str, float]) -> pd.DataFrame:
    """Allocation-weighted sum of direction-adjusted z-scores.

    An instrument's composite is valid only when every factor has a valid
    value for it that day (intersection universe).
    """
    composite = None
    valid_mask = None
    for name, matrix in values.items():
        contribution = zscore(matrix) * int(directions[name]) * float(allocations[name])
        mask = contribution.notna()
        valid_mask = mask if valid_mask is None else (valid_mask & mask)
        filled = contribution.fillna(0.0)
        composite = filled if composite is None else composite.add(filled, fill_value=0.0)
    if composite is None:
        raise ValueError("composite requires at least one factor")
    return composite.where(valid_mask)


def composite_targets(score: pd.DataFrame, profile, min_valid: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unit-gross 50/50 long-short targets on the composite score + diagnostics."""
    count = score.notna().sum(axis=1)
    spread = score.max(axis=1) - score.min(axis=1)
    valid = (count >= min_valid) & (spread > 0.0)
    weights = target_weights(score, profile)["long_short"]
    weights.loc[~valid, :] = 0.0
    weights = weights.fillna(0.0)
    diagnostics = pd.DataFrame(
        {
            "valid_count": count,
            "valid_signal": valid,
            "reason": np.where(
                count < min_valid,
                "insufficient_values",
                np.where(spread > 0, "valid", "no_cross_section_spread"),
            ),
        },
        index=score.index,
    )
    return weights, diagnostics


__all__ = ["zscore", "composite_score", "composite_targets"]
