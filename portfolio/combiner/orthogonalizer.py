"""Symmetric orthogonalization of factor panel (cross-sectional, per date).

For each date slice, computes X_ortho = X @ (X.T @ X)^{-1/2}.
Falls back to no-op when n_instruments < n_factors + 1.
"""
from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

logger = logging.getLogger(__name__)


def _sym_ortho(X: np.ndarray) -> np.ndarray:
    """Return X @ (X.T @ X)^{-1/2} using scipy sqrtm."""
    C = X.T @ X
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C_sqrt_inv = np.linalg.inv(sqrtm(C).real)
    return X @ C_sqrt_inv


def orthogonalize(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
) -> pd.DataFrame:
    out = panel.copy()
    n_factors = len(factor_cols)
    
    # Preserve the date column separately since groupby will drop it
    date_col = out["date"].copy()

    def _apply(grp: pd.DataFrame) -> pd.DataFrame:
        X = grp[list(factor_cols)].values.astype(float)
        n_ins = X.shape[0]
        # Get date from the first row's index (it's in the grouping key)
        date_val = date_col.iloc[grp.index[0]]
        if n_ins < n_factors + 1:
            logger.warning(
                "orthogonalize: degenerate date %s (%d instruments, %d factors) – skipped",
                date_val,
                n_ins,
                n_factors,
            )
            return grp
        try:
            X_ortho = _sym_ortho(X)
            result = grp.copy()
            result[list(factor_cols)] = X_ortho
            return result
        except Exception as exc:
            logger.warning("orthogonalize: failed for date %s – %s. Skipping.", date_val, exc)
            return grp

    result = (
        out.groupby("date", group_keys=False, sort=False)
        .apply(_apply)
    )
    # Restore the date column
    result["date"] = date_col.loc[result.index]
    # Reorder columns to match original
    result = result[panel.columns.tolist()]
    return result
