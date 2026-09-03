"""Factor cluster selector: reduce redundant factors via hierarchical correlation clustering.

For a given aligned factor panel, computes pairwise Pearson correlations,
groups highly-correlated factors into clusters, and returns one representative
per cluster — the factor with the most non-NaN observations (best data coverage).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


def cluster_and_select(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    threshold: float = 0.6,
) -> list[str]:
    """Cluster factors by pairwise |correlation| and select one representative per cluster.

    For each pair of factors where |corr| > threshold, they are merged into one cluster.
    The representative is chosen as the factor with the most non-NaN values (coverage).

    Args:
        panel: DataFrame with columns including all factor_cols.
        factor_cols: Factor column names to cluster.
        threshold: Merge factors whose |corr| exceeds this value (e.g. 0.6).
                   Set to 0.0 or below to disable (returns all factors unchanged).

    Returns:
        Ordered list of selected factor names, one per cluster.
    """
    cols = [c for c in factor_cols if c in panel.columns]
    if len(cols) < 2 or threshold <= 0:
        return list(cols)

    # Pairwise Pearson correlation over the flattened factor value matrix
    corr_mat = panel[cols].corr(method="pearson", min_periods=30)

    # Distance = 1 - |corr|; fill NaN correlations as 1.0 (maximally distant)
    dist_arr = (1.0 - corr_mat.abs().fillna(0.0)).clip(0.0, 1.0).to_numpy().copy()
    np.fill_diagonal(dist_arr, 0.0)

    # Average-linkage hierarchical clustering
    condensed = squareform(dist_arr, checks=False)
    Z = linkage(condensed, method="average")

    # Cut tree: factors within distance (1 - threshold) end up in the same cluster
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    # Group factors by cluster label
    clusters: dict[int, list[str]] = {}
    for factor, cid in zip(cols, labels):
        clusters.setdefault(int(cid), []).append(factor)

    selected: list[str] = []
    for cid in sorted(clusters):
        members = clusters[cid]
        if len(members) == 1:
            selected.append(members[0])
        else:
            # Pick member with best non-NaN coverage
            coverage = {f: int(panel[f].notna().sum()) for f in members}
            best = max(coverage, key=coverage.__getitem__)
            dropped = [f for f in members if f != best]
            logger.info(
                "cluster %d: keeping %r, dropping %r (|corr| > %.2f)",
                cid, best, dropped, threshold,
            )
            selected.append(best)

    n_in, n_out = len(cols), len(selected)
    logger.info("cluster_and_select: %d → %d factors (threshold=%.2f)", n_in, n_out, threshold)
    print(f"  [Cluster] {n_in} → {n_out} factors kept (|corr| threshold={threshold})")
    return selected
