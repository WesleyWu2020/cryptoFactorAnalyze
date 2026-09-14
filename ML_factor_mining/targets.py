"""Evaluation-only target construction and purged sample partitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_common.labels import label_dates, make_labels

from .dataset import rank_panel, stack_matrix


def build_targets(opens: pd.DataFrame, membership: pd.DataFrame, config) -> pd.DataFrame:
    """Build next-open supervised targets, masked by same-day eligibility.

    The returned frame contains evaluation-only future data.  It must remain
    separate from feature construction and ``FeatureMap`` admission.
    """
    clean_opens = opens.where(np.isfinite(opens) & (opens > 0))
    raw = make_labels(clean_opens, config.holding_days)

    eligible = membership.reindex(index=raw.index, columns=raw.columns)
    eligible = eligible.fillna(False).astype(bool)
    raw = raw.where(eligible)
    raw = raw.where(np.isfinite(raw))
    valid_counts = raw.notna().sum(axis=1)
    raw = raw.where(valid_counts >= config.min_pairs, np.nan, axis=0)

    raw_return = stack_matrix(raw)
    if config.target_type == "rank_return":
        target = rank_panel(raw_return)
    elif config.target_type == "raw_return":
        target = raw_return.copy()
    elif config.target_type == "excess_return":
        target = raw_return - raw_return.groupby(level="date", sort=False).transform("mean")
    else:
        raise ValueError(f"unsupported target_type: {config.target_type!r}")

    entry_dates, exit_dates = label_dates(opens.index, config.holding_days)
    row_dates = raw_return.index.get_level_values("date")
    entry_by_signal = pd.Series(entry_dates, index=opens.index)
    exit_by_signal = pd.Series(exit_dates, index=opens.index)
    frame = pd.DataFrame(index=raw_return.index)
    frame["raw_return"] = raw_return
    frame["target"] = target.reindex(raw_return.index)
    frame["entry_date"] = entry_by_signal.reindex(row_dates).to_numpy()
    frame["exit_date"] = exit_by_signal.reindex(row_dates).to_numpy()
    return frame


def partition_masks(index: pd.MultiIndex, fold, holding_days: int) -> dict[str, np.ndarray]:
    """Return fit, validation, and refit masks with exit-time purging."""
    dates = index.get_level_values("date")
    exits = dates + pd.Timedelta(days=holding_days + 1)

    history_start = pd.Timestamp(fold.history_start)
    validation_start = pd.Timestamp(fold.validation_start)
    retrain_at = pd.Timestamp(fold.retrain_at)
    return {
        "fit": (
            (dates >= history_start)
            & (dates < validation_start)
            & (exits < validation_start)
        ),
        "validation": (
            (dates >= validation_start)
            & (dates < retrain_at)
            & (exits < retrain_at)
        ),
        "refit": (
            (dates >= history_start)
            & (dates < retrain_at)
            & (exits < retrain_at)
        ),
    }
