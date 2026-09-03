"""Build equal-weighted Top-N long portfolio from composite scores."""
from __future__ import annotations

import pandas as pd


def build_topn_weights(
    scores: pd.DataFrame,
    universe: dict[pd.Timestamp, set[str]],
    top_n: int = 10,
    rebalance_period: int = 1,
) -> dict[pd.Timestamp, dict[str, float]]:
    """Select top-N instruments by composite_score per rebalance date, equal-weighted.

    Args:
        scores: DataFrame with [date, instrument, composite_score]
        universe: dict mapping date -> set of tradeable instruments
        top_n: number of instruments to select per date
        rebalance_period: only rebalance every N trading days (1=daily, 5=weekly)

    Returns:
        dict mapping rebalance date -> {instrument: weight} with weights summing to 1.0.
        Only includes dates that are rebalance triggers; the backtester handles hold days.
    """
    result: dict[pd.Timestamp, dict[str, float]] = {}

    all_dates = sorted(scores["date"].unique())

    for i, date in enumerate(all_dates):
        if i % rebalance_period != 0:
            continue  # skip non-rebalance days

        group = scores[scores["date"] == date]
        allowed = universe.get(date, set())
        filtered = group[group["instrument"].isin(allowed)]
        if filtered.empty:
            result[date] = {}
            continue
        top = filtered.nlargest(top_n, "composite_score")
        n = len(top)
        result[date] = {row["instrument"]: 1.0 / n for _, row in top.iterrows()}

    return result
