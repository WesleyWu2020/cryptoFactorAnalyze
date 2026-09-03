"""Align multiple factor frames into a single (date, instrument, *factors) panel.

Factors with different rebalance periods are forward-filled (ffill) per instrument
so that a signal issued on day T remains valid until the next rebalance signal
arrives.  This converts sparse per-rebalance data into a dense daily panel and
allows mixing factors with different rebalance cadences without losing most dates.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd


def align_factors(
    factors: Mapping[str, pd.DataFrame],
    universe_by_date: Mapping[pd.Timestamp, set[str]] | None = None,
) -> pd.DataFrame:
    if not factors:
        return pd.DataFrame(columns=["date", "instrument"])
    panel: pd.DataFrame | None = None
    for name, df in factors.items():
        df = df[["date", "instrument", name]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        panel = df if panel is None else panel.merge(
            df, on=["date", "instrument"], how="outer"
        )
    assert panel is not None

    # Forward-fill each factor column within each instrument so that a signal
    # issued on a rebalance day carries forward to subsequent days until the
    # next signal arrives.  Only forward (not backward) fill to avoid leakage.
    panel = panel.sort_values(["date", "instrument"])
    factor_cols = [c for c in panel.columns if c not in ("date", "instrument")]
    if factor_cols:
        panel[factor_cols] = panel.groupby("instrument")[factor_cols].ffill()

    if universe_by_date is not None:
        mask = panel.apply(
            lambda r: r["instrument"] in universe_by_date.get(r["date"], set()),
            axis=1,
        )
        panel = panel.loc[mask].copy()
    return panel.sort_values(["date", "instrument"]).reset_index(drop=True)
