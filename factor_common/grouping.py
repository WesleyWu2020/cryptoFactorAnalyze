"""Deterministic cross-sectional grouping and portfolio target weights.

Grouping contract: on each date, valid (finite) factor values are sorted by
``(factor, instrument)`` ascending so ties always break by instrument name.
The row at sorted position ``p`` of ``count`` valid names joins group
``floor(p * n_groups / count) + 1``; group 1 holds the lowest factor values
and group ``n_groups`` the highest. A date with fewer than ``n_groups``
valid names is reported in the ``insufficient_dates`` diagnostic and keeps
all-NaN groups; the configured group count is never reduced silently.

``target_weights(values, profile)`` returns a dict of weight DataFrames
sharing the input axes, keyed by portfolio name:

- ``"group_1"`` .. ``"group_{n_groups}"``: equal-weight long-only targets for
  each group, each row summing to ``profile.gross_exposure`` on valid dates.
- ``"directional_long"``: the long-only portfolio in the factor's preferred
  direction — the top group (``n_groups``) when ``factor_direction == 1``
  and the bottom group (``1``) when ``factor_direction == -1``.
- ``"long_short"``: a 50/50 dollar-neutral portfolio, long the top group and
  short the bottom group, each leg at half of ``profile.gross_exposure``.

Dates without enough valid names produce all-NaN target rows (no tradeable
signal), never a silently smaller grouping. Everything here uses same-day
signal values only; no future data enters grouping or weights.
"""

import numpy as np
import pandas as pd

from .profiles import BacktestProfile
from .value_engine import _validate_axes


def assign_groups(values: pd.DataFrame, n_groups: int) -> tuple[pd.DataFrame, dict[str, list]]:
    """Assign each valid factor value to a deterministic group 1..n_groups.

    Returns the group matrix (NaN where the value is missing or the date is
    insufficient) and diagnostics listing dates with fewer than n_groups
    valid names under ``"insufficient_dates"``.
    """
    _validate_axes(values, name="values")
    if isinstance(n_groups, bool) or not isinstance(n_groups, int) or n_groups < 2:
        raise ValueError("n_groups must be an integer greater than or equal to 2")
    groups = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    insufficient = []
    for date, row in values.iterrows():
        valid = row.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        count = len(valid)
        if count < n_groups:
            insufficient.append(date)
            continue
        # Stable mergesort on (factor, instrument): ties keep instrument order.
        ordered = valid.sort_values(kind="mergesort")
        assigned = np.floor(np.arange(count) * n_groups / count).astype(int) + 1
        groups.loc[date, ordered.index] = assigned.astype(float)
    return groups, {"insufficient_dates": insufficient}


def _long_leg(groups: pd.DataFrame, group_id: int, exposure: float) -> pd.DataFrame:
    member = groups == group_id
    counts = member.sum(axis=1)
    return member.astype(float).mul(exposure).div(counts.replace(0, np.nan), axis=0)


def target_weights(values: pd.DataFrame, profile: BacktestProfile) -> dict[str, pd.DataFrame]:
    """Build per-group, directional long-only, and 50/50 long-short targets.

    Keys are ``group_1``..``group_{n_groups}``, ``directional_long``, and
    ``long_short`` as documented in the module docstring. The directional
    portfolio picks the top group when ``factor_direction == 1`` and the
    bottom group when ``factor_direction == -1``.
    """
    if not isinstance(profile, BacktestProfile):
        raise TypeError("profile must be a BacktestProfile")
    groups, _ = assign_groups(values, profile.n_groups)
    top_group = profile.n_groups if profile.factor_direction == 1 else 1
    weights = {
        f"group_{group_id}": _long_leg(groups, group_id, profile.gross_exposure)
        for group_id in range(1, profile.n_groups + 1)
    }
    weights["directional_long"] = _long_leg(groups, top_group, profile.gross_exposure)
    half = profile.gross_exposure / 2
    weights["long_short"] = (
        _long_leg(groups, profile.n_groups, half) - _long_leg(groups, 1, half)
    )
    return weights
