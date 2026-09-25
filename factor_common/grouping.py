"""Deterministic cross-sectional grouping and portfolio target weights.

Legacy grouping contract: on each date, valid (finite) factor values are sorted by
``(factor, instrument)`` ascending so ties always break by instrument name.
The row at sorted position ``p`` of ``count`` valid names joins group
``floor(p * n_groups / count) + 1``; group 1 holds the lowest factor values
and group ``n_groups`` the highest. A date with fewer than ``n_groups``
valid names is reported in the ``insufficient_dates`` diagnostic and keeps
all-NaN groups; the configured group count is never reduced silently.

With ``symmetric_fractional``, tied names share the group's rank slots equally.
Each group still has unit exposure, but overlapping long/short legs cancel;
the resulting net portfolio is never re-levered to restore gross exposure.

``target_weights(values, profile)`` returns a dict of weight DataFrames
sharing the input axes, keyed by portfolio name:

- ``"group_1"`` .. ``"group_{n_groups}"``: long-only targets for
  each group, each row summing to ``profile.gross_exposure`` on valid dates.
- ``"directional_long"``: the long-only portfolio in the factor's preferred
  direction — the top group (``n_groups``) when ``factor_direction == 1``
  and the bottom group (``1``) when ``factor_direction == -1``.
- ``"long_short"``: a 50/50 dollar-neutral portfolio in the factor's
  preferred direction — when ``factor_direction == 1`` it is long the top
  group and short the bottom group; when ``factor_direction == -1`` the legs
  are swapped (long bottom, short top). Each leg is half of
  ``profile.gross_exposure``.

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


def group_weights(values: pd.DataFrame, n_groups: int, tie_policy: str = "legacy_instrument"):
    """Unit-sum group weights; tied rank slots are shared without name ordering.

    Fractional membership preserves the legacy slot counts for uneven groups.
    Missing names receive zero; insufficient dates remain NaN. Opposing tied
    legs may cancel, and the remaining exposure must not be re-levered.
    """
    if tie_policy == "legacy_instrument":
        groups, diagnostics = assign_groups(values, n_groups)
        diagnostics["group_tie_policy"] = tie_policy
        return {f"group_{g}": _long_leg(groups, g, 1.0)
                for g in range(1, n_groups + 1)}, diagnostics
    if tie_policy != "symmetric_fractional":
        raise ValueError("unsupported group_tie_policy")
    _validate_axes(values, name="values")
    if isinstance(n_groups, bool) or not isinstance(n_groups, int) or n_groups < 2:
        raise ValueError("n_groups must be an integer greater than or equal to 2")
    result = np.full((n_groups, *values.shape), np.nan)
    insufficient = []
    group_basis = np.eye(n_groups, dtype=np.int64)
    for day, row in enumerate(values.to_numpy(dtype=float)):
        valid = np.flatnonzero(np.isfinite(row))
        count = len(valid)
        if count < n_groups:
            insufficient.append(values.index[day])
            continue
        ordered = valid[np.argsort(row[valid])]
        slots = np.arange(count) * n_groups // count
        totals = np.bincount(slots, minlength=n_groups)
        _, starts, inverse, counts = np.unique(
            row[ordered], return_index=True, return_inverse=True, return_counts=True
        )
        membership = np.add.reduceat(group_basis[:, slots], starts, axis=1)
        block_weights = membership / (totals[:, None] * counts[None, :])
        day_weights = result[:, day, :]
        day_weights[:] = 0.0
        day_weights[:, ordered] = block_weights[:, inverse]
    weights = {f"group_{g + 1}": pd.DataFrame(result[g], index=values.index, columns=values.columns)
               for g in range(n_groups)}
    diagnostics = {"insufficient_dates": insufficient, "group_tie_policy": tie_policy}
    return weights, diagnostics


def target_weights(values: pd.DataFrame, profile: BacktestProfile) -> dict[str, pd.DataFrame]:
    """Build per-group, directional long-only, and 50/50 long-short targets.

    Keys are ``group_1``..``group_{n_groups}``, ``directional_long``, and
    ``long_short`` as documented in the module docstring. Both directional
    portfolios follow ``factor_direction``: direction 1 favors the top group,
    direction -1 favors the bottom group (the long-short legs swap).
    """
    if not isinstance(profile, BacktestProfile):
        raise TypeError("profile must be a BacktestProfile")
    unit_weights, _ = group_weights(values, profile.n_groups, profile.group_tie_policy)
    top_group = profile.n_groups if profile.factor_direction == 1 else 1
    weights = {
        f"group_{group_id}": unit_weights[f"group_{group_id}"] * profile.gross_exposure
        for group_id in range(1, profile.n_groups + 1)
    }
    weights["directional_long"] = weights[f"group_{top_group}"].copy()
    half = profile.gross_exposure / 2
    long_group, short_group = (
        (profile.n_groups, 1) if profile.factor_direction == 1 else (1, profile.n_groups)
    )
    weights["long_short"] = (
        unit_weights[f"group_{long_group}"] * half - unit_weights[f"group_{short_group}"] * half
    )
    return weights
