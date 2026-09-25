"""Deterministic signed targets for fixed portfolio factor members."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from factor_common.grouping import target_weights


def member_targets(values, spec, profile, min_valid):
    """Build unit-gross long/short targets and daily validity diagnostics."""
    clean = values.replace([np.inf, -np.inf], np.nan)
    count = clean.notna().sum(axis=1)
    spread = clean.max(axis=1) - clean.min(axis=1)
    valid = (count >= min_valid) & (spread > 0.0)
    unit_profile = replace(
        profile,
        gross_exposure=1.0,
        factor_direction=spec.setting["factor_direction"],
    )
    weights = target_weights(clean, unit_profile)["long_short"]
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
        index=values.index,
    )
    return weights, diagnostics


def combine_targets(members, allocations, config):
    """Combine fixed member targets and apply one common daily risk scale.

    Member targets are already signed (their factor direction has been applied
    by :func:`member_targets`).  Allocations therefore only weight those
    signed targets; they are never used to re-leverage or redistribute a
    member's positions.
    """
    if not isinstance(members, dict) or not members:
        raise ValueError("members must be a non-empty mapping")
    if not isinstance(allocations, dict) or set(members) != set(allocations):
        raise ValueError("members and allocations must have identical keys")
    allocation_values = {}
    for name, allocation in allocations.items():
        if isinstance(allocation, bool) or not np.isfinite(float(allocation)):
            raise ValueError(f"allocation for {name!r} must be finite")
        allocation_values[name] = float(allocation)
        if allocation_values[name] <= 0:
            raise ValueError("allocations must be finite and positive")
    if not np.isclose(sum(allocation_values.values()), 1.0, rtol=0, atol=1e-12):
        raise ValueError("allocations must sum to 1")

    first = next(iter(members.values()))
    if not isinstance(first, pd.DataFrame):
        raise TypeError("member targets must be pandas DataFrames")
    index, columns = first.index, first.columns
    for name, member in members.items():
        if not isinstance(member, pd.DataFrame):
            raise TypeError(f"member {name!r} must be a pandas DataFrame")
        if not member.index.equals(index) or not member.columns.equals(columns):
            raise ValueError("all member targets must share identical axes")

    # NaN/inf cells cannot become positions.  Treat only those cells as cash;
    # importantly, there is no row fill or redistribution across instruments.
    combined = pd.DataFrame(0.0, index=index, columns=columns)
    invalid_counts = {}
    for name, member in members.items():
        invalid_counts[name] = (~np.isfinite(member.astype(float))).sum(axis=1)
        clean = member.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        combined = combined.add(clean * allocation_values[name], fill_value=0.0)

    pre = _exposures(combined)
    limits = {
        "gross": float(config.gross_limit),
        "long": float(config.long_limit),
        "short": float(config.short_limit),
        "single": float(config.single_limit),
        "net": float(config.net_limit),
    }
    ratios = pd.DataFrame({"identity": 1.0}, index=index)
    for name, limit in limits.items():
        value = pre[name]
        ratios[name] = np.where(value > 0.0, limit / value, np.inf)
    scale = ratios.min(axis=1).clip(lower=0.0, upper=1.0)
    # A zero net limit with non-zero net exposure correctly produces scale 0.
    scale = scale.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    combined = combined.mul(scale, axis=0)
    post = _exposures(combined)

    binding = []
    for position, row in enumerate(ratios.itertuples(index=False, name=None)):
        if all(float(series.iloc[position]) <= 0.0 for series in pre.values()):
            binding.append("identity")
            continue
        minimum = min(row)
        binding.append(",".join(name for name, ratio in zip(ratios.columns, row)
                                if np.isclose(ratio, minimum, rtol=0, atol=1e-12)))
    risk = pd.DataFrame(index=index)
    for name in limits:
        risk[f"pre_{name}"] = pre[name]
    risk["pre_abs_net"] = pre["net"]
    risk["scale"] = scale
    for name in limits:
        risk[f"post_{name}"] = post[name]
    risk["post_abs_net"] = post["net"]
    risk["binding_constraint"] = binding
    # Compatibility alias retained for early B1 callers.
    risk["binding_constraints"] = risk["binding_constraint"]
    for name, counts in invalid_counts.items():
        risk[f"invalid_cells_{name}"] = counts.astype(int)
    risk["invalid_cells"] = sum(invalid_counts.values())
    return combined, risk


def _exposures(weights: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the five fixed portfolio exposure measures by day."""
    return {
        "gross": weights.abs().sum(axis=1),
        "long": weights.clip(lower=0.0).sum(axis=1),
        "short": -weights.clip(upper=0.0).sum(axis=1),
        "single": weights.abs().max(axis=1),
        "net": weights.sum(axis=1).abs(),
    }


__all__ = ["member_targets", "combine_targets"]
