"""Training-signal novelty selection for the genetic search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Comparison:
    candidate_id: str
    reference_id: str
    common_days: int
    mean_abs_daily_spearman: float | None
    compatible: bool
    reason: str | None = None


@dataclass(frozen=True)
class DeduplicationResult(Sequence[Any]):
    accepted: tuple[Any, ...]
    rejected: tuple[Any, ...]
    rejection_reasons: dict[str, tuple[str, ...]]
    comparisons: tuple[Comparison, ...]
    archive_entries: tuple[dict[str, Any], ...]

    def __iter__(self):
        return iter(self.accepted)

    def __len__(self) -> int:
        return len(self.accepted)

    def __getitem__(self, index):
        return self.accepted[index]


def _value(config: Any, name: str, default: Any) -> Any:
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _candidate_id(candidate: Any) -> str:
    return str(candidate.get("expression_id", candidate.get("hash")) if isinstance(candidate, Mapping) else candidate.expression_id)


def _candidate_score(candidate: Any) -> tuple[float, float, int, str]:
    if isinstance(candidate, Mapping):
        score = candidate.get("score", ())
    else:
        score = candidate.score
    mean = _finite(score[0] if len(score) > 0 else None)
    worst = _finite(score[1] if len(score) > 1 else None)
    raw_nodes = score[2] if len(score) > 2 else 0
    nodes = int(-raw_nodes) if raw_nodes is not None else 0
    return (-(worst if worst is not None else float("-inf")), -(mean if mean is not None else float("-inf")), nodes, _candidate_id(candidate))


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _frame(value: Any) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, Mapping):
        for key in ("values", "factor_values", "training_values"):
            if isinstance(value.get(key), pd.DataFrame):
                return value[key]
    return None


def _metadata(candidate: Any, values: Any) -> tuple[str | None, str | None]:
    source = candidate if isinstance(candidate, Mapping) else {}
    if isinstance(values, Mapping):
        source = {**source, **values}
    return source.get("training_fingerprint"), source.get("operator_version")


def _ast(node: Any) -> dict[str, Any]:
    return {
        "op": node.op,
        "field": node.field,
        "window": node.window,
        "children": [_ast(child) for child in node.children],
    }


def _daily_correlation(left: pd.DataFrame, right: pd.DataFrame) -> tuple[int, float | None]:
    dates = left.index.intersection(right.index)
    correlations: list[float] = []
    for date in dates:
        pair = pd.concat([left.loc[date], right.loc[date]], axis=1, keys=("left", "right"))
        pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 20:
            continue
        correlation = pair["left"].corr(pair["right"], method="spearman")
        if pd.notna(correlation):
            correlations.append(abs(float(correlation)))
    return len(correlations), (float(np.mean(correlations)) if correlations else None)


def deduplicate_training(
    candidates: Iterable[Any],
    values_by_id: Mapping[str, Any],
    archive: Iterable[Mapping[str, Any]],
    config: Any,
) -> DeduplicationResult:
    """Order, compare, and cap candidates using training-only novelty evidence."""
    minimum_days = int(_value(config, "min_overlap_days", 120))
    limit = int(_value(config, "validation_limit", 20))
    correlation_limit = float(_value(config, "correlation_limit", 0.90))
    ordered = sorted(tuple(candidates), key=_candidate_score)
    archive_items = tuple(archive)
    accepted: list[Any] = []
    rejected: list[Any] = []
    reasons: dict[str, tuple[str, ...]] = {}
    comparisons: list[Comparison] = []
    archive_entries: list[dict[str, Any]] = []

    for candidate in ordered:
        candidate_id = _candidate_id(candidate)
        left = _frame(values_by_id.get(candidate_id))
        candidate_reasons: list[str] = []
        references: list[tuple[str, Any, Mapping[str, Any] | None]] = []
        for item in archive_items:
            reference_id = str(item.get("expression_id", item.get("hash", "archive")))
            right = _frame(item)
            if right is None:
                right = _frame(values_by_id.get(reference_id))
            reference_fingerprint, reference_operator = _metadata(item, right)
            candidate_fingerprint, candidate_operator = _metadata(candidate, values_by_id.get(candidate_id))
            if (reference_fingerprint is not None and candidate_fingerprint is not None and reference_fingerprint != candidate_fingerprint) or (
                reference_operator is not None and candidate_operator is not None and reference_operator != candidate_operator
            ):
                continue
            references.append((reference_id, right, item))
        references.extend((_candidate_id(item), _frame(values_by_id.get(_candidate_id(item))), None) for item in accepted)

        for reference_id, right, _ in references:
            if left is None or right is None:
                continue
            common_days, mean_abs = _daily_correlation(left, right)
            if common_days < minimum_days:
                reason = f"insufficient overlap with {reference_id}: {common_days} < {minimum_days} valid days"
                candidate_reasons.append(reason)
                comparisons.append(Comparison(candidate_id, reference_id, common_days, mean_abs, False, reason))
            elif mean_abs is not None and mean_abs >= correlation_limit:
                reason = f"duplicate of {reference_id}: mean absolute daily Spearman {mean_abs:.6f} >= {correlation_limit:.6f}"
                candidate_reasons.append(reason)
                comparisons.append(Comparison(candidate_id, reference_id, common_days, mean_abs, False, reason))
            else:
                comparisons.append(Comparison(candidate_id, reference_id, common_days, mean_abs, True))

        if candidate_reasons or len(accepted) >= limit:
            if len(accepted) >= limit:
                candidate_reasons.append(f"selection limit reached: {limit}")
            rejected.append(candidate)
            reasons[candidate_id] = tuple(candidate_reasons)
            continue
        accepted.append(candidate)
        metadata = values_by_id.get(candidate_id)
        fingerprint, operator_version = _metadata(candidate, metadata)
        entry = {
            "expression_id": candidate_id,
            "ast": _ast(candidate.tree) if hasattr(candidate, "tree") else None,
            "training_diagnostics": {
                "score": list(candidate.score) if hasattr(candidate, "score") else None,
            },
        }
        if fingerprint is not None:
            entry["training_fingerprint"] = fingerprint
        if operator_version is not None:
            entry["operator_version"] = operator_version
        archive_entries.append(entry)

    return DeduplicationResult(tuple(accepted), tuple(rejected), reasons, tuple(comparisons), tuple(archive_entries))


__all__ = ["Comparison", "DeduplicationResult", "deduplicate_training"]
