"""Training-signal novelty selection for the genetic search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from .features import expression_families


def trading_similarity(left_returns, right_returns, left_positions, right_positions, config):
    """Training-only net return correlation and same-side holding overlap.

    Positions are quantities: compare membership/sign, never raw quantities
    across differently priced instruments. Ignore jointly flat dates.
    """
    pair = pd.concat([left_returns, right_returns], axis=1).dropna()
    if len(pair) < config.min_overlap_days:
        raise ValueError("insufficient training return overlap")
    correlation = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    if not np.isfinite(correlation):
        raise ValueError("undefined training return correlation")
    dates = left_positions.index.intersection(right_positions.index)
    columns = left_positions.columns.union(right_positions.columns)
    left = np.sign(left_positions.reindex(index=dates, columns=columns, fill_value=0))
    right = np.sign(right_positions.reindex(index=dates, columns=columns, fill_value=0))
    if left.isna().any().any() or right.isna().any().any():
        raise ValueError("missing training positions")
    denominator = (left.ne(0).sum(axis=1) + right.ne(0).sum(axis=1))
    overlap = (2 * (left.eq(right) & left.ne(0)).sum(axis=1) / denominator.where(denominator > 0)).dropna()
    if len(overlap) < config.min_overlap_days:
        raise ValueError("insufficient active training position overlap")
    mean_overlap = float(overlap.mean())
    return {
        "return_correlation": float(correlation), "position_overlap": mean_overlap,
        "too_similar": bool(correlation >= config.return_correlation_limit
                            or mean_overlap >= config.position_overlap_limit),
    }


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


@dataclass(frozen=True)
class ValidationSelectionResult(Sequence[Any]):
    accepted: tuple[Any, ...]
    rejected: tuple[Any, ...]
    rejection_reasons: dict[str, tuple[str, ...]]

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


def _candidate_eligible(candidate: Any) -> bool:
    value = candidate.get("eligible") if isinstance(candidate, Mapping) else candidate.eligible
    return value is True


def _candidate_reasons(candidate: Any) -> tuple[str, ...]:
    value = candidate.get("reasons", ()) if isinstance(candidate, Mapping) else candidate.reasons
    return tuple(str(reason) for reason in value)


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
    source = candidate if isinstance(candidate, Mapping) else {
        name: getattr(candidate, name, None)
        for name in ("training_fingerprint", "operator_version")
    }
    if isinstance(values, Mapping):
        source = {**source, **values}
    return source.get("training_fingerprint"), source.get("operator_version")


def _compatibility_reason(
    scope: str,
    reference_id: str,
    candidate_fingerprint: str | None,
    candidate_operator: str | None,
    reference_fingerprint: str | None,
    reference_operator: str | None,
) -> str | None:
    missing = []
    if not candidate_fingerprint:
        missing.append("candidate training fingerprint")
    if not candidate_operator:
        missing.append("candidate operator version")
    if not reference_fingerprint:
        missing.append("reference training fingerprint")
    if not reference_operator:
        missing.append("reference operator version")
    if missing:
        return (
            f"incompatible {scope} reference {reference_id}: unverifiable; "
            f"missing {', '.join(missing)}"
        )
    if reference_fingerprint != candidate_fingerprint or reference_operator != candidate_operator:
        return (
            f"incompatible {scope} reference {reference_id}: "
            "training fingerprint/operator version does not match"
        )
    return None


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
    limit = min(20, int(_value(config, "validation_limit", 20)))
    correlation_limit = float(_value(config, "correlation_limit", 0.90))
    candidate_items = tuple(candidates)
    candidate_ids = tuple(_candidate_id(candidate) for candidate in candidate_items)
    duplicate_ids = {
        candidate_id
        for candidate_id in candidate_ids
        if candidate_ids.count(candidate_id) > 1
    }
    ineligible_ids = {
        candidate_id
        for candidate, candidate_id in zip(candidate_items, candidate_ids)
        if candidate_id not in duplicate_ids and not _candidate_eligible(candidate)
    }
    ordered = sorted(
        (
            candidate
            for candidate, candidate_id in zip(candidate_items, candidate_ids)
            if candidate_id not in duplicate_ids and candidate_id not in ineligible_ids
        ),
        key=(lambda c: (_candidate_score(c)[1], _candidate_score(c)[0], *_candidate_score(c)[2:]))
        if _value(config, "fitness_mode", "legacy_ic") == "all_costs_sharpe" else _candidate_score,
    )
    archive_items = tuple(archive)
    accepted: list[Any] = []
    rejected: list[Any] = [
        candidate
        for candidate, candidate_id in zip(candidate_items, candidate_ids)
        if candidate_id in duplicate_ids or candidate_id in ineligible_ids
    ]
    reasons: dict[str, tuple[str, ...]] = {
        candidate_id: (f"duplicate expression_id candidate: {candidate_id}",)
        for candidate_id in duplicate_ids
    }
    for candidate, candidate_id in zip(candidate_items, candidate_ids):
        if candidate_id in ineligible_ids:
            detail = "; ".join(_candidate_reasons(candidate)) or "no reason provided"
            reasons[candidate_id] = (f"candidate eligible is not True: {detail}",)
    comparisons: list[Comparison] = []
    archive_entries: list[dict[str, Any]] = []

    for candidate in ordered:
        candidate_id = _candidate_id(candidate)
        left = _frame(values_by_id.get(candidate_id))
        candidate_reasons: list[str] = []
        references: list[tuple[str, str, Any, Any, Any]] = []
        candidate_fingerprint, candidate_operator = _metadata(candidate, values_by_id.get(candidate_id))
        for item in archive_items:
            reference_id = str(item.get("expression_id", item.get("hash", "archive")))
            right = _frame(item)
            reference_fingerprint, reference_operator = _metadata(item, right)
            reason = _compatibility_reason(
                "archive",
                reference_id,
                candidate_fingerprint,
                candidate_operator,
                reference_fingerprint,
                reference_operator,
            )
            if reference_id == candidate_id and reason is None:
                candidate_reasons.append(
                    f"duplicate expression_id in compatible archive: {candidate_id}"
                )
                continue
            if reason is not None:
                comparisons.append(Comparison(candidate_id, reference_id, 0, None, False, reason))
                if reference_id == candidate_id:
                    candidate_reasons.append(
                        f"conflicting expression_id in incompatible archive: {candidate_id}"
                    )
                continue
            references.append(("archive", reference_id, right, item, right))
        references.extend(
            (
                "accepted",
                _candidate_id(item),
                _frame(values_by_id.get(_candidate_id(item))),
                item,
                values_by_id.get(_candidate_id(item)),
            )
            for item in accepted
        )

        for scope, reference_id, right, metadata, metadata_values in references:
            reference_fingerprint, reference_operator = _metadata(
                metadata,
                metadata_values if metadata_values is not None else right,
            )
            reason = _compatibility_reason(
                scope,
                reference_id,
                candidate_fingerprint,
                candidate_operator,
                reference_fingerprint,
                reference_operator,
            )
            if reason is not None:
                comparisons.append(Comparison(candidate_id, reference_id, 0, None, False, reason))
                candidate_reasons.append(reason)
                continue
            if left is None or right is None:
                reason = f"insufficient overlap with {reference_id}: missing value panel"
                candidate_reasons.append(reason)
                comparisons.append(Comparison(candidate_id, reference_id, 0, None, False, reason))
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

        if _value(config, "family_diversity", False):
            tree = candidate["tree"] if isinstance(candidate, Mapping) else candidate.tree
            families = expression_families(tree)
            # Balance the shortlist too: otherwise a global top-20 cap could
            # discard other families before independent validation.
            family_limit = max(1, (limit + 3) // 4)
            for family in families:
                count = sum(family in expression_families(c["tree"] if isinstance(c, Mapping) else c.tree) for c in accepted)
                if count >= family_limit:
                    candidate_reasons.append(f"shortlist family limit reached: {family} ({family_limit})")
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


def select_validation(
    training_candidates: Iterable[Any], validation_results: Mapping[str, Mapping[str, Any]], config: Any,
    *, trading_evidence=None,
) -> ValidationSelectionResult:
    """Freeze only candidates meeting predeclared validation evidence gates."""
    accepted: list[tuple[Any, Mapping[str, Any]]] = []
    rejected: list[Any] = []
    reasons: dict[str, tuple[str, ...]] = {}
    for candidate in training_candidates:
        candidate_id = _candidate_id(candidate)
        result = validation_results.get(candidate_id)
        failures: list[str] = []
        if not isinstance(result, Mapping):
            failures.append("missing validation result")
        else:
            direction = (
                candidate.get("training_direction", candidate.get("direction"))
                if isinstance(candidate, Mapping)
                else getattr(candidate, "training_direction", getattr(candidate, "direction", None))
            )
            if direction not in (-1, 1):
                failures.append("candidate is missing a frozen training direction")
            if result.get("direction") != direction:
                failures.append("validation direction differs from frozen training direction")
            for field, minimum in (
                ("day_coverage", float(_value(config, "min_day_coverage", 0.8))),
                ("cell_coverage", float(_value(config, "min_cell_coverage", 0.8))),
            ):
                value = _finite(result.get(field))
                if value is None or value < minimum:
                    failures.append(f"{field} below {minimum:.0%}")
            quarter_days = result.get("quarter_valid_days", {})
            if not isinstance(quarter_days, Mapping):
                quarter_days = {}
            for quarter in ("Q1", "Q2", "Q3", "Q4"):
                days = quarter_days.get(quarter)
                minimum_days = int(_value(config, "min_quarter_days", 45))
                if not isinstance(days, (int, float)) or days < minimum_days:
                    failures.append(f"{quarter} has fewer than {minimum_days} valid days")
            if _value(config, "fitness_mode", "legacy_ic") != "all_costs_sharpe":
                if (_finite(result.get("mean_ic")) or 0.0) <= 0.0:
                    failures.append("mean IC is not positive")
                quarters = result.get("quarter_means", {})
                if not isinstance(quarters, Mapping):
                    quarters = {}
                if sum(1 for value in quarters.values() if (_finite(value) or 0.0) > 0.0) < 3:
                    failures.append("fewer than 3 positive validation quarters")
            if (_finite(result.get("all_costs_cumulative_return")) or 0.0) <= 0.0:
                failures.append("all_costs cumulative return is not positive")
            net_sharpe = _finite(result.get("net_sharpe"))
            if net_sharpe is None:
                failures.append("net Sharpe is not finite")
            else:
                minimum_sharpe = float(_value(config, "min_all_costs_sharpe", 1.0))
                if net_sharpe <= minimum_sharpe:
                    failures.append(
                        f"all_costs Sharpe is not greater than {minimum_sharpe:g}"
                    )
            for enabled, key in (("reference_factor", "incremental"),
                                 ("validation_parameter_stability", "parameter_stability")):
                if _value(config, enabled, False):
                    check = result.get(key)
                    if not isinstance(check, Mapping) or check.get("passed") is not True:
                        failures.extend((check.get("reasons") if isinstance(check, Mapping) else None)
                                        or [f"missing or failed validation {key}"])
            if _value(config, "exposure_residual_mode", "off") == "gate":
                check = result.get("exposure_residual")
                if not isinstance(check, Mapping) or check.get("passed") is not True:
                    failures.extend((check.get("reasons") if isinstance(check, Mapping) else None)
                                    or ["missing or failed Barra residual validation"])
            if _value(config, "validation_stability", False):
                stability = result.get("cost_stability")
                if not isinstance(stability, Mapping) or stability.get("passed") is not True:
                    failures.extend(stability.get("reasons", ["missing validation cost stability"]) if isinstance(stability, Mapping)
                                    else ["missing validation cost stability"])
                    if not failures:
                        failures.append("validation cost stability failed")
        if failures:
            rejected.append(candidate)
            reasons[candidate_id] = (*reasons.get(candidate_id, ()), *failures)
        else:
            accepted.append((candidate, result))
    accepted.sort(key=lambda pair: (
        -float(pair[1]["net_sharpe"]), float(pair[1].get("turnover", float("inf"))),
        int(pair[0].get("complexity", 0) if isinstance(pair[0], Mapping) else getattr(pair[0], "complexity", 0)),
        _candidate_id(pair[0]),
    ))
    limit = min(5, int(_value(config, "frozen_limit", 5)))
    diversified = []
    for candidate, _ in accepted:
        candidate_id = _candidate_id(candidate)
        failures = []
        if _value(config, "family_diversity", False):
            tree = candidate["tree"] if isinstance(candidate, Mapping) else candidate.tree
            for family in sorted(expression_families(tree)):
                count = sum(family in expression_families(c["tree"] if isinstance(c, Mapping) else c.tree) for c in diversified)
                if count >= _value(config, "family_candidate_limit", 2):
                    failures.append(f"frozen family limit reached: {family}")
        if trading_evidence is not None:
            if candidate_id not in trading_evidence:
                raise ValueError(f"missing training trading evidence: {candidate_id}")
            returns, positions = trading_evidence[candidate_id]
            for reference in diversified:
                reference_id = _candidate_id(reference)
                other_returns, other_positions = trading_evidence[reference_id]
                comparison = trading_similarity(returns, other_returns, positions, other_positions, config)
                if comparison["too_similar"]:
                    failures.append(f"training trading similarity with {reference_id}: "
                                    f"return_correlation={comparison['return_correlation']:.6f}, "
                                    f"position_overlap={comparison['position_overlap']:.6f}")
        if len(diversified) >= limit:
            failures.append(f"selection limit reached: {limit}")
        if failures:
            rejected.append(candidate)
            reasons[candidate_id] = (*reasons.get(candidate_id, ()), *failures)
        else:
            diversified.append(candidate)
    return ValidationSelectionResult(tuple(diversified), tuple(rejected), reasons)


__all__ = ["Comparison", "DeduplicationResult", "ValidationSelectionResult", "deduplicate_training", "select_validation"]
