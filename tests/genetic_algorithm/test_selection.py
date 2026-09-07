from __future__ import annotations

import numpy as np
import pandas as pd

from Genetic_Algorithm.evolution import Candidate
from Genetic_Algorithm.expression import Node
from Genetic_Algorithm.selection import deduplicate_training


def _values(seed: int = 0, *, days: int = 120, symbols: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(days, symbols)),
        index=pd.date_range("2024-01-01", periods=days, freq="D"),
        columns=[f"S{i:02d}" for i in range(symbols)],
    )


def _candidate(name: str, *, mean: float = 0.5, worst: float = 0.4, nodes: int = 3) -> Candidate:
    return Candidate(Node("close"), name, (mean, worst, -nodes))


def _metadata(values: pd.DataFrame, fingerprint: str = "train", operator_version: str = "ops") -> dict:
    return {
        "values": values,
        "training_fingerprint": fingerprint,
        "operator_version": operator_version,
    }


def test_identical_and_sign_inverted_training_values_are_duplicates():
    values = _values()
    candidates = [_candidate("a"), _candidate("b", mean=0.4)]

    result = deduplicate_training(
        candidates,
        {"a": _metadata(values), "b": _metadata(-values)},
        [],
        {"min_overlap_days": 120, "correlation_limit": 0.90, "validation_limit": 20},
    )

    assert [candidate.expression_id for candidate in result] == ["a"]
    assert any("duplicate" in reason for reason in result.rejection_reasons["b"])


def test_one_hundred_nineteen_common_days_are_insufficient_evidence():
    values = _values(days=119)
    result = deduplicate_training(
        [_candidate("a"), _candidate("b")],
        {"a": _metadata(values), "b": _metadata(values.copy())},
        [],
        {"min_overlap_days": 120, "correlation_limit": 0.90, "validation_limit": 20},
    )

    assert [candidate.expression_id for candidate in result] == ["a"]
    assert any("insufficient overlap" in reason for reason in result.rejection_reasons["b"])


def test_one_hundred_twenty_days_with_low_correlation_establish_novelty():
    left = _values(1)
    right = _values(2)
    result = deduplicate_training(
        [_candidate("a"), _candidate("b")],
        {"a": _metadata(left), "b": _metadata(right)},
        [],
        {"min_overlap_days": 120, "correlation_limit": 0.90, "validation_limit": 20},
    )

    assert [candidate.expression_id for candidate in result] == ["a", "b"]


def test_first_candidate_without_archive_comparisons_is_admissible():
    result = deduplicate_training(
        [_candidate("first")],
        {"first": _values()},
        [],
        {"min_overlap_days": 120, "correlation_limit": 0.90, "validation_limit": 20},
    )

    assert [candidate.expression_id for candidate in result] == ["first"]
    assert result.comparisons == ()


def test_selection_order_and_limit_are_deterministic():
    values = {str(i): _metadata(_values(i)) for i in range(25)}
    candidates = [
        _candidate(str(i), mean=0.5, worst=0.4, nodes=3)
        for i in range(25)
    ]
    result = deduplicate_training(candidates, values, [], {"validation_limit": 20})

    assert len(result) == 20
    assert [candidate.expression_id for candidate in result] == sorted(str(i) for i in range(25))[:20]


def test_selection_never_exceeds_absolute_validation_candidate_cap():
    values = {str(i): _metadata(_values(i)) for i in range(25)}
    candidates = [_candidate(str(i)) for i in range(25)]

    result = deduplicate_training(candidates, values, [], {"validation_limit": 25})

    assert len(result) == 20
    assert len(result.rejected) == 5


def test_duplicate_expression_ids_are_rejected_before_selection():
    result = deduplicate_training(
        [_candidate("same-id", mean=1.0), _candidate("same-id", mean=0.1)],
        {"same-id": _metadata(_values())},
        [],
        {"min_overlap_days": 120, "correlation_limit": 0.90, "validation_limit": 20},
    )

    assert result.accepted == ()
    assert [candidate.expression_id for candidate in result.rejected] == ["same-id", "same-id"]
    assert result.rejection_reasons["same-id"] == (
        "duplicate expression_id candidate: same-id",
    )
    assert result.comparisons == ()
    assert result.archive_entries == ()


def test_incompatible_archive_reference_rejects_candidate_before_correlation():
    values = _values()
    result = deduplicate_training(
        [_candidate("candidate")],
        {"candidate": {
            "values": values,
            "training_fingerprint": "current-data",
            "operator_version": "same-ops",
        }},
        [{
            "expression_id": "old",
            "training_fingerprint": "different-data",
            "operator_version": "same-ops",
            "values": values,
        }],
        {"min_overlap_days": 120, "correlation_limit": 0.90, "validation_limit": 20},
    )

    assert [candidate.expression_id for candidate in result] == []
    assert result.comparisons[0].compatible is False
    assert "incompatible" in (result.comparisons[0].reason or "")
    assert "incompatible archive reference old" in result.rejection_reasons["candidate"][0]


def test_matching_archive_expression_id_is_rejected_before_selection():
    values = _values()
    result = deduplicate_training(
        [_candidate("same-id")],
        {"same-id": _metadata(values)},
        [{
            "expression_id": "same-id",
            "training_fingerprint": "train",
            "operator_version": "ops",
            "values": values.copy(),
        }],
        {"min_overlap_days": 120, "correlation_limit": 0.90, "validation_limit": 20},
    )

    assert result.accepted == ()
    assert result.comparisons == ()
    assert result.rejection_reasons["same-id"] == (
        "duplicate expression_id in compatible archive: same-id",
    )


def test_archive_reference_without_values_is_explicitly_unverifiable():
    result = deduplicate_training(
        [_candidate("candidate")],
        {"candidate": _values()},
        [{"expression_id": "old", "training_fingerprint": "same", "operator_version": "ops"}],
        {"min_overlap_days": 120, "correlation_limit": 0.90, "validation_limit": 20},
    )

    assert [candidate.expression_id for candidate in result] == []
    assert result.comparisons[0].compatible is False
    assert "unverifiable" in (result.comparisons[0].reason or "")
    assert "unverifiable" in result.rejection_reasons["candidate"][0]
    assert "old" in result.comparisons[0].reason


def test_incompatible_accepted_reference_rejects_candidate_with_reason():
    values = _values()
    result = deduplicate_training(
        [_candidate("first"), _candidate("second", mean=0.4)],
        {
            "first": {"values": values, "training_fingerprint": "data-a", "operator_version": "ops"},
            "second": {"values": values, "training_fingerprint": "data-b", "operator_version": "ops"},
        },
        [],
        {"min_overlap_days": 120, "correlation_limit": 0.90, "validation_limit": 20},
    )

    assert [candidate.expression_id for candidate in result] == ["first"]
    assert result.comparisons[-1].compatible is False
    assert "incompatible accepted reference first" in (result.comparisons[-1].reason or "")
    assert "incompatible accepted reference first" in result.rejection_reasons["second"][0]


def test_missing_archive_metadata_is_unverifiable_and_never_duplicate():
    values = _values()
    result = deduplicate_training(
        [_candidate("candidate")],
        {"candidate": _metadata(values)},
        [{"expression_id": "old", "values": values}],
        {"min_overlap_days": 120, "correlation_limit": 0.90, "validation_limit": 20},
    )

    assert [candidate.expression_id for candidate in result] == []
    comparison = result.comparisons[0]
    assert comparison.compatible is False
    assert "unverifiable" in (comparison.reason or "")
    assert "training fingerprint" in (comparison.reason or "")
    assert comparison.mean_abs_daily_spearman is None
    assert "unverifiable" in result.rejection_reasons["candidate"][0]


def test_missing_accepted_metadata_is_unverifiable_and_never_duplicate():
    values = _values()
    result = deduplicate_training(
        [_candidate("first"), _candidate("second", mean=0.4)],
        {"first": values, "second": _metadata(values)},
        [],
        {"min_overlap_days": 120, "correlation_limit": 0.90, "validation_limit": 20},
    )

    assert [candidate.expression_id for candidate in result] == ["first"]
    comparison = result.comparisons[-1]
    assert comparison.compatible is False
    assert "unverifiable" in (comparison.reason or "")
    assert "training fingerprint" in (comparison.reason or "")
    assert comparison.mean_abs_daily_spearman is None
    assert "unverifiable" in result.rejection_reasons["second"][0]
