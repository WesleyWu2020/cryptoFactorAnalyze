from __future__ import annotations

import pytest

from Genetic_Algorithm.evolution import (
    Candidate,
    SearchFormationError,
    nondominated_sort,
    search,
)
from Genetic_Algorithm.expression import Node


def _candidate(name: str, score: tuple[float, ...]) -> Candidate:
    return Candidate(Node("close"), name, score)


def test_nondominated_sort_keeps_hand_enumerated_fronts_in_hash_order():
    candidates = [
        _candidate("d", (0.5, 0.5)),
        _candidate("b", (0.8, 0.2)),
        _candidate("a", (0.2, 0.8)),
        _candidate("c", (0.4, 0.4)),
    ]

    fronts = nondominated_sort(candidates)

    assert [[candidate.expression_id for candidate in front] for front in fronts] == [
        ["a", "b", "d"], ["c"]
    ]


def test_equal_scores_use_deterministic_expression_id_ties():
    candidates = [_candidate("z", (1.0, 1.0)), _candidate("a", (1.0, 1.0))]

    assert [item.expression_id for item in nondominated_sort(candidates)[0]] == ["a", "z"]


def test_empty_eligible_result_stays_empty():
    candidates = [_candidate("a", (1.0, 1.0))]

    result = search(
        {"marker": "synthetic"},
        {
            "seed": 1,
            "population": 1,
            "generations": 1,
            "max_depth": 0,
            "max_nodes": 1,
            "evaluate_candidate": lambda tree, stage_data, labels, config: {
                "score": (0.0, 0.0), "eligible": False, "reasons": ("empty",)
            },
            "initial_trees": [candidates[0].tree],
        },
    )

    assert result.candidates == ()


def test_invalid_offspring_is_rejected_and_no_legal_population_raises():
    with pytest.raises(SearchFormationError, match="unique exploratory candidates"):
        search(
            {"marker": "synthetic"},
            {
                "seed": 2,
                "population": 2,
                "generations": 1,
                "max_depth": 0,
                "max_nodes": 0,
                "initial_trees": [Node("not-an-op")],
                "evaluate_candidate": lambda *args: {
                    "score": (1.0, 1.0), "eligible": True, "reasons": ()
                },
            },
        )


def test_repeated_seed_has_identical_search_output():
    config = {
        "seed": 7,
        "population": 3,
        "generations": 2,
        "max_depth": 1,
        "max_nodes": 3,
        "crossover_probability": 0.0,
        "mutation_probability": 0.0,
        "copy_probability": 1.0,
        "initial_trees": [Node("close"), Node("open"), Node("high")],
        "evaluate_candidate": lambda tree, stage_data, labels, config: {
            "score": (float({"close": 3, "open": 2, "high": 1}.get(tree.op, 0)),
                      -float(len(tree.children))),
            "eligible": True,
            "reasons": (),
        },
    }

    first = search({"marker": "synthetic"}, config)
    second = search({"marker": "synthetic"}, dict(config))

    assert first == second
    assert [candidate.tree.op for candidate in first.candidates] == ["close", "open", "high"]


def _variation_config(**overrides):
    config = {
        "seed": 11,
        "population": 2,
        "generations": 2,
        "max_depth": 1,
        "max_nodes": 3,
        "crossover_probability": 1.0,
        "mutation_probability": 0.0,
        "copy_probability": 0.0,
        "initial_trees": [Node("close"), Node("open")],
        "evaluate_candidate": lambda tree, stage_data, labels, config: {
            "score": (1.0, 1.0), "eligible": True, "reasons": ()
        },
    }
    config.update(overrides)
    return config


def test_forced_crossover_is_reproducible_with_fixed_seed():
    first = search({"marker": "synthetic"}, _variation_config())
    second = search({"marker": "synthetic"}, _variation_config())

    assert first == second
    assert first.generation_log[1]["attempted_trees"] >= 2


def test_forced_mutation_is_reproducible_with_fixed_seed():
    config = _variation_config(
        crossover_probability=0.0,
        mutation_probability=1.0,
    )

    first = search({"marker": "synthetic"}, config)
    second = search({"marker": "synthetic"}, dict(config))

    assert first == second
    assert first.generation_log[1]["attempted_trees"] == 2


def test_invalid_offspring_is_resampled_and_reasons_are_logged():
    calls = []

    def evaluate(tree, stage_data, labels, config):
        calls.append(tree)
        return {"score": (None, 1.0), "eligible": False, "reasons": ("bad offspring",)}

    result = search(
        {"marker": "synthetic"},
        _variation_config(
            generations=2,
            initial_trees=[Node("close"), Node("open")],
            max_depth=0,
            evaluate_candidate=evaluate,
        ),
    )

    assert result.generation_log[0]["eligible_population"] == 0
    assert result.generation_log[1]["invalid_reasons"]["bad offspring"] >= 2
    assert result.candidates == ()
    assert len(calls) > 2


def test_duplicate_canonical_trees_do_not_form_population():
    with pytest.raises(SearchFormationError, match="unique exploratory candidates 1/2"):
        search(
            {"marker": "synthetic"},
            {
                "seed": 3,
                "population": 2,
                "generations": 1,
                "max_depth": 0,
                "max_nodes": 1,
                "max_attempts": 2,
                "initial_trees": [Node("close"), Node("close")],
                "evaluate_candidate": lambda *args: {
                    "score": (1.0, 1.0), "eligible": True, "reasons": ()
                },
            },
        )


def test_none_objectives_preserve_ineligible_reasons_and_allow_exploration():
    result = search(
        {"marker": "synthetic"},
        {
            "seed": 4,
            "population": 2,
            "generations": 1,
            "max_depth": 0,
            "max_nodes": 1,
            "initial_trees": [Node("close"), Node("open")],
            "evaluate_candidate": lambda tree, stage_data, labels, config: {
                "score": (None, None, -1),
                "eligible": False,
                "reasons": (f"{tree.op} unavailable",),
            },
        },
    )

    assert result.candidates == ()
    assert result.generation_log[0]["eligible_population"] == 0
    assert result.generation_log[0]["invalid_reasons"] == {
        "close unavailable": 1,
        "open unavailable": 1,
    }
