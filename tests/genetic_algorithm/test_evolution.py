from __future__ import annotations

import pytest

from Genetic_Algorithm.evolution import (
    Candidate,
    SearchFormationError,
    crowding_distance,
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


def test_objective_vectors_must_have_equal_lengths():
    candidates = [_candidate("a", (1.0, 2.0)), _candidate("b", (1.0,))]

    with pytest.raises(ValueError, match="objective vector length"):
        nondominated_sort(candidates)

    with pytest.raises(ValueError, match="objective vector length"):
        crowding_distance(candidates)


def test_crowding_distance_handles_incomplete_objectives_without_nan():
    candidates = [
        _candidate("missing", (None,)),
        _candidate("nan", (float("nan"),)),
        _candidate("negative_infinity", (float("-inf"),)),
        _candidate("valid", (1.0,)),
    ]

    distances = crowding_distance(candidates)

    assert all(not (value != value) for value in distances.values())
    assert distances["valid"] == float("inf")
    assert distances["missing"] == float("inf")
    assert distances["nan"] == 0.0
    assert distances["negative_infinity"] == 0.0


def test_malformed_evaluator_objective_vector_is_rejected():
    with pytest.raises(ValueError, match="objective vector length"):
        search(
            {"marker": "synthetic"},
            {
                "seed": 5,
                "population": 2,
                "generations": 1,
                "max_depth": 0,
                "max_nodes": 1,
                "initial_trees": [Node("close"), Node("open")],
                "evaluate_candidate": lambda tree, stage_data, labels, config: {
                    "score": (1.0,) if tree.op == "close" else (1.0, 2.0),
                    "eligible": True,
                    "reasons": (),
                },
            },
        )


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
                "max_nodes": 1,
                "max_attempts": 2,
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


def test_evaluator_exception_is_logged_and_resampled_within_max_attempts():
    calls = 0

    def evaluate(tree, stage_data, labels, config):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient evaluator failure")
        return {"score": (1.0, 1.0), "eligible": True, "reasons": ()}

    result = search(
        {"marker": "synthetic"},
        {
            "seed": 9,
            "population": 1,
            "generations": 1,
            "max_depth": 0,
            "max_nodes": 1,
            "max_attempts": 2,
            "initial_trees": [Node("close")],
            "evaluate_candidate": evaluate,
        },
    )

    assert calls == 2
    assert result.generation_log[0]["invalid_reasons"] == {
        "RuntimeError: transient evaluator failure": 1,
    }
    assert len(result.candidates) == 1


def test_population_scaled_attempt_cap_limits_evaluator_calls():
    calls = 0

    def evaluate(*args):
        nonlocal calls
        calls += 1
        raise RuntimeError("always fails")

    with pytest.raises(SearchFormationError, match="after 50/50 attempts"):
        search(
            {"marker": "synthetic"},
            {
                "seed": 10,
                "population": 1,
                "generations": 1,
                "max_depth": 0,
                "max_nodes": 1,
                "max_attempts": 100,
                "initial_trees": [Node("close")],
                "evaluate_candidate": evaluate,
            },
        )

    assert calls == 50


def test_copy_probability_selects_copy_branch():
    from Genetic_Algorithm import evolution

    class FixedRng:
        def random(self):
            return 0.95

    parent_a = Node("close")
    parent_b = Node("open")
    result = evolution._variation(
        parent_a,
        parent_b,
        FixedRng(),
        {
            "crossover_probability": 0.2,
            "mutation_probability": 0.3,
            "copy_probability": 0.5,
        },
    )

    assert result == parent_a


def test_duplicate_canonical_trees_do_not_form_population():
    with pytest.raises(SearchFormationError, match="unique exploratory candidates 1/2.*after 2/2 attempts"):
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


def test_partial_eligible_population_does_not_use_exploratory_fallback():
    with pytest.raises(SearchFormationError, match="eligible candidates 1/2.*ineligible"):
        search(
            {"marker": "synthetic"},
            {
                "seed": 8,
                "population": 2,
                "generations": 1,
                "max_depth": 0,
                "max_nodes": 1,
                "max_attempts": 2,
                "initial_trees": [Node("close"), Node("open")],
                "evaluate_candidate": lambda tree, *args: {
                    "score": (1.0, 1.0),
                    "eligible": tree.op == "close",
                    "reasons": () if tree.op == "close" else ("ineligible",),
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
