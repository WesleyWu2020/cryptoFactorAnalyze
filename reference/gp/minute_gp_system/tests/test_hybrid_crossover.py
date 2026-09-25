import importlib

import numpy as np
from pymoo.core.problem import Problem


def _reload_evolution():
    import gp.minute_gp_system.engines.unified_v2.evolution as evolution

    return importlib.reload(evolution)


def _make_problem(evolution):
    return Problem(
        n_var=evolution.N_PARAMS,
        n_obj=1,
        xl=np.asarray(evolution.PARAM_BOUNDS_LOWER, dtype=float),
        xu=np.asarray(evolution.PARAM_BOUNDS_UPPER, dtype=float),
    )


def _make_parent_tensor(evolution, n_matings=6):
    lower = np.asarray(evolution.PARAM_BOUNDS_LOWER, dtype=float)
    upper = np.asarray(evolution.PARAM_BOUNDS_UPPER, dtype=float)
    ordered = np.asarray(evolution._ORDERED_LOCI, dtype=np.int32)
    parent_a = lower.copy()
    parent_b = upper.copy()
    parent_a[ordered] = np.maximum(lower[ordered], 1.0)
    parent_b[ordered] = np.maximum(
        parent_a[ordered] + 1.0,
        upper[ordered] - 1.0,
    )
    parents = np.stack(
        [
            np.tile(parent_a, (n_matings, 1)),
            np.tile(parent_b, (n_matings, 1)),
        ],
        axis=0,
    )
    return parents


def _patch_postprocess_noops(monkeypatch, evolution):
    monkeypatch.setattr(evolution, "build_search_space_state", lambda: {})
    monkeypatch.setattr(
        evolution,
        "project_population_to_search_space",
        lambda pop: np.asarray(np.rint(pop), dtype=np.int32),
    )
    monkeypatch.setattr(evolution, "_force_mode4_only_population", lambda pop: pop)
    monkeypatch.setattr(
        evolution,
        "_enforce_fundamental_required",
        lambda pop, rng=None, state=None: np.asarray(pop, dtype=np.int32),
    )


def test_hybrid_crossover_can_import_and_instantiate():
    evolution = _reload_evolution()

    crossover = evolution.HybridCrossover(prob=0.9, sbx_eta=3.0)

    assert crossover.n_parents == 2
    assert crossover.n_offsprings == 2


def test_hybrid_crossover_preserves_output_shape(monkeypatch):
    evolution = _reload_evolution()
    problem = _make_problem(evolution)
    parents = _make_parent_tensor(evolution, n_matings=4)

    _patch_postprocess_noops(monkeypatch, evolution)

    crossover = evolution.HybridCrossover(prob=1.0, sbx_eta=3.0)
    crossover._rng = np.random.default_rng(7)
    np.random.seed(7)

    offspring = crossover._do(problem, parents)

    assert offspring.shape == parents.shape


def test_hybrid_crossover_prob_one_changes_some_offspring(monkeypatch):
    evolution = _reload_evolution()
    problem = _make_problem(evolution)
    parents = _make_parent_tensor(evolution, n_matings=8)

    _patch_postprocess_noops(monkeypatch, evolution)

    crossover = evolution.HybridCrossover(prob=1.0, sbx_eta=3.0)
    crossover._rng = np.random.default_rng(11)
    np.random.seed(11)

    offspring = crossover._do(problem, parents)

    assert np.any(offspring != parents)


def test_hybrid_crossover_respects_global_seed(monkeypatch):
    evolution = _reload_evolution()
    problem = _make_problem(evolution)
    parents = _make_parent_tensor(evolution, n_matings=6)
    _patch_postprocess_noops(monkeypatch, evolution)

    np.random.seed(23)
    first = evolution.HybridCrossover(prob=1.0, sbx_eta=3.0)._do(problem, parents)

    np.random.seed(23)
    second = evolution.HybridCrossover(prob=1.0, sbx_eta=3.0)._do(problem, parents)

    assert np.array_equal(first, second)


def test_hybrid_crossover_keeps_genes_within_bounds():
    evolution = _reload_evolution()
    problem = _make_problem(evolution)
    parents = _make_parent_tensor(evolution, n_matings=5)
    parents_2d = parents.reshape(-1, evolution.N_PARAMS)
    parents = evolution.SearchSpaceRepair()._do(problem, parents_2d).reshape(parents.shape)

    crossover = evolution.HybridCrossover(prob=1.0, sbx_eta=3.0)
    crossover._rng = np.random.default_rng(13)
    np.random.seed(13)

    offspring = crossover._do(problem, parents)
    lower = np.asarray(evolution.PARAM_BOUNDS_LOWER, dtype=np.int32)
    upper = np.asarray(evolution.PARAM_BOUNDS_UPPER, dtype=np.int32)

    assert np.all(offspring >= lower.reshape(1, 1, -1))
    assert np.all(offspring <= upper.reshape(1, 1, -1))
