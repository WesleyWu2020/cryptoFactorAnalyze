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


def _base_row(evolution):
    row = np.asarray(evolution.PARAM_BOUNDS_LOWER, dtype=float).copy()
    row[0] = 0
    row[1] = 1
    row[2] = 1
    row[3] = 0
    row[4] = 0
    row[5] = evolution.MASK_RULES.index("high_0.8")
    row[6] = 3
    row[7] = 0
    row[8] = 0
    row[13] = 0
    return row


def _patch_identity_pm(monkeypatch, evolution):
    monkeypatch.setattr(
        evolution.PM,
        "_do",
        lambda self, problem, X, **kwargs: np.asarray(X, dtype=float),
    )


def _patch_postprocess_capture(monkeypatch, evolution, captured):
    monkeypatch.setattr(evolution, "_force_mode4_only_population", lambda pop: pop)
    monkeypatch.setattr(
        evolution,
        "_enforce_fundamental_required",
        lambda pop, rng=None, state=None: np.asarray(pop, dtype=np.int32),
    )

    def _capture(pop):
        arr = np.asarray(np.rint(pop), dtype=np.int32)
        captured["project_input"] = arr.copy()
        return arr

    monkeypatch.setattr(evolution, "project_population_to_search_space", _capture)


class _ZeroChoiceRng:
    def random(self, size=None):
        if size is None:
            return 0.0
        return np.zeros(size, dtype=float)

    def integers(self, low, high=None, size=None):
        if size is None:
            return 0
        return np.zeros(size, dtype=np.int64)

    def choice(self, values, size=None, p=None):
        values = np.asarray(values)
        picked = values[0]
        if size is None:
            return picked
        return np.full(size, picked, dtype=values.dtype)

def test_diversity_mutation_preserves_shape(monkeypatch):
    evolution = _reload_evolution()
    problem = _make_problem(evolution)
    rows = np.vstack([_base_row(evolution), _base_row(evolution) + 1.0])
    captured = {}

    _patch_identity_pm(monkeypatch, evolution)
    _patch_postprocess_capture(monkeypatch, evolution, captured)
    monkeypatch.setattr(
        evolution,
        "build_search_space_state",
        lambda: {
            "mode1_op_all": (0,),
            "mode2_op_all": (0,),
            "mode3_op_all": (0,),
            "single_window_all": (0, 1, 2),
            "pair_window_all": (0, 1, 2),
            "mode4_a_field_all": (0, 1),
        },
    )

    mutation = evolution.DiversityMutation(prob=1.0)
    out = mutation._do(problem, rows)

    assert out.shape == rows.shape


def test_diversity_mutation_window_drift_stays_within_active_pool(monkeypatch):
    evolution = _reload_evolution()
    problem = _make_problem(evolution)
    rows = np.vstack([_base_row(evolution)])
    rows[0, 2] = 1
    captured = {}

    _patch_identity_pm(monkeypatch, evolution)
    _patch_postprocess_capture(monkeypatch, evolution, captured)
    monkeypatch.setattr(evolution.DiversityMutation, "OP_FLIP_PROB", 0.0)
    monkeypatch.setattr(evolution.DiversityMutation, "MODE_FLIP_PROB", 0.0)
    monkeypatch.setattr(evolution.DiversityMutation, "WINDOW_DRIFT_PROB", 1.0, raising=False)
    monkeypatch.setattr(evolution.DiversityMutation, "FIELD_SWAP_PROB", 0.0, raising=False)
    monkeypatch.setattr(evolution.DiversityMutation, "MASK_FLIP_PROB", 0.0, raising=False)
    monkeypatch.setattr(
        evolution,
        "build_search_space_state",
        lambda: {
            "mode1_op_all": (0,),
            "mode2_op_all": (0,),
            "mode3_op_all": (0,),
            "pair_window_all": (0, 1, 2),
            "mode4_a_field_all": (0, 1),
        },
    )

    mutation = evolution.DiversityMutation(prob=1.0)
    mutation._rng = _ZeroChoiceRng()
    mutation._do(problem, rows)

    drifted = int(captured["project_input"][0, 2])
    assert drifted in (0, 2)
    assert drifted != 1


def test_diversity_mutation_field_swap_keeps_field_index_valid(monkeypatch):
    evolution = _reload_evolution()
    problem = _make_problem(evolution)
    rows = np.vstack([_base_row(evolution)])
    rows[0, 0] = 1
    captured = {}

    _patch_identity_pm(monkeypatch, evolution)
    _patch_postprocess_capture(monkeypatch, evolution, captured)
    monkeypatch.setattr(evolution.DiversityMutation, "OP_FLIP_PROB", 0.0)
    monkeypatch.setattr(evolution.DiversityMutation, "MODE_FLIP_PROB", 0.0)
    monkeypatch.setattr(evolution.DiversityMutation, "WINDOW_DRIFT_PROB", 0.0, raising=False)
    monkeypatch.setattr(evolution.DiversityMutation, "FIELD_SWAP_PROB", 1.0, raising=False)
    monkeypatch.setattr(evolution.DiversityMutation, "MASK_FLIP_PROB", 0.0, raising=False)
    monkeypatch.setattr(
        evolution,
        "build_search_space_state",
        lambda: {
            "mode1_op_all": (0,),
            "mode2_op_all": (0,),
            "mode3_op_all": (0,),
            "pair_window_all": (0, 1, 2),
            "mode4_a_field_all": (1, 0, 2, 3),
        },
    )
    monkeypatch.setattr(
        evolution,
        "INDICATOR_NAMES",
        ("returns", "return_abs", "money_flow", "turnover"),
    )
    monkeypatch.setattr(
        evolution,
        "pair_a_family_of_field",
        lambda name: {
            "returns": "return_family",
            "return_abs": "return_family",
            "money_flow": "flow_family",
            "turnover": "flow_family",
        }.get(name, "other_family"),
    )

    mutation = evolution.DiversityMutation(prob=1.0)
    mutation._rng = _ZeroChoiceRng()
    mutation._do(problem, rows)

    swapped = int(captured["project_input"][0, 0])
    assert swapped == 0


def test_diversity_mutation_mask_flip_keeps_rule_index_valid(monkeypatch):
    evolution = _reload_evolution()
    problem = _make_problem(evolution)
    rows = np.vstack([_base_row(evolution)])
    rows[0, 5] = evolution.MASK_RULES.index("high_0.7")
    captured = {}

    _patch_identity_pm(monkeypatch, evolution)
    _patch_postprocess_capture(monkeypatch, evolution, captured)
    monkeypatch.setattr(evolution.DiversityMutation, "OP_FLIP_PROB", 0.0)
    monkeypatch.setattr(evolution.DiversityMutation, "MODE_FLIP_PROB", 0.0)
    monkeypatch.setattr(evolution.DiversityMutation, "WINDOW_DRIFT_PROB", 0.0, raising=False)
    monkeypatch.setattr(evolution.DiversityMutation, "FIELD_SWAP_PROB", 0.0, raising=False)
    monkeypatch.setattr(evolution.DiversityMutation, "MASK_FLIP_PROB", 1.0, raising=False)
    monkeypatch.setattr(
        evolution,
        "build_search_space_state",
        lambda: {
            "mode1_op_all": (0,),
            "mode2_op_all": (0,),
            "mode3_op_all": (0,),
            "pair_window_all": (0, 1, 2),
            "mode4_a_field_all": (0, 1),
        },
    )

    mutation = evolution.DiversityMutation(prob=1.0)
    mutation._do(problem, rows)

    flipped = int(captured["project_input"][0, 5])
    assert 0 <= flipped < len(evolution.MASK_RULES)
    assert evolution.MASK_RULES[flipped] == "low_0.7"


def test_diversity_mutation_respects_global_seed(monkeypatch):
    evolution = _reload_evolution()
    problem = _make_problem(evolution)
    rows = np.vstack([_base_row(evolution), _base_row(evolution) + 1.0])
    state = {
        "mode1_op_all": (0,),
        "mode2_op_all": (0,),
        "mode3_op_all": (0,),
        "pair_window_all": (0, 1, 2),
        "mode4_a_field_all": (0, 1, 2, 3),
    }

    _patch_identity_pm(monkeypatch, evolution)
    monkeypatch.setattr(evolution, "build_search_space_state", lambda: state)
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

    np.random.seed(37)
    first = evolution.DiversityMutation(prob=1.0)._do(problem, rows)

    np.random.seed(37)
    second = evolution.DiversityMutation(prob=1.0)._do(problem, rows)

    assert np.array_equal(first, second)
