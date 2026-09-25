import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gp.minute_gp_system.engines.unified_v2 import evolution
from gp.minute_gp_system.engines.unified_v2.selection_controls import (
    _max_abs_spearman,
    select_low_corr_results,
)


def _make_candidate(index: int, composite: float) -> dict:
    return {
        "_candidate_index": index,
        "composite": composite,
        "decoded": {},
    }


def test_max_abs_spearman_is_high_for_identical_vectors():
    unit_values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )

    corr = _max_abs_spearman(1, [0], unit_values)

    assert corr > 0.99


def test_max_abs_spearman_is_high_for_monotonic_nonlinear_transform():
    base = np.array([-3.0, -1.0, -0.5, 0.5, 1.5, 3.0], dtype=np.float32)
    unit_values = np.stack([base, np.tanh(base)], axis=0)

    corr = _max_abs_spearman(1, [0], unit_values)

    assert corr > 0.99


def test_max_abs_spearman_is_low_for_independent_vectors():
    rng = np.random.default_rng(123)
    unit_values = rng.standard_normal((2, 256)).astype(np.float32)

    corr = _max_abs_spearman(1, [0], unit_values)

    assert corr < 0.20


def test_max_abs_spearman_handles_nan_without_crashing():
    unit_values = np.array(
        [
            [1.0, np.nan, 3.0, 4.0, np.inf],
            [2.0, 5.0, np.nan, -np.inf, 1.0],
        ],
        dtype=np.float32,
    )

    corr = _max_abs_spearman(1, [0], unit_values)

    assert np.isfinite(corr)
    assert corr >= 0.0


def test_select_low_corr_results_rejects_high_spearman_candidate():
    phenotypes = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    candidates = [_make_candidate(0, 10.0), _make_candidate(1, 9.0)]

    selected, summary = select_low_corr_results(
        candidates,
        target_max=2,
        phenotypes=phenotypes,
        behavior_signatures=None,
        factor_values=None,
        pool_corr_hard=1.0,
        spearman_corr_hard=0.75,
        behavior_corr_hard=1.0,
        use_style_exposure=False,
    )

    assert [item["_candidate_index"] for item in selected] == [0]
    assert summary["rejected"]["selected_spearman_corr"] == 1
    assert candidates[1]["selected_spearman_corr_max"] > 0.99
    assert candidates[1]["final_low_corr_gate"]["reason"] == "selected_spearman_corr"


def test_elite_init_explicitly_disables_spearman_hard_gate(monkeypatch):
    captured = {}

    monkeypatch.setattr(evolution, "ELITE_INIT_ENABLED", True)
    monkeypatch.setattr(evolution, "_generate_exploration_pop", lambda n, rng, feedback_artifact=None: np.arange(n * 2, dtype=np.int32).reshape(n, 2))
    monkeypatch.setattr(evolution, "_normalize_population", lambda pop: np.asarray(pop))
    monkeypatch.setattr(evolution, "_empty_aux_store", lambda n: {})
    monkeypatch.setattr(evolution, "_store_aux_slice", lambda aux_store, batch_start, batch_end, batch_aux: aux_store.update(batch_aux))
    monkeypatch.setattr(evolution, "set_backend", lambda backend, gpu_id: None)
    monkeypatch.setattr(evolution, "decode_individual", lambda row: {})
    monkeypatch.setattr(evolution, "sample_diversity_features", lambda batch_factors, tradable_mask, max_periods: np.asarray(batch_factors, dtype=np.float32))
    monkeypatch.setattr(evolution, "standardize_phenotypes", lambda arr: np.asarray(arr, dtype=np.float32))

    def fake_compute(batch_factors, fitness_context, return_aux=True, **kwargs):
        n = len(batch_factors)
        objectives = np.full((n, evolution.N_OBJECTIVES), 1.0, dtype=np.float32)
        aux = {
            "rankicir": np.full(n, evolution.ELITE_INIT_MIN_RANKICIR + 0.1, dtype=np.float32),
            "ls_netret_ann": np.full(n, evolution.ELITE_INIT_MIN_NETRET + 1.0, dtype=np.float32),
            "ls_turnover": np.full(n, evolution.ELITE_INIT_MAX_TURNOVER - 0.1, dtype=np.float32),
            "coverage": np.full(n, evolution.ELITE_INIT_MIN_COVERAGE + 0.1, dtype=np.float32),
            "ls_positive_rate": np.full(n, 0.6, dtype=np.float32),
        }
        return objectives, aux

    def fake_select(candidates, **kwargs):
        captured.update(kwargs)
        return candidates[: kwargs["target_max"]], {"rejected": {}}

    monkeypatch.setattr(evolution, "compute_five_objectives_prepared", fake_compute)
    monkeypatch.setattr(evolution, "select_low_corr_results", fake_select)

    problem = SimpleNamespace(
        backend="cpu",
        gpu_id=None,
        eval_batch_size=8,
        _fitness_context_is_eval=object(),
        _fitness_eval_kwargs={},
        period_tradable_mask=None,
        _build_batch_factors=lambda batch_pop: np.asarray(batch_pop, dtype=np.float32),
    )

    result, meta = evolution._build_elite_initial_population(problem, pop_size=2, seed=7)

    assert result is not None
    assert meta["enabled"] is True
    assert captured["spearman_corr_hard"] == pytest.approx(1.0)
