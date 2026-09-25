import os

os.environ.setdefault("GP_BACKEND", "cpu")

import numpy as np

from minute_gp_system.engines.unified_v2 import config as C
from minute_gp_system.engines.unified_v2 import evaluator as E
from minute_gp_system.engines.unified_v2 import evaluate_with_framework as F


def test_new_operator_names_are_registered():
    assert {
        "ewm_residual", "fast_slow_diff", "rolling_tstat", "robust_zscore",
        "persistence_ratio", "event_count", "drawup", "drawdown",
        "half_life_decay",
    }.issubset(C.TS_COMP_OPS)
    assert {
        "cs_robust_zscore", "cs_winsor_zscore", "cs_tail_rank",
        "cs_neutralize_mcap", "cs_neutralize_liquidity",
        "cs_group_rank_mcap", "cs_group_rank_liquidity",
    }.issubset(C.CS_COMP_OPS)
    for name in C.TS_COMP_OPS:
        assert name in E.TS_DISPATCH
    for name in ("cs_robust_zscore", "cs_winsor_zscore", "cs_tail_rank"):
        assert name in E.CS_DISPATCH


def test_temporal_operators_have_expected_shapes_and_finite_values():
    factor = np.array(
        [[-2.0, 0.0], [-1.0, 1.0], [2.0, 2.0], [3.0, 4.0], [1.0, 8.0], [2.0, 9.0], [3.0, 10.0]],
        dtype=np.float32,
    )
    for name in C.TS_COMP_OPS:
        if name == "none":
            continue
        out, _ = E.TS_DISPATCH[name](factor, 3)
        assert out.shape == factor.shape
        assert np.isfinite(out).any(), name


def test_cross_sectional_robust_and_tail_operators():
    factor = np.array([[1.0, 2.0, 100.0, np.nan]], dtype=np.float32)
    robust = E.CS_DISPATCH["cs_robust_zscore"](factor)
    winsor = E.CS_DISPATCH["cs_winsor_zscore"](factor)
    tail = E.CS_DISPATCH["cs_tail_rank"](factor)
    assert robust.shape == winsor.shape == tail.shape == factor.shape
    assert np.isfinite(robust[:, :3]).all()
    assert np.isfinite(winsor[:, :3]).all()
    assert tail[0, 0] < 0 and tail[0, 2] > 0


def test_dependency_operators_are_applied_with_indicator_data():
    factor = np.array([[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    population = np.zeros((1, 13), dtype=np.int32)
    population[0, 10] = C.TS_COMP_OPS.index("none")
    population[0, 12] = C.CS_COMP_OPS.index("cs_neutralize_mcap")
    mcap_idx = C.INDICATOR_INDEX["cm_mcap_pct_lag1"]
    indicator_data = np.full((2, C.N_INDICATORS, 1, 4), np.nan, dtype=np.float32)
    indicator_data[:, mcap_idx, 0, :] = [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    out = E.apply_composition(factor, population, indicator_data=indicator_data)
    assert out.shape == factor.shape
    assert np.isfinite(out).all()
    assert np.allclose(out[0, 0], 0.0, atol=1e-5)


def test_projected_cache_includes_cross_sectional_dependency_fields():
    params = np.zeros(C.N_PARAMS, dtype=np.int32)
    params[[0, 1, 4]] = [1, 2, 3]
    params[12] = C.CS_COMP_OPS.index("cs_neutralize_liquidity")

    fields = F._entries_field_indices([{"params": params.tolist()}])

    assert C.INDICATOR_INDEX["dollar_volume_rank"] in fields


def test_compute_factor_batch_passes_projected_field_indices(monkeypatch):
    class Loader:
        n_active_periods = 1
        n_tradable_coins = 2
        indicator_field_indices = np.array([1, 2, 3, 68], dtype=np.intp)

        def precompute_indicators(self):
            pass

        def n_chunks(self):
            return 1

        def get_cached_chunk(self, chunk_idx):
            return np.empty((1, 4, 1, 2), dtype=np.float32)

        def get_chunk_size(self, chunk_idx):
            return 1

    captured = {}

    def fake_evaluate(population, indicator_data, output_backend="cpu", perf_stats=None, indicator_field_indices=None):
        captured["field_indices"] = indicator_field_indices
        return np.ones((population.shape[0], 1, 2), dtype=np.float32)

    monkeypatch.setattr(
        "gp.minute_gp_system.engines.unified_v2.evaluator.evaluate_population",
        fake_evaluate,
    )

    F._compute_factor_batch(
        [{"params": [1, 2, 0, 0, 3] + [0] * (C.N_PARAMS - 5)}],
        Loader(),
        backend="cpu",
        batch_size=1,
    )

    assert np.array_equal(captured["field_indices"], np.array([1, 2, 3, 68]))
