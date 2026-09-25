import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/crypto-research/users/wesleywu")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gp.minute_gp_system.engines.unified_v2 import evolution as evo
from gp.minute_gp_system.engines.unified_v2.fitness import (
    _postprocess_single,
    _precompute_ret_rank,
    compute_five_objectives_prepared,
    compute_rankic_novelty,
    prepare_fitness_context,
    compute_residual_rankic,
    prepare_residual_basis_context,
)


def _synthetic_residual_setup(seed=7, P=100, S=60):
    rng = np.random.default_rng(seed)
    b1 = rng.standard_normal((P, S)).astype(np.float32)
    g = rng.standard_normal((P, S)).astype(np.float32)
    noise = 0.3 * rng.standard_normal((P, S)).astype(np.float32)
    ret = (b1 + g + noise).astype(np.float32)

    cand_clone = b1 + 0.05 * rng.standard_normal((P, S)).astype(np.float32)
    cand_fresh = g.copy()

    idx = np.arange(P, dtype=np.int32)
    ret_rank_cache = _precompute_ret_rank(ret)
    ctx = (idx, ret, None, ret_rank_cache)
    basis = _postprocess_single(b1)[None]  # (K=1, P, S)
    return ctx, basis, np.stack([cand_clone, cand_fresh])


def test_residual_rankic_separates_clone_from_fresh_signal():
    ctx, basis, candidates = _synthetic_residual_setup()
    basis_ctx = prepare_residual_basis_context(basis, ctx, max_periods=128)
    assert basis_ctx is not None

    rankicir, ic_mean = compute_residual_rankic(
        candidates, basis_ctx, periods_per_year=365, min_periods=10,
    )
    assert np.all(np.isfinite(rankicir))
    # The clone's alpha is fully spanned by the basis; the fresh signal keeps
    # most of its IC after residualization.
    assert abs(rankicir[0]) < 1.0
    assert rankicir[1] > 5.0
    assert ic_mean[1] > 0.1


def test_residual_rankic_sign_flip_tracked():
    ctx, basis, candidates = _synthetic_residual_setup()
    basis_ctx = prepare_residual_basis_context(basis, ctx, max_periods=128)
    flipped = -candidates[1:2]
    rankicir_up, _ = compute_residual_rankic(
        candidates[1:2], basis_ctx, ic_directions=[1.0],
        periods_per_year=365, min_periods=10,
    )
    rankicir_dn, _ = compute_residual_rankic(
        flipped, basis_ctx, ic_directions=[1.0],
        periods_per_year=365, min_periods=10,
    )
    assert rankicir_up[0] > 0.0
    assert rankicir_dn[0] < 0.0
    # Direction arg must re-flip the signed residual back to positive.
    rankicir_fixed, _ = compute_residual_rankic(
        flipped, basis_ctx, ic_directions=[-1.0],
        periods_per_year=365, min_periods=10,
    )
    assert rankicir_fixed[0] > 0.0


def test_prepare_residual_basis_context_rejects_tiny_input():
    rng = np.random.default_rng(0)
    P, S = 6, 30
    ret = rng.standard_normal((P, S)).astype(np.float32)
    idx = np.arange(P, dtype=np.int32)
    ctx = (idx, ret, None, _precompute_ret_rank(ret))
    basis = rng.standard_normal((2, P, S)).astype(np.float32)
    assert prepare_residual_basis_context(basis, ctx) is None


def test_compose_search_objectives_has_novelty_column():
    assert evo.SEARCH_N_OBJECTIVES == 6  # novelty objective on by default
    is_obj = np.array(
        [
            [2.0, 3.0, -0.1, 0.0, 0.8],
            [-1e6, -1e6, -1e6, -1e6, -1e6],
        ],
        dtype=np.float32,
    )
    min_netret = np.array([0.5, -1e6], dtype=np.float32)
    min_sharpe = np.array([1.5, -1e6], dtype=np.float32)
    out = evo._compose_annual_worst_search_objectives(is_obj, min_netret, min_sharpe)
    assert out.shape == (2, 6)
    assert out[0, 5] == 0.0  # neutral placeholder, filled downstream
    assert out[1, 5] == -1e6


def test_combine_search_novelty_uses_min_of_references():
    pool = np.array([0.9, 0.4, 1.0], dtype=np.float32)
    ext_corr = np.array([0.2, 0.7, np.nan], dtype=np.float32)
    valid = np.array([True, True, False])
    out = evo._combine_search_novelty(pool, ext_corr, valid)
    assert abs(out[0] - 0.8) < 1e-6   # min(0.9, 1-0.2)
    assert abs(out[1] - 0.3) < 1e-6   # min(0.4, 1-0.7)
    assert out[2] == -1e6             # invalid row stays sentinel
    # No external reference -> pool novelty passes through.
    out2 = evo._combine_search_novelty(
        pool, np.full(3, np.nan, dtype=np.float32), np.ones(3, dtype=bool))
    assert np.allclose(out2, [0.9, 0.4, 1.0])


def test_residual_ic_multipliers():
    resid = np.array([2.0, 1.0, 0.0, np.nan, -3.0], dtype=np.float32)
    mult = evo._residual_ic_multipliers(
        resid,
        config={"enabled": True, "target": 1.0, "strength": 0.75, "min_multiplier": 0.30},
    )
    assert mult[0] == 1.0            # above target: no penalty
    assert mult[1] == 1.0            # at target: no penalty
    assert abs(mult[2] - 0.30) < 1e-6  # zero residual IC -> floor
    assert mult[3] == 1.0            # NaN (not computed): untouched
    assert abs(mult[4] - 0.30) < 1e-6  # negative residual IC -> floor


def test_search_scores_tolerate_legacy_five_wide_objectives():
    legacy = np.array([[1.0, 2.0, 0.5, 1.0, -0.2]], dtype=np.float32)
    scores = evo._compute_search_scores(legacy)
    expected = legacy[0] @ evo.SEARCH_SCORE_WEIGHTS[:5]
    assert abs(scores[0] - expected) < 1e-4
    six = np.array([[1.0, 2.0, 0.5, 1.0, -0.2, 0.9]], dtype=np.float32)
    scores6 = evo._compute_search_scores(six)
    assert scores6[0] > scores[0]  # novelty column contributes positively



def test_rankic_novelty_matches_absolute_production_correlation():
    library_rankic = np.array(
        [
            [0.0, 5.0],
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [5.0, 0.0],
        ],
        dtype=np.float32,
    )
    candidate_rankic = np.array(
        [
            library_rankic[:, 0],
            -library_rankic[:, 0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    novelty = compute_rankic_novelty(candidate_rankic, library_rankic, min_valid=3)

    assert novelty[0] == 0.0
    assert novelty[1] == 0.0
    expected_corr = max(
        abs(np.corrcoef(candidate_rankic[2], library_rankic[:, j])[0, 1])
        for j in range(library_rankic.shape[1])
    )
    assert np.isclose(novelty[2], 1.0 - expected_corr, atol=1e-6)


def test_rankic_novelty_uses_pairwise_finite_observations():
    candidate_rankic = np.array([[1.0, 2.0, np.nan, 4.0, 5.0, 6.0]], dtype=np.float32)
    library_rankic = np.array(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
        dtype=np.float32,
    )

    novelty = compute_rankic_novelty(candidate_rankic, library_rankic, min_valid=5)

    assert novelty.shape == (1,)
    assert novelty[0] < 1e-6


def test_rankic_novelty_returns_neutral_value_without_usable_library_match():
    candidate_rankic = np.ones((2, 4), dtype=np.float32)
    library_rankic = np.full((4, 1), np.nan, dtype=np.float32)

    novelty = compute_rankic_novelty(candidate_rankic, library_rankic, min_valid=3)

    assert np.allclose(novelty, 1.0)


def test_prepared_fitness_can_return_candidate_rankic_series():
    rng = np.random.default_rng(12)
    period_returns = rng.standard_normal((70, 20)).astype(np.float32)
    context = prepare_fitness_context(period_returns)
    factors = rng.standard_normal((2, 70, 20)).astype(np.float32)

    objectives, aux, rankic_series = compute_five_objectives_prepared(
        factors,
        context,
        periods_per_year=365,
        return_aux=True,
        return_rankic_series=True,
    )

    assert objectives.shape == (2, 5)
    assert aux["ic_mean"].shape == (2,)
    assert rankic_series.shape == (2, 70)


def test_rankic_novelty_takes_precedence_over_behavior_novelty():
    out = evo._combine_search_novelty(
        np.array([0.9], dtype=np.float32),
        np.array([0.1], dtype=np.float32),
        np.array([True]),
        rankic_novelty=np.array([0.2], dtype=np.float32),
    )
    assert np.allclose(out, [0.2])


def test_rankic_novelty_basis_aligns_only_to_is_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(evo, "RANKIC_NOVELTY_MIN_VALID", 2)
    path = tmp_path / "rankics.parquet"
    index = pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2025-01-01"])
    rankics = pd.DataFrame(
        {"base_a": [1.0, 99.0, 3.0, 77.0], "base_b": [10.0, 98.0, 30.0, 66.0]},
        index=index,
    )
    rankics.to_parquet(path)

    class Loader:
        dates_flat = np.array(["20220101", "20220102", "20220103", "20250101"])
        active_period_indices = np.arange(4, dtype=np.int32)

    class Problem:
        loader = Loader()
        _fitness_context_is = (np.array([0, 2], dtype=np.int32), None, None, None)

    basis, meta = evo._load_rankic_novelty_basis(str(path), Problem())

    assert basis.shape == (2, 2)
    assert np.array_equal(basis, np.array([[1.0, 10.0], [3.0, 30.0]], dtype=np.float32))
    assert meta["n_is_periods"] == 2
    assert meta["date_start"] == "2022-01-01"
    assert meta["date_end"] == "2022-01-03"


def test_rankic_novelty_basis_missing_path_is_explicit_failure():
    try:
        evo._load_rankic_novelty_basis("/tmp/definitely_missing_rankic_library.parquet", object())
    except FileNotFoundError as exc:
        assert "GP_RANKIC_LIBRARY_PATH" in str(exc)
    else:
        raise AssertionError("missing RankIC library must be reported")
