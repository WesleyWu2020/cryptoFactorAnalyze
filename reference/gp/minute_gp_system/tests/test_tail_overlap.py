import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gp.minute_gp_system.engines.unified_v2 import evolution as evo
from gp.minute_gp_system.engines.unified_v2.selection_controls import (
    _compute_tail_overlap,
    select_low_corr_results,
)


def _make_candidate(index: int, composite: float) -> dict:
    return {
        "_candidate_index": index,
        "composite": composite,
        "decoded": {},
    }


def test_compute_tail_overlap_is_high_for_identical_factors():
    factor_values = np.array(
        [
            [[4.0, 3.0, 2.0, 1.0, -1.0], [5.0, 4.0, 3.0, 2.0, -2.0]],
            [[4.0, 3.0, 2.0, 1.0, -1.0], [5.0, 4.0, 3.0, 2.0, -2.0]],
        ],
        dtype=np.float32,
    )

    overlap = _compute_tail_overlap(1, [0], factor_values, top_frac=0.20)

    assert overlap > 0.99


def test_compute_tail_overlap_is_low_for_anti_correlated_factors():
    factor_values = np.array(
        [
            [[4.0, 3.0, 2.0, 1.0, -1.0], [5.0, 4.0, 3.0, 2.0, -2.0]],
            [[-4.0, -3.0, -2.0, -1.0, 1.0], [-5.0, -4.0, -3.0, -2.0, 2.0]],
        ],
        dtype=np.float32,
    )

    overlap = _compute_tail_overlap(1, [0], factor_values, top_frac=0.20)

    assert overlap < 0.01


def test_tail_factor_sampling_adjusts_direction_and_masks_non_tradable_assets():
    batch_factors = np.array(
        [
            [[4.0, 3.0, 2.0, 1.0, -1.0], [5.0, 4.0, 3.0, 2.0, -2.0]],
            [[-4.0, -3.0, -2.0, -1.0, 1.0], [-5.0, -4.0, -3.0, -2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    ic_direction = np.array([1.0, -1.0], dtype=np.float32)
    tradable_mask = np.array(
        [[True, True, False, True, True], [True, True, False, True, True]],
        dtype=bool,
    )

    sampled = evo._sample_tail_factor_values(
        batch_factors,
        ic_direction,
        tradable_mask,
        max_periods=10,
    )

    overlap = _compute_tail_overlap(1, [0], sampled, top_frac=0.20)

    assert overlap > 0.99
    assert np.isnan(sampled[:, :, 2]).all()


def test_compute_tail_overlap_handles_nan_inputs():
    factor_values = np.array(
        [
            [[4.0, np.nan, 2.0, 1.0, -1.0], [np.nan, 4.0, 3.0, 2.0, -2.0]],
            [[4.0, 3.0, np.nan, 1.0, -1.0], [5.0, 4.0, 3.0, np.nan, -2.0]],
        ],
        dtype=np.float32,
    )

    overlap = _compute_tail_overlap(1, [0], factor_values, top_frac=0.20)

    assert np.isfinite(overlap)
    assert overlap >= 0.0


def test_compute_tail_overlap_skips_periods_with_insufficient_valid_assets():
    factor_values = np.array(
        [
            [
                [5.0, 4.0, np.nan, np.nan, np.nan],
                [5.0, 4.0, 3.0, 2.0, -2.0],
            ],
            [
                [5.0, 4.0, np.nan, np.nan, np.nan],
                [5.0, 4.0, 3.0, 2.0, -2.0],
            ],
        ],
        dtype=np.float32,
    )

    overlap = _compute_tail_overlap(1, [0], factor_values, top_frac=0.20)

    assert overlap > 0.99


def test_compute_tail_overlap_returns_zero_for_none_inputs():
    assert _compute_tail_overlap(None, [0], None) == 0.0
    assert _compute_tail_overlap(0, [], None) == 0.0


def test_compute_tail_overlap_returns_zero_for_out_of_bounds_index():
    factor_values = np.ones((1, 2, 5), dtype=np.float32)

    assert _compute_tail_overlap(3, [0], factor_values) == 0.0


def test_select_low_corr_results_rejects_high_tail_overlap_candidate():
    factor_values = np.array(
        [
            [[4.0, 3.0, 2.0, 1.0, -1.0], [5.0, 4.0, 3.0, 2.0, -2.0]],
            [[4.0, 3.0, 2.0, 1.0, -1.0], [5.0, 4.0, 3.0, 2.0, -2.0]],
        ],
        dtype=np.float32,
    )
    candidates = [_make_candidate(0, 10.0), _make_candidate(1, 9.0)]

    selected, summary = select_low_corr_results(
        candidates,
        target_max=2,
        phenotypes=None,
        behavior_signatures=None,
        factor_values=factor_values,
        pool_corr_hard=1.0,
        behavior_corr_hard=1.0,
        tail_overlap_hard=0.70,
        use_style_exposure=False,
    )

    assert [item["_candidate_index"] for item in selected] == [0]
    assert summary["rejected"]["selected_tail_overlap"] == 1
    assert candidates[1]["selected_tail_overlap"] > 0.99
    assert candidates[1]["final_low_corr_gate"]["reason"] == "selected_tail_overlap"
