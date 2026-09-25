import numpy as np

from gp.minute_gp_system.engines.unified_v2.operators import MODE4_DISPATCH


def test_group_slope_positive_for_increasing_group_means():
    a = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    b = np.zeros_like(a)
    mask = np.ones_like(a, dtype=np.float32)

    out = MODE4_DISPATCH["group_slope"](a, b, mask, 4)

    assert out.shape == (1, 1)
    assert out[0, 0] > 0.9


def test_group_early_late_diff_uses_late_minus_early():
    a = np.array([[[1.0], [1.0], [5.0], [5.0]]], dtype=np.float32)
    b = np.zeros_like(a)
    mask = np.ones_like(a, dtype=np.float32)

    out = MODE4_DISPATCH["group_early_late_diff"](a, b, mask, 4)

    assert np.allclose(out, [[4.0]], equal_nan=False)


def test_group_top_share_measures_positive_concentration():
    a = np.array([[[1.0], [1.0], [8.0], [0.0]]], dtype=np.float32)
    b = np.zeros_like(a)
    mask = np.ones_like(a, dtype=np.float32)

    out = MODE4_DISPATCH["group_top_share"](a, b, mask, 4)

    assert np.allclose(out, [[0.8]], atol=1e-6)


def test_group_corr_and_beta_use_paired_group_means():
    a = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    b = np.array([[[2.0], [4.0], [6.0], [8.0]]], dtype=np.float32)
    mask = np.ones_like(a, dtype=np.float32)

    corr = MODE4_DISPATCH["group_corr"](a, b, mask, 4)
    beta = MODE4_DISPATCH["group_beta"](a, b, mask, 4)

    assert np.allclose(corr, [[1.0]], atol=1e-5)
    assert np.allclose(beta, [[0.5]], atol=1e-5)


def test_group_path_length_is_path_efficiency():
    # 完全单调：|last - first| == path_length → efficiency = 1
    monotone = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    # 迂回回到起点：net displacement = 0 → efficiency = 0
    meander = np.array([[[1.0], [4.0], [1.0], [1.0]]], dtype=np.float32)
    b = np.zeros_like(monotone)
    mask = np.ones_like(monotone, dtype=np.float32)

    eff_mono = MODE4_DISPATCH["group_path_length"](monotone, b, mask, 4)
    eff_meander = MODE4_DISPATCH["group_path_length"](meander, b, mask, 4)

    assert np.allclose(eff_mono, [[1.0]], atol=1e-5)
    assert np.allclose(eff_meander, [[0.0]], atol=1e-5)


def test_group_dispersion_uses_b_to_define_buckets():
    a = np.array([[[0.0], [0.0], [10.0], [10.0], [20.0], [20.0]]], dtype=np.float32)
    b_sorted = np.array([[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]], dtype=np.float32)
    b_mixed = np.array([[[0.0], [2.0], [4.0], [1.0], [3.0], [5.0]]], dtype=np.float32)
    mask = np.ones_like(a, dtype=np.float32)

    out_sorted = MODE4_DISPATCH["group_dispersion"](a, b_sorted, mask, 3)
    out_mixed = MODE4_DISPATCH["group_dispersion"](a, b_mixed, mask, 3)

    assert np.allclose(out_sorted, [[10.0]], atol=1e-5)
    assert np.allclose(out_mixed, [[5.0]], atol=1e-5)
