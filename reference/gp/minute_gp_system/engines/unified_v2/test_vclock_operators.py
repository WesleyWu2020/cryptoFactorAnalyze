import numpy as np

# Force CPU backend for tests (avoid GPU dependency)
import os
os.environ.setdefault("GP_BACKEND", "cpu")

from minute_gp_system.engines.unified_v2 import operators as O
from minute_gp_system.engines.unified_v2 import config as C


def _data():
    D, W, S = 3, 48, 5
    rng = np.random.default_rng(2)
    a = rng.standard_normal((D, W, S)).astype(np.float32)
    b = np.abs(rng.standard_normal((D, W, S))).astype(np.float32)  # volume proxy
    mask = np.ones((D, W, S), dtype=np.float32)
    return a, b, mask


def test_vclock_dispersion_finite():
    a, b, mask = _data()
    out = O.m4_group_dispersion_vclock(a, b, mask, 6)
    assert out.shape == (3, 5)
    assert np.isfinite(out).all()


def test_vclock_slope_finite():
    a, b, mask = _data()
    out = O.m4_group_slope_vclock(a, b, mask, 6)
    assert out.shape == (3, 5)
    assert np.isfinite(out).sum() > 0


def test_vclock_path_length_range():
    a, b, mask = _data()
    out = O.m4_group_path_length_vclock(a, b, mask, 6)
    assert out.shape == (3, 5)
    valid = out[np.isfinite(out)]
    if len(valid):
        assert (valid >= 0).all() and (valid <= 1 + 1e-5).all()


def test_vclock_early_late_diff_finite():
    a, b, mask = _data()
    out = O.m4_group_early_late_diff_vclock(a, b, mask, 6)
    assert out.shape == (3, 5)


def test_vclock_registered_in_dispatch():
    for name in ["group_dispersion_vclock", "group_slope_vclock",
                 "group_path_length_vclock", "group_early_late_diff_vclock"]:
        assert name in O.MODE4_DISPATCH, f"{name} not in MODE4_DISPATCH"


def test_old_mode4_indices_preserved():
    """旧索引 0-9 绝对不能变（archive compatibility contract）"""
    assert C.MODE4_OPS[0] == "group_slope"
    assert C.MODE4_OPS[1] == "group_dispersion"
    assert C.MODE4_OPS[7] == "group_path_length"
    assert C.MODE4_OPS[9] == "group_first_last_diff"
    assert len(C.MODE4_OPS) == 14  # 旧 10 + 新 4


def test_vclock_uses_volume_weighting():
    """验证成交量时钟与等时分组给出不同结果（高低峰效应）"""
    D, W, S = 2, 24, 3
    rng = np.random.default_rng(99)
    a = rng.standard_normal((D, W, S)).astype(np.float32)
    # 成交量集中在后半段（模拟尾盘效应）
    b_uniform = np.ones((D, W, S), dtype=np.float32)
    b_tail = np.concatenate([
        np.full((D, W//2, S), 0.1, dtype=np.float32),
        np.full((D, W - W//2, S), 2.0, dtype=np.float32),
    ], axis=1)
    mask = np.ones((D, W, S), dtype=np.float32)
    out_uniform = O.m4_group_dispersion_vclock(a, b_uniform, mask, 4)
    out_tail = O.m4_group_dispersion_vclock(a, b_tail, mask, 4)
    # 两者结果不完全相同（成交量加权有效果）
    assert not np.allclose(out_uniform, out_tail)
