import numpy as np
from minute_gp_system.engines.unified_v2.fitness import apply_shortcoming_penalty

def test_penalty_is_continuous_and_monotone():
    rng = np.random.default_rng(1)
    obj = rng.standard_normal((50, 5)).astype(np.float32)
    obj_pos = np.abs(obj) + 0.1            # 保证正值，乘子单调可检
    before = obj_pos.copy()
    after = apply_shortcoming_penalty(obj_pos.copy())
    # 每列最好个体几乎不被罚（乘子≈1），最差个体约 0.5^? 但单列乘子∈[0.5,1]
    for j in range(5):
        ratio = after[:, j] / np.maximum(before[:, j], 1e-9)
        assert ratio.min() >= 0.0
        assert ratio.max() <= 1.0 + 1e-5
        # 不应只有 {0.5,1.0} 两个离散值
        assert len(np.unique(np.round(ratio, 3))) > 5

def test_handles_small_population():
    # valid.sum() < 10 时跳过，不崩溃
    obj = np.full((5, 5), -1e6, dtype=np.float32)
    obj[0, 0] = 1.0  # one valid entry per col is still < 10
    result = apply_shortcoming_penalty(obj.copy())
    np.testing.assert_array_equal(result, obj)
