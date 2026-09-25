# Intraday Group Operators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GP-searchable mode4 operator family that compresses grouped intraday minute information into daily factor values.

**Architecture:** Append mode4 to the existing fixed-width integer genome while keeping old mode1/mode2/mode3 indices stable. Mode4 dispatch lives beside the existing backend-agnostic operators, and the evaluator adds one task path that slices A/B minute data, reuses masks, and calls grouped operators with a selected group count.

**Tech Stack:** Python, NumPy/CuPy-compatible backend wrapper, pytest, existing unified_v2 GP evaluator.

---

## File Structure

- Modify `engines/unified_v2/operators.py` to add grouped helper functions and `MODE4_DISPATCH`.
- Modify `engines/unified_v2/config.py` to add `MODE4_OPS`, `INTRADAY_GROUP_CHOICES`, the new gene slot, bounds, decode/formula output, profile validation, and repair projection.
- Modify `engines/unified_v2/evaluator.py` to evaluate mode4 genomes.
- Modify `engines/unified_v2/evolution.py` to sample and repair mode4 genomes in a small controlled lane.
- Create `tests/test_intraday_group_operators.py` for operator numerics.
- Create `tests/test_intraday_group_evaluator.py` for evaluator and formula smoke coverage.

### Task 1: Operator Unit Tests

**Files:**
- Create: `tests/test_intraday_group_operators.py`

- [ ] **Step 1: Write failing operator tests**

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=/root/crypto-research/users/wesleywu python -m pytest /root/crypto-research/users/wesleywu/gp/minute_gp_system/tests/test_intraday_group_operators.py -q`

Expected: import failure or missing `MODE4_DISPATCH`.

### Task 2: Mode4 Operator Implementation

**Files:**
- Modify: `engines/unified_v2/operators.py`
- Test: `tests/test_intraday_group_operators.py`

- [ ] **Step 1: Add grouped helper and mode4 dispatch**

Add functions that split the minute axis into `min(group_count, W)` contiguous groups, compute masked group means or sums, and implement `group_slope`, `group_dispersion`, `group_early_late_diff`, `group_top_share`, `group_corr`, and `group_beta`.

- [ ] **Step 2: Run operator tests**

Run: `PYTHONPATH=/root/crypto-research/users/wesleywu python -m pytest /root/crypto-research/users/wesleywu/gp/minute_gp_system/tests/test_intraday_group_operators.py -q`

Expected: all operator tests pass.

### Task 3: Config and Formula Tests

**Files:**
- Create: `tests/test_intraday_group_evaluator.py`
- Modify: `engines/unified_v2/config.py`

- [ ] **Step 1: Write failing config/formula tests**

```python
import numpy as np

from gp.minute_gp_system.engines.unified_v2.config import (
    INDICATOR_INDEX,
    INTRADAY_GROUP_CHOICES,
    MODE4_OPS,
    N_PARAMS,
    formula_string,
)


def test_mode4_formula_renders_group_operator():
    row = np.zeros(N_PARAMS, dtype=np.int32)
    row[0] = INDICATOR_INDEX["returns"]
    row[1] = INDICATOR_INDEX["turnover"]
    row[2] = 0
    row[3] = 0
    row[4] = INDICATOR_INDEX["open"]
    row[5] = 0
    row[6] = 3
    row[14] = INTRADAY_GROUP_CHOICES.index(4)

    text = formula_string(row)

    assert MODE4_OPS[0] in text
    assert "groups=4" in text
```

- [ ] **Step 2: Run test and verify failure**

Run: `PYTHONPATH=/root/crypto-research/users/wesleywu python -m pytest /root/crypto-research/users/wesleywu/gp/minute_gp_system/tests/test_intraday_group_evaluator.py::test_mode4_formula_renders_group_operator -q`

Expected: import failure for `MODE4_OPS` or bounds/index failure.

- [ ] **Step 3: Implement config additions**

Append `MODE4_OPS`, `INTRADAY_GROUP_CHOICES`, `intraday_group_count`, increase mode upper bound to 3, update `decode_individual()` and `formula_string()`, and project the new slot in repair.

### Task 4: Evaluator Smoke Test

**Files:**
- Modify: `tests/test_intraday_group_evaluator.py`
- Modify: `engines/unified_v2/evaluator.py`

- [ ] **Step 1: Add failing evaluator test**

```python
def test_evaluate_population_supports_mode4_group_slope():
    from gp.minute_gp_system.engines.unified_v2.config import MODE4_OPS
    from gp.minute_gp_system.engines.unified_v2.evaluator import evaluate_population

    p, n_ind, m, s = 2, max(INDICATOR_INDEX.values()) + 1, 15, 3
    data = np.zeros((p, n_ind, m, s), dtype=np.float32)
    data[:, INDICATOR_INDEX["returns"], :, :] = np.arange(m, dtype=np.float32)[None, :, None]
    data[:, INDICATOR_INDEX["turnover"], :, :] = 1.0
    data[:, INDICATOR_INDEX["open"], :, :] = 1.0

    row = np.zeros(N_PARAMS, dtype=np.int32)
    row[0] = INDICATOR_INDEX["returns"]
    row[1] = INDICATOR_INDEX["turnover"]
    row[2] = 0
    row[3] = 0
    row[4] = INDICATOR_INDEX["open"]
    row[5] = 0
    row[6] = 3
    row[13] = 0
    row[14] = INTRADAY_GROUP_CHOICES.index(4)

    out = evaluate_population(row[None, :], data)

    assert out.shape == (1, p, s)
    assert np.all(out[0] > 0.0)
    assert MODE4_OPS[int(row[13])] == "group_slope"
```

- [ ] **Step 2: Run evaluator test and verify failure**

Run: `PYTHONPATH=/root/crypto-research/users/wesleywu python -m pytest /root/crypto-research/users/wesleywu/gp/minute_gp_system/tests/test_intraday_group_evaluator.py -q`

Expected: evaluator does not route mode4 yet.

- [ ] **Step 3: Implement evaluator mode4 routing**

Import `MODE4_OPS`, `INTRADAY_GROUP_CHOICES`, and `MODE4_DISPATCH`. Include group count in task keys, slice A/B data, and call the selected mode4 function with `group_count`.

### Task 5: Evolution Sampling

**Files:**
- Modify: `engines/unified_v2/evolution.py`
- Test: existing import/compile checks

- [ ] **Step 1: Add controlled mode4 sampling**

Update policy generation and structural exploration so a small share can set `row[6] = 3`, choose mode4 op from `MODE4_OPS`, choose `row[14]` from `INTRADAY_GROUP_CHOICES`, and prefer fields from intraday path, return, volatility, liquidity, and order-flow families.

- [ ] **Step 2: Run compile checks**

Run: `PYTHONPATH=/root/crypto-research/users/wesleywu python -m py_compile /root/crypto-research/users/wesleywu/gp/minute_gp_system/engines/unified_v2/config.py /root/crypto-research/users/wesleywu/gp/minute_gp_system/engines/unified_v2/operators.py /root/crypto-research/users/wesleywu/gp/minute_gp_system/engines/unified_v2/evaluator.py /root/crypto-research/users/wesleywu/gp/minute_gp_system/engines/unified_v2/evolution.py`

Expected: no compile errors.

### Task 6: Full Verification

**Files:**
- Test: `tests/test_intraday_group_operators.py`
- Test: `tests/test_intraday_group_evaluator.py`

- [ ] **Step 1: Run focused tests**

Run: `PYTHONPATH=/root/crypto-research/users/wesleywu python -m pytest /root/crypto-research/users/wesleywu/gp/minute_gp_system/tests/test_intraday_group_operators.py /root/crypto-research/users/wesleywu/gp/minute_gp_system/tests/test_intraday_group_evaluator.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run a package import check**

Run: `PYTHONPATH=/root/crypto-research/users/wesleywu python -c "from gp.minute_gp_system.engines.unified_v2.config import MODE4_OPS, N_PARAMS; from gp.minute_gp_system.engines.unified_v2.evaluator import evaluate_population; print(MODE4_OPS, N_PARAMS)"`

Expected: prints the mode4 operator tuple/list and the updated parameter count.
