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
    row[13] = 0
    row[14] = INTRADAY_GROUP_CHOICES.index(4)

    text = formula_string(row)

    assert MODE4_OPS[0] in text
    assert "groups=4" in text


def test_evaluate_population_supports_mode4_group_slope():
    from gp.minute_gp_system.engines.unified_v2.evaluator import evaluate_population

    p, n_ind, m, s = 2, max(INDICATOR_INDEX.values()) + 1, 1440, 3
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
