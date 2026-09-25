import warnings

import numpy as np

from gp.minute_gp_system.engines.unified_v2.config import CS_COMP_OPS
from gp.minute_gp_system.engines.unified_v2.evaluator import apply_composition
from gp.minute_gp_system.engines.unified_v2.fitness import (
    _postprocess_single,
    standardize_phenotypes,
)


def test_postprocess_single_handles_all_nan_rows_without_runtime_warning():
    factor = np.full((2, 4), np.nan, dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        out = _postprocess_single(factor)

    assert not caught
    assert np.isnan(out).all()


def test_apply_composition_cs_zscore_handles_all_nan_rows_without_runtime_warning():
    factor = np.full((1, 3, 4), np.nan, dtype=np.float32)
    population = np.zeros((1, 13), dtype=np.int32)
    population[0, 12] = CS_COMP_OPS.index('cs_zscore')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        out = apply_composition(factor, population)

    assert not caught
    assert np.isnan(out).all()


def test_standardize_phenotypes_handles_all_nan_rows_without_runtime_warning():
    phenotypes = np.array([
        [np.nan, np.nan, np.nan],
        [1.0, 2.0, 3.0],
    ], dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        out = standardize_phenotypes(phenotypes)

    assert not caught
    assert np.allclose(out[0], 0.0)
    assert np.isfinite(out[1]).all()
