"""Matched-sample style residual evidence in bounded GA validation."""

import numpy as np
import pandas as pd
import pytest

from Genetic_Algorithm.config import SearchConfig, Stage
from Genetic_Algorithm.data import load_stage
from Genetic_Algorithm.exposure import assess_exposure_residual
from Genetic_Algorithm.selection import select_validation


def _config(**overrides):
    return SearchConfig(fitness_mode="all_costs_sharpe", n_groups=5,
                        **{"exposure_residual_mode": "diagnostic", **overrides})


def test_exposure_mode_is_explicit_and_cost_bounded():
    for mode in ("off", "diagnostic", "gate"):
        assert _config(exposure_residual_mode=mode).exposure_residual_mode == mode
    with pytest.raises(ValueError, match="exposure_residual_mode"):
        _config(exposure_residual_mode="unknown")
    with pytest.raises(ValueError, match="all_costs_sharpe"):
        SearchConfig(exposure_residual_mode="gate")


def test_residual_uses_identical_cells_and_certified_ledgers(gp_h5):
    stage = Stage("validation", "2025-01-01", "2025-03-31")
    data = load_stage(gp_h5, stage, 0, ["close"], include_accounting=True)
    rng = np.random.default_rng(414)
    shape = data.opens.shape
    style = pd.DataFrame(rng.normal(size=shape), index=data.opens.index,
                         columns=data.opens.columns)
    style.iloc[::9, 0] = np.nan
    alpha = 2 * style + pd.DataFrame(rng.normal(scale=.3, size=shape),
                                    index=style.index, columns=style.columns)
    evidence = assess_exposure_residual(alpha, data, _config(min_quarter_days=20), 1,
                                        {"style": style})
    assert evidence["status"] == "complete"
    assert evidence["valid_regression_days"] == len(style)
    assert evidence["mean_r2"] > .9
    assert evidence["style_summary"][0]["style"] == "style"
    assert evidence["style_summary"][0]["pearson_abs_mean"] > .9
    assert evidence["matched_cells"] < alpha.size
    assert evidence["matched_days"] == len(style)
    assert evidence["raw_matched_ic"]["days"] == evidence["residual_ic"]["days"]
    assert evidence["ledger_days"] == len(style) - 1
    assert evidence["raw_matched_net"]["sharpe"] is not None
    assert evidence["residual_net"]["sharpe"] is not None


def test_insufficient_or_degenerate_residual_is_evidence_not_a_spurious_pass(gp_h5):
    stage = Stage("validation", "2025-01-01", "2025-02-15")
    data = load_stage(gp_h5, stage, 0, ["close"], include_accounting=True)
    style = pd.DataFrame(1., index=data.opens.index, columns=data.opens.columns)
    alpha = data.opens.copy()
    result = assess_exposure_residual(alpha, data, _config(min_quarter_days=20), 1,
                                      {"constant": style})
    assert result["status"] == "insufficient_data"
    assert result["passed"] is False
    assert result["regression_status_counts"].get("insufficient_data", 0) > 0
    assert result["matched_cells"] == 0

    rng = np.random.default_rng(96)
    independent_style = pd.DataFrame(rng.normal(size=alpha.shape), index=alpha.index,
                                     columns=alpha.columns)
    exact_style_signal = 3 * independent_style
    rounding = assess_exposure_residual(exact_style_signal, data,
                                        _config(min_quarter_days=20), 1,
                                        {"style": independent_style})
    assert rounding["valid_regression_days"] == len(style)
    assert rounding["numerically_zero_residual_days"] == len(style)
    assert rounding["matched_cells"] == 0
    assert rounding["passed"] is False


def test_residual_gate_rejects_failed_evidence_and_diagnostic_only_records_it():
    candidate = {"expression_id": "candidate", "training_direction": 1, "direction": 1,
                 "complexity": 1}
    base = {
        "direction": 1, "day_coverage": 1., "cell_coverage": 1.,
        "quarter_valid_days": {f"Q{i}": 60 for i in range(1, 5)},
        "all_costs_cumulative_return": .2, "net_sharpe": 2., "turnover": .1,
        "exposure_residual": {"passed": False, "reasons": ["Barra residual net return or Sharpe is not positive"]},
    }
    gate = SearchConfig(fitness_mode="all_costs_sharpe", exposure_residual_mode="gate")
    rejected = select_validation([candidate], {"candidate": base}, gate)
    assert not rejected.accepted
    assert "Barra residual" in rejected.rejection_reasons["candidate"][0]
    diagnostic = SearchConfig(fitness_mode="all_costs_sharpe", exposure_residual_mode="diagnostic")
    assert len(select_validation([candidate], {"candidate": base}, diagnostic).accepted) == 1
    passing = {**base, "exposure_residual": {"passed": True, "reasons": []}}
    assert len(select_validation([candidate], {"candidate": passing}, gate).accepted) == 1
