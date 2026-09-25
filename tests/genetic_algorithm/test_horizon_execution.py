"""Prediction-horizon labels and executable rebalance schedules remain separate."""

import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from Genetic_Algorithm.config import STAGES, SearchConfig, Stage
from Genetic_Algorithm.cost_fitness import evaluate_cost_window
from Genetic_Algorithm.data import load_stage
from Genetic_Algorithm.horizon import assess_horizon_execution


def _config(**overrides):
    return SearchConfig(fitness_mode="all_costs_sharpe", n_groups=5,
                        **{"horizon_diagnostics": True, **overrides})


def test_horizon_options_require_fixed_baseline_and_cost_accounting():
    for field, value in (("prediction_horizons", (3, 7)),
                         ("prediction_horizons", (1, 1)),
                         ("execution_intervals", (1, 0)),
                         ("execution_intervals", (1, 31))):
        with pytest.raises(ValueError, match=field):
            _config(**{field: value})
    with pytest.raises(ValueError, match="all_costs_sharpe"):
        SearchConfig(horizon_diagnostics=True)


def test_fixed_direction_horizons_and_one_day_ledger_parity(gp_h5):
    stage = Stage("validation", "2025-01-01", "2025-03-31")
    data = load_stage(gp_h5, stage, 0, ["close"], include_accounting=True)
    values = pd.DataFrame(np.random.default_rng(318).normal(size=data.opens.shape),
                          index=data.opens.index, columns=data.opens.columns)
    config = _config(prediction_horizons=(1, 3, 7), execution_intervals=(1, 3))
    evidence = assess_horizon_execution(values, data, config, -1)
    assert evidence["direction_source"] == "frozen_training"
    assert set(evidence["prediction"]) == {"1", "3", "7"}
    assert evidence["prediction"]["7"]["last_eligible_signal"] == "2025-03-23"
    assert evidence["prediction"]["7"]["valid_days"] <= len(values) - 8
    assert evidence["execution"]["1"]["status"] == "complete"
    assert evidence["execution"]["3"]["status"] == "complete"
    assert evidence["execution"]["3"]["last_eligible_signal"] == "2025-03-27"
    assert evidence["execution"]["3"]["scheduled_entries"] < evidence["execution"]["1"]["scheduled_entries"]
    assert evidence["execution"]["3"]["last_scheduled_signal"] <= "2025-03-27"
    assert evidence["execution"]["3"]["calendar_days"] == len(values)
    baseline, _ = evaluate_cost_window(values, data, config, -1, stage.start, stage.end)
    assert evidence["execution"]["1"]["net_total_return"] == pytest.approx(
        baseline["total_return"], abs=1e-12)
    assert evidence["execution"]["1"]["turnover"] == pytest.approx(baseline["turnover"])
    assert evidence["execution"]["3"]["turnover"] < evidence["execution"]["1"]["turnover"]


def test_horizon_evidence_rejects_non_validation_and_missing_accounting(gp_h5):
    data = load_stage(gp_h5, Stage("validation", "2025-01-01", "2025-01-20"),
                      0, ["close"])
    with pytest.raises(ValueError, match="validation-stage"):
        assess_horizon_execution(data.opens, replace(data, stage=STAGES["train"]), _config(), 1)
    with pytest.raises(ValueError, match="accounting"):
        assess_horizon_execution(data.opens, data, _config(), 1)
