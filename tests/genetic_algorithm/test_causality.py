from __future__ import annotations

import numpy as np
import pandas as pd

from Genetic_Algorithm.expression import Node
from Genetic_Algorithm.evaluator import evaluate_tree
from Genetic_Algorithm.config import STAGES
from Genetic_Algorithm.data import load_stage
from factor_common.data_provider import DataProvider
from factor_common.validation import check_cutoff


def test_daily_gp_prefix_replay_is_identical_at_listing_and_year_cutoffs(gp_h5):
    def compute(cutoff):
        provider = DataProvider(gp_h5, as_of=cutoff)
        end = pd.Timestamp("2026-09-01") if cutoff is None else pd.Timestamp(cutoff)
        close = provider.get_single_data("close", start="2024-07-01", end=end)
        eligible = provider.get_universe(start="2024-07-01", end=end)
        return evaluate_tree(Node("close"), {"close": close}, eligible)

    result = check_cutoff(compute, ["2024-12-31", "2025-01-01", "2026-01-01"], atol=1e-10)
    assert result["status"] == "verified"
    assert [item["max_abs_diff"] for item in result["cutoffs"]] == [0.0, 0.0, 0.0]
    assert all(item["index_equal"] and item["mask_equal"] for item in result["cutoffs"])


def test_cutoff_policy_rejects_finite_column_hidden_by_drop():
    def compute(cutoff):
        index = pd.date_range("2025-01-01", pd.Timestamp(cutoff or "2025-01-03"), freq="D")
        columns = ["A", "B"] if cutoff is None else ["A"]
        return pd.DataFrame(np.ones((len(index), len(columns))), index=index, columns=columns)

    import pytest
    with pytest.raises(AssertionError, match="removed finite historical"):
        check_cutoff(compute, ["2025-01-02"], atol=1e-10)


def test_fixture_exercises_complete_funding_and_intentional_new_listing_gap(gp_h5):
    provider = DataProvider(gp_h5, as_of="2025-01-02")
    funding = provider.get_funding(start="2025-01-01", end="2025-01-02", symbols=["S00USDT", "S29USDT"])
    quality = provider.get_quality(start="2025-01-01", end="2025-01-02", symbols=["S29USDT"])
    assert len(funding) == 4
    assert funding["mark_price_valid"].all()
    assert not bool(quality.loc[(pd.Timestamp("2025-01-02"), "S29USDT"), "has_complete_kline"])


def test_fixture_loads_train_validation_and_test_with_180_day_warmup(gp_h5):
    loaded = {name: load_stage(gp_h5, stage, 180, ["close"]) for name, stage in STAGES.items()}
    assert loaded["train"].opens.index[-1] == STAGES["train"].end
    assert loaded["validation"].opens.index[-1] == STAGES["validation"].end
    assert loaded["test"].opens.index[-1] == STAGES["test"].end
    assert not loaded["test"].quality_eligible.empty
