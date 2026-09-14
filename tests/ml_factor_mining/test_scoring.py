import numpy as np
import pandas as pd
import pytest

from ML_factor_mining.config import Config, quarter_folds
from ML_factor_mining.scoring import (
    CandidateRejected,
    MarketWindow,
    rank_candidates,
    score_validation,
)


def test_composite_preserves_sign_and_excludes_invalid():
    rows = [
        {"candidate_id": "wrong", "status": "valid", "mean_rank_ic": -0.8, "net_sharpe": -3.0},
        {"candidate_id": "right", "status": "valid", "mean_rank_ic": 0.1, "net_sharpe": 1.0},
        {"candidate_id": "broken", "status": "rejected", "mean_rank_ic": 1.0, "net_sharpe": 99.0},
    ]
    winner, table = rank_candidates(rows, Config())
    assert winner == "right"
    assert pd.isna(table.set_index("candidate_id").loc["broken", "selection_score"])


def test_ties_and_single_candidate_are_deterministic():
    one = {"candidate_id": "a", "status": "valid", "mean_rank_ic": 0.1, "net_sharpe": 1.0}
    winner, table = rank_candidates([one], Config())
    assert winner == "a"
    assert table.iloc[0]["selection_score"] == 0.5
    assert rank_candidates([{**one, "candidate_id": "b"}, one], Config())[0] == "a"
    with pytest.raises(CandidateRejected) as rejected:
        rank_candidates([{**one, "net_sharpe": None}], Config())
    assert rejected.value.records[0]["candidate_id"] == "a"


def test_validation_tail_and_profile_are_bounded(monkeypatch):
    config = Config(min_pairs=5, min_val_dates=2)
    fold = quarter_folds(config, "2025-03-31")[0]
    dates = pd.date_range("2024-10-01", "2024-12-31", name="date")
    members = pd.DataFrame(True, index=dates, columns=list("ABCDE"))
    matrix = pd.DataFrame(np.tile(np.arange(5), (len(dates), 1)), index=dates, columns=list("ABCDE"))
    pred = matrix.rename_axis(columns="instrument").stack(future_stack=True)
    targets = pred.rename("raw_return").to_frame()
    market = MarketWindow(matrix + 100, pd.DataFrame(), pd.DataFrame())
    calls = {}

    def fake_backtest(values, opens, events, quality, profile, *, signal_start, signal_end):
        calls.update({"end": signal_end, "profile": profile, "values": values})
        ledger = pd.DataFrame(
            {"return": np.resize([0.01, -0.004], len(dates))}, index=dates
        )
        return {
            "scenarios": {
                "all_costs": {
                    "status": "complete",
                    "ledger": ledger,
                    "diagnostics": {
                        "liquidation_reached": True,
                        "liquidation_date": "2024-12-31",
                        "final_quantities": {},
                    },
                }
            }
        }

    monkeypatch.setattr("ML_factor_mining.scoring.run_backtest", fake_backtest)
    result = score_validation(pred, targets, members, market, fold, config)
    assert calls["end"] == pd.Timestamp("2024-12-27")
    assert calls["values"].loc["2024-12-28":].isna().all().all()
    assert calls["profile"].rebalance_days == 3
    assert result["ic_dates"] >= 2
