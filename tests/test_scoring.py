import numpy as np
import pandas as pd
import pytest

from ML_factor_mining.scoring import CandidateRejected, MarketWindow, rank_candidates, score_validation


def test_rank_candidates_uses_signed_metrics_and_excludes_invalid_rows():
    winner, table = rank_candidates(
        [
            {"candidate_id": "negative", "status": "complete", "mean_rank_ic": -0.9, "net_sharpe": -0.9},
            {"candidate_id": "positive", "status": "complete", "mean_rank_ic": 0.5, "net_sharpe": 0.4},
            {"candidate_id": "bad", "status": "incomplete", "mean_rank_ic": 100.0, "net_sharpe": 100.0},
        ],
        ic_weight=0.7,
        sharpe_weight=0.3,
    )
    assert winner["candidate_id"] == "positive"
    assert pd.isna(table.loc[table.candidate_id == "bad", "selection_score"]).all()


def test_rank_candidates_ties_single_candidate_and_rejection_records():
    winner, table = rank_candidates([
        {"candidate_id": "b", "status": "complete", "mean_rank_ic": 1.0, "net_sharpe": 1.0},
        {"candidate_id": "a", "status": "complete", "mean_rank_ic": 1.0, "net_sharpe": 1.0},
    ])
    assert winner["candidate_id"] == "a"
    assert table.selection_score.nunique() == 1

    winner, table = rank_candidates([
        {"candidate_id": "only", "status": "complete", "mean_rank_ic": 0.0, "net_sharpe": 0.0},
    ])
    assert winner["selection_score"] == 0.0

    with pytest.raises(CandidateRejected) as exc:
        rank_candidates([{"candidate_id": "bad", "status": "incomplete", "mean_rank_ic": 0.1, "net_sharpe": 0.2}])
    assert exc.value.records


def test_score_validation_masks_prediction_tail_and_builds_profile(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    instruments = ["A", "B", "C"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["date", "instrument"])
    predictions = pd.Series(np.tile([1.0, 2.0, 3.0], len(dates)), index=index)
    labels = pd.DataFrame({"raw_return": np.tile([0.01, 0.02, 0.03], len(dates))}, index=index)
    membership = pd.DataFrame(True, index=dates, columns=instruments)
    opens = pd.DataFrame(100.0, index=dates, columns=instruments)
    quality = pd.DataFrame(index=pd.MultiIndex.from_product([dates, instruments], names=["date", "instrument"]))
    quality["has_placeholder_kline"] = False
    events = pd.DataFrame(columns=["funding_time", "instrument", "funding_rate", "mark_price"])
    market = MarketWindow(opens=opens, events=events, quality=quality)

    calls = {}
    def fake_backtest(values, opens, events, quality, profile, *, signal_start, signal_end):
        calls.update(values=values, signal_end=signal_end, profile=profile)
        ledger = pd.DataFrame({"return": [0.01, -0.002], "equity": [1.01, 1.008]}, index=pd.date_range("2024-01-03", periods=2))
        return {"scenarios": {"all_costs": {"status": "complete", "ledger": ledger, "diagnostics": {"liquidation_reached": True, "liquidation_date": "2024-01-04", "final_quantities": {}}}}}
    monkeypatch.setattr("ML_factor_mining.scoring.run_backtest", fake_backtest)
    config = type("C", (), {"holding_days": 1, "fee_rate": 0.001, "slippage": 0.002, "validation_prediction_coverage": 0.5, "min_val_dates": 1, "min_pairs": 3})()
    result = score_validation(predictions, labels, membership, market, config, validation_start=dates[0], retrain_at=dates[-1] + pd.Timedelta(days=1))
    assert result["status"] == "complete"
    assert calls["values"].loc[calls["values"].index > calls["signal_end"]].isna().all().all()
    assert calls["profile"].n_groups == 5

