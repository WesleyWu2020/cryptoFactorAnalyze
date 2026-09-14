import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from ML_factor_mining.config import Config, quarter_folds
from ML_factor_mining.scoring import (
    CandidateRejected,
    MarketWindow,
    backtest_profile,
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


def test_backtest_profile_normalizes_date_like_anchor_to_calendar_date():
    config = SimpleNamespace(
        holding_days=3,
        anchor_date=pd.Timestamp("2024-01-01 12:34:56"),
        fee_rate=0.001,
        slippage=0.002,
    )

    profile = backtest_profile(config)

    assert profile.anchor_date == "2024-01-01"


def test_rank_candidates_requires_explicit_weights_as_a_pair():
    row = {"candidate_id": "a", "status": "valid", "mean_rank_ic": 0.1, "net_sharpe": 1.0}

    with pytest.raises(ValueError, match="together"):
        rank_candidates([row], ic_weight=0.7)


def test_score_validation_rejects_uncertified_ledger_boundaries(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    instruments = ["A", "B", "C"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["date", "instrument"])
    predictions = pd.Series(np.tile([1.0, 2.0, 3.0], len(dates)), index=index)
    labels = pd.DataFrame({"raw_return": np.tile([0.01, 0.02, 0.03], len(dates))}, index=index)
    membership = pd.DataFrame(True, index=dates, columns=instruments)
    opens = pd.DataFrame(100.0, index=dates, columns=instruments)
    quality = pd.DataFrame(index=pd.MultiIndex.from_product([dates, instruments], names=["date", "instrument"]))
    quality["has_placeholder_kline"] = False
    market = MarketWindow(
        opens=opens,
        events=pd.DataFrame(columns=["funding_time", "instrument", "funding_rate", "mark_price"]),
        quality=quality,
    )
    config = SimpleNamespace(
        holding_days=1,
        fee_rate=0.001,
        slippage=0.002,
        validation_prediction_coverage=0.5,
        min_val_dates=1,
        min_pairs=3,
        anchor_date="2024-01-01",
    )
    fold = SimpleNamespace(validation_start=dates[0], retrain_at=dates[-1] + pd.Timedelta(days=1))
    cases = [
        (pd.Index([0, 1]), "DatetimeIndex"),
        (pd.DatetimeIndex(["2024-01-03", "2024-01-05"]), "contiguous daily"),
        (pd.date_range("2024-01-03", periods=2), "last date"),
        (pd.date_range("2023-12-30", periods=2), "validation calendar"),
        (pd.date_range("2024-01-12", periods=3), "after retrain"),
    ]

    def fake_backtest(values, opens, events, quality, profile, *, signal_start, signal_end):
        ledger_index, reason = cases.pop(0)
        if reason == "contiguous daily":
            liquidation_date = "2024-01-05"
        elif reason == "last date":
            liquidation_date = "2024-01-05"
        elif reason == "validation calendar":
            liquidation_date = "2023-12-31"
        else:
            liquidation_date = "2024-01-14"
        ledger = pd.DataFrame({"return": np.zeros(len(ledger_index))}, index=ledger_index)
        return {
            "scenarios": {
                "all_costs": {
                    "status": "complete",
                    "ledger": ledger,
                    "diagnostics": {
                        "liquidation_reached": True,
                        "liquidation_date": liquidation_date,
                        "final_quantities": {},
                    },
                }
            }
        }

    monkeypatch.setattr("ML_factor_mining.scoring.run_backtest", fake_backtest)
    for _ledger_index, reason in cases.copy():
        with pytest.raises(CandidateRejected, match=reason):
            score_validation(predictions, labels, membership, market, fold, config)
