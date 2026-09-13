from types import SimpleNamespace

import pandas as pd
import pytest

from Genetic_Algorithm.artifacts import read_verified_manifest
from Genetic_Algorithm.config import SearchConfig
from Genetic_Algorithm.evolution import Candidate, SearchResult
from Genetic_Algorithm.expression import Node
from Genetic_Algorithm import walk_forward as wf


def test_schedule_is_chronological_disjoint_and_does_not_access_2026():
    folds = wf.fold_schedule("2025-01-01", "2025-12-31")
    assert len(folds) == 4
    assert folds[0][0].start == pd.Timestamp("2023-10-01")
    assert folds[-1][0].end == pd.Timestamp("2025-06-30")
    for train, validation, test in folds:
        assert train.end + pd.Timedelta(days=1) == validation.start
        assert validation.end + pd.Timedelta(days=1) == test.start
    with pytest.raises(ValueError, match="before 2026"):
        wf.fold_schedule("2025-01-01", "2026-03-31")
    holdout = wf.fold_schedule("2026-01-01", "2026-06-30", allow_2026_test=True)
    assert [(train.start, train.end, test.start, test.end) for train, validation, test in holdout] == [
        (pd.Timestamp("2024-10-01"), pd.Timestamp("2025-09-30"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")),
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-30")),
    ]
    with pytest.raises(ValueError, match="only complete"):
        wf.fold_schedule("2026-01-01", "2026-09-30", allow_2026_test=True)
    with pytest.raises(ValueError, match="complete calendar quarters"):
        wf.fold_schedule("2025-01-02", "2025-12-31")


@pytest.mark.parametrize("mode", ["normal", "cash", "failure", "validation_rejected", "duplicate_trading"])
def test_retrains_and_commits_before_every_test_load(monkeypatch, tmp_path, mode):
    sequence = []
    folder = tmp_path / mode
    c = Candidate(Node("close"), "selected_from_train", (2.0, 1.5, -1), direction=1)
    def load(path, stage, warmup, fields, **kwargs):
        sequence.append(stage.name)
        if stage.name == "walk_forward_test":
            fold = len([s for s in sequence if s == "walk_forward_test"])
            committed = read_verified_manifest(folder / f"fold_{fold:02d}" / "frozen.json")
            assert committed["train"]["end"] < committed["test"]["start"]
            assert len(committed["candidates"]) == (1 if mode in {"normal", "duplicate_trading"} else 0)
        if stage.name == "walk_forward_validation":
            fold = sequence.count("walk_forward_validation")
            frozen = read_verified_manifest(folder / f"fold_{fold:02d}" / "shortlist.json")
            assert frozen["train"]["end"] < frozen["validation"]["start"]
        dates = pd.date_range(stage.start, stage.end)
        panel = pd.DataFrame(1.0, index=dates, columns=["A"])
        return SimpleNamespace(stage=stage, opens=panel, features={"close": panel}, eligible=panel.astype(bool),
                               fingerprint="bounded", audit={"accounting_fingerprint": "funding"})
    def search(data, config):
        sequence.append("search")
        assert data.stage.name == "walk_forward_train"
        candidates = (c, Candidate(Node("close"), "duplicate", (1.5, 1.0, -1), direction=1)) if mode == "duplicate_trading" else (c,)
        return SearchResult(() if mode == "cash" else candidates, (), 1)
    def cost(values, data, config, direction, start, end, **kwargs):
        if mode == "failure":
            raise ValueError("uncertified funding")
        r = pd.Series([0.001 if i % 2 else -0.0002 for i in range(len(values))], index=values.index)
        if kwargs.get("include_positions"):
            return wf.summarize_returns(r), r, values
        metrics = wf.summarize_returns(r)
        if mode == "validation_rejected" and data.stage.name == "walk_forward_validation":
            metrics["sharpe"] = 1.0
        return metrics, r
    monkeypatch.setattr(wf, "load_stage", load)
    monkeypatch.setattr(wf, "search", search)
    monkeypatch.setattr(wf, "deduplicate_training", lambda candidates, *args: candidates)
    monkeypatch.setattr(wf, "evaluate_cost_window", cost)
    result = wf.run_walk_forward("unused", folder, SearchConfig(fitness_mode="all_costs_sharpe"),
                                 first_test_start="2025-01-01", last_test_end="2025-06-30")
    assert sequence == ["walk_forward_train", "search", "walk_forward_validation", "walk_forward_test"] * 2
    summary = read_verified_manifest(result["artifact"])
    assert summary["fold_count"] == 2
    if mode == "duplicate_trading":
        evidence = read_verified_manifest(folder / "fold_01" / "validation.json")
        assert evidence["candidates"][1]["reason"] == "training trading similarity"
        assert evidence["candidates"][1]["comparisons"][0]["position_overlap"] == 1
    if mode in {"cash", "validation_rejected"}:
        assert summary["cash_fold_count"] == 2
        assert summary["all_costs_portfolio"]["total_return"] == 0
        assert not summary["passed"]
    if mode == "failure":
        assert summary["failed_fold_count"] == 2
        assert summary["all_costs_portfolio"] is None
        assert not summary["passed"]
    with pytest.raises(FileExistsError):
        wf.run_walk_forward("unused", folder, SearchConfig(fitness_mode="all_costs_sharpe"))


def test_trading_similarity_checks_returns_and_signed_holdings():
    from Genetic_Algorithm.selection import trading_similarity
    dates = pd.date_range("2024-01-01", periods=130)
    r = pd.Series([0.01, -0.02] * 65, index=dates)
    p = pd.DataFrame({"A": 1.0, "B": -2.0}, index=dates)
    c = SearchConfig()
    # Identical exposure despite different quantities, even with opposite PnL.
    result = trading_similarity(r, -r, p, p * 100, c)
    assert result["position_overlap"] == 1
    assert result["too_similar"]
    result = trading_similarity(r, r, p, -p, c)
    assert result["position_overlap"] == 0
    assert result["too_similar"]
    assert not trading_similarity(r, -r, p, -p, c)["too_similar"]
    with pytest.raises(ValueError, match="insufficient"):
        trading_similarity(r.iloc[:10], r.iloc[:10], p, p, c)
