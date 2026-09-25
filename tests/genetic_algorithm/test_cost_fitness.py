from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from Genetic_Algorithm.config import SearchConfig, Stage, STAGES, load_config
from Genetic_Algorithm.cost_fitness import (
    aggregate_cost_score, evaluate_cost_window, score_all_costs,
    summarize_net_stability, training_parameter_stability,
)
from Genetic_Algorithm.data import load_stage
from Genetic_Algorithm.evaluator import evaluate_tree
from Genetic_Algorithm.evolution import search, Candidate, _cost_key
from Genetic_Algorithm.expression import Node
from Genetic_Algorithm.features import RAW_FIELDS
from Genetic_Algorithm.replay import replay
from factor_common.labels import make_labels


def config(**kwargs):
    return SearchConfig(fitness_mode="all_costs_sharpe", n_groups=5, render_reports=False, **kwargs)


def metric(sharpe=2.0, total_return=0.2, max_drawdown=0.05):
    return dict(sharpe=sharpe, total_return=total_return, max_drawdown=max_drawdown)


def test_sharpe_is_primary_and_unstable_or_losing_scores_are_rejected():
    c = config()
    stable, reasons = aggregate_cost_score([metric()] * 3, metric(), c, 3)
    weak, _ = aggregate_cost_score([metric(1.2)] * 3, metric(1.2), c, 3)
    unstable, _ = aggregate_cost_score([metric(0), metric(2), metric(4)], metric(), c, 3)
    assert not reasons
    assert stable[0] > weak[0]
    assert stable[0] > unstable[0]
    assert aggregate_cost_score([metric()] * 3, metric(total_return=-0.1), c, 3)[1]
    assert aggregate_cost_score([metric()] * 3, metric(sharpe=1.0), c, 3)[1]
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_cost_score([metric(float("nan"))] * 3, metric(), c, 3)
    assert _cost_key(Candidate(Node("close"), "a", stable)) < _cost_key(Candidate(Node("close"), "b", weak))


def test_continuous_stability_reconciles_profit_and_removes_best_quarter():
    index = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    returns = pd.Series(np.tile([0.0004, 0.0, 0.0002], 122)[:len(index)], index=index)
    returns.loc[returns.index.quarter == 4] += 0.002
    result = summarize_net_stability(returns, min_quarter_days=45)

    assert result["best_quarter"] == "2025Q4"
    assert result["positive_quarters"] == 4
    assert abs(result["profit_contribution_reconciliation"]) < 1e-12
    assert result["leave_best_quarter_out_days"] == 273
    assert result["leave_best_quarter_out"]["total_return"] > 0


def test_continuous_score_penalizes_excess_concentration():
    c = config(stability_mode="continuous_leave_best_out")
    stable = {
        "positive_quarter_profit_concentration": 0.40,
        "leave_best_quarter_out": {"sharpe": 1.0},
    }
    concentrated = {
        "positive_quarter_profit_concentration": 0.90,
        "leave_best_quarter_out": {"sharpe": 1.0},
    }
    stable_score, _ = aggregate_cost_score([metric()] * 3, metric(), c, 3, stability=stable)
    concentrated_score, _ = aggregate_cost_score(
        [metric()] * 3, metric(), c, 3, stability=concentrated,
    )
    assert stable_score[0] > concentrated_score[0]


def test_training_parameter_stability_uses_2024_and_fixed_direction(monkeypatch):
    from types import SimpleNamespace
    from Genetic_Algorithm import cost_fitness

    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    data = SimpleNamespace(
        stage=STAGES["train"], features={}, eligible=None,
        opens=pd.DataFrame(index=dates),
    )
    monkeypatch.setattr(cost_fitness, "evaluate_tree", None, raising=False)
    from Genetic_Algorithm import evaluator
    monkeypatch.setattr(
        evaluator, "evaluate_tree", lambda *args, **kwargs: pd.DataFrame({"x": 1.0}, index=dates),
    )
    calls = []
    metrics = [
        {"total_return": 0.10, "sharpe": 1.0},
        {"total_return": -0.10, "sharpe": -1.0},
    ]
    def evaluate(values, stage, cfg, direction, start, end):
        assert direction == -1
        assert start == STAGES["train"].start and end == STAGES["train"].end
        item = metrics[len(calls)]
        calls.append(item)
        return item, pd.Series(0.0, index=dates)
    monkeypatch.setattr(cost_fitness, "evaluate_cost_window", evaluate)

    result = training_parameter_stability(
        Node("rolling_mean", (Node("close"),), window=10), -1, data,
        config(parameter_perturbation=0.20, parameter_min_positive_fraction=0.75),
        {"total_return": 0.20, "sharpe": 2.0},
    )

    assert result["variant_count"] == 2
    assert result["backtest_count"] == 2
    assert result["positive_fraction"] == 0.5
    assert result["return_degradation"] == pytest.approx(0.20)
    assert result["sharpe_degradation"] == pytest.approx(2.0)
    assert result["positive_fraction_shortfall"] == pytest.approx(0.25)
    assert result["penalty_units"] == pytest.approx(2.45)


def test_training_parameter_stability_checks_only_top_k_without_changing_formula_budget(monkeypatch):
    from types import SimpleNamespace
    from Genetic_Algorithm import cost_fitness
    from Genetic_Algorithm.expression import expression_hash

    trees = (
        Node("rolling_mean", (Node("close"),), window=10),
        Node("rolling_mean", (Node("open"),), window=10),
    )
    diagnostics = {
        expression_hash(tree): {"full_training_period": {"total_return": .2, "sharpe": 2.0}}
        for tree in trees
    }
    stage = SimpleNamespace(stage=None, opens=None, fitness_diagnostics=diagnostics)
    calls = []
    def evaluate(tree, data, labels, cfg):
        return {"score": ((2.0 if tree.children[0].op == "close" else 1.0), .5, -1),
                "eligible": True, "direction": 1}
    def stability(tree, direction, data, cfg, baseline):
        calls.append(tree.children[0].op)
        return {"penalty_units": .4, "backtest_count": 2}
    monkeypatch.setattr(cost_fitness, "training_parameter_stability", stability)

    result = search(stage, config(
        population=2, generations=1, max_attempts=2, max_depth=1, max_nodes=2,
        initial_trees=trees, evaluate_candidate=evaluate,
        training_parameter_stability=True,
        training_parameter_stability_top_k=1,
        training_parameter_stability_penalty=1.0,
    ))

    assert result.evaluations == 2
    assert calls == ["close"]
    assert result.generation_log[0]["parameter_stability_candidates"] == 1
    assert result.generation_log[0]["parameter_stability_backtests"] == 2
    evidence = diagnostics[expression_hash(trees[0])]["training_parameter_stability"]
    assert evidence["base_score"][0] == 2.0
    assert evidence["adjusted_score"][0] == pytest.approx(1.6)


def test_cost_mode_population_can_evolve_without_filling_profitable_quota():
    counter = []
    def evaluate(tree, stage, labels, c):
        counter.append(tree)
        return {"score": (0.3, 0.2, -1), "eligible": tree.op == "close", "direction": 1}
    result = search({}, config(population=2, generations=2, max_attempts=10,
                              max_depth=0, initial_trees=(Node("close"), Node("open")), evaluate_candidate=evaluate))
    assert result.generation_log[0]["attempted_trees"] == 2
    assert all(c.eligible for c in result.candidates)


def test_search_reports_each_completed_generation():
    records = []
    def evaluate(tree, stage, labels, c):
        return {"score": (0.3, 0.2, -1), "eligible": True, "direction": 1}
    search(
        {}, config(population=2, generations=2, max_attempts=10, max_depth=0,
                   initial_trees=(Node("close"), Node("open")), evaluate_candidate=evaluate),
        progress_callback=records.append,
    )
    assert [record["generation"] for record in records] == [0, 1]


def test_cost_deduplication_prioritizes_primary_sharpe_objective():
    from Genetic_Algorithm.selection import deduplicate_training
    rng = np.random.default_rng(8)
    first = Candidate(Node("close"), "first", (2.0, 0.1, -1), direction=1)
    second = Candidate(Node("open"), "second", (1.0, 0.5, -1), direction=1)
    panels = {c.expression_id: {
        "values": pd.DataFrame(rng.normal(size=(150, 30)), index=pd.date_range("2024-01-01", periods=150)),
        "training_fingerprint": "same_fold", "operator_version": "same_code",
    } for c in (first, second)}
    selection = deduplicate_training((second, first), panels, (), config())
    assert [c.expression_id for c in selection] == ["first", "second"]


def test_accounting_matches_replay_and_fails_closed(gp_h5, tmp_path):
    c = config()
    stage = Stage("parity", "2024-04-01", "2024-06-30")
    data = load_stage(gp_h5, stage, 21, sorted(RAW_FIELDS), include_accounting=True)
    tree = Node("close")
    values = evaluate_tree(tree, data.features, data.eligible).loc[data.opens.index]
    metrics, returns = evaluate_cost_window(values, data, c, 1, stage.start, stage.end)
    outcome = replay({"tree": tree}, stage, gp_h5, tmp_path / "replay", direction=1, n_groups=5, render_reports=False)
    for name in ("sharpe", "total_return", "max_drawdown", "turnover"):
        assert metrics[name] == pytest.approx(outcome["metrics"][name], abs=1e-12)
    events = data.funding_events.copy()
    events.loc[events.instrument.isin([f"S{i:02d}USDT" for i in range(24, 29)]), "funding_rate"] = 0.001
    expensive, _ = evaluate_cost_window(values, replace(data, funding_events=events), c, 1, stage.start, stage.end)
    assert expensive["total_return"] < metrics["total_return"]
    assert expensive["sharpe"] < metrics["sharpe"]
    assert returns.index.max() <= stage.end
    with pytest.raises(ValueError, match="bounded funding"):
        evaluate_cost_window(values, replace(data, funding_events=None), c, 1, stage.start, stage.end)
    with pytest.raises(ValueError, match="bounded stage"):
        evaluate_cost_window(values, data, c, 1, stage.start, stage.end + pd.Timedelta(days=1))
    quality = data.accounting_quality.copy()
    quality.loc[:, "funding_coverage_status"] = "unknown"
    with pytest.raises(ValueError, match="uncertified"):
        evaluate_cost_window(values, replace(data, accounting_quality=quality), c, 1, stage.start, stage.end)


def test_real_training_cost_path_and_boundary_purge(gp_h5, tmp_path):
    c = config(population=1, generations=1, max_attempts=1, initial_trees=(Node("close"),))
    data = load_stage(gp_h5, STAGES["train"], 21, sorted(RAW_FIELDS), include_accounting=True)
    values = evaluate_tree(Node("close"), data.features, data.eligible).loc[data.opens.index]
    labels = make_labels(data.opens, 1)
    baseline = score_all_costs(values, labels, data, c, 1)
    assert "segments" in baseline[-1]
    altered = labels.copy()
    tail = labels.index + pd.Timedelta(days=2) > labels.index.to_period("Q").end_time.normalize()
    altered.loc[tail] = -1e9
    changed = score_all_costs(values, altered, data, c, 1)
    assert baseline == changed
    result = search(data, c)
    assert len(result.fitness_diagnostics) == 1
    assert all("all_costs" in d["objective"] for d in result.fitness_diagnostics.values())
    from Genetic_Algorithm.walk_forward import run_walk_forward
    from Genetic_Algorithm.artifacts import read_verified_manifest
    outcome = run_walk_forward(gp_h5, tmp_path / "walk", c,
                               first_test_start="2025-01-01", last_test_end="2025-03-31")
    summary = read_verified_manifest(outcome["artifact"])
    assert summary["status"] == "complete"
    assert summary["fold_count"] == 1
    assert (tmp_path / "walk" / "fold_01" / "frozen.json").exists()
