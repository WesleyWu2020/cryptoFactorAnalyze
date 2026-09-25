from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from Genetic_Algorithm.config import SearchConfig, STAGES, load_config
from Genetic_Algorithm.evaluator import evaluate_tree
from Genetic_Algorithm.evolution import (
    Candidate, _cost_key, _random_tree, _semantic_valid, _structured_mutation, search,
)
from Genetic_Algorithm.expression import Node, expression_dimension, expression_hash, node_count
from Genetic_Algorithm.research import (
    behavior_cell, behavior_evidence, dsr_report, return_moments, select_behavior,
)


def config(**kwargs):
    return SearchConfig(fitness_mode="all_costs_sharpe", **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"mutation_weights": [0.3, 0.3, 0.3]},
    {"mutation_weights": [0.3, 0.3, 0.3, float("nan")]},
    {"dimensionless_probability": float("nan")},
    {"behavior_cell_capacity": True},
    {"dsr_effective_trials": [True]},
    {"structured_mutation": 1},
    {"layered_elites": True},
    {"layered_elites": True, "training_parameter_stability": True, "exploration_fraction": 1.0},
])
def test_config_rejects_invalid_research_options(kwargs):
    with pytest.raises((ValueError, TypeError)):
        config(**kwargs)


def test_opt_in_config_preserves_validation_rules():
    old = load_config("Genetic_Algorithm/configs/daily_5groups_continuous_multiseed_parameter_stability.json")
    new = load_config("Genetic_Algorithm/configs/daily_5groups_behavior_elites.json")
    for name in old.__dataclass_fields__:
        if name.startswith("validation_") or name in {"frozen_limit", "n_groups", "parameter_perturbation"}:
            assert getattr(old, name) == getattr(new, name)
    assert new.structured_mutation and new.semantic_generation and new.layered_elites


def test_window_mutation_changes_only_one_window():
    original = Node("rolling_mean", (Node("return_1d"),), window=20)
    changed = _structured_mutation(original, "window", np.random.default_rng(1), config())
    assert changed.children == original.children and changed.op == original.op
    assert changed.window in (10, 40)


def test_field_mutation_preserves_family_and_dimension():
    original = Node("rolling_mean", (Node("volume_relative_20"),), window=20)
    changed = _structured_mutation(original, "field", np.random.default_rng(1), config())
    assert changed.op == original.op and changed.window == original.window
    assert changed.children == (Node("quote_volume_relative_20"),)
    assert expression_dimension(changed) == expression_dimension(original)


def test_pruning_reduces_complexity_without_changing_units():
    original = Node("rolling_mean", (Node("return_1d"),), window=20)
    changed = _structured_mutation(original, "prune", np.random.default_rng(1), config())
    assert changed == Node("return_1d")
    assert node_count(changed) < node_count(original)
    assert _structured_mutation(Node("return_1d"), "prune", np.random.default_rng(1), config()) == Node("return_1d")


def test_subtree_mutation_and_semantic_generation_are_legal_and_repeatable():
    c = config(semantic_generation=True, dimensionless_probability=1.0, family_diversity=True)
    def generate():
        rng = np.random.default_rng(9)
        return [_random_tree(rng, c) for _ in range(30)]
    trees = generate()
    assert trees == generate()
    assert len({expression_hash(t) for t in trees}) > 10
    for tree in trees:
        _semantic_valid(tree, c)
        assert expression_dimension(tree) == "ratio"
    changed = _structured_mutation(Node("return_1d"), "subtree", np.random.default_rng(4), c)
    assert expression_dimension(changed) == "ratio"
    _semantic_valid(changed, c)
    with pytest.raises(ValueError, match="mixed"):
        _semantic_valid(Node("safe_log", (Node("close"),)), c)


def test_behavior_archive_keeps_different_lower_scoring_niche():
    candidates = [Candidate(Node("return_1d"), name, (score, 1.0, -1))
                  for name, score in (("a", 10), ("b", 9), ("c", 2))]
    evidence = {"a": {"behavior": {"turnover": .01}}, "b": {"behavior": {"turnover": .02}},
                "c": {"behavior": {"turnover": .5}}}
    picked = select_behavior(candidates, 2, evidence, _cost_key, capacity=2)
    assert [c.expression_id for c in picked] == ["a", "c"]
    assert picked == select_behavior(reversed(candidates), 2, evidence, _cost_key, capacity=2)
    assert behavior_cell({}) != behavior_cell({"market_beta": 0.0})


def test_behavior_market_proxy_does_not_fill_missing_prices():
    dates = pd.date_range("2024-04-01", periods=20)
    market = pd.Series(np.sin(np.arange(20)) * .01, index=dates)
    opens = pd.DataFrame({"a": 100 * (1 + market).cumprod()}, index=dates)
    data = SimpleNamespace(opens=opens, eligible=opens.notna())
    evidence = behavior_evidence({"turnover": .1}, market * 2, data, [])
    assert evidence["market_beta"] == pytest.approx(2.0)
    assert evidence["reference_correlation"] is None
    opens.loc[dates[7], "a"] = np.nan
    assert opens.pct_change(fill_method=None).loc[dates[8]].isna().all()


def synthetic_search(monkeypatch, **overrides):
    from Genetic_Algorithm import cost_fitness
    diagnostics = {}
    data = SimpleNamespace(stage=STAGES["train"], opens=None, fitness_diagnostics=diagnostics)
    calls = []
    def evaluate(tree, *args):
        score = 10.0 if tree.children and tree.children[0].op == "return_1d" else 9.0
        diagnostics[expression_hash(tree)] = {
            "full_training_period": {"total_return": .2, "sharpe": 2.0},
            "behavior": {"turnover": .1},
            "return_moments": return_moments(np.sin(np.arange(100)) * .01 + .001),
        }
        return {"score": (score, 1.0, -node_count(tree)), "eligible": True, "direction": -1}
    def stability(tree, direction, data, cfg, baseline):
        assert direction == -1 and data.stage == STAGES["train"]
        calls.append(expression_hash(tree))
        return {"penalty_units": 8.0, "backtest_count": 2}
    monkeypatch.setattr(cost_fitness, "training_parameter_stability", stability)
    options = dict(
        population=4, generations=1, max_depth=2,
        initial_trees=tuple(Node("rolling_mean", (Node(name),), window=10)
                            for name in ("return_1d", "body_relative", "range_relative", "volume_relative_20")),
        evaluate_candidate=evaluate, layered_elites=True,
        training_parameter_stability=True, training_parameter_stability_top_k=1,
        exploration_fraction=.5,
    )
    options.update(overrides)
    return search(data, config(**options)), calls


def test_unchecked_high_score_cannot_displace_checked_elite(monkeypatch):
    result, calls = synthetic_search(monkeypatch)
    assert len(result.candidates) == 1
    assert result.candidates[0].score[0] == 2.0
    assert result.candidates[0].expression_id in calls
    assert result.generation_log[0]["elite_population"] == 1
    assert result.generation_log[0]["exploration_population"] == 3
    assert result.evaluations == 4


def test_layered_search_checks_once_reserves_budget_and_reports_cumulative_counts(monkeypatch):
    result, calls = synthetic_search(
        monkeypatch, generations=3, structured_mutation=True, semantic_generation=True,
        behavior_diversity=True, dsr_diagnostics=True,
    )
    assert len(calls) == len(set(calls)) == 3
    assert all(c.expression_id in calls for c in result.candidates)
    assert result.generation_log[-1]["cumulative_parameter_stability_candidates"] == 3
    assert result.generation_log[-1]["cumulative_parameter_stability_backtests"] == 6
    assert result.generation_log[-1]["cumulative_evaluations"] == result.evaluations
    assert all(row["random_exploration_proposals"] == 2 for row in result.generation_log[1:])
    assert result.research_diagnostics["dsr"]["selection_gate"] is False
    repeated, _ = synthetic_search(
        monkeypatch, generations=3, structured_mutation=True, semantic_generation=True,
        behavior_diversity=True, dsr_diagnostics=True,
    )
    assert result.candidates == repeated.candidates
    assert result.generation_log == repeated.generation_log


@pytest.mark.parametrize("stage", [STAGES["validation"], STAGES["test"]])
def test_research_search_rejects_nontraining_stages(stage):
    with pytest.raises(ValueError, match="2024"):
        search(SimpleNamespace(stage=stage), config(dsr_diagnostics=True))


def test_dsr_uses_daily_sharpe_and_effective_trial_scenarios_not_formula_count():
    moments = {"status": "complete", "n": 200, "daily_sharpe": .1, "skew": 0.0, "kurtosis": 3.0}
    diagnostics = {str(i): {"return_moments": dict(moments, daily_sharpe=.01 * i)} for i in range(20)}
    report = dsr_report(diagnostics, (1, 10, 100), 1000)
    expected = norm.cdf(.1 * np.sqrt(199 / (1 + .5 * .1 ** 2)))
    assert report["scenarios"][0]["conditional_dsr"]["10"] == pytest.approx(expected)
    assert report["scenarios"][1]["conditional_dsr"]["10"] < expected
    assert report["scenarios"][2]["status"] == "unavailable"
    assert report["attempted_evaluations"] == 1000 and report["usable_trials"] == 20
    assert report["diagnostic_only"] and not report["selection_gate"]
    assert return_moments(np.ones(30))["status"] == "unavailable"


def test_cli_persists_research_diagnostics_and_cumulative_progress(monkeypatch, tmp_path):
    import json
    from Genetic_Algorithm import cli
    from Genetic_Algorithm.artifacts import read_verified_manifest
    from Genetic_Algorithm.evolution import SearchResult
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"fitness_mode": "all_costs_sharpe", "dsr_diagnostics": True}))
    log = dict(generation=0, attempted_trees=4, unique_evaluations=4, eligible_population=2,
               cumulative_evaluations=4, cumulative_parameter_stability_candidates=2,
               cumulative_parameter_stability_backtests=8, elite_population=2, exploration_population=2)
    result = SearchResult((), (log,), 4, research_diagnostics={"dsr": {"diagnostic_only": True}})
    def run(*args, **kwargs):
        kwargs["search_stage"](None)
        return result
    def evolution(data, cfg, progress_callback):
        progress_callback(log)
        return result
    monkeypatch.setattr(cli, "run_search", run)
    monkeypatch.setattr(cli, "load_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "evolution_search", evolution)
    destination = tmp_path / "run"
    progress = tmp_path / "progress.json"
    assert cli.main(["search", "--config", str(config_path), "--h5", "unused",
                     "--run-dir", str(destination), "--progress", str(progress)]) == 0
    assert read_verified_manifest(destination / "search_research.json")["dsr"]["diagnostic_only"]
    assert json.loads(progress.read_text())["cumulative_parameter_stability_backtests"] == 8


def test_generated_formulas_preserve_cutoff_causality():
    from Genetic_Algorithm.features import RAW_FIELDS
    rng = np.random.default_rng(24)
    dates = pd.date_range("2023-01-01", periods=400)
    features = {name: pd.DataFrame(rng.uniform(10, 20, (400, 20)), index=dates) for name in RAW_FIELDS}
    eligible = pd.DataFrame(True, index=dates, columns=range(20))
    cutoff = dates[320]
    c = config(semantic_generation=True, max_depth=2, max_nodes=5)
    for _ in range(12):
        tree = _random_tree(rng, c)
        full = evaluate_tree(tree, features, eligible).loc[:cutoff]
        truncated = evaluate_tree(tree, {k: v.loc[:cutoff] for k, v in features.items()}, eligible.loc[:cutoff])
        np.testing.assert_allclose(full, truncated, rtol=1e-10, atol=1e-10, equal_nan=True)
