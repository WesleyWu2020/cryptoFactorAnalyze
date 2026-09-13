from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from Genetic_Algorithm import incremental as inc
from Genetic_Algorithm.config import SearchConfig, STAGES
from Genetic_Algorithm.expression import Node, expression_hash
from Genetic_Algorithm.evaluator import evaluate_tree
from Genetic_Algorithm.selection import select_validation


def config(**kwargs):
    return SearchConfig(fitness_mode="all_costs_sharpe", **kwargs)


def test_reference_identity_and_causal_prefix():
    tree = inc.reference_tree()
    assert expression_hash(tree) == "064185107a8f267e818c0432cf3aa2e479ba8fbd31cda3079ff795365192e043"
    assert expression_hash(inc.second_reference_tree()) == "a37c60cf4492e9f1b2475353414b99723cf09e2f23a2c9fe79eecd7c765794b4"
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=400)
    close = pd.DataFrame(rng.uniform(10, 20, (400, 5)), index=dates)
    features = {"close": close, "high": close + 2, "low": close - 1}
    eligible = close.notna()
    for candidate in [tree, *inc.neighboring_trees(tree, .2)]:
        full = evaluate_tree(candidate, features, eligible).iloc[:300]
        prefix = evaluate_tree(candidate, {k: v.iloc[:300] for k, v in features.items()}, eligible.iloc[:300])
        pd.testing.assert_frame_equal(full, prefix)


def test_neighbors_change_one_window_and_preserve_structure():
    tree = inc.reference_tree()
    variants = inc.neighboring_trees(tree, .2)
    assert len(variants) == 8
    def windows(t):
        return [t.window] + sum((windows(c) for c in t.children), [])
    for variant in variants:
        assert sum(a != b for a, b in zip(windows(tree), windows(variant))) == 1
    assert inc.neighboring_trees(Node("close"), .2) == []


def test_reference_rejects_test_stage_before_reading_features():
    with pytest.raises(ValueError, match="train/validation"):
        inc.reference_evidence(SimpleNamespace(stage=STAGES["test"]), config(reference_factor="064185107a8f267e"))
    with pytest.raises(ValueError, match="2025"):
        inc.parameter_stability(Node("close"), 1, SimpleNamespace(stage=STAGES["test"]), config())


def test_incremental_fixed_capital_combination_and_duplicate_gate(monkeypatch):
    index = pd.date_range("2025-01-01", periods=150)
    base = pd.Series(np.tile([.02, -.01, .003], 50), index=index)
    candidate = pd.Series(np.tile([-.01, .02, .003], 50), index=index)
    positions = pd.DataFrame({"A": 1., "B": -1.}, index=index)
    other = pd.DataFrame({"C": 1., "D": -1.}, index=index)
    monkeypatch.setattr(inc, "reference_evidence", lambda *a: [{
        "reference_id": "064185107a8f267e", "metrics": {},
        "returns": base, "positions": positions}])
    c = config(reference_factor="064185107a8f267e")
    result = inc.incremental_evidence(candidate, other, None, c)
    assert result["passed"]
    expected = ((1 + base).prod() + (1 + candidate).prod()) / 2 - 1
    assert result["combination"]["total_return"] == pytest.approx(expected)
    assert not inc.incremental_evidence(base, positions, None, c)["passed"]
    with pytest.raises(ValueError, match="calendars"):
        inc.incremental_evidence(candidate.iloc[1:], other, None, c)


def test_dual_reference_uses_three_equal_initial_sleeves(monkeypatch):
    index = pd.date_range("2025-01-01", periods=150)
    first = pd.Series(np.tile([.02, -.01, .003], 50), index=index)
    second = pd.Series(np.tile([.003, .02, -.01], 50), index=index)
    candidate = pd.Series(np.tile([-.01, .003, .02], 50), index=index)
    frames = [pd.DataFrame({name: 1.}, index=index) for name in ("A", "B", "C")]
    monkeypatch.setattr(inc, "reference_evidence", lambda *a: [
        {"reference_id": "one", "metrics": {}, "returns": first, "positions": frames[0]},
        {"reference_id": "two", "metrics": {}, "returns": second, "positions": frames[1]},
    ])
    c = config(reference_factor="064185107a8f267e+a37c60cf4492e9f1")
    result = inc.incremental_evidence(candidate, frames[2], None, c)
    expected = sum((1 + item).prod() for item in (first, second, candidate)) / 3 - 1
    assert result["combination"]["total_return"] == pytest.approx(expected)
    assert len(result["comparisons"]) == 2


def test_parameter_stability_counts_failures_and_keeps_direction(monkeypatch):
    from Genetic_Algorithm import cost_fitness
    data = SimpleNamespace(stage=STAGES["validation"], features={}, eligible=None,
                           opens=pd.DataFrame(index=pd.date_range("2025-01-01", periods=365)))
    monkeypatch.setattr(inc, "evaluate_tree", lambda *a: data.opens)
    calls = []
    def evaluate(values, stage, cfg, direction, start, end):
        assert direction == -1 and start.year == end.year == 2025
        calls.append(1)
        return {"total_return": .1 if len(calls) <= 6 else -.1, "sharpe": 1.}, pd.Series(np.zeros(365))
    monkeypatch.setattr(cost_fitness, "evaluate_cost_window", evaluate)
    result = inc.parameter_stability(inc.reference_tree(), -1, data, config())
    assert result["passed"] and result["positive_fraction"] == .75
    assert not inc.parameter_stability(inc.reference_tree(), -1, data, config())["passed"]


def test_validation_missing_new_evidence_fails_closed():
    c = config(reference_factor="064185107a8f267e", validation_parameter_stability=True)
    candidate = {"expression_id": "x", "tree": Node("close"), "direction": 1}
    evidence = {"direction": 1, "day_coverage": 1., "cell_coverage": 1.,
                "quarter_valid_days": {f"Q{i}": 90 for i in range(1, 5)},
                "all_costs_cumulative_return": .2, "net_sharpe": 2.}
    result = select_validation([candidate], {"x": evidence}, c)
    assert not result.accepted
    assert any("incremental" in r for r in result.rejection_reasons["x"])
    assert any("parameter_stability" in r for r in result.rejection_reasons["x"])


@pytest.mark.parametrize("kwargs", [{"reference_factor": "unknown"},
    {"parameter_perturbation": float("nan")}, {"parameter_min_positive_fraction": 0.},
    {"reference_similarity_penalty": -1.}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_real_accounting_training_penalty_and_validation_neighbors(gp_h5):
    from Genetic_Algorithm.data import load_stage
    from Genetic_Algorithm.features import RAW_FIELDS
    from Genetic_Algorithm.cost_fitness import score_all_costs, evaluate_cost_window
    from factor_common.labels import make_labels
    c = config(reference_factor="064185107a8f267e", n_groups=5)
    train = load_stage(gp_h5, STAGES["train"], 180, sorted(RAW_FIELDS), include_accounting=True)
    values = evaluate_tree(Node("close"), train.features, train.eligible).loc[train.opens.index]
    outcome = score_all_costs(values, make_labels(train.opens, 1), train, c, 1)
    assert "reference_comparison" in outcome[-1]
    assert outcome[-1]["reference_comparison"]["penalty"] >= 0
    assert len(train.reference_evidence) == 1
    assert inc.reference_evidence(train, c) is inc.reference_evidence(train, c)
    validation = load_stage(gp_h5, STAGES["validation"], 180, sorted(RAW_FIELDS), include_accounting=True)
    tree = Node("rolling_mean", (Node("close"),), window=10)
    values = evaluate_tree(tree, validation.features, validation.eligible).loc[validation.opens.index]
    _, returns, positions = evaluate_cost_window(values, validation, c, 1,
        validation.stage.start, validation.stage.end, include_positions=True)
    evidence = inc.incremental_evidence(returns, positions, validation, c)
    assert isinstance(evidence["passed"], bool)
    stability = inc.parameter_stability(tree, 1, validation, c)
    assert len(stability["variants"]) == 2
    assert all("metrics" in record for record in stability["variants"])
