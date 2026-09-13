from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from Genetic_Algorithm.config import SearchConfig, STAGES
from Genetic_Algorithm.expression import Node
from Genetic_Algorithm.features import FEATURE_FAMILIES, TERMINAL_FIELDS, expression_families
from Genetic_Algorithm.features import evaluate_terminal
from Genetic_Algorithm.evolution import Candidate, _random_tree, _select_cost_population
from Genetic_Algorithm.selection import select_validation, deduplicate_training
from Genetic_Algorithm import cost_fitness


def test_all_fields_remain_available_and_mixed_trees_count_every_family():
    assert set().union(*FEATURE_FAMILIES.values()) == TERMINAL_FIELDS
    mixed = Node("add", (Node("range_relative"), Node("taker_quote_ratio")))
    assert expression_families(mixed) == {"volatility", "buy_flow"}
    assert "volatility" in expression_families(Node("rolling_std", (Node("close"),), window=20))
    assert expression_families(Node("rolling_std", (Node("quote_per_trade"),), window=20)) == {"activity"}


def test_nonprice_derived_fields_do_not_read_undeclared_close():
    dates = pd.date_range("2024-01-01", periods=20)
    data = {name: pd.DataFrame(2.0, index=dates, columns=["A"])
            for name in ("volume", "quote_volume", "trade_count",
                         "taker_buy_base_volume", "taker_buy_quote_volume")}
    for field in ("volume_relative_20", "quote_volume_relative_20", "taker_base_ratio",
                  "taker_quote_ratio", "quote_per_trade"):
        result = evaluate_terminal(field, data)
        assert list(result.columns) == ["A"]


def test_random_generation_and_survival_preserve_nonvolatility_exploration():
    config = SearchConfig(fitness_mode="all_costs_sharpe", family_diversity=True)
    rng = np.random.default_rng(42)
    families = set().union(*(expression_families(_random_tree(rng, config)) for _ in range(100)))
    assert families == set(FEATURE_FAMILIES)
    candidates = [Candidate(Node("range_relative"), f"v{i}", (10., 5., -1)) for i in range(20)]
    candidates += [Candidate(Node(field), field, (0., 0., -1), eligible=False)
                   for field in ("return_1d", "quote_volume", "taker_quote_ratio")]
    selected = _select_cost_population(candidates, 8, config)
    assert len(selected) == 8
    assert {"return_1d", "quote_volume", "taker_quote_ratio"} <= {c.expression_id for c in selected}


def evidence(sharpe):
    return {"direction": 1, "day_coverage": 1., "cell_coverage": 1.,
            "quarter_valid_days": {f"Q{i}": 90 for i in range(1, 5)},
            "all_costs_cumulative_return": .1, "net_sharpe": sharpe}


def test_final_family_cap_and_stability_gate_allow_fewer_than_five():
    c = SearchConfig(fitness_mode="all_costs_sharpe", family_diversity=True,
                     family_candidate_limit=2, validation_stability=True)
    candidates = [{"expression_id": str(i), "tree": Node(field), "direction": 1}
                  for i, field in enumerate(["range_relative"] * 3 + ["taker_quote_ratio", "quote_volume"])]
    outcomes = {str(i): {**evidence(5-i/2), "cost_stability": {"passed": True}} for i in range(5)}
    outcomes["4"]["cost_stability"] = {"passed": False, "reasons": ["cost stress failed"]}
    result = select_validation(candidates, outcomes, c)
    assert [x["expression_id"] for x in result] == ["0", "1", "3"]
    assert "family limit" in result.rejection_reasons["2"][0]
    assert "cost stress" in result.rejection_reasons["4"][0]


def test_shortlist_does_not_fill_with_one_family_before_validation():
    config = SearchConfig(fitness_mode="all_costs_sharpe", family_diversity=True,
                          validation_limit=4, min_overlap_days=1)
    candidates = [Candidate(Node(field), str(i), (10.-i, 1., -1))
                  for i, field in enumerate(["range_relative"] * 3 + ["taker_quote_ratio", "quote_volume"])]
    # Independent signals bypass ordinary correlation dedup, exercising family cap.
    rng = np.random.default_rng(13)
    values = {c.expression_id: {"values": pd.DataFrame(rng.normal(size=(3, 30))),
                              "training_fingerprint": "t", "operator_version": "v"} for c in candidates}
    result = deduplicate_training(candidates, values, (), config)
    assert [x.expression_id for x in result] == ["0", "3", "4"]


@pytest.mark.parametrize("profits,stress,reason", [
    ([.1, .1, .1, -.02], .02, None),
    ([.1, .1, -.01, -.01], .02, "positive net"),
    ([.1, .1, .1, -.11], .02, "loss exceeds"),
    ([.8, .01, .01, -.01], .02, "concentrated"),
    ([.1, .1, .1, -.02], -.01, "cost stress"),
])
def test_validation_stability_uses_only_2025_and_fixed_stress(monkeypatch, profits, stress, reason):
    calls = []
    def evaluate(values, data, config, direction, start, end, **kwargs):
        assert pd.Timestamp(start).year == pd.Timestamp(end).year == 2025
        calls.append((start, end, kwargs))
        value = stress if kwargs else profits[len(calls)-1]
        return {"total_return": value, "sharpe": 1 if value > 0 else -1}, pd.Series(0., index=pd.date_range(start, end))
    monkeypatch.setattr(cost_fitness, "evaluate_cost_window", evaluate)
    data = SimpleNamespace(stage=STAGES["validation"], audit={"accounting_fingerprint": "v"})
    result = cost_fitness.validation_cost_stability(None, data, SearchConfig(), 1)
    assert result["passed"] == (reason is None)
    if reason:
        assert reason in " ".join(result["reasons"])
    assert len(calls) == 5
    assert calls[-1][2] == {"cost_multiplier": 1.5}
    data.stage = STAGES["test"]
    with pytest.raises(ValueError, match="2025"):
        cost_fitness.validation_cost_stability(None, data, SearchConfig(), 1)


def test_cost_pressure_changes_fees_not_funding_or_direction():
    base = cost_fitness.cost_profile(SearchConfig(), -1)
    stress = cost_fitness.cost_profile(SearchConfig(), -1, cost_multiplier=1.5)
    assert stress.fee_rate == pytest.approx(base.fee_rate * 1.5)
    assert stress.slippage == pytest.approx(base.slippage * 1.5)
    assert stress.include_funding == base.include_funding
    assert stress.factor_direction == base.factor_direction


def test_real_validation_ledger_stability_and_cost_stress(gp_h5):
    from Genetic_Algorithm.data import load_stage
    from Genetic_Algorithm.features import RAW_FIELDS
    from Genetic_Algorithm.evaluator import evaluate_tree
    config = SearchConfig(fitness_mode="all_costs_sharpe", n_groups=5, validation_stability=True)
    data = load_stage(gp_h5, STAGES["validation"], 21, sorted(RAW_FIELDS), include_accounting=True)
    values = evaluate_tree(Node("close"), data.features, data.eligible).loc[data.opens.index]
    result = cost_fitness.validation_cost_stability(values, data, config, 1)
    baseline, _ = cost_fitness.evaluate_cost_window(values, data, config, 1, data.stage.start, data.stage.end)
    assert set(result["quarters"]) == {"2025Q1", "2025Q2", "2025Q3", "2025Q4"}
    assert result["stressed_full_year"]["total_return"] < baseline["total_return"]
    assert result["accounting_fingerprint"] == data.audit["accounting_fingerprint"]


def test_family_evolution_is_deterministic_and_never_exports_ineligible():
    from Genetic_Algorithm.evolution import search
    from Genetic_Algorithm.expression import node_count
    def evaluate(tree, *args):
        return {"score": (1., .5, -node_count(tree)), "direction": 1,
                "eligible": "volatility" not in expression_families(tree)}
    config = SearchConfig(fitness_mode="all_costs_sharpe", family_diversity=True,
                          population=12, generations=2, max_attempts=300, evaluate_candidate=evaluate)
    first, second = search({}, config), search({}, config)
    assert first.candidates == second.candidates
    assert first.candidates
    assert all(c.eligible and "volatility" not in expression_families(c.tree) for c in first.candidates)
    assert first.generation_log[1]["family_population"]


@pytest.mark.parametrize("kwargs", [{"family_candidate_limit": 0}, {"validation_min_positive_quarters": 5},
                                   {"validation_cost_multiplier": float("nan")}, {"validation_max_profit_concentration": 0.}])
def test_invalid_research_thresholds_fail_before_data_access(kwargs):
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)
