import numpy as np
import pandas as pd
import pytest

from Genetic_Algorithm.config import SearchConfig, STAGES
from Genetic_Algorithm.evolution import search
from Genetic_Algorithm.fitness import score_training
from Genetic_Algorithm.expression import Node
from Genetic_Algorithm.evaluator import evaluate_tree
from factor_common.validation import check_cutoff


def panels():
    dates = pd.date_range("2024-01-01", "2024-12-31")
    rng = np.random.default_rng(13)
    values = pd.DataFrame(rng.normal(size=(len(dates), 30)), index=dates)
    return values, values.copy(), pd.DataFrame(True, index=dates, columns=values.columns)


def score(values, labels, quality, **kwargs):
    return score_training(values, labels, quality, {
        "fitness_mode": "robust_ic", "node_count": 3, **kwargs,
    })


def test_robust_objective_and_complexity_penalty():
    values, labels, quality = panels()
    result = score(values, labels, quality)
    assert result.eligible
    assert result.valid_days == 358
    assert result.objective_vector == pytest.approx((0.997, 1.0, -3))
    bigger = score(values, labels, quality, node_count=10)
    assert bigger.robust_ic == pytest.approx(result.robust_ic - 0.007)


def test_future_quarters_cannot_flip_calibrated_direction():
    values, labels, quality = panels()
    labels.loc[labels.index.quarter > 1] *= -1
    result = score(values, labels, quality)
    assert result.direction == 1
    assert not result.eligible
    assert result.robust_ic < 0


def test_quarter_boundary_future_labels_are_ignored():
    values, labels, quality = panels()
    baseline = score(values, labels, quality)
    exits = labels.index + pd.Timedelta(days=2)
    tail = exits > labels.index.to_period("Q").end_time.normalize()
    labels.loc[tail] *= -1e9
    assert score(values, labels, quality) == baseline


def test_unstable_quarters_are_penalized():
    values, labels, quality = panels()
    labels.loc[labels.index.quarter == 4] *= -1
    penalized = score(values, labels, quality)
    unpenalized = score(values, labels, quality, stability_penalty=0.0, worst_quarter_penalty=0.0)
    assert penalized.robust_ic < unpenalized.robust_ic
    assert not penalized.eligible


@pytest.mark.parametrize("name", ["stability_penalty", "worst_quarter_penalty", "complexity_penalty"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True])
def test_invalid_penalties_rejected(name, value):
    with pytest.raises(ValueError):
        SearchConfig(**{name: value})


def test_search_rejects_validation_as_training():
    with pytest.raises(ValueError, match="training stage"):
        search({"stage": STAGES["validation"]}, SearchConfig(
            population=1, generations=1, initial_trees=(Node("close"),),
        ))


def test_robust_search_key_path_on_bounded_training_panels():
    values, _, quality = panels()
    ranks = np.arange(30, dtype=float)
    values.loc[:, :] = np.broadcast_to(ranks, values.shape)
    opens = pd.DataFrame(
        100 * (1 + ranks * 0.0001) ** np.arange(len(values))[:, None],
        index=values.index, columns=values.columns,
    )
    result = search({
        "stage": STAGES["train"], "opens": opens,
        "features": {"close": values}, "eligible": quality,
        "quality_eligible": quality,
    }, SearchConfig(
        fitness_mode="robust_ic", population=1, generations=1,
        max_attempts=1, initial_trees=(Node("close"),),
    ))
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.direction == 1
    assert candidate.score == pytest.approx((0.999, 1.0, -1))


def test_rolling_expression_is_prefix_invariant():
    values, _, quality = panels()
    tree = Node("rolling_mean", (Node("close"),), window=5)
    def compute(cutoff):
        panel = values if cutoff is None else values.loc[:cutoff]
        return evaluate_tree(tree, {"close": panel}, quality.loc[panel.index])
    result = check_cutoff(compute, ["2024-03-31", "2024-06-30", "2024-09-30"])
    assert all(item["max_abs_diff"] == 0 for item in result["cutoffs"])
