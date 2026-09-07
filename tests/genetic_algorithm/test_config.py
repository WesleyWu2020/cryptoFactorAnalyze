import json

import pandas as pd
import pytest

from Genetic_Algorithm.config import STAGES, Stage, SearchConfig, load_config


@pytest.mark.parametrize("name,last", [
    ("train", "2024-12-29"),
    ("validation", "2025-12-29"),
    ("test", "2026-08-30"),
])
def test_signal_end(name, last):
    assert STAGES[name].signal_end == pd.Timestamp(last)


def test_stages_are_immutable_and_have_inclusive_boundaries():
    assert STAGES["train"].start == pd.Timestamp("2024-01-01")
    assert STAGES["train"].end == pd.Timestamp("2024-12-31")
    assert STAGES["validation"].start == pd.Timestamp("2025-01-01")
    assert STAGES["validation"].end == pd.Timestamp("2025-12-31")
    assert STAGES["test"].start == pd.Timestamp("2026-01-01")
    assert STAGES["test"].end == pd.Timestamp("2026-09-01")
    with pytest.raises((AttributeError, TypeError)):
        STAGES["train"].name = "changed"


def test_default_search_config_values_are_frozen():
    config = SearchConfig()
    assert config.seed == 42
    assert config.population == 200
    assert config.generations == 20
    assert config.windows == (3, 5, 10, 20, 40, 60)
    assert config.lags == (1, 3, 5, 10)
    assert config.enable_funding_features is False
    with pytest.raises((AttributeError, TypeError)):
        config.population = 12


@pytest.mark.parametrize("field,value", [
    ("seed", True),
    ("population", True),
    ("generations", True),
    ("max_depth", True),
    ("max_nodes", True),
    ("max_history", True),
    ("min_pairs", True),
    ("min_quarter_days", True),
    ("min_overlap_days", True),
    ("validation_limit", True),
    ("frozen_limit", True),
    ("cache_bytes", True),
])
def test_integer_fields_reject_booleans(field, value):
    with pytest.raises(TypeError):
        SearchConfig(**{field: value})


@pytest.mark.parametrize("field", [
    "population", "generations", "max_depth", "max_nodes", "max_history",
    "min_pairs", "min_quarter_days", "min_overlap_days", "validation_limit",
    "frozen_limit", "cache_bytes",
])
def test_integer_budgets_reject_negative_values(field):
    with pytest.raises(ValueError):
        SearchConfig(**{field: -1})


def test_probabilities_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        SearchConfig(crossover_probability=0.6, mutation_probability=0.3, copy_probability=0.2)


def test_probabilities_allow_float_rounding_error():
    SearchConfig(crossover_probability=0.01, mutation_probability=0.29, copy_probability=0.70)


@pytest.mark.parametrize("field", [
    "crossover_probability", "mutation_probability", "copy_probability",
])
def test_probability_fields_reject_integers(field):
    values = {
        "crossover_probability": 0.0,
        "mutation_probability": 0.0,
        "copy_probability": 0.0,
    }
    values[field] = 1
    with pytest.raises(TypeError):
        SearchConfig(**values)


@pytest.mark.parametrize("field", [
    "min_day_coverage", "min_cell_coverage", "correlation_limit",
])
def test_coverage_fields_reject_integers(field):
    with pytest.raises(TypeError):
        SearchConfig(**{field: 1})


def test_unknown_keys_are_rejected():
    with pytest.raises(TypeError, match="Unknown configuration keys"):
        SearchConfig(unknown_key=1)


def test_stage_rejects_reversed_dates():
    with pytest.raises(ValueError, match="start must be on or before end"):
        Stage("broken", "2025-01-02", "2025-01-01")


def test_load_config_rejects_unsupported_funding_features(tmp_path):
    path = tmp_path / "funding.json"
    path.write_text(json.dumps({"enable_funding_features": True}))
    with pytest.raises(ValueError, match="Funding features are unsupported"):
        load_config(path)
