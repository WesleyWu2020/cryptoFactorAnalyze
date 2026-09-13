"""Frozen stages and validated configuration for daily cross-sectional GP."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from .expression import Node


@dataclass(frozen=True)
class Stage:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        start = pd.Timestamp(self.start)
        end = pd.Timestamp(self.end)
        if start > end:
            raise ValueError("stage start must be on or before end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def signal_end(self) -> pd.Timestamp:
        return self.end - pd.Timedelta(days=2)


STAGES = MappingProxyType({
    "train": Stage("train", "2024-01-01", "2024-12-31"),
    "validation": Stage("validation", "2025-01-01", "2025-12-31"),
    "test": Stage("test", "2026-01-01", "2026-09-01"),
})


@dataclass(frozen=True, init=False)
class SearchConfig:
    seed: int = 42
    population: int = 200
    generations: int = 20
    max_depth: int = 4
    max_nodes: int = 15
    max_history: int = 180
    windows: tuple[int, ...] = (3, 5, 10, 20, 40, 60)
    lags: tuple[int, ...] = (1, 3, 5, 10)
    crossover_probability: float = 0.6
    mutation_probability: float = 0.3
    copy_probability: float = 0.1
    min_pairs: int = 20
    min_quarter_days: int = 45
    min_day_coverage: float = 0.8
    min_cell_coverage: float = 0.8
    min_all_costs_sharpe: float = 1.0
    correlation_limit: float = 0.9
    return_correlation_limit: float = 0.8
    position_overlap_limit: float = 0.7
    min_overlap_days: int = 120
    validation_limit: int = 20
    frozen_limit: int = 5
    n_groups: int = 10
    render_reports: bool = True
    cache_bytes: int = 268435456
    enable_funding_features: bool = False
    max_attempts: int = 10000
    tournament_size: int = 2
    initial_trees: tuple[Node, ...] = ()
    evaluate_candidate: Any = None
    hold_days: int = 1
    fitness_mode: str = "legacy_ic"
    stability_penalty: float = 0.5
    worst_quarter_penalty: float = 1.0
    complexity_penalty: float = 0.001
    drawdown_penalty: float = 1.0
    family_diversity: bool = False
    family_candidate_limit: int = 2
    validation_stability: bool = False
    validation_min_positive_quarters: int = 3
    validation_max_quarter_loss: float = 0.10
    validation_max_profit_concentration: float = 0.60
    validation_cost_multiplier: float = 1.5
    reference_factor: str | None = None
    reference_similarity_penalty: float = 1.0
    validation_parameter_stability: bool = False
    parameter_perturbation: float = 0.20
    parameter_min_positive_fraction: float = 0.75
    validation_min_incremental_sharpe: float = 0.0

    def __init__(self, **overrides: Any) -> None:
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown configuration keys: {names}")
        for name, field in self.__dataclass_fields__.items():
            value = overrides.get(name, field.default)
            if name in {"windows", "lags", "initial_trees"}:
                if not isinstance(value, (list, tuple)):
                    raise TypeError(f"{name} must be a list or tuple")
                value = tuple(value)
            object.__setattr__(self, name, value)
        self._validate()

    def _validate(self) -> None:
        if self.reference_factor not in (None, "064185107a8f267e",
                                         "064185107a8f267e+a37c60cf4492e9f1"):
            raise ValueError("unsupported reference_factor")
        required_reference_history = (166 if self.reference_factor and "a37c60cf4492e9f1" in self.reference_factor else 116)
        if self.reference_factor and (type(self.max_history) is not int or self.max_history < required_reference_history):
            raise ValueError(f"reference_factor requires max_history >= {required_reference_history}")
        if type(self.validation_parameter_stability) is not bool:
            raise TypeError("validation_parameter_stability must be boolean")
        if (self.reference_factor or self.validation_parameter_stability) and self.fitness_mode != "all_costs_sharpe":
            raise ValueError("incremental research requires all_costs_sharpe")
        for name in ("reference_similarity_penalty", "validation_min_incremental_sharpe"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("parameter_perturbation", "parameter_min_positive_fraction"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be in (0, 1)")
        for name in ("family_diversity", "validation_stability"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        if type(self.family_candidate_limit) is not int or self.family_candidate_limit < 1:
            raise ValueError("family_candidate_limit must be a positive integer")
        if type(self.validation_min_positive_quarters) is not int or not 1 <= self.validation_min_positive_quarters <= 4:
            raise ValueError("validation_min_positive_quarters must be between 1 and 4")
        for name in ("validation_max_quarter_loss", "validation_max_profit_concentration"):
            v = getattr(self, name)
            if type(v) is not float or not math.isfinite(v) or not 0 < v <= 1:
                raise ValueError(f"{name} must be a finite float in (0, 1]")
        if type(self.validation_cost_multiplier) is not float or not math.isfinite(self.validation_cost_multiplier) or self.validation_cost_multiplier < 1:
            raise ValueError("validation_cost_multiplier must be a finite float >= 1")
        if (self.family_diversity or self.validation_stability) and self.fitness_mode != "all_costs_sharpe":
            raise ValueError("family diversity and validation stability require all_costs_sharpe")
        if self.fitness_mode not in {"legacy_ic", "robust_ic", "all_costs_sharpe"}:
            raise ValueError("fitness_mode must be legacy_ic, robust_ic or all_costs_sharpe")
        if self.fitness_mode == "all_costs_sharpe" and self.hold_days != 1:
            raise ValueError("all_costs_sharpe currently requires hold_days=1 to match replay")
        for name in ("stability_penalty", "worst_quarter_penalty", "complexity_penalty", "drawdown_penalty"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative float")
        integer_fields = {
            "seed", "population", "generations", "max_depth", "max_nodes",
            "max_history", "min_pairs", "min_quarter_days", "min_overlap_days",
            "validation_limit", "frozen_limit", "n_groups", "cache_bytes", "max_attempts",
            "tournament_size", "hold_days",
        }
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")

        positive_fields = {
            "population", "generations", "max_nodes", "max_history", "min_pairs",
            "min_quarter_days", "min_overlap_days", "validation_limit", "frozen_limit", "n_groups",
            "cache_bytes", "max_attempts", "tournament_size", "hold_days",
        }
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        if self.seed < 0 or self.max_depth < 0:
            raise ValueError("seed and max_depth must be non-negative")

        for name in ("windows", "lags"):
            values = getattr(self, name)
            if not values or any(type(value) is not int or value <= 0 for value in values):
                raise ValueError(f"{name} must contain positive integers")
        if any(not isinstance(tree, Node) for tree in self.initial_trees):
            raise TypeError("initial_trees must contain Node instances")
        if self.evaluate_candidate is not None and not callable(self.evaluate_candidate):
            raise TypeError("evaluate_candidate must be callable or None")

        probability_fields = (
            "crossover_probability", "mutation_probability", "copy_probability",
        )
        for name in probability_fields:
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a float")
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        probability_sum = sum(getattr(self, name) for name in probability_fields)
        if not math.isclose(probability_sum, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("probabilities must sum to 1")

        for name in ("min_day_coverage", "min_cell_coverage", "correlation_limit", "return_correlation_limit", "position_overlap_limit"):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a float")
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if type(self.min_all_costs_sharpe) is not float:
            raise TypeError("min_all_costs_sharpe must be a float")
        if not math.isfinite(self.min_all_costs_sharpe):
            raise ValueError("min_all_costs_sharpe must be finite")
        if type(self.enable_funding_features) is not bool:
            raise TypeError("enable_funding_features must be a boolean")
        if type(self.render_reports) is not bool:
            raise TypeError("render_reports must be a boolean")
        if self.enable_funding_features:
            raise ValueError("Funding features are unsupported until their data contract is validated")


def load_config(path: str | Path) -> SearchConfig:
    """Load and validate a search configuration from a JSON object."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("configuration JSON must contain an object")
    return SearchConfig(**payload)
