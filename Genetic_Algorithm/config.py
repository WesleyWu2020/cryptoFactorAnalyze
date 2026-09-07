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
    correlation_limit: float = 0.9
    min_overlap_days: int = 120
    validation_limit: int = 20
    frozen_limit: int = 5
    cache_bytes: int = 268435456
    enable_funding_features: bool = False
    max_attempts: int = 10000
    tournament_size: int = 2
    initial_trees: tuple[Node, ...] = ()
    evaluate_candidate: Any = None
    hold_days: int = 1

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
        integer_fields = {
            "seed", "population", "generations", "max_depth", "max_nodes",
            "max_history", "min_pairs", "min_quarter_days", "min_overlap_days",
            "validation_limit", "frozen_limit", "cache_bytes", "max_attempts",
            "tournament_size", "hold_days",
        }
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")

        positive_fields = {
            "population", "generations", "max_nodes", "max_history", "min_pairs",
            "min_quarter_days", "min_overlap_days", "validation_limit", "frozen_limit",
            "cache_bytes", "max_attempts", "tournament_size", "hold_days",
        }
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
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

        for name in ("min_day_coverage", "min_cell_coverage", "correlation_limit"):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a float")
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if type(self.enable_funding_features) is not bool:
            raise TypeError("enable_funding_features must be a boolean")
        if self.enable_funding_features:
            raise ValueError("Funding features are unsupported until their data contract is validated")


def load_config(path: str | Path) -> SearchConfig:
    """Load and validate a search configuration from a JSON object."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("configuration JSON must contain an object")
    return SearchConfig(**payload)
