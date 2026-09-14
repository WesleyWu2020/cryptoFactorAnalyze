"""Configuration and quarter scheduling for ML factor research."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import math
from numbers import Integral, Real
from collections.abc import Mapping
from typing import Any

import pandas as pd


MODELS = ("lightgbm", "xgboost", "ridge", "linear")
_TARGET_TYPES = ("rank_return", "raw_return", "excess_return")
_POSITIVE_INT_FIELDS = (
    "rolling_months",
    "holding_days",
    "min_fit_dates",
    "min_val_dates",
    "min_pairs",
    "max_rounds",
    "patience",
    "seed",
    "threads",
)


def _as_calendar_date(value: Any, name: str) -> date:
    """Convert a date-like value while rejecting timezone and time-of-day data."""
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError(f"{name} must be timezone-naive")
        if value.time() != datetime.min.time():
            raise ValueError(f"{name} must be normalized to midnight")
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO calendar date") from exc
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            raise ValueError(f"{name} must be timezone-naive")
        if parsed.time() != datetime.min.time():
            raise ValueError(f"{name} must be normalized to midnight")
        return parsed.date()
    # pandas.Timestamp and compatible date-like objects are common at call sites.
    if isinstance(value, pd.Timestamp):
        if value.tz is not None:
            raise ValueError(f"{name} must be timezone-naive")
        if value != value.normalize():
            raise ValueError(f"{name} must be normalized to midnight")
        return value.date()
    raise TypeError(f"{name} must be a date or ISO calendar date string")


def _as_tuple(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, not a string")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc


@dataclass(frozen=True)
class Config:
    train_start: date | str = "2024-01-01"
    oos_start: date | str = "2025-01-01"
    end: date | str = "2026-12-31"
    mode: str = "expanding"
    rolling_months: int = 12
    holding_days: int = 3
    target_type: str = "rank_return"
    models: tuple[str, ...] = ("lightgbm",)
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    coverage: float = 0.9
    min_fit_dates: int = 120
    min_val_dates: int = 45
    min_pairs: int = 20
    validation_prediction_coverage: float = 0.9
    ic_weight: float = 0.5
    sharpe_weight: float = 0.5
    max_rounds: int = 1000
    patience: int = 50
    seed: int = 42
    threads: int = 1
    anchor_date: date | str = "2024-01-01"
    fee_rate: float = 0.0005
    slippage: float = 0.001
    candidate_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("train_start", "oos_start", "end", "anchor_date"):
            object.__setattr__(self, name, _as_calendar_date(getattr(self, name), name))

        if not isinstance(self.candidate_overrides, Mapping):
            raise TypeError("candidate_overrides must be a mapping")
        object.__setattr__(self, "candidate_overrides", dict(self.candidate_overrides))

        if self.train_start >= self.oos_start:
            raise ValueError("train_start must be before oos_start")
        if self.oos_start > self.end:
            raise ValueError("oos_start must not be after end")
        if self.oos_start.month not in (1, 4, 7, 10) or self.oos_start.day != 1:
            raise ValueError("oos_start must be the first day of a calendar quarter")

        if self.mode not in ("expanding", "rolling"):
            raise ValueError("mode must be 'expanding' or 'rolling'")
        if self.target_type not in _TARGET_TYPES:
            raise ValueError(f"target_type must be one of {_TARGET_TYPES}")

        for name in _POSITIVE_INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.rolling_months <= 3:
            raise ValueError("rolling_months must be greater than 3")
        if self.min_pairs < 5:
            raise ValueError("min_pairs must be at least 5")

        models = _as_tuple(self.models, "models")
        include = _as_tuple(self.include, "include")
        exclude = _as_tuple(self.exclude, "exclude")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "include", include)
        object.__setattr__(self, "exclude", exclude)
        if not models:
            raise ValueError("models must be nonempty")
        if len(set(models)) != len(models):
            raise ValueError("models must be unique")
        if not set(models).issubset(MODELS):
            raise ValueError(f"models must be a subset of {MODELS}")
        if set(include).intersection(exclude):
            raise ValueError("include and exclude must be disjoint")

        for name in ("coverage", "validation_prediction_coverage"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("ic_weight", "sharpe_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isclose(self.ic_weight + self.sharpe_weight, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("ic_weight and sharpe_weight must sum to one")
        for name in ("fee_rate", "slippage"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "Config":
        """Build a config from JSON-compatible values."""
        data = dict(values)
        for name in ("models", "include", "exclude"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)


@dataclass(frozen=True)
class Fold:
    quarter: str
    history_start: pd.Timestamp
    validation_start: pd.Timestamp
    retrain_at: pd.Timestamp
    prediction_end: pd.Timestamp


def quarter_folds(config: Config, evidence_end: date | str) -> list[Fold]:
    """Generate quarter folds through the available evidence date."""
    stop = min(pd.Timestamp(config.end), pd.Timestamp(_as_calendar_date(evidence_end, "evidence_end")))
    if stop < pd.Timestamp(config.oos_start):
        return []

    periods = pd.period_range(start=config.oos_start, end=stop, freq="Q")
    folds: list[Fold] = []
    for period in periods:
        quarter_start = period.start_time.normalize()
        validation_start = (quarter_start - pd.DateOffset(months=3)).normalize()
        if config.mode == "expanding":
            history_start = pd.Timestamp(config.train_start)
        else:
            history_start = max(
                pd.Timestamp(config.train_start),
                (quarter_start - pd.DateOffset(months=config.rolling_months)).normalize(),
            )
        if history_start >= validation_start:
            raise ValueError("no fit interval before validation")
        prediction_end = min(stop, period.end_time.normalize())
        folds.append(
            Fold(
                quarter=str(period),
                history_start=history_start,
                validation_start=validation_start,
                retrain_at=quarter_start,
                prediction_end=prediction_end,
            )
        )
    return folds
