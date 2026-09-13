"""Training fitness and coverage accounting for daily GP signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from factor_common.metrics import _daily_ic


_STD_FLOOR = 1e-12


@dataclass(frozen=True)
class TrainingScore:
    direction: int
    mean_ic: float | None
    worst_quarter_ic: float | None
    icir: float | None
    quarter_means: dict[str, float | None]
    valid_days: int
    day_coverage: float
    cell_coverage: float
    node_count: int
    eligible: bool
    reasons: tuple[str, ...]
    fitness_mode: str = "legacy_ic"
    robust_ic: float | None = None

    @property
    def objective_vector(self) -> tuple[float | None, float | None, int]:
        if self.fitness_mode == "robust_ic":
            return (self.robust_ic, self.worst_quarter_ic, -self.node_count)
        return (self.mean_ic, self.worst_quarter_ic, -self.node_count)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _training_bounds(config: Any) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    stage = _config_value(config, "stage")
    start = _config_value(config, "train_start", _config_value(config, "training_start"))
    end = _config_value(config, "train_end", _config_value(config, "training_end"))
    if stage is not None:
        start = getattr(stage, "start", start)
        end = getattr(stage, "end", end)
    training_range = _config_value(config, "training_range")
    if training_range is not None:
        start, end = training_range
    return (
        None if start is None else pd.Timestamp(start).normalize(),
        None if end is None else pd.Timestamp(end).normalize(),
    )


def _validate_axes(values: pd.DataFrame, labels: pd.DataFrame, quality: pd.DataFrame) -> None:
    if not all(isinstance(frame, pd.DataFrame) for frame in (values, labels, quality)):
        raise TypeError("values, labels, and quality_eligible must be DataFrames")
    if not values.index.equals(labels.index) or not values.index.equals(quality.index):
        raise ValueError("training panels must have identical date axes")
    if not values.columns.equals(labels.columns) or not values.columns.equals(quality.columns):
        raise ValueError("training panels must have identical instrument axes")
    if values.index.has_duplicates or values.columns.has_duplicates:
        raise ValueError("training panels must not contain duplicate axes")


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def score_training(
    values: pd.DataFrame,
    labels: pd.DataFrame,
    quality_eligible: pd.DataFrame,
    config: Any,
) -> TrainingScore:
    """Score a signal on its bounded training panel.

    Labels are evaluation-only forward returns. They are never used to alter
    the signal or the quality-eligibility denominator.
    """
    _validate_axes(values, labels, quality_eligible)
    start, end = _training_bounds(config)
    if start is not None and values.index.min() < start:
        raise ValueError("signal outside training range")
    if end is not None and values.index.max() > end:
        raise ValueError("signal outside training range")

    quality = quality_eligible.fillna(False).astype(bool)
    finite_values = values.replace([np.inf, -np.inf], np.nan)
    finite_labels = labels.replace([np.inf, -np.inf], np.nan)
    robust = _config_value(config, "fitness_mode", "legacy_ic") == "robust_ic"
    if robust:
        if not isinstance(values.index, pd.DatetimeIndex) or not values.index.is_monotonic_increasing:
            raise ValueError("robust fitness requires ordered daily dates")
        if len(set(values.index.year)) != 1:
            raise ValueError("robust fitness requires one training calendar year")
        # Labels alone may look forward. Purge every quarter's execution tail,
        # even when a caller supplies precomputed labels containing later prices.
        exits = values.index + pd.Timedelta(days=1 + int(_config_value(config, "hold_days", 1)))
        quarter_ends = values.index.to_period("Q").end_time.normalize()
        boundary = values.index.max() if end is None else end
        usable = (exits <= quarter_ends) & (exits <= boundary)
        finite_labels = finite_labels.copy()
        finite_labels.loc[~usable] = np.nan
    signal_days = quality.any(axis=1)
    signal_values = finite_values.where(quality)
    daily = _daily_ic(signal_values.loc[signal_days], finite_labels.loc[signal_days])
    min_pairs = max(20, int(_config_value(config, "min_pairs", 20)))
    daily = daily.loc[
        (daily["n_pairs"] >= min_pairs)
        & pd.to_numeric(daily["rank_ic"], errors="coerce").notna()
    ].copy()

    valid_days = int(len(daily))
    total_days = int(signal_days.sum())
    day_coverage = valid_days / total_days if total_days else 0.0
    denominator = int(quality.loc[signal_days].to_numpy(dtype=bool).sum())
    observed = int(signal_values.loc[signal_days].notna().to_numpy().sum())
    cell_coverage = observed / denominator if denominator else 0.0

    raw_mean = _finite(daily["rank_ic"].mean()) if valid_days else None
    if robust:
        # Direction calibration uses Q1 only. Q2-Q4 never reorient the signal.
        first = daily.loc[pd.to_datetime(daily["date"]).dt.quarter == 1, "rank_ic"]
        raw_mean = _finite(first.mean())
    if raw_mean == 0.0:
        return TrainingScore(
            direction=0,
            mean_ic=None,
            worst_quarter_ic=None,
            icir=None,
            quarter_means={f"Q{quarter}": None for quarter in range(1, 5)},
            valid_days=valid_days,
            day_coverage=float(day_coverage),
            cell_coverage=float(cell_coverage),
            node_count=int(_config_value(config, "node_count", 0)),
            eligible=False,
            reasons=("zero raw training mean; direction is undefined",),
            fitness_mode="robust_ic" if robust else "legacy_ic",
        )
    direction = 1 if raw_mean is None or raw_mean >= 0 else -1
    directed = daily["rank_ic"] * direction
    mean_ic = _finite(directed.mean()) if valid_days else None
    std = float(directed.std(ddof=1)) if valid_days >= 2 else 0.0
    icir = _finite(float(mean_ic) / std) if mean_ic is not None and std > _STD_FLOOR else None

    quarter_means: dict[str, float | None] = {}
    daily_quarters = pd.to_datetime(daily["date"], errors="coerce").dt.quarter
    for quarter in range(1, 5):
        key = f"Q{quarter}"
        quarter_rows = daily.loc[daily_quarters == quarter, "rank_ic"]
        quarter_means[key] = _finite(quarter_rows.mean() * direction) if len(quarter_rows) else None
    finite_quarters = [value for value in quarter_means.values() if value is not None]
    worst_quarter_ic = min(finite_quarters) if len(finite_quarters) == 4 else None

    reasons: list[str] = []
    min_quarter_days = int(_config_value(config, "min_quarter_days", 45))
    for quarter in range(1, 5):
        if int((daily_quarters == quarter).sum()) < min_quarter_days:
            reasons.append(f"Q{quarter} has fewer than {min_quarter_days} valid IC days")
    if valid_days == 0:
        reasons.append("no valid daily IC with minimum pairs")
    min_day_coverage = float(_config_value(config, "min_day_coverage", 0.8))
    min_cell_coverage = float(_config_value(config, "min_cell_coverage", 0.8))
    if day_coverage < min_day_coverage:
        reasons.append("day coverage below threshold")
    if cell_coverage < min_cell_coverage:
        reasons.append("cell coverage below threshold")
    if sum(value is not None and value > 0 for value in quarter_means.values()) < 3:
        reasons.append("fewer than three positive quarters")

    node_count = int(_config_value(config, "node_count", 0))
    robust_ic = None
    if robust:
        later = [quarter_means[key] for key in ("Q2", "Q3", "Q4")]
        if raw_mean is None:
            reasons.append("Q1 direction calibration unavailable")
        if all(value is not None for value in later):
            worst_quarter_ic = min(later)
            robust_ic = float(
                np.median(later)
                - float(_config_value(config, "stability_penalty", 0.5)) * np.std(later)
                - float(_config_value(config, "worst_quarter_penalty", 1.0)) * max(0.0, -min(later))
                - float(_config_value(config, "complexity_penalty", 0.001)) * node_count
            )
            if robust_ic <= 0:
                reasons.append("robust IC objective is not positive")
            if sum(value > 0 for value in later) < 2:
                reasons.append("fewer than two positive post-calibration quarters")
        else:
            worst_quarter_ic = None
            reasons.append("incomplete post-calibration quarters")
    return TrainingScore(
        direction=direction,
        mean_ic=mean_ic,
        worst_quarter_ic=worst_quarter_ic,
        icir=icir,
        quarter_means=quarter_means,
        valid_days=valid_days,
        day_coverage=float(day_coverage),
        cell_coverage=float(cell_coverage),
        node_count=node_count,
        eligible=not reasons,
        reasons=tuple(reasons),
        fitness_mode="robust_ic" if robust else "legacy_ic",
        robust_ic=robust_ic,
    )


__all__ = ["TrainingScore", "score_training"]
