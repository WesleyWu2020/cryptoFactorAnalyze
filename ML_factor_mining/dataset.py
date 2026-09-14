"""Causal panel construction and fit-only feature admission."""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


MISSING_PREFIX = "__missing__"


def _freeze(value: Any) -> Any:
    """Recursively copy mutable diagnostics into immutable containers."""
    if isinstance(value, ABCMapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def stack_matrix(matrix: pd.DataFrame) -> pd.Series:
    """Stack a date-by-instrument matrix with stable, explicit axis names."""
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix must be a pandas DataFrame")
    return matrix.rename_axis(index="date", columns="instrument").stack(future_stack=True)


def clean_factor_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize a long factor table."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("factor table must be a pandas DataFrame")
    expected = ["date", "instrument", "factor"]
    if list(table.columns) != expected:
        raise ValueError(f"factor table columns must be exactly {expected}")
    result = table.copy()
    try:
        dates = pd.to_datetime(result["date"], utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("date values must be valid timestamps") from exc
    if dates.isna().any():
        raise ValueError("date values cannot be missing")
    dates = dates.dt.tz_localize(None)
    if (dates.dt.normalize() != dates).any():
        raise ValueError("intraday timestamps are not supported")
    result["date"] = dates

    if result["instrument"].isna().any() or not result["instrument"].map(
        lambda value: isinstance(value, str) and bool(value)
    ).all():
        raise ValueError("instrument values must be nonmissing strings")
    try:
        result["factor"] = pd.to_numeric(result["factor"], errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("factor values must be numeric") from exc
    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan)
    if result.duplicated(["date", "instrument"]).any():
        raise ValueError("duplicate date/instrument rows")
    return result.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)


def rank_panel(raw: pd.Series) -> pd.Series:
    """Rank factor values within each date on [-1, 1], retaining missingness."""
    if not isinstance(raw, pd.Series):
        raise TypeError("raw factor panel must be a pandas Series")
    if not isinstance(raw.index, pd.MultiIndex) or raw.index.nlevels != 2:
        raise ValueError("raw factor panel must have date/instrument MultiIndex")
    values = pd.to_numeric(raw, errors="raise").astype("float64")
    date_level = values.index.names.index("date") if "date" in values.index.names else 0
    counts = values.notna().groupby(level=date_level, sort=False).transform("sum")
    ranks = values.groupby(level=date_level, sort=False).rank(method="average")
    ranked = 2.0 * (ranks - 1.0) / (counts - 1.0) - 1.0
    ranked = ranked.where(counts > 1, 0.0)
    ranked = ranked.where(values.notna())
    ranked.index = ranked.index.set_names(["date", "instrument"])
    return ranked.sort_index()


def _eligible_index(membership: pd.DataFrame) -> pd.MultiIndex:
    if not isinstance(membership, pd.DataFrame):
        raise TypeError("membership must be a pandas DataFrame")
    if membership.index.has_duplicates or membership.columns.has_duplicates:
        raise ValueError("membership date/instrument axes contain duplicate labels")
    membership = membership.fillna(False).astype(bool)
    eligible = stack_matrix(membership)
    eligible = eligible[eligible]
    if eligible.empty:
        raise ValueError("membership has no eligible date/instrument rows")
    return eligible.index.sort_values()


def build_panel(catalog: Iterable[Mapping[str, Any]], membership: pd.DataFrame) -> pd.DataFrame:
    """Load, mask, and cross-sectionally rank catalog factors."""
    eligible = _eligible_index(membership)
    records = list(catalog)
    if not records:
        raise ValueError("factor catalog is empty")
    factor_ids = [record.get("factor_id") for record in records]
    if any(not isinstance(identifier, str) or not identifier for identifier in factor_ids):
        raise ValueError("catalog factor_id must be nonempty strings")
    if len(set(factor_ids)) != len(factor_ids):
        raise ValueError("catalog contains duplicate factor_id")

    result: dict[str, pd.Series] = {}
    for record in records:
        values_path = record.get("values")
        if values_path is None:
            raise ValueError(f"missing values path for {record['factor_id']!r}")
        cleaned = clean_factor_table(pd.read_parquet(values_path))
        raw = cleaned.set_index(["date", "instrument"])["factor"].reindex(eligible)
        result[record["factor_id"]] = rank_panel(raw)
    return pd.DataFrame(result, index=eligible).sort_index()


@dataclass(frozen=True)
class FeatureMap:
    """A fit-only, immutable map from ranked factors to model columns."""

    factors: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    @property
    def columns(self) -> tuple[str, ...]:
        return self.factors + tuple(f"{MISSING_PREFIX}{factor}" for factor in self.factors)

    @classmethod
    def fit(cls, fit_panel: pd.DataFrame, coverage: float) -> "FeatureMap":
        if not isinstance(fit_panel, pd.DataFrame) or fit_panel.empty or fit_panel.shape[1] == 0:
            raise ValueError("fit panel must be nonempty")
        if not isinstance(coverage, (int, float)) or isinstance(coverage, bool) or not 0 < coverage <= 1:
            raise ValueError("coverage must be in (0, 1]")
        names = list(fit_panel.columns)
        if any(not isinstance(name, str) or name.startswith(MISSING_PREFIX) for name in names):
            raise ValueError(f"factor names cannot use reserved prefix {MISSING_PREFIX!r}")
        rates = fit_panel.notna().mean()
        distinct = fit_panel.nunique(dropna=True)
        selected = tuple(sorted(name for name in names if rates[name] >= coverage and distinct[name] > 1))
        diagnostics = {
            "coverage": {name: float(rates[name]) for name in sorted(names)},
            "distinct": {name: int(distinct[name]) for name in sorted(names)},
            "selected": list(selected),
        }
        if not selected:
            raise ValueError("no factors meet fit-only coverage and distinct-value thresholds")
        return cls(factors=selected, diagnostics=diagnostics)

    def transform(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if not self.factors:
            raise ValueError("FeatureMap has no admitted factors")
        if not isinstance(panel, pd.DataFrame):
            raise TypeError("panel must be a pandas DataFrame")
        missing_factors = [factor for factor in self.factors if factor not in panel.columns]
        if missing_factors:
            raise ValueError("panel is missing factors: " + ", ".join(missing_factors))
        values = panel.loc[:, list(self.factors)].astype("float64")
        present = values.notna().any(axis=1).rename("present")
        indicators = values.isna().astype("float64")
        indicators.columns = [f"{MISSING_PREFIX}{factor}" for factor in self.factors]
        features = pd.concat([values.fillna(0.0), indicators], axis=1)
        features = features.loc[:, list(self.columns)].astype("float64")
        return features, present
