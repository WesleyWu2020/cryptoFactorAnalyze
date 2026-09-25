"""Strict, reproducible configuration for the fixed portfolio backtest."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import math
import re
from numbers import Real
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import pandas as pd


def daily_date(value: Any) -> pd.Timestamp:
    """Parse a naive midnight timestamp suitable for a daily date."""
    ts = pd.Timestamp(value)
    if pd.isna(ts) or ts.tzinfo is not None:
        raise ValueError("date must be a naive, non-NaT daily timestamp")
    if ts != ts.normalize():
        raise ValueError("date must be at midnight")
    return ts.normalize()


class _FrozenDict(dict):
    """Dict-compatible immutable mapping that remains dataclasses.asdict-safe."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("code_hashes is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> "_FrozenDict":
        return self


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise TypeError(f"{name} must be a finite number")
    return float(value)


def _date_only(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError(f"{name} must be a UTC date-only ISO string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a UTC date-only ISO string") from exc
    canonical = parsed.isoformat()
    if value != canonical:
        raise ValueError(f"{name} must be a UTC date-only ISO string")
    return canonical


@dataclass(frozen=True)
class FactorAllocation:
    path: str
    allocation: float

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("factor path must be nonempty")
        allocation = _number(self.allocation, "allocation")
        if allocation <= 0:
            raise ValueError("allocation must be > 0")
        object.__setattr__(self, "path", self.path.strip())
        object.__setattr__(self, "allocation", allocation)


@dataclass(frozen=True)
class FixedConfig:
    h5_path: str
    signal_start: str
    signal_end: str
    as_of: str
    factors: tuple[FactorAllocation, ...]
    code_hashes: dict[str, str]
    n_groups: int = 5
    min_valid_instruments: int = 10
    gross_limit: float = 0.5
    long_limit: float = 0.25
    short_limit: float = 0.25
    single_limit: float = 0.05
    net_limit: float = 0.05
    fee_rate: float = 0.0005
    slippage: float = 0.001
    schema_version: int = 1
    anchor_date: str = "2024-01-01"
    rebalance_days: int = 1
    market_mode: str = "perp_long_short"
    initial_equity: float = 1.0
    frozen_at: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(self.h5_path, str) or not self.h5_path.strip():
            raise ValueError("h5_path must be nonempty")
        object.__setattr__(self, "h5_path", self.h5_path.strip())
        starts = _date_only(self.signal_start, "signal_start")
        ends = _date_only(self.signal_end, "signal_end")
        as_of = _date_only(self.as_of, "as_of")
        object.__setattr__(self, "anchor_date", _date_only(self.anchor_date, "anchor_date"))
        if not starts <= ends or as_of < starts:
            raise ValueError("signal_start <= signal_end and as_of >= signal_start are required")
        object.__setattr__(self, "signal_start", starts)
        object.__setattr__(self, "signal_end", ends)
        object.__setattr__(self, "as_of", as_of)

        if not isinstance(self.factors, tuple) or not self.factors:
            raise TypeError("constructor factors must be a nonempty tuple")
        normalized: list[FactorAllocation] = []
        for item in self.factors:
            if isinstance(item, FactorAllocation):
                normalized.append(item)
            else:
                raise TypeError("constructor factors entries must be FactorAllocation")
        if not math.isclose(sum(x.allocation for x in normalized), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("factor allocations must sum to 1")
        resolved = [str(Path(x.path).expanduser().resolve()) for x in normalized]
        if len(set(resolved)) != len(resolved):
            raise ValueError("duplicate factor paths are not allowed")
        object.__setattr__(self, "factors", tuple(normalized))

        if not isinstance(self.code_hashes, Mapping):
            raise TypeError("code_hashes must be a mapping")
        if not self.code_hashes:
            raise ValueError("code_hashes must be nonempty")
        object.__setattr__(self, "code_hashes", _FrozenDict(self.code_hashes))
        for path, digest in self.code_hashes.items():
            if not isinstance(path, str):
                raise TypeError("code_hashes keys must be strings")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError(f"code_hashes[{path!r}] must be a 64-character hex digest")
        for name in ("n_groups", "min_valid_instruments", "rebalance_days"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < (2 if name == "n_groups" else 1 if name == "min_valid_instruments" else 1):
                raise ValueError(f"{name} is out of range")
        if self.min_valid_instruments < self.n_groups:
            raise ValueError("min_valid_instruments must be >= n_groups")
        if self.market_mode != "perp_long_short":
            raise ValueError("market_mode must be 'perp_long_short'")
        for name in ("gross_limit", "long_limit", "short_limit", "single_limit", "net_limit"):
            value = _number(getattr(self, name), name)
            lower = 0 if name == "net_limit" else 0
            if (value < lower if name == "net_limit" else value <= 0) or value > 1:
                raise ValueError(f"{name} must be in {'[0, 1]' if name == 'net_limit' else '(0, 1]'}")
            object.__setattr__(self, name, value)
        for name in ("fee_rate", "slippage"):
            value = _number(getattr(self, name), name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
            object.__setattr__(self, name, value)
        equity = _number(self.initial_equity, "initial_equity")
        if equity <= 0:
            raise ValueError("initial_equity must be > 0")
        object.__setattr__(self, "initial_equity", equity)
        if not isinstance(self.frozen_at, str):
            raise TypeError("frozen_at must be a string")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FixedConfig":
        if not isinstance(data, dict):
            raise TypeError("config must be a dict")
        valid = set(cls.__dataclass_fields__)
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
        payload = dict(data)
        if "factors" in payload:
            entries = payload["factors"]
            if not isinstance(entries, (tuple, list)):
                raise TypeError("factors must be a tuple or list")
            converted = []
            for item in entries:
                if not isinstance(item, dict) or set(item) != {"path", "allocation"}:
                    raise TypeError("from_dict factors entries must be dicts with path and allocation")
                converted.append(FactorAllocation(**item))
            payload["factors"] = tuple(converted)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["code_hashes"] = dict(result["code_hashes"])
        return result
