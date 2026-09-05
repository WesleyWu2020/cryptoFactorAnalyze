"""Immutable definitions shared by factor calculation and evaluation code."""

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

import pandas as pd


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    meta: dict
    setting: dict
    calc_factor: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]
    source_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", _freeze(deepcopy(self.meta)))
        object.__setattr__(self, "setting", _freeze(deepcopy(self.setting)))
