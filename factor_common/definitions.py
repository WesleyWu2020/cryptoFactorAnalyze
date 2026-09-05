"""Immutable definitions shared by factor calculation and evaluation code."""

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

import pandas as pd


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {_freeze(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise TypeError(
        "Unsupported mutable value in FactorSpec metadata/settings: "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    meta: dict
    setting: dict
    calc_factor: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]
    source_sha256: str

    def __post_init__(self) -> None:
        meta = deepcopy(self.meta)
        setting = deepcopy(self.setting)
        if not isinstance(meta, dict):
            raise TypeError("FactorSpec meta must be a dict")
        if not isinstance(setting, dict):
            raise TypeError("FactorSpec setting must be a dict")
        object.__setattr__(self, "meta", _freeze(meta))
        object.__setattr__(self, "setting", _freeze(setting))
