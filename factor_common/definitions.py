"""Immutable definitions shared by factor calculation and evaluation code."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from copy import deepcopy


@dataclass(frozen=True)
class FactorSpec:
    """Stable metadata and settings for one daily factor definition."""

    factor_id: str
    level: str = "daily"
    description: str = ""
    required_fields: tuple[str, ...] = ()
    warmup_days: int = 0
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_fields", tuple(self.required_fields))
        object.__setattr__(self, "params", MappingProxyType(deepcopy(dict(self.params))))
