"""Load and validate daily factor modules without touching market data."""
from __future__ import annotations

import hashlib
import importlib.util
import re
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from .data_provider import MARKET_FIELDS
from .definitions import FactorSpec


_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_META_FIELDS = ("factor_name", "author", "level", "category", "description")
_SETTING_FIELDS = (
    "data_needed",
    "universe",
    "warmup_bars",
    "preprocessing",
    "params",
    "factor_direction",
)
_SUPPORTED_PREPROCESSING = frozenset({"none", "mad_rank"})
_SUPPORTED_FIELDS = frozenset(MARKET_FIELDS)


def _validate_identifier(value: Any, *, field_name: str = "factor identifier") -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a safe identifier matching "
            r"^[A-Za-z][A-Za-z0-9_]*$"
        )
    return value


def _mapping_attribute(module: ModuleType, name: str) -> dict[str, Any]:
    value = getattr(module, name, None)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _validate_meta(meta: dict[str, Any], factor_id: str) -> None:
    for field_name in _META_FIELDS:
        if field_name not in meta:
            raise ValueError(f"META is missing required field: {field_name}")
        if not isinstance(meta[field_name], str) or not meta[field_name].strip():
            raise ValueError(f"META.{field_name} must be a non-empty string")

    meta_factor_name = _validate_identifier(meta["factor_name"], field_name="META.factor_name")
    if meta_factor_name != factor_id:
        raise ValueError(
            f"META.factor_name {meta_factor_name!r} must match factor filename identifier {factor_id!r}"
        )
    if meta["level"] != "daily":
        raise ValueError(
            f"Only daily factors are supported; META.level={meta['level']!r}. "
            "Use level='daily' instead of minute or intraday frequency."
        )
    if "frequency" in meta and meta["frequency"] != "daily":
        raise ValueError(
            f"Only daily formula frequency is supported; META.frequency={meta['frequency']!r}"
        )


def _validate_settings(setting: dict[str, Any]) -> None:
    missing = [field_name for field_name in _SETTING_FIELDS if field_name not in setting]
    if missing:
        raise ValueError(f"SETTING is missing required field: {missing[0]}")

    data_needed = setting["data_needed"]
    if not isinstance(data_needed, (list, tuple)) or not data_needed:
        raise ValueError("SETTING.data_needed must be a non-empty list of field names")
    if any(not isinstance(field, str) or not field for field in data_needed):
        raise ValueError("SETTING.data_needed must contain only non-empty strings")
    if len(set(data_needed)) != len(data_needed):
        raise ValueError("SETTING.data_needed must not contain duplicate fields")
    unsupported = sorted(set(data_needed) - _SUPPORTED_FIELDS)
    if unsupported:
        raise ValueError(f"SETTING.data_needed contains unsupported field(s): {unsupported}")

    if not isinstance(setting["universe"], str) or not setting["universe"].strip():
        raise ValueError("SETTING.universe must be a non-empty string")

    warmup = setting["warmup_bars"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("SETTING.warmup_bars must be an integer greater than or equal to 0")

    preprocessing = setting["preprocessing"]
    if preprocessing not in _SUPPORTED_PREPROCESSING:
        choices = ", ".join(sorted(_SUPPORTED_PREPROCESSING))
        raise ValueError(f"SETTING.preprocessing must be one of: {choices}")

    if not isinstance(setting["params"], Mapping):
        raise ValueError("SETTING.params must be a mapping")

    direction = setting["factor_direction"]
    if isinstance(direction, bool) or not isinstance(direction, int) or direction not in (-1, 1):
        raise ValueError("SETTING.factor_direction must be either -1 or 1")

    if "frequency" in setting and setting["frequency"] != "daily":
        raise ValueError(
            f"Only daily formula frequency is supported; SETTING.frequency={setting['frequency']!r}"
        )


def _load_module(path: Path, factor_id: str, source_sha256: str) -> ModuleType:
    module_name = f"_factor_common_{factor_id}_{source_sha256[:16]}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"Could not create a Python module spec for {path}")
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        raise ValueError(f"Could not import factor module {path}: {exc}") from exc
    return module


def load_factor(path: str | Path) -> FactorSpec:
    """Load one trusted local daily ``.py`` factor module.

    The module is executed as Python code, so this loader does not provide a
    sandbox. It only accepts a local file path and validates the resulting
    module contract before returning its immutable ``FactorSpec``.
    """

    factor_path = Path(path)
    if factor_path.suffix != ".py":
        raise ValueError("factor path must point to a local .py file")
    try:
        factor_path = factor_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"factor file does not exist: {path}") from exc
    if not factor_path.is_file():
        raise ValueError(f"factor path is not a file: {path}")

    factor_id = _validate_identifier(factor_path.stem, field_name="factor filename")
    source = factor_path.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    module = _load_module(factor_path, factor_id, source_sha256)

    if getattr(module, "TYPE", None) != "regular":
        raise ValueError("factor TYPE must be 'regular'")
    meta = _mapping_attribute(module, "META")
    setting = _mapping_attribute(module, "SETTING")
    _validate_meta(meta, factor_id)
    _validate_settings(setting)
    calc_factor = getattr(module, "calc_factor", None)
    if not callable(calc_factor):
        raise ValueError("factor module must define callable calc_factor(data_ctx)")

    return FactorSpec(factor_id, meta, setting, calc_factor, source_sha256)


def validate_factor_output(output: pd.DataFrame) -> pd.DataFrame:
    """Validate the unique date/instrument axes of a factor result."""

    if not isinstance(output, pd.DataFrame):
        raise TypeError("factor output must be a pandas DataFrame")
    if output.index.has_duplicates:
        raise ValueError("factor output has a duplicate date axis")
    if output.columns.has_duplicates:
        raise ValueError("factor output has a duplicate instrument axis")
    if {"date", "instrument"}.issubset(output.columns):
        if output.duplicated(["date", "instrument"]).any():
            raise ValueError("factor output has a duplicate date/instrument axis")
    return output


def create_template(factor_name: str, directory: str | Path = ".") -> Path:
    """Create a daily regular-factor module without overwriting a file."""

    factor_name = _validate_identifier(factor_name)
    target_directory = Path(directory)
    if not target_directory.is_dir():
        raise ValueError(f"template directory is not an existing directory: {directory}")
    target = target_directory / f"{factor_name}.py"
    template_path = Path(__file__).with_name("templates") / "regular_daily.py.tmpl"
    content = template_path.read_text(encoding="utf-8").replace("example_momentum", factor_name)
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing factor file: {target}") from None
    return target


__all__ = ["create_template", "load_factor", "validate_factor_output"]
