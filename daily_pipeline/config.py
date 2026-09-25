"""Validated configuration for the daily target-weight producer."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _iso_date(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be an ISO date")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


@dataclass(frozen=True)
class FactorMember:
    path: Path
    allocation: float
    sha256: str


@dataclass(frozen=True)
class DailyStrategyConfig:
    strategy_id: str
    strategy_version: int
    source_report: Path
    source_report_config_sha256: str
    h5_path: Path
    calculation_start: str
    anchor_date: str
    rebalance_days: int
    signal_delay_days: int
    n_groups: int
    min_valid_instruments: int
    gross_limit: float
    long_limit: float
    short_limit: float
    single_limit: float
    net_limit: float
    factors: tuple[FactorMember, ...]
    combine_mode: str = "members"
    parallel_workers: int = 1
    use_factor_cache: bool = False


def _resolve(path: str, *, root: Path) -> Path:
    candidate = Path(path).expanduser()
    return (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def load_strategy_config(path: str | Path) -> DailyStrategyConfig:
    """Load a compact production strategy and verify its frozen provenance."""
    config_path = Path(path).expanduser().resolve(strict=True)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    root = PROJECT_ROOT
    factors = tuple(
        FactorMember(
            path=_resolve(item["path"], root=root),
            allocation=_positive(item["allocation"], "factor allocation"),
            sha256=str(item["sha256"]),
        )
        for item in raw["factors"]
    )
    if not factors or not math.isclose(
        sum(item.allocation for item in factors), 1.0, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("factor allocations must sum to 1")
    for item in factors:
        if not item.path.is_file():
            raise FileNotFoundError(f"factor file does not exist: {item.path}")
        digest = hashlib.sha256(item.path.read_bytes()).hexdigest()
        if digest != item.sha256:
            raise ValueError(f"factor source changed: {item.path}")

    source_report = _resolve(raw["source_report"], root=root)
    source_config = source_report / "config.json"
    if not source_config.is_file():
        raise FileNotFoundError(f"source report config does not exist: {source_config}")
    digest = hashlib.sha256(source_config.read_bytes()).hexdigest()
    if digest != raw["source_report_config_sha256"]:
        raise ValueError("source report configuration changed")

    integer_fields = ("strategy_version", "rebalance_days", "signal_delay_days", "n_groups", "min_valid_instruments")
    for name in integer_fields:
        value = raw[name]
        minimum = 0 if name == "signal_delay_days" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if raw["n_groups"] < 2 or raw["min_valid_instruments"] < raw["n_groups"]:
        raise ValueError("invalid group/minimum-instrument configuration")

    limits = {name: float(raw[name]) for name in (
        "gross_limit", "long_limit", "short_limit", "single_limit", "net_limit"
    )}
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in limits.values()):
        raise ValueError("portfolio limits must be finite values in [0, 1]")
    if any(limits[name] == 0 for name in ("gross_limit", "long_limit", "short_limit", "single_limit")):
        raise ValueError("gross/long/short/single limits must be positive")

    combine_mode = str(raw.get("combine_mode", "members"))
    if combine_mode not in ("members", "composite"):
        raise ValueError("combine_mode must be 'members' or 'composite'")

    parallel_workers = raw.get("parallel_workers", 1)
    if isinstance(parallel_workers, bool) or not isinstance(parallel_workers, int) or parallel_workers < 1:
        raise ValueError("parallel_workers must be an integer >= 1")

    use_factor_cache = raw.get("use_factor_cache", False)
    if not isinstance(use_factor_cache, bool):
        raise ValueError("use_factor_cache must be a boolean")

    return DailyStrategyConfig(
        strategy_id=str(raw["strategy_id"]),
        strategy_version=raw["strategy_version"],
        source_report=source_report,
        source_report_config_sha256=raw["source_report_config_sha256"],
        h5_path=_resolve(raw["h5_path"], root=root),
        calculation_start=_iso_date(raw["calculation_start"], "calculation_start"),
        anchor_date=_iso_date(raw["anchor_date"], "anchor_date"),
        rebalance_days=raw["rebalance_days"],
        signal_delay_days=raw["signal_delay_days"],
        n_groups=raw["n_groups"],
        min_valid_instruments=raw["min_valid_instruments"],
        factors=factors,
        combine_mode=combine_mode,
        parallel_workers=parallel_workers,
        use_factor_cache=use_factor_cache,
        **limits,
    )
