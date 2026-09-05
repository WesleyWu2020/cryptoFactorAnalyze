"""Validated immutable execution profiles for factor backtests."""

from copy import deepcopy
from dataclasses import dataclass, fields, replace
from datetime import date
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class BacktestProfile:
    profile_id: str = "perp_1d"
    rebalance_days: int = 1
    anchor_date: str = "2024-01-01"
    signal_delay_days: int = 1
    price_field: str = "open"
    n_groups: int = 10
    factor_direction: int = 1
    initial_equity: float = 1.0
    gross_exposure: float = 1.0
    fee_rate: float = 0.0003
    include_funding: bool = True
    funding_price_mode: str = "strict"
    out_of_sample_days: int = 180
    split_date: str | None = None
    periods_per_year: int = 365


_SUPPORTED_PROFILE = "perp_1d"
_FUNDING_PRICE_MODES = frozenset({"strict", "daily_open_approx"})
_PROFILE_FIELDS = frozenset(field.name for field in fields(BacktestProfile))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_iso_date(field_name: str, value: str | None) -> None:
    if value is None and field_name == "split_date":
        return
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO date")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc


def _validate_profile(profile: BacktestProfile, requested_profile_id: str) -> None:
    if profile.profile_id != requested_profile_id or profile.profile_id != _SUPPORTED_PROFILE:
        raise ValueError(f"Unsupported profile: {profile.profile_id}")
    if profile.rebalance_days != 1:
        raise ValueError("perp_1d is a daily profile and requires rebalance_days=1")
    if profile.signal_delay_days != 1:
        raise ValueError("daily profile requires a one-day signal delay")
    if profile.price_field != "open":
        raise ValueError("daily profile requires price_field='open'")
    if not _is_int(profile.rebalance_days) or profile.rebalance_days < 1:
        raise ValueError("rebalance_days must be a positive integer")
    if not _is_int(profile.signal_delay_days) or profile.signal_delay_days < 0:
        raise ValueError("signal_delay_days must be a non-negative integer")
    if not isinstance(profile.price_field, str):
        raise ValueError("price_field must be a string")
    _validate_iso_date("anchor_date", profile.anchor_date)
    _validate_iso_date("split_date", profile.split_date)
    if not _is_int(profile.n_groups) or profile.n_groups < 2:
        raise ValueError("n_groups must be an integer greater than or equal to 2")
    if profile.factor_direction not in (-1, 1) or not _is_int(profile.factor_direction):
        raise ValueError("factor_direction must be either -1 or 1")
    if not _is_finite_number(profile.initial_equity) or profile.initial_equity <= 0:
        raise ValueError("initial_equity must be finite and greater than 0")
    if not _is_finite_number(profile.gross_exposure) or not 0 < profile.gross_exposure <= 1:
        raise ValueError("gross_exposure must be finite and in (0, 1]")
    if not _is_finite_number(profile.fee_rate) or not 0 <= profile.fee_rate < 1:
        raise ValueError("fee_rate must be finite and in [0, 1)")
    if not isinstance(profile.include_funding, bool):
        raise ValueError("include_funding must be a boolean")
    if profile.funding_price_mode not in _FUNDING_PRICE_MODES:
        raise ValueError(
            "funding_price_mode must be one of: strict, daily_open_approx"
        )
    if not _is_int(profile.out_of_sample_days) or profile.out_of_sample_days <= 0:
        raise ValueError("out_of_sample_days must be a positive integer")
    if not _is_int(profile.periods_per_year) or profile.periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")


def resolve_profile(profile_id: str, overrides: Mapping[str, Any] | None) -> BacktestProfile:
    """Resolve and validate a supported profile without retaining mutable inputs."""

    if not isinstance(profile_id, str):
        raise ValueError(f"Unsupported profile: {profile_id}")
    if profile_id in {"perp_4h", "perp_8h"}:
        raise ValueError("Only the daily profile is supported; intraday profiles are rejected")
    if profile_id != _SUPPORTED_PROFILE:
        raise ValueError(f"Unsupported profile: {profile_id}")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping")

    copied_overrides = deepcopy(dict(overrides))
    unknown = sorted(set(copied_overrides) - _PROFILE_FIELDS)
    if unknown:
        raise ValueError(f"Unknown override: {unknown[0]}")

    profile = replace(BacktestProfile(profile_id=profile_id), **copied_overrides)
    _validate_profile(profile, profile_id)
    return profile
