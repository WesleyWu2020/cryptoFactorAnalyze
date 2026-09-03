"""Immutable runtime configuration for the crypto-quant data pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from numbers import Integral, Real
from pathlib import Path


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_real(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class PipelineConfig:
    repo_root: Path
    store_path: Path
    staging_path: Path
    lock_path: Path
    rules_path: Path
    universe_start: date = date(2024, 1, 1)
    warmup_days: int = 180
    top_n: int = 50
    cmc_page_days: int = 10
    cmc_overlap_days: int = 10
    binance_overlap_days: int = 7
    request_timeout_seconds: float = 20.0
    max_attempts: int = 8
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    cmc_url: str = "https://pro-api.coinmarketcap.com/public-api/v3/index/cmc100-historical"
    exchange_info_url: str = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    klines_url: str = "https://fapi.binance.com/fapi/v1/klines"
    funding_url: str = "https://fapi.binance.com/fapi/v1/fundingRate"

    def __post_init__(self) -> None:
        for field_name in (
            "repo_root",
            "store_path",
            "staging_path",
            "lock_path",
            "rules_path",
        ):
            value = Path(getattr(self, field_name)).expanduser().resolve()
            object.__setattr__(self, field_name, value)

        _require_positive_int("warmup_days", self.warmup_days)
        _require_positive_int("top_n", self.top_n)
        _require_positive_int("cmc_page_days", self.cmc_page_days)
        _require_nonnegative_int("cmc_overlap_days", self.cmc_overlap_days)
        _require_nonnegative_int("binance_overlap_days", self.binance_overlap_days)
        _require_positive_real(
            "request_timeout_seconds", self.request_timeout_seconds
        )
        _require_positive_int("max_attempts", self.max_attempts)
        _require_positive_real("base_backoff_seconds", self.base_backoff_seconds)
        _require_positive_real("max_backoff_seconds", self.max_backoff_seconds)
        if self.max_backoff_seconds < self.base_backoff_seconds:
            raise ValueError(
                "max_backoff_seconds must be greater than or equal to "
                "base_backoff_seconds"
            )

    @classmethod
    def default(cls, repo_root: Path | None = None) -> "PipelineConfig":
        root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
        data_dir = root / "data"
        return cls(
            repo_root=root,
            store_path=data_dir / "crypto_quant.h5",
            staging_path=data_dir / ".crypto_quant.staging.h5",
            lock_path=data_dir / ".crypto_quant.lock",
            rules_path=data_dir / "crypto_quant_rules.json",
        )
