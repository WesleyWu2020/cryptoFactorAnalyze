"""Immutable runtime configuration for the crypto-quant data pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


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
