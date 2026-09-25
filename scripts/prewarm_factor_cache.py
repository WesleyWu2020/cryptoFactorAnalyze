"""Pre-warm daily_pipeline factor caches for re-exported GP factors.

Computes each factor's full-history matrix exactly as the pipeline's serial
cache path would (same provider, same range, same parquet layout), so the
recovery pipeline runs start from a warm cache and finish within minutes.

Usage: ./.venv/bin/python scripts/prewarm_factor_cache.py GP_xxx [GP_yyy ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from daily_pipeline.config import load_strategy_config  # noqa: E402
from daily_pipeline.pipeline import _factor_cache_path, _write_factor_cache  # noqa: E402
from factor_common.data_provider import DataProvider  # noqa: E402
from factor_common.loader import load_factor  # noqa: E402
from factor_common.value_engine import compute_factor  # noqa: E402

CACHE_END = "2026-09-21"  # matches the last successful run's data cutoff


def main(argv: list[str]) -> int:
    config = load_strategy_config(
        PROJECT_ROOT / "daily_pipeline" / "configs" / "portfolio_composite_daily1.json"
    )
    wanted = set(argv)
    provider = DataProvider(config.h5_path, as_of=pd.Timestamp(CACHE_END))
    for member in config.factors:
        stem = member.path.stem
        if stem not in wanted:
            continue
        spec = load_factor(member.path)
        matrix, _ = compute_factor(
            spec, provider, start=config.calculation_start, end=CACHE_END
        )
        path = _factor_cache_path(config, member, spec)
        _write_factor_cache(path, matrix)
        print(f"CACHED {stem} -> {path.name} shape={matrix.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
