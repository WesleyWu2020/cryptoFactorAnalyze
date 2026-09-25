"""Full-history parallel rebuild check: workers must reproduce prod caches.

Two members rebuild from calculation_start=2024-01-01 (the production full
history) in an isolated cache root with parallel_workers=2. Each resulting
matrix is compared NaN-safe bitwise against the existing production factor
cache for the same sha, and peak child RSS is sampled.

Usage: ./.venv/bin/python daily_pipeline/check_full_rebuild.py
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import daily_pipeline.pipeline as pipeline
from daily_pipeline.config import load_strategy_config
from daily_pipeline.test_parallel_rebuild import _ChildRssSampler, _frames_identical


def main() -> int:
    base = load_strategy_config(PROJECT_ROOT / "daily_pipeline/configs/portfolio_composite_v1.json")
    members = tuple(
        m for m in base.factors
        if m.path.name in ("GP_0006a47b612eaf9b.py", "GP_77282e814dac81a8.py")
    )
    cfg = replace(base, factors=members, parallel_workers=2)
    day = "2026-09-24"
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        pipeline.FACTOR_CACHE_ROOT = Path(tmp)
        with _ChildRssSampler(interval=0.3) as sampler:
            t0 = time.monotonic()
            results = pipeline._compute_factor_matrices(
                cfg, pd.Timestamp(day),
                pipeline.DataProvider(cfg.h5_path, as_of=pd.Timestamp(day)),
            )
            elapsed = time.monotonic() - t0
        for member in members:
            spec, matrix, diag = results[str(member.path)]
            prod_path = (
                PROJECT_ROOT / "outputs/daily_pipeline/factor_cache" / spec.factor_id
                / f"{member.sha256[:16]}_{cfg.calculation_start}.parquet"
            )
            prod = pd.read_parquet(prod_path)
            same = _frames_identical(prod.loc[:day], matrix)
            ok &= same
            print(f"{spec.factor_id}: identical_to_prod_cache={same} "
                  f"diag_cache={diag.get('factor_cache')} shape={matrix.shape}")
    print(f"elapsed={elapsed:.1f}s peak_child_RSS={sampler.peak_mb:,.0f} MB -> "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
