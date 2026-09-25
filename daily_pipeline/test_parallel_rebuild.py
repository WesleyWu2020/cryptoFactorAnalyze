"""Verify parallel cache-miss rebuilds match the serial path bit-for-bit.

Runs compute_daily_signal twice over a short history with an isolated
factor-cache root: once serial (parallel_workers=1) and once parallel
(parallel_workers=3). Asserts NaN-safe exact equality of every factor
matrix, confirms rebuild workers persist their own cache files, and samples
peak child-process RSS so parallel_workers can be sized for this machine.

Usage: ./.venv/bin/python daily_pipeline/test_parallel_rebuild.py
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import daily_pipeline.pipeline as pipeline
from daily_pipeline.config import load_strategy_config


def _frames_identical(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    right = right.reindex(index=left.index, columns=left.columns)
    if right.shape != left.shape:
        return False
    equal = (left == right) | (left.isna() & right.isna())
    return bool(equal.all().all())


class _ChildRssSampler:
    """Track peak total RSS (MB) of direct child processes."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["pgrep", "-P", str(os.getpid())],
                    capture_output=True, text=True, check=False,
                ).stdout.split()
                total = 0.0
                for pid in out:
                    rss = subprocess.run(
                        ["ps", "-o", "rss=", "-p", pid],
                        capture_output=True, text=True, check=False,
                    ).stdout.strip()
                    if rss:
                        total += float(rss) / 1024.0
                self.peak_mb = max(self.peak_mb, total)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        return False


def _run(config, cache_root: Path, day: str) -> dict:
    pipeline.FACTOR_CACHE_ROOT = cache_root
    return pipeline._compute_factor_matrices(
        config, pd.Timestamp(day),
        pipeline.DataProvider(config.h5_path, as_of=pd.Timestamp(day)),
    )


def main() -> int:
    base = load_strategy_config(PROJECT_ROOT / "daily_pipeline/configs/portfolio_composite_v1.json")
    members = base.factors[:3]
    calc_start = "2026-06-01"
    day = "2026-09-24"

    from dataclasses import replace

    serial_cfg = replace(base, factors=members, calculation_start=calc_start, parallel_workers=1)
    parallel_cfg = replace(base, factors=members, calculation_start=calc_start, parallel_workers=3)

    with tempfile.TemporaryDirectory() as tmp:
        root_a = Path(tmp) / "serial"
        root_b = Path(tmp) / "parallel"

        t0 = time.monotonic()
        serial = _run(serial_cfg, root_a, day)
        serial_s = time.monotonic() - t0

        with _ChildRssSampler() as sampler:
            t0 = time.monotonic()
            parallel = _run(parallel_cfg, root_b, day)
            parallel_s = time.monotonic() - t0
        cache_files = sorted(p.relative_to(root_b) for p in root_b.glob("*/*.parquet"))

    failures = []
    for member in members:
        key = str(member.path)
        s_spec, s_matrix, _ = serial[key]
        p_spec, p_matrix, p_diag = parallel[key]
        if s_spec.factor_id != p_spec.factor_id:
            failures.append(f"{key}: factor_id mismatch")
        if not _frames_identical(s_matrix, p_matrix):
            failures.append(f"{key}: matrix differs between serial and parallel")
        if p_diag.get("factor_cache") != "rebuilt":
            failures.append(f"{key}: parallel diag missing factor_cache=rebuilt")
    if len(cache_files) != len(members):
        failures.append(f"expected {len(members)} cache files written by workers, got {len(cache_files)}")

    print(f"serial   : {serial_s:6.1f}s")
    print(f"parallel : {parallel_s:6.1f}s  (workers=3)")
    print(f"peak child RSS: {sampler.peak_mb:,.0f} MB")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: parallel rebuild output is bit-identical to serial")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
