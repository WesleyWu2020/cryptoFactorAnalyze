"""Smoke-test one factor module against the factor_common contract.

Usage:
    ./.venv/bin/python scripts/smoke_alpha_factor.py <factor_path> [--start 2025-05-01] [--end 2025-07-01]

Loads the module via loader.load_factor, computes values through
value_engine.compute_factor over a short window, and prints diagnostics.
Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("factor_path")
    parser.add_argument("--start", default="2025-05-01")
    parser.add_argument("--end", default="2025-07-01")
    args = parser.parse_args()

    from factor_common.data_provider import DataProvider
    from factor_common.loader import load_factor
    from factor_common.value_engine import compute_factor

    factor_path = Path(args.factor_path).expanduser()
    if not factor_path.is_absolute():
        factor_path = (Path.cwd() / factor_path).resolve()

    spec = load_factor(factor_path)
    print(f"loaded: {spec.factor_id} setting={dict(spec.setting)}")

    dp = DataProvider(PROJECT_ROOT / "data" / "crypto_quant.h5")
    matrix, diagnostics = compute_factor(spec, dp, start=args.start, end=args.end)
    print(f"matrix: shape={matrix.shape} diagnostics={diagnostics}")

    if diagnostics["valid_count"] == 0:
        print("FAIL: no valid factor values in window")
        return 1
    coverage = diagnostics["valid_count"] / max(diagnostics["eligible_count"], 1)
    print(f"coverage(valid/eligible)={coverage:.3f}")
    if coverage < 0.5:
        print("FAIL: coverage below 0.5")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
