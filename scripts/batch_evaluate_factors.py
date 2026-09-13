"""Batch-evaluate all new-template factors and render HTML reports.

Runs every factor_mining module (new factor_common contract) through
FactorManager.evaluate with one standard parameter set, then writes
reports/<factor_stem>.html via plot_result. Intended usage:

    ./.venv/bin/python scripts/batch_evaluate_factors.py [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXCLUDE = {"util_factor", "operator_utils"}

PARAMS = {
    "start": "2024-01-01",
    "end": "2026-09-01",
    "rebalance_days": 1,
    "n_groups": 5,
    "fee_rate": 0.0005,
    "slippage": 0.001,
    "include_funding": True,
}


def evaluate_one(stem: str) -> dict[str, object]:
    from factor_common import FactorManager

    manager = FactorManager(
        h5_path=PROJECT_ROOT / "data" / "crypto_quant.h5",
        base_dir=PROJECT_ROOT / "data" / "factor_results",
        reports_dir=PROJECT_ROOT / "reports",
    )
    factor_path = PROJECT_ROOT / "factor_analyse" / "factor_mining" / f"{stem}.py"
    summary: dict[str, object] = {"factor": stem}
    try:
        result = manager.evaluate(factor_path, params=dict(PARAMS), plot=False)
        summary["status"] = result["status"]
        summary["run_id"] = result["run_id"]
        ic = result["factor_performance"]["samples"]["full"]["ic"]
        summary["ic_mean"] = ic["ic_mean"]
        summary["rank_ic_mean"] = ic["rank_ic_mean"]
        validation = result["diagnostics"]["validation"]
        summary["static_scan"] = validation["static_scan"]["status"]
        summary["cutoff"] = validation["cutoff"]["status"]
        info = manager.plot_result(
            result, output_path=PROJECT_ROOT / "reports" / f"{stem}.html"
        )
        summary["report"] = str(info["output_path"])
    except Exception as exc:  # noqa: BLE001 - batch must not die on one factor
        summary["status"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    factor_dir = PROJECT_ROOT / "factor_analyse" / "factor_mining"
    stems = sorted(
        p.stem for p in factor_dir.glob("*.py") if p.stem not in EXCLUDE
    )
    print(f"factors: {len(stems)}  workers: {args.workers}", flush=True)

    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_one, stem): stem for stem in stems}
        for future in as_completed(futures):
            summary = future.result()
            results.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    failures = [r for r in results if r.get("status") != "complete"]
    print(f"done: {len(results) - len(failures)} complete, {len(failures)} not complete")
    for item in failures:
        print(f"  {item['factor']}: {item.get('status')} {item.get('error', '')}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
