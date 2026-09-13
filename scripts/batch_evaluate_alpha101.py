"""Batch-evaluate all Alpha101 factors (new factor_common contract).

Runs every factor module under factor_analyse/Alpha101/{first50,last51}
through FactorManager.evaluate over 2024-01-01..2026-09-01, renders an HTML
report per factor into reports/, and dumps per-factor performance metrics to
outputs/alpha101_results.json for the summary notebook.

Usage:
    ./.venv/bin/python scripts/batch_evaluate_alpha101.py [--workers 6] [--no-report]
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

FACTOR_DIRS = (
    PROJECT_ROOT / "factor_analyse" / "Alpha101" / "first50",
    PROJECT_ROOT / "factor_analyse" / "Alpha101" / "last51",
)

PARAMS = {
    "start": "2024-01-01",
    "end": "2026-09-01",
    "rebalance_days": 1,
    "n_groups": 5,
    "fee_rate": 0.0005,
    "slippage": 0.001,
    "include_funding": True,
}

RESULTS_PATH = PROJECT_ROOT / "outputs" / "alpha101_results.json"

RETURN_KEYS = ("total_return", "annual_return", "sharpe", "max_drawdown", "win_rate")


def _block_get(block, keys):
    if not isinstance(block, dict):
        return {key: None for key in keys}
    return {key: block.get(key) for key in keys}


def evaluate_one(path_str: str, render: bool) -> dict[str, object]:
    from factor_common import FactorManager

    factor_path = Path(path_str)
    stem = factor_path.stem
    manager = FactorManager(
        h5_path=PROJECT_ROOT / "data" / "crypto_quant.h5",
        base_dir=PROJECT_ROOT / "data" / "factor_results",
        reports_dir=PROJECT_ROOT / "reports",
    )
    summary: dict[str, object] = {"factor": stem, "path": str(factor_path)}
    try:
        result = manager.evaluate(factor_path, params=dict(PARAMS), plot=False)
        summary["status"] = result["status"]
        summary["run_id"] = result["run_id"]

        samples = result["factor_performance"]["samples"]
        for sample_name in ("full", "in_sample", "out_of_sample"):
            ic = samples.get(sample_name, {}).get("ic") or {}
            for key in ("ic_mean", "rank_ic_mean", "icir", "t_stat", "p_value"):
                summary[f"{sample_name}.{key}"] = ic.get(key)

        scenarios = result["factor_performance"].get("scenarios", {})
        for scenario in ("gross", "trading_net", "all_costs"):
            block = scenarios.get(scenario, {}).get("full")
            for key, value in _block_get(block, RETURN_KEYS).items():
                summary[f"{scenario}.{key}"] = value

        validation = result["diagnostics"]["validation"]
        summary["static_scan"] = validation["static_scan"]["status"]
        summary["cutoff"] = validation["cutoff"]["status"]

        if render:
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
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-report", action="store_true", help="跳过 HTML 报告渲染")
    args = parser.parse_args()

    paths = sorted(p for d in FACTOR_DIRS for p in d.glob("*.py"))
    print(f"factors: {len(paths)}  workers: {args.workers}", flush=True)

    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_one, str(p), not args.no_report): p for p in paths
        }
        for future in as_completed(futures):
            summary = future.result()
            results.append(summary)
            print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)

    results.sort(key=lambda r: r["factor"])
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {"params": PARAMS, "results": results},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    print(f"results written: {RESULTS_PATH}")

    failures = [r for r in results if r.get("status") not in ("complete", "incomplete")]
    print(f"done: {len(results) - len(failures)} ok, {len(failures)} failed")
    for item in failures:
        print(f"  {item['factor']}: {item.get('status')} {item.get('error', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
