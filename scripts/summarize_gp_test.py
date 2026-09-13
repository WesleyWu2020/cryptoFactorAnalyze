#!/usr/bin/env python3
"""Summarize immutable GP validation and 2026 test artifacts without selecting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Genetic_Algorithm.artifacts import read_verified_manifest, write_artifact


def _passed_test(metrics: object, min_sharpe: float) -> bool:
    if not isinstance(metrics, dict):
        return False
    total_return = metrics.get("total_return")
    sharpe = metrics.get("sharpe")
    return (
        isinstance(total_return, (int, float))
        and isinstance(sharpe, (int, float))
        and math.isfinite(total_return)
        and math.isfinite(sharpe)
        and total_return > 0.0
        and sharpe > min_sharpe
    )


def summarize(run_dir: Path) -> Path:
    config = read_verified_manifest(run_dir / "config.json")
    validation = read_verified_manifest(run_dir / "validation.json")
    receipts = sorted((run_dir / "test_receipts").glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
    if not receipts:
        raise FileNotFoundError("no 2026 test receipt exists")
    receipt = read_verified_manifest(receipts[-1])
    min_sharpe = float(config["min_all_costs_sharpe"])
    validation_accepted = set(validation.get("accepted", []))
    candidates = []
    for item in receipt.get("outcomes", []):
        expression_id = item["expression_id"]
        outcome = item.get("outcome", {})
        metrics = outcome.get("all_costs_metrics") if isinstance(outcome, dict) else None
        test_passed = item.get("status") == "complete" and _passed_test(metrics, min_sharpe)
        candidates.append({
            "expression_id": expression_id,
            "passed_2025_validation": expression_id in validation_accepted,
            "passed_2026_test_gate": test_passed,
            "passed_both": expression_id in validation_accepted and test_passed,
            "test_metrics": metrics,
            "test_status": item.get("status"),
        })
    output = run_dir / "test_summary.json"
    return write_artifact(output, {
        "summary_version": 1,
        "selection_disclosure": "2026 metrics are reported only and do not change frozen candidate selection",
        "research_disclosure": "2026 was previously inspected in this project; these results are repeat research, not unseen holdout evidence",
        "test_receipt": str(receipts[-1]),
        "min_all_costs_sharpe": min_sharpe,
        "frozen_candidate_count": len(validation_accepted),
        "passed_both_count": sum(item["passed_both"] for item in candidates),
        "candidates": candidates,
    }, immutable=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(summarize(Path(args.run_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
