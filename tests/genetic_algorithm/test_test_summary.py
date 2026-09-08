from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from Genetic_Algorithm.artifacts import write_artifact


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "summarize_gp_test", ROOT / "scripts" / "summarize_gp_test.py"
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_summary_reports_both_year_gate_without_changing_frozen_selection(tmp_path):
    write_artifact(
        tmp_path / "config.json", {"min_all_costs_sharpe": 1.0}, immutable=True
    )
    write_artifact(
        tmp_path / "validation.json", {"accepted": ["good", "bad"]}, immutable=True
    )
    receipt_dir = tmp_path / "test_receipts"
    write_artifact(
        receipt_dir / "receipt.json",
        {
            "outcomes": [
                {"expression_id": "good", "status": "complete", "outcome": {"all_costs_metrics": {"total_return": 0.2, "sharpe": 1.1}}},
                {"expression_id": "bad", "status": "complete", "outcome": {"all_costs_metrics": {"total_return": 0.3, "sharpe": 0.9}}},
            ]
        },
        immutable=True,
    )

    output = SUMMARY.summarize(tmp_path)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["frozen_candidate_count"] == 2
    assert payload["passed_both_count"] == 1
    assert [item["expression_id"] for item in payload["candidates"] if item["passed_both"]] == ["good"]
    assert "do not change frozen candidate selection" in payload["selection_disclosure"]
