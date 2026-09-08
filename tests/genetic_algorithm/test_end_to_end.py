from __future__ import annotations

import json
import subprocess
import sys

from Genetic_Algorithm.export import export_factor
from Genetic_Algorithm.expression import Node


def test_verifier_writes_machine_readable_result_for_no_exports(gp_h5, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = subprocess.run([sys.executable, "scripts/verify_daily_gp.py", "--run-dir", str(run_dir), "--h5", str(gp_h5)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((run_dir / "verification.json").read_text())
    assert report["status"] == "passed"
    assert report["static_findings"] == []
    assert report["failures"] == []


def test_verifier_rejects_static_future_operation_and_retains_json(gp_h5, tmp_path):
    run_dir = tmp_path / "run"; run_dir.mkdir()
    (run_dir / "GP_bad.py").write_text("import pandas as pd\ndef bad(x): return x.bfill()\n")
    result = subprocess.run([sys.executable, "scripts/verify_daily_gp.py", "--run-dir", str(run_dir), "--h5", str(gp_h5)], text=True, capture_output=True)
    report = json.loads((run_dir / "verification.json").read_text())
    assert result.returncode != 0
    assert report["status"] == "failed"
    assert report["static_findings"][0]["pattern"] == "bfill"


def test_verifier_executes_wrapper_and_retains_dynamic_failure(gp_h5, tmp_path):
    run_dir = tmp_path / "run"; exports = run_dir / "exports"
    export_factor({"ast": {"op": "close", "field": None, "window": None, "children": []}}, exports, direction=1)
    wrapper = next(exports.glob("GP_*.py"))
    # Valid Python with no banned static pattern: only wrapper execution can
    # expose this fault, which guards against AST-only verifier shortcuts.
    wrapper.write_text("def calc_factor(data_ctx):\n    raise RuntimeError('wrapper executed')\n")
    result = subprocess.run([sys.executable, "scripts/verify_daily_gp.py", "--run-dir", str(run_dir), "--h5", str(gp_h5)], text=True, capture_output=True)
    report = json.loads((run_dir / "verification.json").read_text())
    assert result.returncode != 0
    assert report["status"] == "failed"
    assert report["failures"]
    assert "wrapper sha256" in report["failures"][0]["error"]


def test_verifier_rejects_tampered_export_manifest_before_integrity_checks(gp_h5, tmp_path):
    run_dir = tmp_path / "run"; exports = run_dir / "exports"
    export_factor({"ast": {"op": "close", "field": None, "window": None, "children": []}}, exports, direction=1)
    manifest = next(exports.glob("*.export_manifest.json"))
    payload = json.loads(manifest.read_text())
    payload["wrapper_sha256"] = "0" * 64  # leave signed digest stale
    manifest.write_text(json.dumps(payload))
    result = subprocess.run([sys.executable, "scripts/verify_daily_gp.py", "--run-dir", str(run_dir), "--h5", str(gp_h5)], text=True, capture_output=True)
    report = json.loads((run_dir / "verification.json").read_text())
    assert result.returncode != 0
    assert report["status"] == "failed"
    assert "hash verification" in report["failures"][0]["error"]


def test_verifier_happy_path_executes_exported_wrapper_with_zero_cutoff_diff(gp_h5, tmp_path):
    run_dir = tmp_path / "run"; exports = run_dir / "exports"
    export_factor({"ast": {"op": "close", "field": None, "window": None, "children": []}}, exports, direction=1)
    result = subprocess.run([sys.executable, "scripts/verify_daily_gp.py", "--run-dir", str(run_dir), "--h5", str(gp_h5)], text=True, capture_output=True)
    report = json.loads((run_dir / "verification.json").read_text())
    assert result.returncode == 0, result.stderr
    assert report["status"] == "passed"
    assert report["dynamic"][0]["status"] == "verified"
    assert [entry["max_abs_diff"] for entry in report["dynamic"][0]["cutoffs"]] == [0.0, 0.0, 0.0]


def test_synthetic_no_candidate_cli_lifecycle_retains_audit_validation_and_baseline(gp_h5, tmp_path):
    """Exercise the actual audit/validate CLIs without bypassing holdout gates."""
    from dataclasses import asdict
    from Genetic_Algorithm.artifacts import write_artifact
    from Genetic_Algorithm.config import SearchConfig

    run_dir = tmp_path / "no_candidates"; run_dir.mkdir()
    audit = subprocess.run([sys.executable, "-m", "Genetic_Algorithm", "audit", "--stage", "train", "--h5", str(gp_h5), "--output", str(run_dir / "audit_train.json"), "--warmup-days", "180"], text=True, capture_output=True)
    assert audit.returncode == 0, audit.stderr
    config = asdict(SearchConfig(population=2, generations=1))
    write_artifact(run_dir / "config.json", config, immutable=True)
    write_artifact(run_dir / "training_candidates.json", {"candidates": []}, immutable=True)
    write_artifact(run_dir / "provenance.json", {"stage_content_hashes": {"train": "synthetic-train"}, "backtest_profile": {}}, immutable=True)
    validate = subprocess.run([sys.executable, "-m", "Genetic_Algorithm", "validate", "--run-dir", str(run_dir), "--h5", str(gp_h5)], text=True, capture_output=True)
    assert validate.returncode == 0, validate.stderr
    validation = json.loads((run_dir / "validation.json").read_text())
    assert validation["status"] == "no_candidates"
    assert validation["accepted"] == []
    assert validation["validation_fingerprint"]
    freeze = subprocess.run([sys.executable, "-m", "Genetic_Algorithm", "freeze", "--run-dir", str(run_dir)], text=True, capture_output=True)
    assert freeze.returncode == 0, freeze.stderr
    frozen = json.loads((run_dir / "frozen.json").read_text())
    assert frozen["candidates"] == []
    test = subprocess.run([sys.executable, "-m", "Genetic_Algorithm", "test", "--manifest", str(run_dir / "frozen.json"), "--h5", str(gp_h5)], text=True, capture_output=True)
    assert test.returncode == 0, test.stderr
    exported = subprocess.run([sys.executable, "-m", "Genetic_Algorithm", "export", "--manifest", str(run_dir / "frozen.json"), "--output-dir", str(run_dir / "empty_exports")], text=True, capture_output=True)
    assert exported.returncode == 0, exported.stderr
    # Fixed causal baseline is exported then independently cutoff-verified;
    # it is not admitted as a GP survivor or used to relax validation gates.
    exports = run_dir / "baseline_export"
    export_factor({"ast": {"op": "close", "field": None, "window": None, "children": []}}, exports, direction=1)
    baseline = subprocess.run([sys.executable, "scripts/verify_daily_gp.py", "--run-dir", str(run_dir), "--h5", str(gp_h5)], text=True, capture_output=True)
    assert baseline.returncode == 0, baseline.stderr
    assert json.loads((run_dir / "verification.json").read_text())["status"] == "passed"
