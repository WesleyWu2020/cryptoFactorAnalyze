"""Black-box contract tests for the Genetic_Algorithm command line."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
from pathlib import Path

from Genetic_Algorithm.artifacts import write_artifact
from Genetic_Algorithm import cli
import pytest


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"


def _cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON), "-m", "Genetic_Algorithm", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_lists_the_six_pipeline_commands():
    result = _cli("--help")

    assert result.returncode == 0
    for command in ("audit", "search", "validate", "freeze", "test", "export"):
        assert command in result.stdout


def test_test_rejects_a_missing_manifest(tmp_path):
    result = _cli("test", "--manifest", str(tmp_path / "missing.json"), "--h5", "missing.h5")

    assert result.returncode != 0
    assert "manifest" in result.stderr.lower()


def test_search_rejects_invalid_configuration_before_reading_data(tmp_path):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({"population": 0}), encoding="utf-8")

    result = _cli("search", "--config", str(config), "--h5", "missing.h5", "--run-dir", str(tmp_path / "run"))

    assert result.returncode != 0
    assert "population" in result.stderr.lower()


def test_audit_rejects_non_training_stage(tmp_path):
    result = _cli("audit", "--stage", "validation", "--h5", "missing.h5", "--output", str(tmp_path / "audit.json"))

    assert result.returncode != 0
    assert "train" in result.stderr.lower()


def test_validate_empty_training_archive_is_a_successful_no_candidate_outcome(tmp_path, gp_h5):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_artifact(run_dir / "training_candidates.json", {"training_only": True, "candidates": []}, immutable=True)
    write_artifact(run_dir / "provenance.json", {"backtest_profile": {}}, immutable=True)
    write_artifact(run_dir / "config.json", json.loads((ROOT / "Genetic_Algorithm/configs/smoke.json").read_text()), immutable=True)

    result = _cli("validate", "--run-dir", str(run_dir), "--h5", str(gp_h5))

    assert result.returncode == 0, result.stderr
    assert "no_candidates" in result.stdout
    assert (run_dir / "validation.json").is_file()
    assert json.loads((run_dir / "validation.json").read_text())["validation_fingerprint"]


@pytest.mark.parametrize("status", ["partial", "failed"])
def test_test_command_returns_nonzero_for_incomplete_replay_outcomes(monkeypatch, status):
    monkeypatch.setattr(cli, "_test", lambda args: {"status": status, "receipt": "controlled"})
    assert cli.main(["test", "--manifest", "controlled.json", "--h5", "controlled.h5"]) == 2


def test_freeze_rejects_tampered_validation_artifact(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_artifact(run_dir / "validation.json", {"accepted": [], "validation_fingerprint": "validation"}, immutable=True)
    validation = run_dir / "validation.json"
    payload = json.loads(validation.read_text())
    payload["accepted"] = ["injected"]
    validation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash verification"):
        cli._freeze(argparse.Namespace(run_dir=str(run_dir)))


def test_validation_metrics_use_all_costs_total_return_not_a_missing_alias():
    metrics = cli._all_costs_validation_metrics(
        {"total_return": 0.911753, "sharpe": 1.8495, "turnover": 0.1088}
    )

    assert metrics == {
        "all_costs_cumulative_return": 0.911753,
        "net_sharpe": 1.8495,
        "turnover": 0.1088,
    }
