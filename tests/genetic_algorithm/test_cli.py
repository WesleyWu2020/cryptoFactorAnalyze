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


def test_cost_fixed_year_validation_binds_training_and_never_loads_test(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from dataclasses import asdict
    import pandas as pd
    from Genetic_Algorithm.config import SearchConfig
    from Genetic_Algorithm.expression import Node
    from Genetic_Algorithm.artifacts import read_verified_manifest
    from Genetic_Algorithm import cost_fitness
    from Genetic_Algorithm.selection import ValidationSelectionResult

    config = SearchConfig(fitness_mode="all_costs_sharpe", n_groups=5, render_reports=False, validation_stability=True)
    write_artifact(tmp_path / "config.json", asdict(config), immutable=True)
    write_artifact(tmp_path / "training_candidates.json", {"candidates": [
        {"expression_id": "a", "ast": asdict(Node("close")), "training_direction": 1}]}, immutable=True)
    write_artifact(tmp_path / "provenance.json", {"stage_content_hashes": {"train": "train-fingerprint"}}, immutable=True)
    write_artifact(tmp_path / "cost_fitness.json", {"candidates": {"a": {"accounting_fingerprint": "accounting"}}}, immutable=True)
    stages = []
    def load(path, stage, *args, **kwargs):
        stages.append(stage.name)
        assert stage.end < pd.Timestamp("2026-01-01")
        panel = pd.DataFrame(1.0, index=pd.date_range(stage.start, stage.end), columns=["A"])
        return SimpleNamespace(fingerprint=f"{stage.name}-fingerprint", features={"close": panel},
                               opens=panel, eligible=panel.astype(bool), audit={"accounting_fingerprint": "accounting"})
    def cost(values, data, config, direction, start, end, **kwargs):
        assert start == pd.Timestamp("2024-01-01") and end == pd.Timestamp("2024-12-31")
        assert kwargs["include_positions"]
        return {}, values.iloc[:, 0], values
    def select(candidates, outcomes, config, *, trading_evidence):
        assert trading_evidence["a"][0].index.max().year == 2024
        assert outcomes["a"]["net_sharpe"] == 1.5
        assert outcomes["a"]["cost_stability"]["passed"] is True
        return ValidationSelectionResult(tuple(candidates), (), {})
    def stability(values, data, config, direction):
        assert values.index.min() == pd.Timestamp("2025-01-01")
        assert values.index.max() == pd.Timestamp("2025-12-31")
        return {"passed": True, "reasons": []}
    monkeypatch.setattr(cli, "load_stage", load)
    monkeypatch.setattr(cost_fitness, "evaluate_cost_window", cost)
    monkeypatch.setattr(cost_fitness, "validation_cost_stability", stability)
    monkeypatch.setattr(cli, "replay", lambda *a, **k: {"metrics": {"total_return": 0.1, "sharpe": 1.5, "turnover": 0.1}, "artifact_path": "replay.json"})
    monkeypatch.setattr(cli, "_validation_evidence", lambda *a: {"direction": 1})
    monkeypatch.setattr(cli, "select_validation", select)
    cli._validate(argparse.Namespace(run_dir=tmp_path, h5="unused"))
    assert stages == ["train", "validation"]
    result = read_verified_manifest(tmp_path / "validation.json")
    assert result["accepted"] == ["a"]
    assert result["training_accounting_fingerprint"] == "accounting"


def test_test_rejects_a_missing_manifest(tmp_path):
    result = _cli("test", "--manifest", str(tmp_path / "missing.json"), "--h5", "missing.h5")

    assert result.returncode != 0
    assert "manifest" in result.stderr.lower()


@pytest.mark.parametrize("fail_validation", [False, True])
def test_full_bash_defaults_to_fixed_years_and_stops_before_test_on_failure(tmp_path, fail_validation):
    import os
    executable = tmp_path / "fake_python"
    log = tmp_path / "calls"
    executable.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$GP_CALL_LOG\"\n"
                          "if [ \"$3\" = 'validate' ] && [ \"$FAIL_VALIDATION\" = 'yes' ]; then exit 2; fi\n")
    executable.chmod(0o755)
    run = tmp_path / "new-run"
    env = {k: v for k, v in os.environ.items() if not k.startswith("GP_")}
    env.update(GP_PYTHON=str(executable), GP_H5=str(executable), GP_RUN_DIR=str(run),
               GP_CALL_LOG=str(log), FAIL_VALIDATION="yes" if fail_validation else "no")
    result = subprocess.run(["bash", str(ROOT / "tmp/run_full_daily_gp.sh")], cwd=tmp_path,
                            env=env, text=True, capture_output=True)
    calls = log.read_text()
    assert "Genetic_Algorithm walk-forward" not in calls
    assert "Genetic_Algorithm audit --stage train" in calls
    assert "Genetic_Algorithm search" in calls
    assert "Genetic_Algorithm validate" in calls
    if fail_validation:
        assert result.returncode == 2
        assert "Genetic_Algorithm freeze" not in calls
        assert "Genetic_Algorithm test" not in calls
    else:
        assert result.returncode == 0, result.stderr
        assert calls.index("Genetic_Algorithm freeze") < calls.index("Genetic_Algorithm test")
        assert "summarize_gp_test.py --run-dir" in calls


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
