from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ML_factor_mining.cli import build_parser, canonical_relative, main
from ML_factor_mining.validation import replay_model, scan_source


def test_parser_has_run_and_verify_commands():
    parser = build_parser()
    assert parser.parse_args(["run", "--config", "config.json", "--factor-dir", "f", "--h5", "x.h5", "--output", "run"]).command == "run"
    assert parser.parse_args(["verify", "run"]).command == "verify"


def test_canonical_relative_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        canonical_relative(tmp_path.parent / "outside", tmp_path)


def test_scan_source_finds_only_forbidden_feature_patterns(tmp_path):
    source = tmp_path / "factor.py"
    source.write_text("factor = close.rolling(3, center=True).mean()\n", encoding="utf-8")
    findings = scan_source(source)
    assert findings and findings[0]["pattern"] == "rolling_center"


def test_replay_model_retrains_at_cutoff_instead_of_slicing():
    calls = []

    def retrain(cutoff):
        calls.append(pd.Timestamp(cutoff))
        return pd.DataFrame(
            {"date": pd.to_datetime(["2025-01-01"]), "instrument": ["A"], "factor": [float(len(calls))]}
        )

    result = replay_model(retrain, "2025-01-01")
    assert calls == [pd.Timestamp("2025-01-01")]
    assert result.iloc[0]["factor"] == 1.0


def test_main_failure_writes_state_json(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"models": ["missing"]}), encoding="utf-8")
    run_dir = tmp_path / "run"
    assert main(["run", "--config", str(config), "--factor-dir", str(tmp_path), "--h5", str(tmp_path / "missing.h5"), "--output", str(run_dir)]) != 0
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["error"]["type"]
