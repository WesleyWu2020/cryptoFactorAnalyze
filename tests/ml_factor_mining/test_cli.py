from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from ML_factor_mining.cli import (
    _resolve_path,
    _runtime_source_files,
    _validate_cutoffs,
    build_parser,
    canonical_relative,
    main,
)
from ML_factor_mining.validation import (
    assert_frozen_inputs,
    replay_model,
    scan_source,
    sha256,
    verify_hashes,
)


def test_parser_has_run_and_verify_commands():
    parser = build_parser()
    assert parser.parse_args(["run", "--config", "config.json", "--factor-dir", "f", "--h5", "x.h5", "--output", "run"]).command == "run"
    assert parser.parse_args(["verify", "run"]).command == "verify"


def test_canonical_relative_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        canonical_relative(tmp_path.parent / "outside", tmp_path)


def test_cli_relative_paths_use_repository_root_not_cwd(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert _resolve_path("data/file", repo) == repo / "data/file"


def test_verify_static_scan_excludes_framework_provenance_sources(tmp_path):
    manifest = {
        "code": {
            "files": [
                {"path": "ML_factor_mining/training.py", "sha256": "x"},
                {"path": "factor_common/labels.py", "sha256": "y"},
            ]
        }
    }
    assert _runtime_source_files(manifest, tmp_path) == [tmp_path / "ML_factor_mining/training.py"]


def test_verify_rejects_cutoff_before_oos_start():
    with pytest.raises(ValueError, match="oos_start"):
        _validate_cutoffs(["2024-12-31"], "2025-01-01")


def test_compare_cutoff_does_not_verify_empty_prefixes():
    from ML_factor_mining.validation import compare_cutoff

    empty = pd.DataFrame(columns=["date", "instrument", "factor"])
    result = compare_cutoff(empty, empty, "2024-12-31")
    assert result["status"] == "failed"


def test_scan_source_finds_only_forbidden_feature_patterns(tmp_path):
    source = tmp_path / "factor.py"
    source.write_text("factor = close.rolling(3, center=True).mean()\n", encoding="utf-8")
    findings = scan_source(source)
    assert findings and findings[0]["pattern"] == "rolling_center"


def test_scan_source_finds_negative_shift_keyword(tmp_path):
    source = tmp_path / "factor.py"
    source.write_text("factor = close.shift(periods=-1)\n", encoding="utf-8")
    findings = scan_source(source)
    assert findings and findings[0]["pattern"] == "negative_shift"


def _valid_manifest(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    input_file = inputs / "x"
    input_file.write_text("snapshot", encoding="utf-8")
    code_file = tmp_path / "code.py"
    code_file.write_text("print('ok')\n", encoding="utf-8")
    config = {"models": ["linear"]}
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "config": config,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "hashes": {"x": sha256(input_file)},
        "code": {"files": [{"path": "code.py", "sha256": sha256(code_file)}]},
    }


@pytest.mark.parametrize("manifest", [
    {"hashes": {}, "code": {"files": [{"path": "code.py", "sha256": "x"}]}},
    {"hashes": {"x": "x"}, "code": {"files": []}},
])
def test_verify_hashes_rejects_incomplete_provenance(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="(hashes|code.files)"):
        verify_hashes(tmp_path, code_root=tmp_path)


def test_verify_hashes_detects_manifest_config_tampering(tmp_path):
    manifest = _valid_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_hashes(tmp_path, code_root=tmp_path)["status"] == "verified"
    manifest["config"]["models"] = ["ridge"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_hashes(tmp_path, code_root=tmp_path)
    assert result["status"] == "failed"
    assert any(item["path"] == "manifest.config" for item in result["failures"])
    with pytest.raises(ValueError, match="manifest.config"):
        assert_frozen_inputs(tmp_path)


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
