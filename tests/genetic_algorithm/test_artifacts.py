from __future__ import annotations

import json
import math

import pytest

from Genetic_Algorithm.artifacts import read_verified_manifest, write_artifact


def test_manifest_is_immutable_and_verified(tmp_path):
    path = tmp_path / "run.manifest.json"
    write_artifact(path, {"metrics": {"ic": float("nan")}, "seed": 7})

    payload = read_verified_manifest(path)
    assert payload["metrics"]["ic"] is None
    with pytest.raises(FileExistsError):
        write_artifact(path, {"seed": 8})

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["seed"] = 8
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        read_verified_manifest(path)

def test_nonfinite_values_are_serialized_as_null_and_progress_is_replaceable(tmp_path):
    path = tmp_path / "progress.json"
    write_artifact(path, {"nan": math.nan, "positive": math.inf, "negative": -math.inf})
    write_artifact(path, {"step": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"step": 2}


def test_read_manifest_rejects_corrupt_digest(tmp_path):
    path = tmp_path / "manifest.json"
    write_artifact(path, {"value": 1})
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["sha256"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        read_verified_manifest(path)
