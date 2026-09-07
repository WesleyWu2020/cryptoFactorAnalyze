from __future__ import annotations

import json
import math

import pytest

from Genetic_Algorithm.artifacts import (
    read_verified_manifest,
    read_verified_value_artifact,
    write_artifact,
    write_value_artifact,
)

import numpy as np
import pandas as pd


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


def test_value_artifact_is_canonical_hashed_and_immutable(tmp_path):
    values = {
        "candidate": pd.DataFrame(
            [[1.0, np.nan], [2.0, 3.0]],
            index=pd.date_range("2024-01-01", periods=2),
            columns=["A", "B"],
        )
    }
    path = tmp_path / "training_values.json"

    artifact = write_value_artifact(path, values)
    assert artifact.path == path
    assert len(artifact.sha256) == 64
    assert artifact.sha256 in path.read_text(encoding="utf-8")
    loaded = read_verified_value_artifact(path, expected_sha256=artifact.sha256)
    pd.testing.assert_frame_equal(loaded["candidate"], values["candidate"])

    with pytest.raises(FileExistsError):
        write_value_artifact(path, values)

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["panels"]["candidate"]["values"][0][0] = 99.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        read_verified_value_artifact(path)
