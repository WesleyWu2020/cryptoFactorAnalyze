from __future__ import annotations

import json
import math
import os

import pytest

from Genetic_Algorithm.artifacts import (
    build_provenance,
    read_verified_manifest,
    read_verified_value_artifact,
    write_artifact,
    write_value_artifact,
)
from Genetic_Algorithm.expression import Node

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


def test_value_artifact_rejects_serialized_column_label_collisions(tmp_path):
    values = pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.date_range("2024-01-01", periods=1),
        columns=["1", 1],
    )

    with pytest.raises(ValueError, match="column label collision"):
        write_value_artifact(tmp_path / "values.json", {"candidate": values})


def test_immutable_write_is_atomic_and_cleans_temp_on_link_failure(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"

    def fail_link(*args, **kwargs):
        raise OSError("simulated link failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="simulated link failure"):
        write_artifact(path, {"value": 1}, immutable=True)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_provenance_rejects_code_paths_outside_or_through_symlink(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    inside = repository / "inside.py"
    inside.write_text("value = 1\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("value = 2\n", encoding="utf-8")
    escape = repository / "escape.py"
    escape.symlink_to(outside)
    monkeypatch.setattr("Genetic_Algorithm.artifacts.working_tree_patch_hash", lambda root: "patch")

    kwargs = dict(
        config={}, stage_content_hashes={}, repository_root=repository,
        package_names=(), seed=1, operator_version="ops", backtest_profile={},
        experiment_id="exp",
    )
    with pytest.raises(ValueError, match="outside repository"):
        build_provenance(selected_code_paths=[outside], **kwargs)
    with pytest.raises(ValueError, match="traverse"):
        build_provenance(selected_code_paths=["../outside.py"], **kwargs)
    with pytest.raises(ValueError, match="outside repository"):
        build_provenance(selected_code_paths=[escape], **kwargs)


def test_provenance_preserves_runtime_inputs_with_deterministic_identities(tmp_path, monkeypatch):
    monkeypatch.setattr("Genetic_Algorithm.artifacts.working_tree_patch_hash", lambda root: "patch")

    def evaluator(tree, stage_data, labels, config):
        return {"score": (1.0,), "values": pd.DataFrame()}

    from Genetic_Algorithm.search import _resolved_config

    resolved = _resolved_config({"initial_trees": [Node("close")], "evaluate_candidate": evaluator})

    assert resolved["initial_trees"]["sha256"]
    assert resolved["initial_trees"]["trees"] == [{
        "op": "close", "field": None, "window": None, "children": []
    }]
    identity = resolved["evaluate_candidate"]
    assert identity["module"] == __name__
    assert identity["qualname"].endswith("evaluator")
    assert identity["source"]
    assert identity["sha256"]
