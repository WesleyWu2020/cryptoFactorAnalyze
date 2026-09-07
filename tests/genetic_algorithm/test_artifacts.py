from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess

import pytest

from Genetic_Algorithm.artifacts import (
    build_provenance,
    read_verified_manifest,
    read_verified_value_artifact,
    write_artifact,
    write_value_artifact,
    working_tree_patch_hash,
)
from Genetic_Algorithm.expression import Node

import numpy as np
import pandas as pd


def test_working_tree_patch_hash_is_deterministic_without_git(tmp_path):
    first = working_tree_patch_hash(tmp_path)
    second = working_tree_patch_hash(tmp_path)

    assert first == second
    assert len(first) == 64


def test_working_tree_patch_hash_changes_for_different_non_git_trees(tmp_path):
    first_tree = tmp_path / "first"
    second_tree = tmp_path / "second"
    first_tree.mkdir()
    second_tree.mkdir()
    (first_tree / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    (second_tree / "source.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert working_tree_patch_hash(first_tree) != working_tree_patch_hash(second_tree)


def test_non_git_hash_remains_sensitive_to_late_files(tmp_path):
    (tmp_path / "early.txt").write_text("a", encoding="utf-8")
    late = tmp_path / "late.txt"
    late.write_text("before", encoding="utf-8")

    before = working_tree_patch_hash(tmp_path)
    late.write_text("after!", encoding="utf-8")

    assert working_tree_patch_hash(tmp_path) != before


def test_non_git_hash_includes_symlink_target_and_does_not_follow_symlink_directory(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("one", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    linked_dir = tmp_path / "linked-dir"
    linked_dir.mkdir()
    (linked_dir / "hidden.txt").write_text("hidden", encoding="utf-8")
    directory_link = tmp_path / "directory-link"
    directory_link.symlink_to(linked_dir, target_is_directory=True)

    first = working_tree_patch_hash(tmp_path)
    link.unlink()
    link.symlink_to(tmp_path / "other.txt")

    assert working_tree_patch_hash(tmp_path) != first


def test_git_hash_does_not_read_untracked_symlink_target_outside_repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "--allow-empty", "-qm", "fixture"], check=True)
    outside = tmp_path.parent / "outside-content.txt"
    outside.write_text("one", encoding="utf-8")
    link = tmp_path / "untracked-link.txt"
    link.symlink_to(outside)

    first = working_tree_patch_hash(tmp_path)
    outside.write_text("two", encoding="utf-8")

    assert working_tree_patch_hash(tmp_path) == first


def test_git_hash_streams_large_untracked_files_without_read_bytes(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "--allow-empty", "-qm", "fixture"], check=True)
    payload = bytes(range(256)) * 8192
    (tmp_path / "large.bin").write_bytes(payload)

    def fail_read_bytes(self):
        raise AssertionError("working-tree hashing must stream file contents")

    monkeypatch.setattr(type(tmp_path), "read_bytes", fail_read_bytes)

    first = working_tree_patch_hash(tmp_path)
    (tmp_path / "large.bin").write_bytes(payload + b"changed")

    assert working_tree_patch_hash(tmp_path) != first


def test_non_git_hash_ignores_timestamp_only_changes(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("content", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(source)

    before = working_tree_patch_hash(tmp_path)
    os.utime(source, ns=(1_000_000_000, 1_000_000_000))
    os.utime(link, ns=(2_000_000_000, 2_000_000_000), follow_symlinks=False)

    assert working_tree_patch_hash(tmp_path) == before


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


def test_manifest_verification_rejects_nonfinite_json_token_even_with_normalized_digest(tmp_path):
    path = tmp_path / "manifest.json"
    unsigned = {"metrics": {"ic": None}, "seed": 7}
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    path.write_text('{"metrics":{"ic":NaN},"seed":7,"sha256":"' + digest + '"}', encoding="utf-8")

    with pytest.raises(ValueError, match="nonfinite"):
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


def test_value_artifact_verification_rejects_nonfinite_json_token_even_with_normalized_digest(tmp_path):
    path = tmp_path / "training_values.json"
    artifact = write_value_artifact(
        path,
        {"candidate": pd.DataFrame([[1.0]], index=pd.date_range("2024-01-01", periods=1), columns=["A"])},
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["panels"]["candidate"]["values"][0][0] = None
    unsigned = json.loads(json.dumps(raw))
    unsigned.pop("sha256")
    raw["panels"]["candidate"]["values"][0][0] = float("nan")
    raw["sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    path.write_text(json.dumps(raw, allow_nan=True), encoding="utf-8")

    with pytest.raises(ValueError, match="nonfinite"):
        read_verified_value_artifact(path, expected_sha256=raw["sha256"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("validation_metrics", {"ic": 0.9}),
        ("metadata", {"test_metrics": {"ic": 0.8}, "labels": [0.1]}),
    ],
)
def test_value_artifact_rejects_non_training_panel_fields_even_with_recomputed_digest(
    tmp_path, field, value
):
    path = tmp_path / "training_values.json"
    write_value_artifact(
        path,
        {"candidate": pd.DataFrame([[1.0]], index=pd.date_range("2024-01-01", periods=1), columns=["A"])},
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["panels"]["candidate"][field] = value
    unsigned = dict(raw)
    unsigned.pop("sha256")
    raw["sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="panel contains non-training fields"):
        read_verified_value_artifact(path)


def test_value_artifact_rejects_serialized_column_label_collisions(tmp_path):
    values = pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.date_range("2024-01-01", periods=1),
        columns=["1", 1],
    )

    with pytest.raises(ValueError, match="column label collision"):
        write_value_artifact(tmp_path / "values.json", {"candidate": values})


def test_value_artifact_normalizes_mixed_column_labels_before_sorting(tmp_path):
    values = pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.date_range("2024-01-01", periods=1),
        columns=["A", 1],
    )

    artifact = write_value_artifact(tmp_path / "values.json", {"candidate": values})

    assert read_verified_value_artifact(artifact.path)["candidate"].columns.tolist() == ["1", "A"]


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
