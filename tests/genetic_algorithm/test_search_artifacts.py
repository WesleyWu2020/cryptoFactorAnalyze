from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Genetic_Algorithm.evolution import Candidate, SearchResult
from Genetic_Algorithm.expression import Node
from Genetic_Algorithm.config import STAGES
from Genetic_Algorithm.evolution import search as evolution_search
from Genetic_Algorithm import search as search_module
from Genetic_Algorithm.artifacts import working_tree_patch_hash, write_artifact
from Genetic_Algorithm.search import _load_archive


REPOSITORY_ROOT = __file__.split("/tests/")[0]


def _valid_archive_entry(identifier="candidate", **overrides):
    entry = {
        "expression_id": identifier,
        "training_only": True,
        "ast": {"op": "close", "field": None, "window": None, "children": []},
        "training_fingerprint": "train-fingerprint",
        "operator_version": "ops-v1",
        "training_diagnostics": {"score": [1.0], "eligible": True, "reasons": []},
    }
    entry.update(overrides)
    return entry


def _repack_archive_with_safe_value_name(source, destination):
    document = json.loads(source.read_text(encoding="utf-8"))
    for entry in document["candidates"]:
        reference = entry.get("value_artifact")
        if reference:
            shutil.copy2(source.parent / reference["path"], destination.parent / "prior-values.json")
            entry["value_artifact"] = {"path": "prior-values.json", "sha256": reference["sha256"]}
    write_artifact(destination, document, immutable=True)


def test_run_search_writes_training_artifacts_without_validation(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    candidate = Candidate(Node("close"), "candidate-id", (0.5, 0.4, -1))
    result = SearchResult((candidate,), (), 1)
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    returned = search_module.run_search(
        "unused.h5",
        audit_path,
        stage=STAGES["train"],
        warmup_days=0,
        fields=["close"],
        search_stage=lambda path: result,
        artifact_dir=tmp_path / "run",
        config={"population": 1},
        repository_root=REPOSITORY_ROOT,
        selected_code_paths=(),
        seed=7,
        operator_version="ops-v1",
        backtest_profile={"fee_rate": 0.001},
        experiment_id="train-exp",
    )

    assert returned.candidates == ()
    provenance = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    archive = json.loads((tmp_path / "run" / "training_candidates.json").read_text(encoding="utf-8"))
    assert provenance["training_experiment_id"] == "train-exp"
    assert provenance["validation"]["attempts"] == 0
    assert provenance["selected_code_content_hashes"]
    assert provenance["package_versions"]
    assert archive["candidates"] == []


def test_run_search_persists_real_search_panels_and_compares_on_second_run(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    values = pd.DataFrame(
        np.arange(120 * 24, dtype="float64").reshape(120, 24),
        index=dates,
        columns=[f"S{i:02d}" for i in range(24)],
    )
    search_config = {
        "seed": 3,
        "population": 1,
        "generations": 1,
        "max_depth": 0,
        "max_nodes": 1,
        "initial_trees": [Node("close")],
        "evaluate_candidate": lambda tree, stage_data, labels, config: {
            "score": (1.0, 1.0, -1), "eligible": True, "reasons": (), "values": values,
        },
    }
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    def search_stage(_audit):
        return evolution_search({"marker": "synthetic"}, search_config)

    first = search_module.run_search(
        "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0, fields=["close"],
        search_stage=search_stage, artifact_dir=tmp_path / "first", repository_root=REPOSITORY_ROOT,
        operator_version="ops-v1",
    )
    assert first.values_by_id
    first_archive = tmp_path / "first" / "training_candidates.json"
    safe_archive = tmp_path / "safe-archive.json"
    _repack_archive_with_safe_value_name(first_archive, safe_archive)
    second = search_module.run_search(
        "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0, fields=["close"],
        search_stage=search_stage, artifact_dir=tmp_path / "second", archive_path=safe_archive,
        repository_root=REPOSITORY_ROOT, operator_version="ops-v1",
    )

    assert second.candidates == ()
    dedup = json.loads((tmp_path / "second" / "deduplication.json").read_text(encoding="utf-8"))
    assert any(
        "duplicate" in reason
        for reasons in dedup["rejection_reasons"].values()
        for reason in reasons
    )


def test_run_search_second_archive_is_cumulative_and_preserves_prior_value_reference(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    values = pd.DataFrame(
        np.arange(120 * 24, dtype="float64").reshape(120, 24),
        index=dates,
        columns=[f"S{i:02d}" for i in range(24)],
    )
    first = Candidate(Node("close"), "first", (1.0, 1.0, -1))
    second = Candidate(Node("open"), "second", (1.0, 1.0, -1))
    results = iter(
        (
            SearchResult((first,), (), 1, values_by_id={"first": {"values": values}}),
            SearchResult(
                (second,), (), 1,
                values_by_id={
                    "second": {
                        "values": pd.DataFrame(
                            np.random.default_rng(7).normal(size=(120, 24)),
                            index=dates,
                            columns=values.columns,
                        )
                    }
                },
            ),
        )
    )
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    def run(destination, archive_path=None):
        return search_module.run_search(
            "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0, fields=["close"],
            search_stage=lambda path: next(results), artifact_dir=destination,
            archive_path=archive_path, repository_root=REPOSITORY_ROOT, operator_version="ops-v1",
        )

    run_dir = tmp_path / "first"
    run(run_dir)
    first_document = json.loads((run_dir / "training_candidates.json").read_text(encoding="utf-8"))
    first_entry = first_document["candidates"][0]
    second_dir = tmp_path / "second"
    safe_archive = tmp_path / "safe-archive.json"
    _repack_archive_with_safe_value_name(run_dir / "training_candidates.json", safe_archive)
    run(second_dir, safe_archive)

    document = json.loads((second_dir / "training_candidates.json").read_text(encoding="utf-8"))
    assert [entry["expression_id"] for entry in document["candidates"]] == ["first", "second"]
    assert document["candidates"][0]["value_artifact"] == {
        "path": "prior-values.json", "sha256": first_entry["value_artifact"]["sha256"]
    }


def test_load_archive_rejects_duplicate_expression_ids(tmp_path):
    archive_path = tmp_path / "archive.json"
    write_artifact(
        archive_path,
        {"training_only": True, "candidates": [
            _valid_archive_entry("same"), _valid_archive_entry("same"),
        ]},
        immutable=True,
    )

    with pytest.raises(ValueError, match="duplicate expression_id.*same"):
        _load_archive(archive_path)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"expression_id": "candidate"}, "training_only"),
        ({"expression_id": "candidate", "training_only": True, "validation_metrics": {}}, "non-training"),
    ],
)
def test_load_archive_rejects_unmarked_or_non_training_candidate_entries(
    tmp_path, entry, message
):
    archive_path = tmp_path / "archive.json"
    write_artifact(
        archive_path,
        {"training_only": True, "candidates": [entry]},
        immutable=True,
    )

    with pytest.raises(ValueError, match=message):
        _load_archive(archive_path)


def test_load_archive_rejects_unmarked_value_artifact(tmp_path):
    values_path = tmp_path / "values.json"
    from Genetic_Algorithm.artifacts import write_value_artifact

    value_artifact = write_value_artifact(
        values_path,
        {"candidate": pd.DataFrame([[1.0]], index=pd.date_range("2024-01-01", periods=1), columns=["S0"])},
    )
    raw_value = json.loads(values_path.read_text(encoding="utf-8"))
    raw_value.pop("training_only")
    from Genetic_Algorithm.artifacts import _manifest_payload
    values_path.write_text(json.dumps(_manifest_payload(raw_value)), encoding="utf-8")
    archive_path = tmp_path / "archive.json"
    write_artifact(
        archive_path,
        {"training_only": True, "candidates": [{
            **_valid_archive_entry("candidate"),
            "value_artifact": {"path": values_path.name, "sha256": value_artifact.sha256},
        }]},
        immutable=True,
    )

    with pytest.raises(ValueError, match="training_only"):
        _load_archive(archive_path)

@pytest.mark.parametrize(
    "missing",
    ["expression_id", "training_only", "ast", "training_fingerprint", "operator_version", "training_diagnostics"],
)
def test_load_archive_rejects_missing_required_candidate_fields(tmp_path, missing):
    entry = _valid_archive_entry()
    entry.pop(missing)
    archive_path = tmp_path / "archive.json"
    write_artifact(archive_path, {"training_only": True, "candidates": [entry]}, immutable=True)

    with pytest.raises(ValueError, match="required"):
        _load_archive(archive_path)


@pytest.mark.parametrize(
    "entry",
    [
        _valid_archive_entry(ast={"op": "close", "field": None, "window": None, "children": [], "validation_metrics": {}}),
        _valid_archive_entry(training_diagnostics={"score": [1.0], "eligible": True, "reasons": [], "nested": {"test_metrics": {}}}),
    ],
)
def test_load_archive_rejects_nested_validation_or_test_data(tmp_path, entry):
    archive_path = tmp_path / "archive.json"
    write_artifact(archive_path, {"training_only": True, "candidates": [entry]}, immutable=True)

    with pytest.raises(ValueError, match="non-training"):
        _load_archive(archive_path)


def test_run_search_does_not_publish_partial_directory_when_commit_fails(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    destination = tmp_path / "run"
    destination.mkdir()
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    def fail_replace(*args, **kwargs):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(search_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated publish failure"):
        search_module.run_search(
            "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0,
            fields=["close"], search_stage=lambda path: SearchResult((), (), 0),
            artifact_dir=destination, repository_root=REPOSITORY_ROOT,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".run.*"))


def test_existing_published_run_survives_failure_between_backup_and_stage_publish(
    tmp_path, monkeypatch
):
    audit_path = tmp_path / "run" / "audit_train.json"
    audit_path.parent.mkdir()
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "old-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    destination = audit_path.parent
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)
    real_replace = search_module.os.replace
    failed = False

    def fail_after_backup(source, target):
        nonlocal failed
        if Path(source).is_dir() and Path(target) == destination:
            if not failed:
                failed = True
                raise OSError("simulated crash between backup and stage publish")
        return real_replace(source, target)

    monkeypatch.setattr(search_module.os, "replace", fail_after_backup)
    with pytest.raises(OSError, match="simulated crash"):
        search_module.run_search(
            "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0,
            fields=["close"], search_stage=lambda path: SearchResult((), (), 0),
            artifact_dir=destination, repository_root=REPOSITORY_ROOT,
        )

    assert "old-fingerprint" in (destination / "audit_train.json").read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".run.audit-backup"))
    assert not list(tmp_path.glob(".run.publish.json"))


def test_publish_recovery_restores_backup_after_crash_before_stage_rename(tmp_path):
    destination = tmp_path / "run"
    backup = tmp_path / ".run.audit-backup"
    staging = tmp_path / ".run.staging"
    backup.mkdir()
    (backup / "old").write_text("old", encoding="utf-8")
    staging.mkdir()
    (staging / "partial").write_text("partial", encoding="utf-8")
    journal = tmp_path / ".run.publish.json"
    journal.write_text(
        json.dumps({
            "version": 1,
            "destination": destination.name,
            "backup": backup.name,
            "staging": staging.name,
        }),
        encoding="utf-8",
    )

    search_module._recover_publish_destination(destination)

    assert (destination / "old").read_text(encoding="utf-8") == "old"
    assert not backup.exists()
    assert not staging.exists()
    assert not journal.exists()


def test_publish_recovery_accepts_relative_destination(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    destination = Path("run")
    backup = tmp_path / ".run.audit-backup"
    staging = tmp_path / ".run.staging"
    backup.mkdir()
    (backup / "old").write_text("old", encoding="utf-8")
    staging.mkdir()
    journal = tmp_path / ".run.publish.json"
    journal.write_text(
        json.dumps({
            "version": 1,
            "destination": destination.name,
            "backup": backup.name,
            "staging": staging.name,
        }),
        encoding="utf-8",
    )

    search_module._recover_publish_destination(destination)

    assert (destination / "old").read_text(encoding="utf-8") == "old"
    assert not backup.exists()
    assert not staging.exists()
    assert not journal.exists()


def test_publish_recovery_normalizes_destination_alias(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    destination = alias_parent / "run"
    backup = real_parent / ".run.audit-backup"
    staging = real_parent / ".run.staging"
    backup.mkdir()
    (backup / "old").write_text("old", encoding="utf-8")
    staging.mkdir()
    journal = real_parent / ".run.publish.json"
    journal.write_text(
        json.dumps({
            "version": 1,
            "destination": destination.name,
            "backup": backup.name,
            "staging": staging.name,
        }),
        encoding="utf-8",
    )

    search_module._recover_publish_destination(destination)

    assert (destination / "old").read_text(encoding="utf-8") == "old"
    assert not backup.exists()
    assert not staging.exists()
    assert not journal.exists()


@pytest.mark.parametrize(
    ("backup", "staging"),
    [
        ("run", ".run.staging"),
        (".run.audit-backup", ".run.audit-backup"),
        (".run.audit-backup", "../outside"),
        (".run.publish.json", ".run.staging"),
        (".run.audit-backup", ".run.publish.json"),
    ],
)
def test_publish_recovery_rejects_invalid_journal_paths(tmp_path, backup, staging):
    destination = tmp_path / "run"
    journal = tmp_path / ".run.publish.json"
    journal.write_text(
        json.dumps({
            "version": 1,
            "destination": destination.name,
            "backup": backup,
            "staging": staging,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="publish journal"):
        search_module._recover_publish_destination(destination)

    assert journal.exists()


def test_publish_recovery_rejects_symlink_alias_to_journal_without_deleting_it(tmp_path):
    destination = tmp_path / "run"
    journal = tmp_path / ".run.publish.json"
    alias = tmp_path / ".run.staging"
    alias.symlink_to(journal)
    journal.write_text(
        json.dumps({
            "version": 1,
            "destination": destination.name,
            "backup": None,
            "staging": alias.name,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="publish journal"):
        search_module._recover_publish_destination(destination)

    assert journal.exists()


def test_publish_recovery_rejects_journal_symlink_to_unrelated_parent_file(tmp_path):
    destination = tmp_path / "run"
    journal = tmp_path / ".run.publish.json"
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text(json.dumps({"version": 1}), encoding="utf-8")
    journal.symlink_to(unrelated)

    with pytest.raises(ValueError, match="publish journal"):
        search_module._recover_publish_destination(destination)

    assert journal.is_symlink()
    assert unrelated.exists()


@pytest.mark.parametrize("journal_target_field", ["staging", "backup"])
def test_publish_recovery_rejects_journal_symlink_aliasing_recovery_path(
    tmp_path, journal_target_field
):
    destination = tmp_path / "run"
    staging = tmp_path / ".run.staging"
    backup = tmp_path / ".run.audit-backup"
    staging.write_text("staging", encoding="utf-8")
    backup.write_text("backup", encoding="utf-8")
    journal = tmp_path / ".run.publish.json"
    journal_target = staging if journal_target_field == "staging" else backup
    journal_target.write_text(
        json.dumps({
            "version": 1,
            "destination": destination.name,
            "backup": backup.name,
            "staging": staging.name,
        }),
        encoding="utf-8",
    )
    journal.symlink_to(journal_target)

    with pytest.raises(ValueError, match="publish journal"):
        search_module._recover_publish_destination(destination)

    assert journal.is_symlink()
    assert journal_target.exists()


@pytest.mark.parametrize("journal_target_field", ["staging", "backup"])
@pytest.mark.parametrize("target_kind", ["directory", "file"])
def test_publish_recovery_rejects_symlink_alias_to_sensitive_target(
    tmp_path, journal_target_field, target_kind
):
    destination = tmp_path / "run"
    journal = tmp_path / ".run.publish.json"
    sensitive = tmp_path / f"sensitive-{target_kind}"
    if target_kind == "directory":
        sensitive.mkdir()
        (sensitive / "sentinel").write_text("keep", encoding="utf-8")
    else:
        sensitive.write_text("keep", encoding="utf-8")

    staging = tmp_path / ".run.staging"
    backup = tmp_path / ".run.audit-backup"
    if journal_target_field == "staging":
        staging.symlink_to(sensitive, target_is_directory=target_kind == "directory")
        backup_name = None
    else:
        staging.mkdir()
        backup.symlink_to(sensitive, target_is_directory=target_kind == "directory")
        backup_name = backup.name
    journal.write_text(
        json.dumps({
            "version": 1,
            "destination": destination.name,
            "backup": backup_name,
            "staging": staging.name,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="publish journal"):
        search_module._recover_publish_destination(destination)

    assert sensitive.exists()
    if target_kind == "directory":
        assert (sensitive / "sentinel").read_text(encoding="utf-8") == "keep"
    else:
        assert sensitive.read_text(encoding="utf-8") == "keep"
    assert journal.exists()


def test_run_search_rejects_same_expression_id_with_incompatible_provenance(
    tmp_path, monkeypatch
):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    columns = [f"S{i:02d}" for i in range(24)]
    results = iter(
        (
            SearchResult(
                (Candidate(Node("close"), "same-id", (1.0, 1.0, -1)),), (), 1,
                values_by_id={"same-id": {"values": pd.DataFrame(np.arange(120 * 24, dtype="float64").reshape(120, 24), index=dates, columns=columns)}},
            ),
            SearchResult(
                (Candidate(Node("open"), "same-id", (1.0, 1.0, -1)),), (), 1,
                values_by_id={"same-id": {"values": pd.DataFrame(np.random.default_rng(9).normal(size=(120, 24)), index=dates, columns=columns)}},
            ),
        )
    )
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    def run(destination, archive_path=None, operator_version="ops-v1"):
        return search_module.run_search(
            "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0, fields=["close"],
            search_stage=lambda path: next(results), artifact_dir=destination,
            archive_path=archive_path, repository_root=REPOSITORY_ROOT,
            operator_version=operator_version,
        )

    first_dir = tmp_path / "first"
    run(first_dir)
    second_dir = tmp_path / "second"
    safe_archive = tmp_path / "safe-archive.json"
    _repack_archive_with_safe_value_name(first_dir / "training_candidates.json", safe_archive)
    second = run(second_dir, safe_archive, operator_version="ops-v2")

    assert second.candidates == ()
    dedup = json.loads((second_dir / "deduplication.json").read_text(encoding="utf-8"))
    reason = dedup["rejection_reasons"]["same-id"][0]
    assert "incompatible archive reference same-id" in reason
    archive = json.loads((second_dir / "training_candidates.json").read_text(encoding="utf-8"))
    assert [entry["expression_id"] for entry in archive["candidates"]] == ["same-id"]


@pytest.mark.parametrize("reference", ["../outside.json", "/tmp/outside.json"])
def test_run_search_rejects_archive_artifact_path_outside_archive_directory(tmp_path, reference, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    archive_path = tmp_path / "archive" / "training_candidates.json"
    archive_path.parent.mkdir()
    write_artifact(
        archive_path,
        {"training_only": True, "candidates": [{
            **_valid_archive_entry("old", value_artifact={"path": reference, "sha256": "0" * 64}),
        }]},
        immutable=True,
    )
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    with pytest.raises(ValueError, match="path"):
        search_module.run_search(
            "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0, fields=["close"],
            search_stage=lambda path: SearchResult((), (), 0), artifact_dir=tmp_path / "run",
            archive_path=archive_path, repository_root=REPOSITORY_ROOT,
        )


@pytest.mark.parametrize("reserved_name", [
    "deduplication.json",
    "training_candidates.json",
    "training_values.json",
    "training_values_archive.json",
    "provenance.json",
    ".run.publish.json",
])
def test_run_search_rejects_archive_artifact_path_reserved_by_publication(
    tmp_path, reserved_name, monkeypatch
):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    values_path = archive_dir / reserved_name
    from Genetic_Algorithm.artifacts import write_value_artifact

    value_artifact = write_value_artifact(
        values_path,
        {"old": pd.DataFrame([[1.0]], index=pd.date_range("2024-01-01", periods=1), columns=["S0"])},
    )
    archive_path = archive_dir / "source-archive.json"
    write_artifact(
        archive_path,
        {"training_only": True, "candidates": [{
            **_valid_archive_entry("old"),
            "value_artifact": {"path": reserved_name, "sha256": value_artifact.sha256},
        }]},
        immutable=True,
    )
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    with pytest.raises(ValueError, match="reserved"):
        search_module.run_search(
            "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0,
            fields=["close"], search_stage=lambda path: SearchResult((), (), 0),
            artifact_dir=tmp_path / "run", archive_path=archive_path,
            repository_root=REPOSITORY_ROOT,
        )


def test_run_search_rejects_archive_artifact_path_shared_by_candidates(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    values_path = archive_dir / "shared-values.json"
    from Genetic_Algorithm.artifacts import write_value_artifact

    value_artifact = write_value_artifact(
        values_path,
        {
            "first": pd.DataFrame([[1.0]], index=pd.date_range("2024-01-01", periods=1), columns=["S0"]),
            "second": pd.DataFrame([[2.0]], index=pd.date_range("2024-01-01", periods=1), columns=["S0"]),
        },
    )
    archive_path = archive_dir / "training_candidates.json"
    write_artifact(
        archive_path,
        {"training_only": True, "candidates": [
            _valid_archive_entry("first", value_artifact={"path": values_path.name, "sha256": value_artifact.sha256}),
            _valid_archive_entry("second", value_artifact={"path": values_path.name, "sha256": value_artifact.sha256}),
        ]},
        immutable=True,
    )
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    with pytest.raises(ValueError, match="multiple archive candidates"):
        search_module.run_search(
            "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0,
            fields=["close"], search_stage=lambda path: SearchResult((), (), 0),
            artifact_dir=tmp_path / "run", archive_path=archive_path,
            repository_root=REPOSITORY_ROOT,
        )


def test_run_search_rejects_selected_candidate_without_value_panel(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    candidate = Candidate(Node("close"), "missing-panel", (1.0, 1.0, -1))
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    search_module.run_search(
        "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0, fields=["close"],
        search_stage=lambda path: SearchResult((candidate,), (), 1), artifact_dir=tmp_path / "run",
        repository_root=REPOSITORY_ROOT, operator_version="ops-v1",
    )

    document = json.loads((tmp_path / "run" / "training_candidates.json").read_text(encoding="utf-8"))
    dedup = json.loads((tmp_path / "run" / "deduplication.json").read_text(encoding="utf-8"))
    assert document["candidates"] == []
    assert dedup["accepted"] == []
    assert dedup["rejected"] == ["missing-panel"]
    assert "missing value panel" in dedup["rejection_reasons"]["missing-panel"][0]


def test_run_search_uses_deterministic_default_artifact_dir(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)
    monkeypatch.chdir(tmp_path)

    search_module.run_search(
        "unused.h5",
        audit_path,
        stage=STAGES["train"],
        warmup_days=0,
        fields=["close"],
        search_stage=lambda path: SearchResult((), (), 0),
        repository_root=REPOSITORY_ROOT,
        experiment_id="train-exp",
        operator_version="ops-v1",
    )

    default_dir = tmp_path / "Genetic_Algorithm" / "runs" / "train-exp-train-fingerprint"
    assert (default_dir / "provenance.json").exists()
    assert (default_dir / "training_candidates.json").exists()


def test_run_search_writes_value_artifact_and_deduplicates_with_archive(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    values = pd.DataFrame(
        np.arange(120 * 24, dtype="float64").reshape(120, 24),
        index=dates,
        columns=[f"S{i:02d}" for i in range(24)],
    )
    first = Candidate(Node("close"), "first", (0.8, 0.7, -1))
    duplicate = Candidate(Node("open"), "duplicate", (0.7, 0.6, -1))
    result = SearchResult(
        (first, duplicate), (), 2,
        values_by_id={"first": {"values": values}, "duplicate": {"values": values.copy()}},
    )
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    returned = search_module.run_search(
        "unused.h5",
        audit_path,
        stage=STAGES["train"],
        warmup_days=0,
        fields=["close"],
        search_stage=lambda path: result,
        artifact_dir=tmp_path / "run",
        config={"validation_limit": 1, "min_overlap_days": 120, "correlation_limit": 0.9},
        repository_root=REPOSITORY_ROOT,
        operator_version="ops-v1",
    )

    assert [item.expression_id for item in returned.candidates] == ["first"]
    archive = json.loads((tmp_path / "run" / "training_candidates.json").read_text(encoding="utf-8"))
    entry = archive["candidates"][0]
    reference = entry["value_artifact"]
    assert reference["path"] == "training_values_archive.json"
    assert len(reference["sha256"]) == 64
    assert (tmp_path / "run" / reference["path"]).exists()


def test_run_search_resolves_default_backtest_profile_and_merges_overrides(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    search_module.run_search(
        "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0, fields=["close"],
        search_stage=lambda path: SearchResult((), (), 0), artifact_dir=tmp_path / "run",
        repository_root=REPOSITORY_ROOT, backtest_profile={"fee_rate": 0.001},
    )

    provenance = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    profile = provenance["backtest_profile"]
    assert profile["profile_id"] == "perp_1d"
    assert profile["fee_rate"] == 0.001
    assert profile["slippage"] == 0.001
    assert profile["include_funding"] is True


def test_run_search_provenance_records_effective_config_and_complete_code_paths(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    search_module.run_search(
        "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0, fields=["close"],
        search_stage=lambda path: SearchResult((), (), 0), artifact_dir=tmp_path / "run",
        config={"population": 7}, repository_root=REPOSITORY_ROOT,
    )

    provenance = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["config"]["population"] == 7
    assert provenance["config"]["generations"] == 20
    code_paths = provenance["selected_code_content_hashes"]
    assert "Genetic_Algorithm/config.py" in code_paths
    assert "Genetic_Algorithm/configs/default.json" in code_paths
    assert "Genetic_Algorithm/configs/smoke.json" in code_paths


def test_run_search_fails_before_new_immutable_writes_on_manifest_rerun(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    destination = tmp_path / "run"
    destination.mkdir()
    write_artifact(destination / "provenance.json", {"old": True}, immutable=True)
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    with pytest.raises(FileExistsError):
        search_module.run_search(
            "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0,
            fields=["close"], search_stage=lambda path: SearchResult((), (), 0),
            artifact_dir=destination, repository_root=REPOSITORY_ROOT,
        )

    assert not (destination / "training_values.json").exists()


def test_run_search_provenance_fingerprints_stage_read_sources_and_merges_package_defaults(
    tmp_path, monkeypatch
):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    search_module.run_search(
        "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0, fields=["close"],
        search_stage=lambda path: SearchResult((), (), 0), artifact_dir=tmp_path / "run",
        repository_root=REPOSITORY_ROOT, package_names=("custom-provenance-package",),
    )

    provenance = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    code_paths = provenance["selected_code_content_hashes"]
    for path in (
        "Genetic_Algorithm/data.py",
        "factor_common/data_provider.py",
        "data/crypto_quant/reader.py",
        "data/crypto_quant/store.py",
        "data/crypto_quant/panel.py",
        "data/crypto_quant/schemas.py",
        "data/crypto_quant/config.py",
        "Genetic_Algorithm/config.py",
        "Genetic_Algorithm/configs/default.json",
        "Genetic_Algorithm/configs/smoke.json",
    ):
        assert path in code_paths

    package_versions = provenance["package_versions"]
    for package in (
        "pandas", "numpy", "scipy", "pyecharts", "matplotlib", "python-binance",
        "tqdm", "flask", "requests", "tables", "pyarrow", "pytest",
        "nbformat", "nbclient", "ipykernel", "custom-provenance-package",
    ):
        assert package in package_versions


def test_working_tree_patch_hash_changes_for_untracked_file(tmp_path):
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "fixture"], check=True)
    after = working_tree_patch_hash(tmp_path)
    marker = tmp_path / "untracked-marker.txt"
    marker.write_text("new relevant source\n", encoding="utf-8")
    added = working_tree_patch_hash(tmp_path)
    marker.write_text("changed relevant source\n", encoding="utf-8")
    changed = working_tree_patch_hash(tmp_path)
    assert after != added
    assert added != changed


def test_run_search_loads_existing_archive_for_novelty(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit_train.json"
    audit_path.write_text(
        json.dumps({"training_only": True, "fingerprint": "train-fingerprint", "stage": {"name": "train"}}),
        encoding="utf-8",
    )
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    values = pd.DataFrame(
        np.arange(120 * 24, dtype="float64").reshape(120, 24),
        index=dates,
        columns=[f"S{i:02d}" for i in range(24)],
    )
    values_path = tmp_path / "old-values.json"
    from Genetic_Algorithm.artifacts import write_value_artifact
    value_artifact = write_value_artifact(values_path, {"old": values})
    archive_path = tmp_path / "old-archive.json"
    write_artifact(
        archive_path,
        {"training_only": True, "candidates": [{
            **_valid_archive_entry("old"),
            "value_artifact": {"path": values_path.name, "sha256": value_artifact.sha256},
        }]},
        immutable=True,
    )
    candidate = Candidate(Node("close"), "new", (0.8, 0.7, -1))
    result = SearchResult((candidate,), (), 1, values_by_id={"new": {"values": values}})
    monkeypatch.setattr(search_module, "run_training_audit", lambda *args, **kwargs: audit_path)

    returned = search_module.run_search(
        "unused.h5", audit_path, stage=STAGES["train"], warmup_days=0, fields=["close"],
        search_stage=lambda path: result, artifact_dir=tmp_path / "run", archive_path=archive_path,
        config={"validation_limit": 20, "min_overlap_days": 120}, repository_root=REPOSITORY_ROOT,
        operator_version="ops-v1",
    )

    assert returned.candidates == ()
    dedup = json.loads((tmp_path / "run" / "deduplication.json").read_text(encoding="utf-8"))
    assert "duplicate" in dedup["rejection_reasons"]["new"][0]
