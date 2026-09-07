from __future__ import annotations

import json

import numpy as np
import pandas as pd

from Genetic_Algorithm.evolution import Candidate, SearchResult
from Genetic_Algorithm.expression import Node
from Genetic_Algorithm.config import STAGES
from Genetic_Algorithm import search as search_module
from Genetic_Algorithm.artifacts import write_artifact


REPOSITORY_ROOT = __file__.split("/tests/")[0]


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
        backtest_profile={"fees": 0.001},
        experiment_id="train-exp",
    )

    assert returned is result
    provenance = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    archive = json.loads((tmp_path / "run" / "training_candidates.json").read_text(encoding="utf-8"))
    assert provenance["training_experiment_id"] == "train-exp"
    assert provenance["validation"]["attempts"] == 0
    assert provenance["selected_code_content_hashes"]
    assert provenance["package_versions"]
    assert archive["candidates"][0]["expression_id"] == "candidate-id"


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
    assert reference["path"] == "training_values.json"
    assert len(reference["sha256"]) == 64
    assert (tmp_path / "run" / reference["path"]).exists()


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
            "expression_id": "old",
            "training_fingerprint": "train-fingerprint",
            "operator_version": "ops-v1",
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
