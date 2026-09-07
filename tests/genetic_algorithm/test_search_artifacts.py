from __future__ import annotations

import json

from Genetic_Algorithm.evolution import Candidate, SearchResult
from Genetic_Algorithm.expression import Node
from Genetic_Algorithm.config import STAGES
from Genetic_Algorithm import search as search_module


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
