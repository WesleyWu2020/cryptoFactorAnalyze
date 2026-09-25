from dataclasses import asdict
import json

import numpy as np
import pandas as pd

from Genetic_Algorithm import experiment
from Genetic_Algorithm.artifacts import (
    read_verified_manifest, training_archive_entry, write_artifact,
    write_value_artifact,
)
from Genetic_Algorithm.config import SearchConfig
from Genetic_Algorithm.evolution import Candidate
from Genetic_Algorithm.expression import Node


def test_multi_seed_merge_uses_only_current_candidates_and_tracks_sources(tmp_path, monkeypatch):
    config = SearchConfig(
        fitness_mode="all_costs_sharpe", n_groups=5, render_reports=False,
        stability_mode="continuous_leave_best_out", population=2, generations=1,
        validation_limit=20,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")

    def audit(_h5, output, **_kwargs):
        write_artifact(output, {"fingerprint": "train-fingerprint"}, immutable=False)
        return output

    monkeypatch.setattr(experiment, "run_training_audit", audit)
    root = tmp_path / "experiment"
    experiment.initialize_experiment(config_path, "unused.h5", root, [7, 8])

    dates = pd.date_range("2024-01-01", periods=150)
    rng = np.random.default_rng(9)
    for seed, op in ((7, "close"), (8, "open")):
        run = root / f"seed_{seed}" / "run"
        run.mkdir()
        seed_config = {**asdict(config), "seed": seed}
        write_artifact(run / "config.json", seed_config, immutable=True)
        write_artifact(run / "provenance.json", {
            "stage_content_hashes": {"train": "train-fingerprint"},
            "backtest_profile": {"n_groups": 5},
        }, immutable=True)
        candidate = Candidate(Node(op), f"candidate-{seed}", (1.0 + seed / 100, 0.5, -1), direction=1)
        values = pd.DataFrame(rng.normal(size=(150, 24)), index=dates)
        value_artifact = write_value_artifact(
            run / "training_values_archive.json", {
                candidate.expression_id: values,
                f"historical-{seed}": values,
            },
        )
        entry = training_archive_entry(
            candidate, training_fingerprint="train-fingerprint", operator_version="ops-v1",
            diagnostics={"score": list(candidate.score), "eligible": True, "reasons": []},
            value_artifact={"path": value_artifact.path.name, "sha256": value_artifact.sha256},
        )
        write_artifact(run / "training_candidates.json", {
            "training_only": True,
            "candidates": [
                {**entry, "expression_id": f"historical-{seed}"},
                entry,
            ],
        }, immutable=True)
        write_artifact(run / "deduplication.json", {
            "accepted": [candidate.expression_id], "rejected": [],
            "rejection_reasons": {}, "comparisons": [], "loaded_archive_entries": 1,
        }, immutable=False)
        write_artifact(run / "cost_fitness.json", {
            "candidates": {candidate.expression_id: {"accounting_fingerprint": "accounting"}},
        }, immutable=True)

    result = experiment.merge_seed_runs(root)

    assert result["candidate_count"] == 2
    pooled = read_verified_manifest(root / "pooled" / "training_candidates.json")
    assert {item["expression_id"] for item in pooled["candidates"]} == {
        "candidate-7", "candidate-8",
    }
    sources = read_verified_manifest(root / "pooled" / "candidate_sources.json")
    assert sources["candidates"]["candidate-7"][0]["seed"] == 7
