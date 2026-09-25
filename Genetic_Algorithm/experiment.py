"""Predeclared multi-seed GP experiments with one pooled validation pass."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .artifacts import (
    read_verified_manifest, write_artifact, write_value_artifact,
)
from .config import STAGES, SearchConfig, load_config
from .data import run_training_audit
from .evolution import Candidate
from .expression import Node
from .features import RAW_FIELDS
from .search import _load_archive
from .selection import deduplicate_training


def _source_hashes() -> dict[str, str]:
    from .search import _default_code_paths
    package = Path(__file__).resolve().parent
    return {
        path: hashlib.sha256((package.parent / path).read_bytes()).hexdigest()
        for path in _default_code_paths(package.parent)
    }


def _tree(payload: dict[str, Any]) -> Node:
    return Node(
        payload["op"], tuple(_tree(child) for child in payload["children"]),
        payload["field"], payload["window"],
    )


def _plain_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def initialize_experiment(
    config_path: str | Path, h5_path: str | Path, experiment_dir: str | Path,
    seeds: Iterable[int], *, archive_path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze the budget and create one independent configuration per seed."""
    root = Path(experiment_dir)
    if root.exists():
        raise FileExistsError(f"refusing to reuse experiment directory: {root}")
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list or len(seed_list) != len(set(seed_list)) or any(seed < 0 for seed in seed_list):
        raise ValueError("experiment seeds must be unique non-negative integers")
    base = load_config(config_path)
    if base.stability_mode != "continuous_leave_best_out":
        raise ValueError("multi-seed experiment requires continuous_leave_best_out stability")
    root.mkdir(parents=True)
    audit_path = root / "audit_train.json"
    run_training_audit(
        h5_path, audit_path, stage=STAGES["train"], warmup_days=base.max_history,
        fields=sorted(RAW_FIELDS),
    )
    audit = _plain_json(audit_path)
    archive_sha256 = None
    if archive_path is not None:
        archive_sha256 = read_verified_manifest(archive_path)["sha256"]
    raw = asdict(base)
    raw.pop("initial_trees", None)
    raw.pop("evaluate_candidate", None)
    for seed in seed_list:
        payload = {**raw, "seed": seed}
        SearchConfig(**payload)
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir()
        write_artifact(seed_dir / "search_config.json", payload, immutable=False)
    config_bytes = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "experiment_version": 1,
        "status": "declared",
        "seeds": list(seed_list),
        "search_budget": {
            "population": base.population,
            "generations": base.generations,
            "validation_limit": base.validation_limit,
        },
        "base_config": raw,
        "base_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "training_fingerprint": audit["fingerprint"],
        "archive_path": str(Path(archive_path).resolve()) if archive_path else None,
        "archive_sha256": archive_sha256,
        "source_hashes": _source_hashes(),
        "research_disclosure": (
            "2026 was previously inspected; it is repeat research and cannot alter selection"
        ),
    }
    write_artifact(root / "experiment_manifest.json", manifest, immutable=True)
    update_experiment_progress(root)
    return {"status": "declared", "experiment_dir": str(root), "seeds": list(seed_list)}


def update_experiment_progress(experiment_dir: str | Path) -> dict[str, Any]:
    root = Path(experiment_dir)
    manifest = read_verified_manifest(root / "experiment_manifest.json")
    if manifest.get("source_hashes") != _source_hashes():
        raise ValueError("experiment source code changed after the manifest was frozen")
    records = []
    for seed in manifest["seeds"]:
        run = root / f"seed_{seed}" / "run"
        live_progress = root / f"seed_{seed}" / "search_progress.json"
        generation_path = run / "generations.jsonl"
        generations = 0
        if generation_path.is_file():
            with generation_path.open(encoding="utf-8") as handle:
                generations = sum(1 for line in handle if line.strip())
        live = _plain_json(live_progress) if live_progress.is_file() else {}
        generations = max(generations, int(live.get("completed_generations", 0)))
        records.append({
            "seed": seed,
            "status": "complete" if (run / "training_candidates.json").is_file()
                      else ("running" if run.exists() else "pending"),
            "completed_generations": generations,
            "target_generations": int(manifest["search_budget"]["generations"]),
            "eligible_population": live.get("eligible_population"),
            "unique_evaluations": live.get("unique_evaluations"),
            "run_dir": str(run),
        })
    payload = {
        "experiment_manifest_sha256": manifest["sha256"],
        "seeds": records,
        "completed_seed_count": sum(item["status"] == "complete" for item in records),
        "total_seed_count": len(records),
        "pool_status": "complete" if (root / "pooled" / "training_candidates.json").is_file()
                       else "pending",
    }
    write_artifact(root / "experiment_progress.json", payload, immutable=False)
    return payload


def merge_seed_runs(experiment_dir: str | Path) -> dict[str, Any]:
    """Pool only current accepted candidates and deduplicate on 2024 values."""
    root = Path(experiment_dir)
    pooled = root / "pooled"
    if pooled.exists():
        raise FileExistsError(f"refusing to reuse pooled run directory: {pooled}")
    manifest = read_verified_manifest(root / "experiment_manifest.json")
    if manifest.get("source_hashes") != _source_hashes():
        raise ValueError("experiment source code changed after the manifest was frozen")
    base_config = dict(manifest["base_config"])
    config = SearchConfig(**base_config)
    candidates_by_id: dict[str, Candidate] = {}
    entries_by_id: dict[str, dict[str, Any]] = {}
    values_by_id: dict[str, Any] = {}
    costs_by_id: dict[str, Any] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    first_provenance = None
    for seed in manifest["seeds"]:
        run = root / f"seed_{seed}" / "run"
        run_config = read_verified_manifest(run / "config.json")
        unsigned_config = {key: value for key, value in run_config.items() if key != "sha256"}
        expected_config = {**base_config, "seed": seed}
        if asdict(SearchConfig(**unsigned_config)) != asdict(SearchConfig(**expected_config)):
            raise ValueError(f"seed {seed} configuration differs from experiment manifest")
        provenance = read_verified_manifest(run / "provenance.json")
        if provenance["stage_content_hashes"]["train"] != manifest["training_fingerprint"]:
            raise ValueError(f"seed {seed} training fingerprint differs from experiment manifest")
        if first_provenance is None:
            first_provenance = provenance
        entries, values = _load_archive(run / "training_candidates.json")
        deduplication = _plain_json(run / "deduplication.json")
        accepted_ids = deduplication.get("accepted")
        if not isinstance(accepted_ids, list):
            raise ValueError(f"seed {seed} deduplication result is malformed")
        entry_map = {entry["expression_id"]: entry for entry in entries}
        cost_document = read_verified_manifest(run / "cost_fitness.json")
        for expression_id in accepted_ids:
            if expression_id not in entry_map or expression_id not in values:
                raise ValueError(f"seed {seed} accepted candidate lacks training evidence")
            entry = entry_map[expression_id]
            diagnostic = entry["training_diagnostics"]
            candidate = Candidate(
                _tree(entry["ast"]), expression_id, tuple(diagnostic["score"]),
                eligible=diagnostic["eligible"], reasons=tuple(diagnostic["reasons"]),
                direction=entry.get("training_direction"),
            )
            previous = candidates_by_id.get(expression_id)
            if previous is not None and previous != candidate:
                raise ValueError(f"candidate {expression_id} differs across seeds")
            candidates_by_id[expression_id] = candidate
            entries_by_id[expression_id] = entry
            values_by_id[expression_id] = values[expression_id]
            if expression_id not in cost_document["candidates"]:
                raise ValueError(f"seed {seed} candidate lacks cost_fitness evidence")
            costs_by_id[expression_id] = cost_document["candidates"][expression_id]
            sources.setdefault(expression_id, []).append({
                "seed": seed, "run_dir": str(run),
                "training_score": list(candidate.score),
            })
    selection = deduplicate_training(
        candidates_by_id.values(), values_by_id, (), config,
    )
    pooled.mkdir(parents=True)
    selected = tuple(selection.accepted)
    panels = {
        candidate.expression_id: values_by_id[candidate.expression_id]["values"]
        for candidate in selected
    }
    value_artifact = write_value_artifact(pooled / "training_values_archive.json", panels)
    pooled_entries = []
    for candidate in selected:
        entry = dict(entries_by_id[candidate.expression_id])
        entry["value_artifact"] = {
            "path": value_artifact.path.name, "sha256": value_artifact.sha256,
        }
        pooled_entries.append(entry)
    write_artifact(
        pooled / "training_candidates.json",
        {"training_only": True, "candidates": pooled_entries}, immutable=True,
    )
    write_artifact(pooled / "deduplication.json", {
        "accepted": [candidate.expression_id for candidate in selected],
        "rejected": [candidate.expression_id for candidate in selection.rejected],
        "rejection_reasons": selection.rejection_reasons,
        "comparisons": [comparison.__dict__ for comparison in selection.comparisons],
        "loaded_archive_entries": 0,
    }, immutable=False)
    write_artifact(pooled / "cost_fitness.json", {
        "config": asdict(config),
        "candidates": {
            candidate.expression_id: costs_by_id[candidate.expression_id]
            for candidate in selected
        },
    }, immutable=True)
    pooled_config = {**base_config, "seed": manifest["seeds"][0]}
    write_artifact(pooled / "config.json", pooled_config, immutable=True)
    provenance = {
        **{key: value for key, value in first_provenance.items() if key != "sha256"},
        "training_experiment_id": root.name,
        "multi_seed_experiment": {
            "manifest_sha256": manifest["sha256"], "seeds": manifest["seeds"],
        },
    }
    write_artifact(pooled / "provenance.json", provenance, immutable=True)
    write_artifact(pooled / "candidate_sources.json", {
        "experiment_manifest_sha256": manifest["sha256"],
        "candidates": {
            candidate.expression_id: sources[candidate.expression_id]
            for candidate in selected
        },
    }, immutable=True)
    update_experiment_progress(root)
    return {
        "status": "complete" if selected else "no_candidates",
        "pooled_run_dir": str(pooled), "candidate_count": len(selected),
    }


__all__ = ["initialize_experiment", "merge_seed_runs", "update_experiment_progress"]
