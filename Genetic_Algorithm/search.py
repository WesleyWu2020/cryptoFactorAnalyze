"""Audit-first boundary for the future daily GP search stage."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeVar

import pandas as pd

from .artifacts import (
    build_provenance,
    read_verified_manifest,
    read_verified_value_artifact,
    training_archive_entry,
    write_artifact,
    write_value_artifact,
)
from .config import Stage
from .data import run_training_audit
from .selection import deduplicate_training


Result = TypeVar("Result")


def _default_artifact_dir(audit_path: str | Path, experiment_id: str) -> Path:
    with Path(audit_path).open(encoding="utf-8") as handle:
        audit = json.load(handle)
    fingerprint = str(audit["fingerprint"])
    safe_experiment_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_id).strip("._") or "training-search"
    return Path("Genetic_Algorithm") / "runs" / f"{safe_experiment_id}-{fingerprint}"


def run_search(
    h5_path: str | Path,
    audit_path: str | Path,
    *,
    stage: Stage,
    warmup_days: int,
    fields: Iterable[str],
    search_stage: Callable[[Path], Result],
    artifact_dir: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    repository_root: str | Path = ".",
    selected_code_paths: Iterable[str | Path] = (),
    package_names: Iterable[str] = (),
    seed: int | None = None,
    operator_version: str = "daily-gp-search-v1",
    backtest_profile: Mapping[str, Any] | None = None,
    experiment_id: str = "training-search",
    validation_attempts: int = 0,
    validation_experiment_id: str | None = None,
    archive_path: str | Path | None = None,
) -> Result:
    """Persist the train audit before handing off to the future GP search.

    This is the Task 2 boundary for Task 6/10 integration. The callback is
    intentionally injected: this module does not implement GP search and it
    does not load validation or test rows.
    """
    persisted_audit = run_training_audit(
        h5_path,
        audit_path,
        stage=stage,
        warmup_days=warmup_days,
        fields=fields,
    )
    result = search_stage(persisted_audit)
    destination = Path(artifact_dir) if artifact_dir is not None else _default_artifact_dir(
        persisted_audit, experiment_id
    )
    selected_result = _write_training_artifacts(
        result,
        persisted_audit,
        artifact_dir=destination,
        config=config or {},
        repository_root=repository_root,
        selected_code_paths=tuple(selected_code_paths) or _default_code_paths(repository_root),
        package_names=tuple(package_names) or ("numpy", "pandas", "tables"),
        seed=(seed if seed is not None else int((config or {}).get("seed", 42))),
        operator_version=operator_version,
        backtest_profile=backtest_profile or {},
        experiment_id=experiment_id,
        validation_attempts=validation_attempts,
        validation_experiment_id=validation_experiment_id,
        archive_path=archive_path,
    )
    return selected_result


def _frame(value: Any) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, Mapping):
        for key in ("values", "factor_values", "training_values"):
            if isinstance(value.get(key), pd.DataFrame):
                return value[key]
    return None


def _result_values(result: Any) -> dict[str, Any]:
    values = getattr(result, "values_by_id", None)
    if isinstance(values, Mapping):
        return {str(identifier): value for identifier, value in values.items()}
    values = getattr(result, "training_values", None)
    if isinstance(values, Mapping):
        return {str(identifier): value for identifier, value in values.items()}
    return {}


def _load_archive(path: str | Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {}
    archive_path = Path(path)
    document = read_verified_manifest(archive_path)
    entries = document.get("candidates", ())
    if not isinstance(entries, list):
        raise ValueError("training archive candidates must be a list")
    values: dict[str, Any] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("training archive entry is malformed")
        identifier = str(entry.get("expression_id", entry.get("hash", "archive")))
        reference = entry.get("value_artifact")
        if not isinstance(reference, Mapping):
            values[identifier] = {
                "training_fingerprint": entry.get("training_fingerprint"),
                "operator_version": entry.get("operator_version"),
            }
            continue
        relative, digest = reference.get("path"), reference.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError(f"archive value artifact reference is malformed for {identifier}")
        panels = read_verified_value_artifact(archive_path.parent / relative, expected_sha256=digest)
        values[identifier] = {
            "values": panels.get(identifier),
            "training_fingerprint": entry.get("training_fingerprint"),
            "operator_version": entry.get("operator_version"),
        }
    return [dict(entry) for entry in entries], values


def _default_code_paths(repository_root: str | Path) -> tuple[str, ...]:
    candidates = (
        "Genetic_Algorithm/artifacts.py", "Genetic_Algorithm/search.py",
        "Genetic_Algorithm/selection.py", "Genetic_Algorithm/evolution.py",
        "Genetic_Algorithm/expression.py", "Genetic_Algorithm/evaluator.py",
        "Genetic_Algorithm/fitness.py", "Genetic_Algorithm/operators.py",
        "Genetic_Algorithm/features.py", "factor_common/labels.py",
    )
    root = Path(repository_root)
    return tuple(path for path in candidates if (root / path).is_file())


def _resolved_config(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dataclass_fields__"):
        return {name: getattr(config, name) for name in config.__dataclass_fields__}
    return {}


def _write_training_artifacts(
    result: Any,
    audit_path: str | Path,
    *,
    artifact_dir: str | Path,
    config: Mapping[str, Any],
    repository_root: str | Path,
    selected_code_paths: Iterable[str | Path],
    package_names: Iterable[str],
    seed: int,
    operator_version: str,
    backtest_profile: Mapping[str, Any],
    experiment_id: str,
    validation_attempts: int,
    validation_experiment_id: str | None,
    archive_path: str | Path | None,
) -> Any:
    """Persist training outputs after search, without consulting validation."""
    with Path(audit_path).open(encoding="utf-8") as handle:
        audit = json.load(handle)
    training_fingerprint = str(audit["fingerprint"])
    candidates = tuple(getattr(result, "candidates", ()))
    current_values = _result_values(result)
    archive_entries, archive_values = _load_archive(archive_path)
    for identifier, value in list(current_values.items()):
        metadata = dict(value) if isinstance(value, Mapping) else {}
        frame = _frame(value)
        if frame is not None:
            metadata["values"] = frame
        metadata.setdefault("training_fingerprint", training_fingerprint)
        metadata.setdefault("operator_version", operator_version)
        current_values[identifier] = metadata
    archive_for_selection = [
        {
            **entry,
            "values": _frame(archive_values.get(str(entry.get("expression_id", entry.get("hash", "archive"))))),
        }
        for entry in archive_entries
    ]
    selection = deduplicate_training(candidates, current_values, archive_for_selection, config or {})
    selected = selection.accepted
    if hasattr(result, "candidates") and tuple(selected) != candidates:
        result = replace(result, candidates=tuple(selected))
    provenance = build_provenance(
        config=_resolved_config(config),
        stage_content_hashes={"train": training_fingerprint},
        selected_code_paths=selected_code_paths,
        repository_root=repository_root,
        package_names=package_names,
        seed=seed,
        operator_version=operator_version,
        backtest_profile=backtest_profile,
        experiment_id=experiment_id,
        validation_attempts=validation_attempts,
        validation_experiment_id=validation_experiment_id,
    )
    destination = Path(artifact_dir)
    value_panels = {
        candidate.expression_id: _frame(current_values.get(candidate.expression_id))
        for candidate in selected
        if _frame(current_values.get(candidate.expression_id)) is not None
    }
    value_artifact = write_value_artifact(destination / "training_values.json", value_panels)
    write_artifact(destination / "provenance.json", provenance, immutable=True)
    archive = [
        training_archive_entry(
            candidate,
            training_fingerprint=training_fingerprint,
            operator_version=operator_version,
            diagnostics={
                "score": list(candidate.score),
                "eligible": candidate.eligible,
                "reasons": list(candidate.reasons),
            },
            value_artifact={"path": value_artifact.path.name, "sha256": value_artifact.sha256},
        )
        for candidate in selected
    ]
    write_artifact(
        destination / "training_candidates.json",
        {"training_only": True, "candidates": archive},
        immutable=True,
    )
    write_artifact(
        destination / "deduplication.json",
        {
            "accepted": [candidate.expression_id for candidate in selection.accepted],
            "rejected": [candidate.expression_id for candidate in selection.rejected],
            "rejection_reasons": selection.rejection_reasons,
            "comparisons": [comparison.__dict__ for comparison in selection.comparisons],
            "loaded_archive_entries": len(archive_entries),
        },
    )
    return result


__all__ = ["run_search"]
