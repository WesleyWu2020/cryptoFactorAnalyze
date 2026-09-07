"""Audit-first boundary for the future daily GP search stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeVar

from .artifacts import build_provenance, training_archive_entry, write_artifact
from .config import Stage
from .data import run_training_audit


Result = TypeVar("Result")


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
    seed: int = 0,
    operator_version: str = "unknown",
    backtest_profile: Mapping[str, Any] | None = None,
    experiment_id: str = "training-search",
    validation_attempts: int = 0,
    validation_experiment_id: str | None = None,
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
    if artifact_dir is not None:
        _write_training_artifacts(
            result,
            persisted_audit,
            artifact_dir=artifact_dir,
            config=config or {},
            repository_root=repository_root,
            selected_code_paths=selected_code_paths,
            package_names=package_names,
            seed=seed,
            operator_version=operator_version,
            backtest_profile=backtest_profile or {},
            experiment_id=experiment_id,
            validation_attempts=validation_attempts,
            validation_experiment_id=validation_experiment_id,
        )
    return result


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
) -> None:
    """Persist training outputs after search, without consulting validation."""
    with Path(audit_path).open(encoding="utf-8") as handle:
        audit = json.load(handle)
    training_fingerprint = str(audit["fingerprint"])
    candidates = tuple(getattr(result, "candidates", ()))
    provenance = build_provenance(
        config=config,
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
        )
        for candidate in candidates
    ]
    write_artifact(
        destination / "training_candidates.json",
        {"training_only": True, "candidates": archive},
        immutable=True,
    )


__all__ = ["run_search"]
