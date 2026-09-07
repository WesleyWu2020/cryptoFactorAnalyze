"""Audit-first boundary for the future daily GP search stage."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, replace
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
from .config import SearchConfig, Stage, load_config
from .data import run_training_audit
from .selection import deduplicate_training
from factor_common.profiles import resolve_profile


Result = TypeVar("Result")


def _resolve_backtest_profile(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    profile = resolve_profile("perp_1d", {} if overrides is None else overrides)
    return asdict(profile)


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
        selected_code_paths=tuple(
            dict.fromkeys((*tuple(selected_code_paths), *_default_code_paths(repository_root)))
        ),
        package_names=tuple(package_names),
        seed=(seed if seed is not None else int((config or {}).get("seed", 42))),
        operator_version=operator_version,
        backtest_profile=_resolve_backtest_profile(backtest_profile),
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


def _confined_reference(root: Path, reference: str, *, label: str) -> Path:
    """Resolve a manifest reference while keeping it inside its owning directory."""
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must be relative and cannot traverse: {reference!r}")
    root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        common = Path(os.path.commonpath((str(root), str(resolved))))
    except ValueError as exc:
        raise ValueError(f"{label} path is outside its owning directory: {reference!r}") from exc
    if common != root:
        raise ValueError(f"{label} path is outside its owning directory: {reference!r}")
    return resolved


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
        artifact_path = _confined_reference(
            archive_path.parent, relative, label="archive value artifact"
        )
        panels = read_verified_value_artifact(artifact_path, expected_sha256=digest)
        values[identifier] = {
            "values": panels.get(identifier),
            "training_fingerprint": entry.get("training_fingerprint"),
            "operator_version": entry.get("operator_version"),
        }
    return [dict(entry) for entry in entries], values


def _default_code_paths(repository_root: str | Path) -> tuple[str, ...]:
    candidates = (
        "Genetic_Algorithm/data.py",
        "Genetic_Algorithm/artifacts.py", "Genetic_Algorithm/search.py",
        "Genetic_Algorithm/selection.py", "Genetic_Algorithm/evolution.py",
        "Genetic_Algorithm/expression.py", "Genetic_Algorithm/evaluator.py",
        "Genetic_Algorithm/fitness.py", "Genetic_Algorithm/operators.py",
        "Genetic_Algorithm/features.py", "factor_common/labels.py",
        "factor_common/data_provider.py",
        "data/crypto_quant/reader.py", "data/crypto_quant/store.py",
        "data/crypto_quant/panel.py", "data/crypto_quant/schemas.py",
        "data/crypto_quant/config.py",
        "Genetic_Algorithm/config.py", "Genetic_Algorithm/configs/default.json",
        "Genetic_Algorithm/configs/smoke.json",
    )
    root = Path(repository_root)
    return tuple(path for path in candidates if (root / path).is_file())


def _resolved_config(config: Any) -> Mapping[str, Any]:
    overrides = dict(config) if isinstance(config, Mapping) else {}
    if hasattr(config, "__dataclass_fields__"):
        overrides = {
            name: getattr(config, name)
            for name in config.__dataclass_fields__
            if name not in {"initial_trees", "evaluate_candidate"}
        }
    runtime_fields = {"initial_trees", "evaluate_candidate"}
    default_config = load_config(Path(__file__).with_name("configs") / "default.json")
    defaults = {
        name: getattr(default_config, name)
        for name in default_config.__dataclass_fields__
        if name not in runtime_fields
    }
    defaults.update({key: value for key, value in overrides.items() if key not in runtime_fields})
    effective = SearchConfig(**defaults)
    return {
        name: getattr(effective, name)
        for name in effective.__dataclass_fields__
        if name not in runtime_fields
    }


def _copy_archive_value_artifacts(
    archive_path: Path | None,
    archive_entries: Iterable[Mapping[str, Any]],
    destination: Path,
) -> None:
    if archive_path is None:
        return
    for entry in archive_entries:
        reference = entry.get("value_artifact")
        if not isinstance(reference, Mapping):
            continue
        relative = reference.get("path")
        if not isinstance(relative, str):
            raise ValueError("archive value artifact reference is malformed")
        source = _confined_reference(
            archive_path.parent, relative, label="archive value artifact"
        )
        target = _confined_reference(destination, relative, label="destination value artifact")
        if not source.is_file():
            raise ValueError(f"archive value artifact is missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"archive value artifact path collision: {target}")
        else:
            shutil.copy2(source, target)


def _archive_key(entry: Mapping[str, Any]) -> str:
    return str(entry.get("expression_id", entry.get("hash", "archive")))


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
    backtest_profile = _resolve_backtest_profile(backtest_profile)
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
    selected = []
    rejected = list(selection.rejected)
    rejection_reasons = dict(selection.rejection_reasons)
    for candidate in selection.accepted:
        if _frame(current_values.get(candidate.expression_id)) is None:
            rejected.append(candidate)
            rejection_reasons[candidate.expression_id] = (
                *rejection_reasons.get(candidate.expression_id, ()),
                "missing value panel: selected candidates must have a training value panel",
            )
        else:
            selected.append(candidate)
    selected = tuple(selected)
    if hasattr(result, "candidates") and tuple(selected) != candidates:
        result = replace(result, candidates=tuple(selected))
    destination = Path(artifact_dir)
    _copy_archive_value_artifacts(archive_path, archive_entries, destination)
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
    value_panels = {
        candidate.expression_id: _frame(current_values.get(candidate.expression_id))
        for candidate in selected
        if _frame(current_values.get(candidate.expression_id)) is not None
    }
    value_artifact_path = destination / "training_values.json"
    if value_artifact_path.exists():
        value_artifact_path = destination / "training_values.new.json"
    value_artifact = write_value_artifact(value_artifact_path, value_panels)
    write_artifact(destination / "provenance.json", provenance, immutable=True)
    new_archive = [
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
    archive = []
    seen_ids: set[str] = set()
    for entry in (*archive_entries, *sorted(new_archive, key=_archive_key)):
        identifier = _archive_key(entry)
        if identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        archive.append(dict(entry))
    write_artifact(
        destination / "training_candidates.json",
        {"training_only": True, "candidates": archive},
        immutable=True,
    )
    write_artifact(
        destination / "deduplication.json",
        {
            "accepted": [candidate.expression_id for candidate in selected],
            "rejected": [candidate.expression_id for candidate in rejected],
            "rejection_reasons": rejection_reasons,
            "comparisons": [comparison.__dict__ for comparison in selection.comparisons],
            "loaded_archive_entries": len(archive_entries),
        },
    )
    return result


__all__ = ["run_search"]
