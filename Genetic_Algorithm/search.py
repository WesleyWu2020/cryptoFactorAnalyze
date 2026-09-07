"""Audit-first boundary for the future daily GP search stage."""

from __future__ import annotations

import json
import hashlib
import inspect
import os
import re
import shutil
import stat
import tempfile
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
from .expression import Node, canonical_tree, validate_node_attributes
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


_ARCHIVE_REQUIRED_ENTRY_KEYS = {
    "expression_id", "training_only", "ast", "training_fingerprint",
    "operator_version", "training_diagnostics",
}
_ARCHIVE_OPTIONAL_ENTRY_KEYS = {"value_artifact"}
_ARCHIVE_AST_KEYS = {"op", "field", "window", "children"}
_ARCHIVE_FORBIDDEN_KEY_MARKERS = (
    "validation", "test", "future", "holdout", "out_of_sample", "oos",
)


def _reject_archive_future_key(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(marker in normalized for marker in _ARCHIVE_FORBIDDEN_KEY_MARKERS):
                raise ValueError(f"training archive contains non-training field: {key}")
            _reject_archive_future_key(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_archive_future_key(item)


def _load_archive_ast(value: Any, *, identifier: str) -> Node:
    if not isinstance(value, Mapping) or set(value) != _ARCHIVE_AST_KEYS:
        raise ValueError(f"training archive AST is malformed for {identifier}")
    op, field, window, children = (
        value["op"], value["field"], value["window"], value["children"]
    )
    if not isinstance(op, str) or (field is not None and not isinstance(field, str)):
        raise ValueError(f"training archive AST is malformed for {identifier}")
    if window is not None and type(window) is not int:
        raise ValueError(f"training archive AST is malformed for {identifier}")
    if not isinstance(children, list):
        raise ValueError(f"training archive AST is malformed for {identifier}")
    node = Node(
        op,
        tuple(_load_archive_ast(child, identifier=identifier) for child in children),
        field,
        window,
    )
    try:
        validate_node_attributes(node)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"training archive AST is malformed for {identifier}") from exc
    return node


def _validate_archive_entry(entry: Mapping[str, Any]) -> str:
    _reject_archive_future_key(entry)
    missing = _ARCHIVE_REQUIRED_ENTRY_KEYS - set(entry)
    if missing:
        raise ValueError(
            "training archive candidate is missing required fields: "
            + ", ".join(sorted(missing))
        )
    unexpected = set(entry) - (_ARCHIVE_REQUIRED_ENTRY_KEYS | _ARCHIVE_OPTIONAL_ENTRY_KEYS)
    if unexpected:
        raise ValueError(
            "training archive candidate contains non-training fields: "
            + ", ".join(sorted(unexpected))
        )
    identifier = entry["expression_id"]
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("training archive candidate expression_id is malformed")
    if entry["training_only"] is not True:
        raise ValueError("training archive candidate must be marked training_only")
    for field in ("training_fingerprint", "operator_version"):
        if not isinstance(entry[field], str) or not entry[field]:
            raise ValueError(f"training archive candidate {field} is malformed")
    _load_archive_ast(entry["ast"], identifier=identifier)
    diagnostics = entry["training_diagnostics"]
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != {"score", "eligible", "reasons"}:
        raise ValueError("training archive diagnostics contain non-training fields")
    if not isinstance(diagnostics["score"], list):
        raise ValueError("training archive diagnostic score is malformed")
    if not isinstance(diagnostics["eligible"], bool):
        raise ValueError("training archive diagnostic eligibility is malformed")
    if not isinstance(diagnostics["reasons"], list) or not all(
        isinstance(reason, str) for reason in diagnostics["reasons"]
    ):
        raise ValueError("training archive diagnostic reasons are malformed")
    reference = entry.get("value_artifact")
    if reference is not None and (
        not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}
    ):
        raise ValueError(f"archive value artifact reference is malformed for {identifier}")
    return identifier


def _load_archive(path: str | Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {}
    archive_path = Path(path)
    document = read_verified_manifest(archive_path)
    _reject_archive_future_key(document)
    if document.get("training_only") is not True:
        raise ValueError("training archive must be marked training_only")
    allowed_document_keys = {"sha256", "training_only", "candidates"}
    unexpected_document_keys = set(document) - allowed_document_keys
    if unexpected_document_keys:
        raise ValueError(
            "training archive contains non-training fields: "
            + ", ".join(sorted(unexpected_document_keys))
        )
    entries = document.get("candidates", ())
    if not isinstance(entries, list):
        raise ValueError("training archive candidates must be a list")
    values: dict[str, Any] = {}
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("training archive entry is malformed")
        identifier = _validate_archive_entry(entry)
        if identifier in seen_ids:
            raise ValueError(f"duplicate expression_id in training archive: {identifier}")
        seen_ids.add(identifier)
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
    runtime_inputs = {"initial_trees": (), "evaluate_candidate": None}
    if hasattr(config, "__dataclass_fields__"):
        runtime_inputs.update({name: getattr(config, name) for name in runtime_inputs})
        overrides = {
            name: getattr(config, name)
            for name in config.__dataclass_fields__
            if name not in runtime_inputs
        }
    elif isinstance(config, Mapping):
        runtime_inputs.update({name: config.get(name, default) for name, default in runtime_inputs.items()})
    runtime_fields = set(runtime_inputs)
    default_config = load_config(Path(__file__).with_name("configs") / "default.json")
    defaults = {
        name: getattr(default_config, name)
        for name in default_config.__dataclass_fields__
        if name not in runtime_fields
    }
    defaults.update({key: value for key, value in overrides.items() if key not in runtime_fields})
    effective = SearchConfig(**defaults)
    resolved = {
        name: getattr(effective, name)
        for name in effective.__dataclass_fields__
        if name not in runtime_fields
    }
    trees = [canonical_tree(tree) for tree in runtime_inputs["initial_trees"]]
    tree_payload = [_tree_payload(tree) for tree in trees]
    tree_bytes = json.dumps(tree_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    resolved["initial_trees"] = {
        "canonical": tree_bytes.decode("utf-8"),
        "trees": tree_payload,
        "sha256": hashlib.sha256(tree_bytes).hexdigest(),
    }
    evaluator = runtime_inputs["evaluate_candidate"]
    if evaluator is None:
        identity = {
            "identity": "none", "module": None, "qualname": None,
            "source": None, "source_hash": None,
        }
    else:
        module = getattr(evaluator, "__module__", type(evaluator).__module__)
        qualname = getattr(evaluator, "__qualname__", type(evaluator).__qualname__)
        try:
            source = inspect.getsource(evaluator)
        except (OSError, TypeError):
            source = None
        identity = {
            "identity": f"{module}:{qualname}",
            "module": module,
            "qualname": qualname,
            "source": source,
            "source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest()
            if source is not None else None,
        }
    identity_bytes = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    identity["sha256"] = hashlib.sha256(identity_bytes).hexdigest()
    resolved["evaluate_candidate"] = identity
    return resolved


def _tree_payload(tree: Node) -> dict[str, Any]:
    return {
        "op": tree.op,
        "field": tree.field,
        "window": tree.window,
        "children": [_tree_payload(child) for child in tree.children],
    }


def _publish_journal_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.publish.json")


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _recover_publish_destination(destination: Path) -> None:
    """Resolve an interrupted immutable directory publication deterministically."""
    destination = Path(destination).resolve()
    journal_path = _publish_journal_path(destination)
    if not os.path.lexists(journal_path):
        return
    if journal_path.is_symlink() or not journal_path.is_file():
        raise ValueError(f"publish journal path is unsafe: {journal_path}")
    try:
        journal_resolved = journal_path.resolve(strict=True)
        journal_resolved.relative_to(destination.parent)
    except (OSError, ValueError) as exc:
        raise ValueError(f"publish journal path is unsafe: {journal_path}") from exc
    try:
        with journal_resolved.open(encoding="utf-8") as handle:
            journal = json.load(handle)
    except OSError as exc:
        raise ValueError(f"publish journal is unreadable: {journal_path}") from exc
    if journal.get("version") != 1:
        raise ValueError(f"unsupported publish journal: {journal_path}")
    parent = destination.parent.resolve()
    paths = {}
    for key in ("destination", "staging"):
        value = journal.get(key)
        if not isinstance(value, str):
            raise ValueError(f"invalid publish journal field: {key}")
        path = Path(value)
        if path.is_absolute() or path.parent != Path("."):
            raise ValueError(f"publish journal path must be a sibling name: {key}")
        original = parent / path
        if key in {"staging", "backup"} and os.path.lexists(original):
            try:
                original_mode = os.lstat(original).st_mode
            except OSError as exc:
                raise ValueError(f"publish journal path is unsafe: {original}") from exc
            if stat.S_ISLNK(original_mode):
                raise ValueError(f"publish journal path is unsafe: {original}")
        resolved = original.resolve()
        try:
            resolved.relative_to(parent)
        except ValueError as exc:
            raise ValueError(f"publish journal path escapes parent: {key}") from exc
        if resolved == parent:
            raise ValueError(f"publish journal path must be a sibling name: {key}")
        paths[key] = resolved
    backup_name = journal.get("backup")
    if backup_name is not None:
        if not isinstance(backup_name, str):
            raise ValueError("invalid publish journal field: backup")
        backup_path = Path(backup_name)
        if backup_path.is_absolute() or backup_path.parent != Path("."):
            raise ValueError("publish journal path must be a sibling name: backup")
        original = parent / backup_path
        if os.path.lexists(original):
            try:
                original_mode = os.lstat(original).st_mode
            except OSError as exc:
                raise ValueError(f"publish journal path is unsafe: {original}") from exc
            if stat.S_ISLNK(original_mode):
                raise ValueError(f"publish journal path is unsafe: {original}")
        resolved = original.resolve()
        try:
            resolved.relative_to(parent)
        except ValueError as exc:
            raise ValueError("publish journal path escapes parent: backup") from exc
        if resolved == parent:
            raise ValueError("publish journal path must be a sibling name: backup")
        paths["backup"] = resolved
    if paths["destination"] != destination:
        raise ValueError(f"publish journal destination mismatch: {journal_path}")
    if journal_resolved in {paths["destination"], paths["staging"]}:
        raise ValueError(f"publish journal collides with a recovery path: {journal_path}")
    if paths["staging"] in {paths["destination"], journal_resolved}:
        raise ValueError(f"publish journal staging collides with destination: {journal_path}")
    if "backup" in paths and paths["backup"] in {
        paths["destination"], paths["staging"], journal_resolved
    }:
        raise ValueError(f"publish journal backup collides with another path: {journal_path}")

    if destination.exists():
        # The staged directory is already visible. The publication committed;
        # only cleanup may have been interrupted.
        if "backup" in paths:
            _remove_path(paths["backup"])
        _remove_path(paths["staging"])
    elif "backup" in paths and paths["backup"].exists():
        # The old run is the only published value until the staged rename wins.
        os.replace(paths["backup"], destination)
        _remove_path(paths["staging"])
    else:
        # No old run was moved, so discard an unpublished staging directory.
        _remove_path(paths["staging"])
    journal_path.unlink()


def _write_publish_journal(destination: Path, staging: Path, backup: Path | None) -> Path:
    journal_path = _publish_journal_path(destination)
    payload = {
        "version": 1,
        "destination": destination.name,
        "backup": backup.name if backup is not None else None,
        "staging": staging.name,
    }
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.publish.", dir=destination.parent
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        with temporary.open("r+", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, journal_path)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        _remove_path(temporary)
        raise
    return journal_path


def _prepare_publish_destination(destination: Path, audit_path: Path) -> tuple[Path, Path | None]:
    """Create an isolated staging directory without changing the published run."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    _recover_publish_destination(destination)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    backup = None
    if destination.exists():
        if not destination.is_dir():
            staging.rmdir()
            raise FileExistsError(f"artifact destination is not a directory: {destination}")
        entries = list(destination.iterdir())
        audit_path = audit_path.resolve()
        destination_resolved = destination.resolve()
        audit_inside = destination_resolved in audit_path.parents
        allowed = audit_path if audit_inside else None
        if any(path.resolve() != allowed for path in entries):
            staging.rmdir()
            raise FileExistsError(f"immutable training artifact already exists: {destination}")
        if allowed is not None:
            relative = audit_path.relative_to(destination_resolved)
            preserved = staging / relative
            preserved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audit_path, preserved)
        if allowed is not None:
            backup = destination.with_name(f".{destination.name}.audit-backup")
            if backup.exists():
                staging.rmdir()
                raise FileExistsError(f"stale artifact backup already exists: {backup}")
        else:
            destination.rmdir()
    return staging, backup


def _publish_staging_directory(
    staging: Path, destination: Path, backup: Path | None = None
) -> None:
    """Publish a complete immutable run directory with recoverable replacement."""
    journal_path = _write_publish_journal(destination, staging, backup)
    if backup is not None:
        os.replace(destination, backup)
    os.replace(staging, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if backup is not None:
        _remove_path(backup)
    journal_path.unlink()


def _archive_key(entry: Mapping[str, Any]) -> str:
    return str(entry.get("expression_id", entry.get("hash", "archive")))


def _copy_archive_value_artifacts(
    archive_path: Path | None,
    archive_entries: Iterable[Mapping[str, Any]],
    destination: Path,
) -> None:
    if archive_path is None:
        return
    reserved_names = {
        "deduplication.json", "training_candidates.json", "training_values.json",
        "training_values_archive.json",
        "provenance.json", "config.json", "audit_train.json", "validation.json",
        "frozen.json", "generations.jsonl", "progress.json",
        _publish_journal_path(destination).name,
    }
    references: list[tuple[Path, Path, str]] = []
    targets: dict[Path, str] = {}
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
        if target.parent == destination and (
            target.name in reserved_names or target.name.endswith(".publish.json")
        ):
            raise ValueError(f"archive value artifact path is reserved: {target.name}")
        previous = targets.get(target)
        if previous is not None:
            raise ValueError(
                "archive value artifact path is used by multiple archive candidates: "
                f"{previous} and {_archive_key(entry)}"
            )
        targets[target] = _archive_key(entry)
        if not source.is_file():
            raise ValueError(f"archive value artifact is missing: {source}")
        references.append((source, target, _archive_key(entry)))
    for source, target, _identifier in references:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"archive value artifact path collision: {target}")
        else:
            shutil.copy2(source, target)


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
            "values": _frame(archive_values.get(_archive_key(entry))),
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
    staging, destination_backup = _prepare_publish_destination(destination, Path(audit_path))
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
    try:
        _copy_archive_value_artifacts(archive_path, archive_entries, staging)
        value_artifact_path = staging / "training_values_archive.json"
        if value_artifact_path.exists():
            value_artifact_path = staging / "training_values.new.json"
        value_artifact = write_value_artifact(value_artifact_path, value_panels)
        write_artifact(staging / "provenance.json", provenance, immutable=True)
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
                raise ValueError(f"duplicate expression_id during archive merge: {identifier}")
            seen_ids.add(identifier)
            published = dict(entry)
            if identifier in value_panels:
                published["value_artifact"] = {
                    "path": value_artifact.path.name,
                    "sha256": value_artifact.sha256,
                }
            archive.append(published)
        write_artifact(
            staging / "training_candidates.json",
            {"training_only": True, "candidates": archive},
            immutable=True,
        )
        write_artifact(
            staging / "deduplication.json",
            {
                "accepted": [candidate.expression_id for candidate in selected],
                "rejected": [candidate.expression_id for candidate in rejected],
                "rejection_reasons": rejection_reasons,
                "comparisons": [comparison.__dict__ for comparison in selection.comparisons],
                "loaded_archive_entries": len(archive_entries),
            },
        )
        _publish_staging_directory(staging, destination, destination_backup)
    except Exception:
        _recover_publish_destination(destination)
        if staging.exists():
            _remove_path(staging)
        raise
    return result


__all__ = ["run_search"]
