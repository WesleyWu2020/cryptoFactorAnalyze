"""Immutable manifests and training-only provenance artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


_DIGEST_FIELD = "sha256"
_VALUE_ARTIFACT_VERSION = 1
_VALUE_PANEL_FIELDS = frozenset(
    {"index", "index_name", "index_freq", "columns", "columns_name", "values", "nan_mask"}
)
_NON_GIT_MAX_PATH_BYTES = 4096
_NON_GIT_MAX_SYMLINK_TARGET_BYTES = 4096
_NON_GIT_READ_CHUNK = 1024 * 1024


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value is not allowed: {value}")


def _requirement_names(repository_root: Path) -> tuple[str, ...]:
    """Return direct distribution names declared by the project requirements."""
    names: dict[str, str] = {}
    visited: set[Path] = set()

    def read(path: Path) -> None:
        path = path.resolve()
        if path in visited or not path.is_file():
            return
        visited.add(path)
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith(('-r ', '--requirement ')):
                read(path.parent / line.split(maxsplit=1)[1].strip())
                continue
            if line.startswith(('-', '--')):
                continue
            match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", line)
            if match:
                name = match.group(1)
                names.setdefault(name.lower(), name)

    read(repository_root / "requirements.txt")
    read(repository_root / "requirements-dev.txt")
    return tuple(names[key] for key in sorted(names))


@dataclass(frozen=True)
class ValueArtifact:
    path: Path
    sha256: str


def _value_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, Mapping):
        for key in ("values", "factor_values", "training_values"):
            if isinstance(value.get(key), pd.DataFrame):
                return value[key]
    raise TypeError("training value artifacts require pandas DataFrame panels")


def _value_payload(frame: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
        raise ValueError("training value panels require a unique DatetimeIndex")
    if frame.columns.has_duplicates:
        raise ValueError("training value panels require unique instrument columns")
    serialized_columns = [str(value) for value in frame.columns]
    if len(serialized_columns) != len(set(serialized_columns)):
        raise ValueError("training value panel column label collision after serialization")
    frame = frame.copy()
    frame.columns = serialized_columns
    frame = frame.sort_index().sort_index(axis=1)
    numeric = frame.astype("float64")
    array = numeric.to_numpy(copy=True)
    if np.isinf(array).any():
        raise ValueError("training value panels cannot contain infinity")
    missing = np.isnan(array)
    array[missing] = 0.0
    return {
        "index": [pd.Timestamp(value).isoformat() for value in frame.index],
        "index_name": frame.index.name,
        "index_freq": frame.index.freqstr,
        "columns": [str(value) for value in frame.columns],
        "columns_name": frame.columns.name,
        "values": array.tolist(),
        "nan_mask": missing.astype(bool).tolist(),
    }


def _value_document(panels: Mapping[str, Any]) -> dict[str, Any]:
    identifiers = [str(identifier) for identifier in panels]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("training value panel identifier collision after string normalization")
    payload = {
        "artifact_version": _VALUE_ARTIFACT_VERSION,
        "training_only": True,
        "panels": {
            str(identifier): _value_payload(_value_frame(panels[identifier]))
            for identifier in sorted(panels, key=str)
        },
    }
    return _manifest_payload(payload)


def write_value_artifact(path: str | Path, panels: Mapping[str, Any]) -> ValueArtifact:
    """Write a canonical, immutable training value panel artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = _value_document(panels)
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    _write_exclusive_atomic(target, encoded)
    return ValueArtifact(target, str(document[_DIGEST_FIELD]))


def read_verified_value_artifact(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, pd.DataFrame]:
    """Verify and decode a canonical training value panel artifact."""
    target = Path(path)
    with target.open(encoding="utf-8") as handle:
        document = json.load(handle, parse_constant=_reject_nonfinite_json_constant)
    if not isinstance(document, dict) or document.get("artifact_version") != _VALUE_ARTIFACT_VERSION:
        raise ValueError("unsupported value artifact")
    if document.get("training_only") is not True:
        raise ValueError("value artifact must be marked training_only")
    unexpected_keys = set(document) - {"artifact_version", "training_only", "panels", _DIGEST_FIELD}
    if unexpected_keys:
        raise ValueError(
            "value artifact contains non-training fields: "
            + ", ".join(sorted(unexpected_keys))
        )
    supplied = document.get(_DIGEST_FIELD)
    if not isinstance(supplied, str):
        raise ValueError("value artifact hash is missing")
    if expected_sha256 is not None and supplied != expected_sha256:
        raise ValueError("value artifact hash does not match archive reference")
    unsigned = dict(document)
    unsigned.pop(_DIGEST_FIELD, None)
    expected = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if supplied != expected:
        raise ValueError("value artifact hash verification failed")
    panels = document.get("panels")
    if not isinstance(panels, dict):
        raise ValueError("value artifact panels are missing")
    decoded: dict[str, pd.DataFrame] = {}
    for identifier, payload in panels.items():
        if not isinstance(payload, dict):
            raise ValueError("value artifact panel is malformed")
        unexpected_panel_fields = set(payload) - _VALUE_PANEL_FIELDS
        missing_panel_fields = _VALUE_PANEL_FIELDS - set(payload)
        if unexpected_panel_fields or missing_panel_fields:
            details = []
            if unexpected_panel_fields:
                details.append("unexpected=" + ",".join(sorted(unexpected_panel_fields)))
            if missing_panel_fields:
                details.append("missing=" + ",".join(sorted(missing_panel_fields)))
            raise ValueError(
                "value artifact panel contains non-training fields or is incomplete: "
                + "; ".join(details)
            )
        if not isinstance(payload["index"], list) or not all(
            isinstance(value, str) for value in payload["index"]
        ):
            raise ValueError("value artifact panel index is invalid")
        if not isinstance(payload["columns"], list) or not all(
            isinstance(value, str) for value in payload["columns"]
        ):
            raise ValueError("value artifact panel columns are invalid")
        for name in ("index_name", "index_freq", "columns_name"):
            if payload[name] is not None and not isinstance(payload[name], str):
                raise ValueError(f"value artifact panel {name} is invalid")
        if not isinstance(payload["values"], list) or not all(
            isinstance(row, list) for row in payload["values"]
        ) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for row in payload["values"]
            for value in row
        ):
            raise ValueError("value artifact panel values are invalid")
        if not isinstance(payload["nan_mask"], list) or not all(
            isinstance(row, list) for row in payload["nan_mask"]
        ) or not all(
            isinstance(value, bool) for row in payload["nan_mask"] for value in row
        ):
            raise ValueError("value artifact panel nan_mask is invalid")
        try:
            index = pd.DatetimeIndex(pd.to_datetime(payload["index"]), name=payload["index_name"])
        except (TypeError, ValueError) as exc:
            raise ValueError("value artifact panel index is invalid") from exc
        if payload["index_freq"]:
            try:
                index.freq = pd.tseries.frequencies.to_offset(payload["index_freq"])
            except (TypeError, ValueError) as exc:
                raise ValueError("value artifact panel index_freq is invalid") from exc
        columns = pd.Index(payload["columns"], name=payload["columns_name"])
        values = np.asarray(payload["values"], dtype="float64")
        missing = np.asarray(payload["nan_mask"], dtype=bool)
        if values.shape != missing.shape or values.shape != (len(index), len(columns)):
            raise ValueError("value artifact panel shape is invalid")
        if index.has_duplicates or columns.has_duplicates:
            raise ValueError("value artifact panel axes are duplicated")
        values[missing] = np.nan
        decoded[str(identifier)] = pd.DataFrame(values, index=index, columns=columns)
    return decoded


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    return value


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(_json_safe(payload))
    clean.pop(_DIGEST_FIELD, None)
    clean["sha256"] = hashlib.sha256(_canonical_json(clean)).hexdigest()
    return clean


def _write_exclusive_atomic(target: Path, encoded: bytes) -> None:
    """Publish an immutable file without exposing a partial final path."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.unlink(temporary)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_artifact(path: str | Path, payload: Mapping[str, Any], *, immutable: bool | None = None) -> Path:
    """Write JSON safely; manifests are immutable and progress files are replaceable.

    A filename containing ``manifest`` is treated as an immutable manifest by
    default. Callers can override that inference with ``immutable``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    is_manifest = immutable if immutable is not None else "manifest" in target.name
    document = _manifest_payload(payload) if is_manifest else _json_safe(payload)
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    if is_manifest:
        _write_exclusive_atomic(target, encoded)
    else:
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return target


def read_verified_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest only when its SHA-256 matches its canonical payload."""
    target = Path(path)
    with target.open(encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_nonfinite_json_constant)
    if not isinstance(payload, dict) or not isinstance(payload.get(_DIGEST_FIELD), str):
        raise ValueError("manifest hash is missing")
    supplied = payload[_DIGEST_FIELD]
    unsigned = dict(payload)
    unsigned.pop(_DIGEST_FIELD, None)
    expected = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if supplied != expected:
        raise ValueError("manifest hash verification failed")
    return _json_safe(payload)


def _tree_payload(tree: Any) -> dict[str, Any]:
    return {
        "op": tree.op,
        "field": tree.field,
        "window": tree.window,
        "children": [_tree_payload(child) for child in tree.children],
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            content = handle.read(_NON_GIT_READ_CHUNK)
            if not content:
                break
            digest.update(content)
    return digest.hexdigest()


def _confined_code_path(repository_root: Path, path: str | Path) -> tuple[str, Path]:
    root = repository_root.resolve(strict=True)
    candidate = Path(path)
    if ".." in candidate.parts:
        raise ValueError(f"selected code path cannot traverse: {path!r}")
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        common = Path(os.path.commonpath((str(root), str(resolved))))
    except ValueError as exc:
        raise ValueError(f"selected code path is outside repository: {path!r}") from exc
    if common != root:
        raise ValueError(f"selected code path is outside repository: {path!r}")
    if not resolved.is_file():
        raise ValueError(f"selected code path is not a file: {path!r}")
    return resolved.relative_to(root).as_posix(), resolved


def working_tree_patch_hash(repository_root: str | Path) -> str:
    """Hash tracked patches and untracked files in the working tree."""
    root = Path(repository_root)
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return _non_git_working_tree_hash(root)

    digest = hashlib.sha256()
    _hash_git_output(digest, ["git", "diff", "--binary", "--cached"], cwd=root)
    digest.update(b"\0")
    _hash_git_output(digest, ["git", "diff", "--binary"], cwd=root)
    digest.update(b"\0")
    untracked_paths = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for raw_path in sorted(path for path in untracked_paths if path):
        path = root / os.fsdecode(raw_path)
        digest.update(raw_path)
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.fsencode(os.readlink(path)))
        else:
            with path.open("rb") as handle:
                while True:
                    content = handle.read(_NON_GIT_READ_CHUNK)
                    if not content:
                        break
                    digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_git_output(digest: Any, command: list[str], *, cwd: Path) -> None:
    """Hash Git output without retaining the complete output in memory."""
    with tempfile.TemporaryFile() as handle:
        subprocess.run(command, cwd=cwd, check=True, stdout=handle)
        handle.seek(0)
        while True:
            content = handle.read(_NON_GIT_READ_CHUNK)
            if not content:
                break
            digest.update(content)


def _non_git_working_tree_hash(root: Path) -> str:
    """Hash every relevant entry without following symlink directories."""
    digest = hashlib.sha256(b"non-git-working-tree-v1\0")
    file_count = 0
    byte_count = 0
    root = root.resolve()
    if not root.is_dir():
        digest.update(b"missing-root\0")
        return digest.hexdigest()

    def visit(directory: Path) -> None:
        nonlocal file_count, byte_count
        entries = []
        try:
            with os.scandir(directory) as handle:
                entries = sorted(handle, key=lambda entry: os.fsencode(entry.name))
        except OSError as exc:
            raise ValueError(f"cannot snapshot working-tree directory: {directory}") from exc
        for entry in entries:
            if entry.name == ".git":
                continue
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix().encode("utf-8", "surrogateescape")
            if len(relative) > _NON_GIT_MAX_PATH_BYTES:
                raise ValueError(f"working-tree path is too long: {relative!r}")
            digest.update(b"path\0" + relative + b"\0")
            if entry.is_symlink():
                target = os.readlink(path)
                target_bytes = os.fsencode(target)
                if len(target_bytes) > _NON_GIT_MAX_SYMLINK_TARGET_BYTES:
                    raise ValueError(f"working-tree symlink target is too long: {path}")
                digest.update(b"symlink\0" + target_bytes + b"\0")
            elif entry.is_dir(follow_symlinks=False):
                digest.update(b"directory\0")
                visit(path)
            elif entry.is_file(follow_symlinks=False):
                file_count += 1
                digest.update(b"file\0")
                with path.open("rb") as handle:
                    while True:
                        content = handle.read(_NON_GIT_READ_CHUNK)
                        if not content:
                            break
                        digest.update(content)
                        byte_count += len(content)
                digest.update(b"\0")
            else:
                digest.update(b"unsupported\0")

    visit(root)
    digest.update(f"entries:{file_count};bytes:{byte_count}\0".encode())
    return digest.hexdigest()


def build_provenance(
    *,
    config: Mapping[str, Any],
    stage_content_hashes: Mapping[str, str],
    selected_code_paths: Iterable[str | Path],
    repository_root: str | Path,
    package_names: Iterable[str] = (),
    seed: int,
    operator_version: str,
    backtest_profile: Mapping[str, Any],
    experiment_id: str,
    validation_attempts: int = 0,
    validation_experiment_id: str | None = None,
) -> dict[str, Any]:
    """Build provenance with training and validation identities kept separate."""
    root = Path(repository_root)
    code_hashes = {}
    for path in selected_code_paths:
        relative, resolved = _confined_code_path(root, path)
        code_hashes[relative] = _file_hash(resolved)
    versions = {}
    package_names = tuple(
        dict.fromkeys((*_requirement_names(root), *(str(name) for name in package_names)))
    )
    for name in sorted(package_names):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "config": _json_safe(config),
        "stage_content_hashes": dict(sorted(stage_content_hashes.items())),
        "selected_code_content_hashes": code_hashes,
        "working_tree_patch_hash": working_tree_patch_hash(root),
        "package_versions": versions,
        "seed": seed,
        "operator_version": operator_version,
        "backtest_profile": _json_safe(backtest_profile),
        "training_experiment_id": experiment_id,
        "validation": {
            "experiment_id": validation_experiment_id,
            "attempts": int(validation_attempts),
        },
    }


def training_archive_entry(
    candidate: Any,
    *,
    training_fingerprint: str,
    operator_version: str,
    diagnostics: Mapping[str, Any],
    value_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an archive-safe record containing only the AST and train diagnostics."""
    entry = {
        "expression_id": str(candidate.expression_id),
        "training_only": True,
        "ast": _tree_payload(candidate.tree),
        "training_fingerprint": training_fingerprint,
        "operator_version": operator_version,
        "training_diagnostics": _json_safe(diagnostics),
    }
    direction = getattr(candidate, "direction", None)
    if direction in (-1, 1):
        entry["training_direction"] = direction
    if value_artifact is not None:
        entry["value_artifact"] = _json_safe(value_artifact)
    return entry


def freeze_candidates(
    candidates: Iterable[Any],
    *,
    training_fingerprint: str,
    validation_fingerprint: str,
    config: Mapping[str, Any] | Any,
    profile: Mapping[str, Any] | Any,
    runtime_source_hashes: Mapping[str, str],
    selection_results: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Serialize frozen candidates and provenance without reading market data."""
    records = []
    candidate_ids: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            candidate_id = str(candidate.get("expression_id", candidate.get("hash")))
            tree = candidate.get("tree")
            training_direction = candidate.get("training_direction")
            direction = candidate.get("direction")
            complexity = candidate.get("complexity")
        else:
            candidate_id = str(candidate.expression_id)
            tree = candidate.tree
            training_direction = getattr(candidate, "training_direction", None)
            direction = getattr(candidate, "direction", None)
            complexity = getattr(candidate, "complexity", None)
        if training_direction is not None and direction is not None and training_direction != direction:
            raise ValueError("frozen candidates have conflicting training_direction and direction")
        direction = training_direction if training_direction is not None else direction
        if tree is None or direction not in (-1, 1):
            raise ValueError("frozen candidates require a full AST and direction of -1 or 1")
        if not candidate_id or candidate_id == "None":
            raise ValueError("frozen candidates require non-null expression IDs")
        candidate_ids.append(candidate_id)
        records.append({
            "expression_id": candidate_id,
            "ast": _tree_payload(tree),
            "direction": direction,
            "complexity": complexity,
        })
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("frozen candidates require unique expression IDs")
    accepted = selection_results.get("accepted") if isinstance(selection_results, Mapping) else None
    if not isinstance(accepted, (list, tuple)):
        raise ValueError("selection results must include accepted candidate IDs")
    accepted_ids = [str(item.get("expression_id")) if isinstance(item, Mapping) else str(item) for item in accepted]
    if any(not identifier or identifier == "None" for identifier in accepted_ids) or len(set(accepted_ids)) != len(accepted_ids):
        raise ValueError("selection accepted IDs must be unique and non-null")
    if set(accepted_ids) != set(candidate_ids):
        raise ValueError("frozen candidate set must match accepted validation selection IDs")
    config_payload = dict(config) if isinstance(config, Mapping) else dict(vars(config))
    profile_payload = dict(profile) if isinstance(profile, Mapping) else dict(vars(profile))
    return write_artifact(path, {
        "workflow_version": 1,
        "stage_fingerprints": {"training": str(training_fingerprint), "validation": str(validation_fingerprint)},
        "candidates": records,
        "config": config_payload,
        "profile": profile_payload,
        "runtime_source_hashes": dict(sorted(runtime_source_hashes.items())),
        "selection_results": _json_safe(selection_results),
    }, immutable=True)


__all__ = [
    "build_provenance",
    "freeze_candidates",
    "read_verified_manifest",
    "read_verified_value_artifact",
    "training_archive_entry",
    "ValueArtifact",
    "working_tree_patch_hash",
    "write_artifact",
    "write_value_artifact",
]
