"""Immutable manifests and training-only provenance artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


_DIGEST_FIELD = "sha256"
_VALUE_ARTIFACT_VERSION = 1


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
    payload = {
        "artifact_version": _VALUE_ARTIFACT_VERSION,
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
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.write(b"\n")
    finally:
        if descriptor != -1:
            os.close(descriptor)
    return ValueArtifact(target, str(document[_DIGEST_FIELD]))


def read_verified_value_artifact(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, pd.DataFrame]:
    """Verify and decode a canonical training value panel artifact."""
    target = Path(path)
    with target.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or document.get("artifact_version") != _VALUE_ARTIFACT_VERSION:
        raise ValueError("unsupported value artifact")
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
        index = pd.DatetimeIndex(pd.to_datetime(payload["index"]), name=payload.get("index_name"))
        if payload.get("index_freq"):
            index.freq = pd.tseries.frequencies.to_offset(payload["index_freq"])
        columns = pd.Index(payload["columns"], name=payload.get("columns_name"))
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
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(target, flags, 0o644)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(encoded)
                handle.write(b"\n")
        finally:
            if descriptor != -1:
                os.close(descriptor)
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
        payload = json.load(handle)
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def working_tree_patch_hash(repository_root: str | Path) -> str:
    """Hash the working-tree patch, including staged and unstaged changes."""
    root = Path(repository_root)
    staged = subprocess.run(
        ["git", "diff", "--binary", "--cached"], cwd=root, check=True, capture_output=True
    ).stdout
    unstaged = subprocess.run(
        ["git", "diff", "--binary"], cwd=root, check=True, capture_output=True
    ).stdout
    return hashlib.sha256(staged + b"\0" + unstaged).hexdigest()


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
    code_hashes = {
        str(Path(path)): _file_hash(root / path) for path in sorted(map(str, selected_code_paths))
    }
    versions = {}
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
        "ast": _tree_payload(candidate.tree),
        "training_fingerprint": training_fingerprint,
        "operator_version": operator_version,
        "training_diagnostics": _json_safe(diagnostics),
    }
    if value_artifact is not None:
        entry["value_artifact"] = _json_safe(value_artifact)
    return entry


__all__ = [
    "build_provenance",
    "read_verified_manifest",
    "read_verified_value_artifact",
    "training_archive_entry",
    "ValueArtifact",
    "working_tree_patch_hash",
    "write_artifact",
    "write_value_artifact",
]
