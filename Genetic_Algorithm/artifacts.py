"""Immutable manifests and training-only provenance artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


_DIGEST_FIELD = "sha256"


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
) -> dict[str, Any]:
    """Return an archive-safe record containing only the AST and train diagnostics."""
    return {
        "expression_id": str(candidate.expression_id),
        "ast": _tree_payload(candidate.tree),
        "training_fingerprint": training_fingerprint,
        "operator_version": operator_version,
        "training_diagnostics": _json_safe(diagnostics),
    }


__all__ = [
    "build_provenance",
    "read_verified_manifest",
    "training_archive_entry",
    "working_tree_patch_hash",
    "write_artifact",
]
