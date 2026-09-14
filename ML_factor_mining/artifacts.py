"""Immutable input catalog and run snapshot helpers for ML factor research."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


_CHUNK_SIZE = 1024 * 1024


def sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of *path*, reading it in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_replace(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: str | Path, payload: Any) -> None:
    """Write deterministic, finite JSON to *path* atomically."""
    target = Path(path)

    def _write(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    _atomic_replace(target, _write)


def write_parquet(path: str | Path, frame) -> None:
    """Write a parquet DataFrame atomically."""
    target = Path(path)
    _atomic_replace(target, lambda temporary: frame.to_parquet(temporary, index=False))


def _metadata_value(metadata: Mapping[str, Any], name: str, default: Any = None) -> Any:
    """Read metadata fields from either the catalog envelope or its payload."""
    if name in metadata:
        return metadata[name]
    nested = metadata.get("metadata")
    if isinstance(nested, Mapping) and name in nested:
        return nested[name]
    return default


def _settings(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("settings", "setting"):
        value = _metadata_value(metadata, key)
        if isinstance(value, Mapping):
            return value
    return {}


def discover_catalog(directory: str | Path, config: Any) -> list[dict[str, Any]]:
    """Discover valid factor value/metadata pairs in a deterministic order."""
    root = Path(directory)
    include = set(getattr(config, "include", ()) or ())
    exclude = set(getattr(config, "exclude", ()) or ())
    found: dict[str, dict[str, Any]] = {}
    included_ids: set[str] = set()

    for metadata_path in sorted(root.glob("*.meta.json")):
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"metadata must be an object: {metadata_path}")

        # Evaluation tables have sidecar metadata too.  They are deliberately
        # not part of the immutable factor catalog.
        catalog_name = _metadata_value(metadata, "catalog")
        if catalog_name in {"evaluation", "eval", "evaluation_sidecar"}:
            continue

        factor_id = metadata.get("factor_id")
        if not isinstance(factor_id, str) or not factor_id:
            raise ValueError(f"metadata missing factor_id: {metadata_path}")
        if factor_id in exclude:
            continue
        if include and factor_id not in include:
            continue
        included_ids.add(factor_id)

        if factor_id in found:
            raise ValueError(f"duplicate factor_id {factor_id!r}")
        settings = _settings(metadata)
        if settings.get("universe") != "historical_top50":
            raise ValueError(
                f"factor {factor_id!r} settings.universe must be 'historical_top50'"
            )
        if _metadata_value(metadata, "source_type") != "module":
            raise ValueError(f"factor {factor_id!r} source_type must be 'module'")

        values_path = metadata_path.with_name(
            metadata_path.name.removesuffix(".meta.json") + ".parquet"
        )
        if not values_path.is_file():
            raise FileNotFoundError(f"missing parquet for {factor_id!r}: {values_path}")
        found[factor_id] = {
            "factor_id": factor_id,
            "values": values_path,
            "metadata": metadata_path,
        }

    missing = include - set(found)
    if missing:
        raise ValueError("requested factors missing from catalog: " + ", ".join(sorted(missing)))
    if not found:
        raise ValueError("factor catalog is empty")
    return [found[key] for key in sorted(found)]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _package_versions() -> dict[str, str]:
    names = ("pandas", "numpy", "scipy", "pyarrow", "scikit-learn", "lightgbm", "xgboost")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def snapshot_inputs(
    factor_dir: str | Path,
    h5_path: str | Path,
    run_dir: str | Path,
    config: Any,
) -> Path:
    """Create an immutable run input snapshot and its provenance manifest.

    The destination is created exclusively and intentionally remains in place
    if any source changes or a copy/manifest operation fails.
    """
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=False)
    inputs = destination / "inputs"
    factors_destination = inputs / "factors"
    inputs.mkdir()
    factors_destination.mkdir()

    catalog = discover_catalog(factor_dir, config)
    source_paths = [Path(h5_path).resolve()]
    for record in catalog:
        source_paths.extend(
            [Path(record["values"]).resolve(), Path(record["metadata"]).resolve()]
        )
    if any(not path.is_file() for path in source_paths):
        missing = next(path for path in source_paths if not path.is_file())
        raise FileNotFoundError(missing)

    source_hashes = {str(path): sha256(path) for path in source_paths}
    target_paths: dict[Path, Path] = {}
    h5_source = source_paths[0]
    target_paths[h5_source] = inputs / "crypto_quant.h5"
    for source in source_paths[1:]:
        target_paths[source] = factors_destination / source.name

    target_hashes: dict[str, str] = {}
    for source, target in target_paths.items():
        shutil.copy2(source, target)
        if sha256(target) != source_hashes[str(source)]:
            raise IOError(f"snapshot hash mismatch for {source}")
        target_hashes[target.relative_to(inputs).as_posix()] = source_hashes[str(source)]

    source_hashes_after = {str(path): sha256(path) for path in source_paths}
    if source_hashes_after != source_hashes:
        raise RuntimeError("source changed while snapshotting inputs")

    manifest = {
        "config": _json_safe(
            asdict(config)
            if is_dataclass(config)
            else vars(config)
            if hasattr(config, "__dict__")
            else config
        ),
        "hashes": target_hashes,
        "research_scope": "model_level_oos",
        "catalog_timing": "retrospective_snapshot",
        "source_paths": [str(path) for path in source_paths],
        "versions": _package_versions(),
    }
    write_json(destination / "manifest.json", manifest)
    return inputs
