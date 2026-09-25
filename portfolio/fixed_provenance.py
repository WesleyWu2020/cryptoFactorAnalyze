"""Reproducibility metadata for fixed-portfolio research runs."""
from __future__ import annotations

import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

from factor_common.storage import _json_safe, snapshot_source_stats

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = (
    "factor_common",
    "portfolio",
    "Genetic_Algorithm",
    "data/crypto_quant",
    "factor_analyse/factor_mining",
)


def code_snapshot(extra_paths: Iterable[str | Path] = ()) -> dict[str, str]:
    """Return SHA-256 hashes for the research code and explicit extra files."""
    paths = {Path(path).resolve(strict=True) for path in extra_paths}
    for directory in CODE_DIRS:
        root = ROOT / directory
        if root.exists():
            paths.update(path.resolve() for path in root.rglob("*.py"))
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def frame_hash(frame: Any) -> str:
    """Hash a frame's axes, column order, and values without writing a file."""
    return hashlib.sha256(frame.to_csv().encode("utf-8")).hexdigest()


class RecordingProvider:
    """Delegate provider reads while retaining compact content fingerprints."""

    def __init__(self, provider: Any):
        self.provider = provider
        self.reads: list[dict[str, Any]] = []

    @property
    def symbols(self):
        return self.provider.symbols

    def _record(self, method: str, *args: Any, **kwargs: Any):
        frame = getattr(self.provider, method)(*args, **kwargs)
        self.reads.append(
            {
                "method": method,
                "args": _json_safe(args),
                "kwargs": _json_safe(kwargs),
                "sha256": frame_hash(frame),
                "shape": list(frame.shape),
            }
        )
        return frame

    def get_single_data(self, field, *, start, end):
        return self._record("get_single_data", field, start=start, end=end)

    def get_universe(self, *, start, end):
        return self._record("get_universe", start=start, end=end)

    def get_quality(self, *, start, end, symbols):
        return self._record("get_quality", start=start, end=end, symbols=list(symbols))

    def get_funding(self, *, start, end, symbols):
        return self._record("get_funding", start=start, end=end, symbols=list(symbols))


def environment_metadata() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("pandas", "numpy", "scipy", "tables", "pyarrow", "fastparquet"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def package_versions() -> dict[str, str | None]:
    """Alias used by callers that want only environment package metadata."""
    return environment_metadata()


def assert_source_unchanged(paths: Iterable[str | Path], before) -> None:
    if snapshot_source_stats(paths) != before:
        raise RuntimeError("input source changed during portfolio evaluation")


__all__ = [
    "ROOT", "CODE_DIRS", "code_snapshot", "frame_hash", "RecordingProvider",
    "environment_metadata", "package_versions", "assert_source_unchanged",
]
