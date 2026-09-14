"""Reproducibility and no-look-ahead checks for ML factor experiments.

The validator deliberately treats a cutoff replay as a new experiment.  It
does not make a copy of a full-run prediction table and trim its rows: callers
must retrain at the cutoff and provide the resulting predictions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import discover_catalog, sha256, write_json
from .config import Config, quarter_folds


FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("negative_shift", re.compile(r"\.shift\s*\(\s*-\s*\d+")),
    ("rolling_center", re.compile(r"\.rolling\s*\([^\n]*\bcenter\s*=\s*True")),
    ("backfill", re.compile(r"\.(?:bfill|backfill)\s*\(")),
    ("forward_asof", re.compile(r"merge_asof\s*\([^\n]*direction\s*=\s*[\"']forward[\"']")),
)


def scan_source(path: str | Path) -> list[dict[str, Any]]:
    """Return source locations containing prohibited causal patterns."""
    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")
    findings: list[dict[str, Any]] = []
    for pattern_name, expression in FORBIDDEN_PATTERNS:
        for match in expression.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            line_text = text.splitlines()[line - 1].strip()
            findings.append(
                {
                    "path": str(source_path),
                    "line": line,
                    "pattern": pattern_name,
                    "text": line_text,
                }
            )
    return sorted(findings, key=lambda item: (item["path"], item["line"], item["pattern"]))


def static_scan(paths: str | Path | Iterable[str | Path]) -> dict[str, Any]:
    """Scan one or more Python files and return a JSON-safe report."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    findings: list[dict[str, Any]] = []
    scanned: list[str] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            candidates = sorted(path.rglob("*.py"))
        else:
            candidates = [path]
        for candidate in candidates:
            scanned.append(str(candidate))
            findings.extend(scan_source(candidate))
    return {
        "status": "verified" if not findings else "failed",
        "scanned": sorted(set(scanned)),
        "findings": findings,
    }


def _normalise_cutoff(cutoff: Any) -> pd.Timestamp:
    value = pd.Timestamp(cutoff)
    if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
        raise ValueError("cutoff must be a timezone-naive calendar date")
    return value.normalize()


def _validate_predictions(result: Any) -> pd.DataFrame:
    if not isinstance(result, pd.DataFrame):
        raise TypeError("replay callback must return a pandas DataFrame")
    required = {"date", "instrument", "factor"}
    if not required.issubset(result.columns):
        raise ValueError("replay predictions must contain date, instrument, and factor")
    frame = result.loc[:, ["date", "instrument", "factor"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True).dt.tz_localize(None).dt.normalize()
    frame["factor"] = pd.to_numeric(frame["factor"], errors="raise")
    if not np.isfinite(frame["factor"].to_numpy(dtype=float)).all():
        raise ValueError("replay predictions must contain finite factors")
    if frame.duplicated(["date", "instrument"]).any():
        raise ValueError("replay predictions contain duplicate date/instrument rows")
    return frame.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)


def replay_model(
    run_or_retrain: str | Path | Callable[[pd.Timestamp], pd.DataFrame],
    cutoff: Any,
    retrain_fn: Callable[[pd.Timestamp], pd.DataFrame] | None = None,
    *,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Retrain a model at *cutoff* and return its predictions.

    The callback form is intentionally public and easy to test.  For a run
    directory, the default implementation reconstructs the catalog from its
    immutable snapshot and executes ``prepare_fold`` + ``select_and_refit``
    for every fold through the cutoff.  Filtering to the requested cutoff is
    only done after that retraining has completed.
    """
    cutoff_ts = _normalise_cutoff(cutoff)
    if callable(run_or_retrain):
        callback = run_or_retrain
    elif retrain_fn is not None:
        callback = retrain_fn
    else:
        run_dir = Path(run_or_retrain)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        config = Config.from_dict(manifest["config"])
        inputs = run_dir / "inputs"
        factor_dir = inputs / "factors"
        h5_path = inputs / "crypto_quant.h5"
        catalog = discover_catalog(factor_dir, config)
        from .training import prepare_fold, predict_quarter, select_and_refit

        names = [model_name] if model_name else list(config.models)
        predictions: list[pd.DataFrame] = []
        for name in names:
            for fold in quarter_folds(config, cutoff_ts.date()):
                if fold.retrain_at > cutoff_ts:
                    continue
                prepared = prepare_fold(catalog, h5_path, fold, config)
                model, _table, audit = select_and_refit(prepared, fold, config, name)
                frame, _coverage = predict_quarter(model, prepared.features, catalog, h5_path, fold, audit)
                predictions.append(frame)
        if not predictions:
            return pd.DataFrame(columns=["date", "instrument", "factor"])
        callback = lambda _cutoff: pd.concat(predictions, ignore_index=True)

    # Calling the callback is the essential replay operation.  It is never
    # replaced by slicing a saved output table.
    result = callback(cutoff_ts)
    return _validate_predictions(result)


def compare_cutoff(
    full_predictions: pd.DataFrame,
    replay_predictions: pd.DataFrame,
    cutoff: Any,
    *,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Compare full-run and independently retrained prefix predictions."""
    cutoff_ts = _normalise_cutoff(cutoff)
    full = _validate_predictions(full_predictions)
    replay = _validate_predictions(replay_predictions)
    full = full.loc[full["date"] <= cutoff_ts]
    replay = replay.loc[replay["date"] <= cutoff_ts]
    keys = ["date", "instrument"]
    merged = full.merge(replay, on=keys, how="outer", suffixes=("_full", "_replay"), indicator=True)
    missing = merged.loc[merged["_merge"] != "both", keys + ["_merge"]]
    both = merged.loc[merged["_merge"] == "both"]
    if both.empty:
        max_abs_diff = 0.0 if full.empty and replay.empty else float("inf")
    else:
        max_abs_diff = float(np.max(np.abs(both["factor_full"] - both["factor_replay"])))
    return {
        "cutoff": cutoff_ts.date().isoformat(),
        "status": "verified" if missing.empty and max_abs_diff <= atol else "failed",
        "max_abs_diff": max_abs_diff,
        "full_rows": int(len(full)),
        "replay_rows": int(len(replay)),
        "missing_rows": int(len(missing)),
        "atol": atol,
    }


def verify_hashes(run_dir: str | Path, *, code_root: str | Path | None = None) -> dict[str, Any]:
    """Verify every immutable snapshot and code-provenance hash."""
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    failures: list[dict[str, str]] = []
    for relative, expected in manifest.get("hashes", {}).items():
        path = root / "inputs" / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            failures.append({"path": relative, "expected": str(expected), "actual": str(actual)})
    code_files = manifest.get("code", {}).get("files", [])
    if code_root is None and code_files:
        # Runs are commonly written below ``outputs/``.  Find the nearest
        # ancestor containing the recorded relative source paths so that the
        # manifest remains canonical and portable without absolute paths.
        for ancestor in (root, *root.parents):
            if all((ancestor / item["path"]).is_file() for item in code_files):
                code_root = ancestor
                break
    code_base = Path(code_root) if code_root is not None else root
    for item in code_files:
        path = code_base / item["path"]
        actual = sha256(path) if path.is_file() else None
        if actual != item.get("sha256"):
            failures.append({"path": item["path"], "expected": str(item.get("sha256")), "actual": str(actual)})
    return {"status": "verified" if not failures else "failed", "failures": failures}


__all__ = [
    "FORBIDDEN_PATTERNS",
    "scan_source",
    "static_scan",
    "replay_model",
    "compare_cutoff",
    "verify_hashes",
]
