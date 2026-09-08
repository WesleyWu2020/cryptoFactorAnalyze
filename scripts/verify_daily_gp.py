#!/usr/bin/env python3
"""Static and cutoff-replay verification for daily GP runtime and exports."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Genetic_Algorithm.evaluator import evaluate_tree
from Genetic_Algorithm.export import _node_from_payload
from Genetic_Algorithm.expression import required_fields
from Genetic_Algorithm.artifacts import read_verified_manifest
from factor_common.data_provider import DataProvider
from factor_common.validation import EVALUATION_ONLY_ALLOWANCE, check_cutoff, scan_future_leaks


def _runtime_paths(run_dir: Path) -> list[Path]:
    paths = sorted((ROOT / "Genetic_Algorithm").glob("*.py"))
    paths.append(Path(__file__).resolve())
    # Exported wrappers are runtime code too; only scan generated GP wrappers.
    paths.extend(sorted(path for path in run_dir.rglob("GP_*.py") if path.is_file()))
    return paths


def _exports(run_dir: Path) -> list[dict]:
    result = []
    for path in sorted(run_dir.rglob("*.export_manifest.json")):
        payload = read_verified_manifest(path)
        if not isinstance(payload.get("ast"), dict):
            raise ValueError(f"malformed export manifest: {path}")
        result.append({"manifest": path, **payload})
    return result


def _load_wrapper(path: Path):
    """Execute the exported module, so verification exercises its contract."""
    name = f"_daily_gp_verify_{path.stem}_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load exported wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "calc_factor", None)):
        raise ValueError(f"exported wrapper lacks calc_factor: {path}")
    return module


def _verify_export_integrity(payload: dict) -> None:
    """Bind dynamic execution to the exported wrapper and runtime closure."""
    wrapper = payload["wrapper"]
    expected_wrapper = payload.get("wrapper_sha256")
    actual_wrapper = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    if not isinstance(expected_wrapper, str) or actual_wrapper != expected_wrapper:
        raise ValueError("export wrapper sha256 does not match manifest")
    expected_runtime = payload.get("runtime_module_hashes")
    if not isinstance(expected_runtime, dict):
        raise ValueError("export runtime module hash mapping is missing")
    for relative, expected in expected_runtime.items():
        path = ROOT / relative
        if not path.is_file() or not isinstance(expected, str):
            raise ValueError(f"export runtime module hash mapping is invalid: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"export runtime module hash mismatch: {relative}")


def _compute_export(h5: Path, payload: dict, cutoff) -> pd.DataFrame:
    end = pd.Timestamp(cutoff) if cutoff is not None else DataProvider(h5).get_time_range()[1]
    if end is None:
        raise ValueError("H5 contains no market history")
    _verify_export_integrity(payload)
    tree = _node_from_payload(payload["ast"])
    provider = DataProvider(h5, as_of=cutoff)
    warmup = int(payload.get("warmup_bars", 0))
    start = pd.Timestamp("2024-01-01") - pd.Timedelta(days=warmup)
    start = max(start, provider.get_time_range()[0])
    fields = sorted(required_fields(tree))
    context = {field: provider.get_single_data(field, start=start, end=end) for field in fields}
    eligible = provider.get_universe(start=start, end=end)
    wrapper = _load_wrapper(payload["wrapper"])
    values = wrapper.calc_factor({**context, "__eligible__": eligible})
    # HDF backends can surface the identical daily axis at different datetime
    # resolutions. The factor contract is daily, so canonicalize resolution
    # before the strict cutoff-axis comparison.
    values = values.copy()
    values.index = pd.DatetimeIndex(values.index).astype("datetime64[ns]")
    return values


def verify(run_dir: Path, h5: Path) -> dict:
    paths = _runtime_paths(run_dir)
    findings = scan_future_leaks(paths)
    allowances = sorted(EVALUATION_ONLY_ALLOWANCE)
    exports = _exports(run_dir)
    cutoffs = ["2024-12-31", "2025-01-01", "2026-01-01"]
    dynamic, failures = [], []
    for exported in exports:
        try:
            wrapper = exported["manifest"].with_name(exported["identifier"] + ".py")
            result = check_cutoff(lambda cutoff, item={**exported, "wrapper": wrapper}: _compute_export(h5, item, cutoff), cutoffs, atol=1e-10)
            dynamic.append({"manifest": str(exported["manifest"]), **result})
        except Exception as exc:  # retain evidence in JSON and fail below
            failures.append({"manifest": str(exported["manifest"]), "error": f"{type(exc).__name__}: {exc}"})
    payload = {
        "runtime_paths": [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path) for path in paths],
        "static_findings": findings,
        "allowances": allowances,
        "cutoffs": cutoffs,
        "dynamic": dynamic,
        "failures": failures,
        "status": "passed" if not findings and not failures else "failed",
    }
    (run_dir / "verification.json").write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--h5", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.run_dir, args.h5)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, SyntaxError) as exc:
        # Even an unreadable source/manifest leaves audit evidence at the
        # requested destination instead of only an ephemeral stderr message.
        args.run_dir.mkdir(parents=True, exist_ok=True)
        failure = {"status": "failed", "static_findings": [], "allowances": sorted(EVALUATION_ONLY_ALLOWANCE), "cutoffs": [], "dynamic": [], "failures": [{"error": f"{type(exc).__name__}: {exc}"}]}
        (args.run_dir / "verification.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "verification": str(args.run_dir / "verification.json")}, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
