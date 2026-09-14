"""Command line entry point for reproducible quarterly ML factor runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid
from typing import Any

import pandas as pd

from .artifacts import _json_safe, discover_catalog, sha256, snapshot_inputs, write_json
from .config import Config, quarter_folds
from .models import require_backends
from .validation import compare_cutoff, replay_model, sha256_bytes, static_scan, verify_hashes


def _resolve_path(value: str | Path, root: str | Path) -> Path:
    path = Path(value).expanduser()
    base = Path(root).expanduser().resolve(strict=False)
    return path if path.is_absolute() else base / path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_source_files(manifest: dict[str, Any], code_root: Path) -> list[Path]:
    """Select ML runtime sources for static scans, retaining framework hashes."""
    code_info = manifest.get("code", {})
    code_files = code_info.get("files", []) if isinstance(code_info, dict) else []
    return [
        code_root / item["path"]
        for item in code_files
        if isinstance(item, dict) and Path(item.get("path", "")).parts[:1] == ("ML_factor_mining",)
    ]


def _validate_cutoffs(cutoffs: list[str], oos_start: Any) -> list[pd.Timestamp]:
    """Normalize verification cutoffs and reject dates before the OOS window."""
    start = pd.Timestamp(oos_start)
    if pd.isna(start) or start.tzinfo is not None or start != start.normalize():
        raise ValueError("config.oos_start must be a timezone-naive calendar date")
    validated: list[pd.Timestamp] = []
    for value in cutoffs:
        cutoff = pd.Timestamp(value)
        if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
            raise ValueError("cutoffs must be timezone-naive calendar dates")
        if cutoff < start:
            raise ValueError("cutoffs must be on or after config.oos_start")
        validated.append(cutoff.normalize())
    return validated


def canonical_relative(path: str | Path, root: str | Path) -> str:
    """Return a canonical POSIX path, rejecting paths outside *root*."""
    candidate = Path(path).expanduser().resolve(strict=False)
    base = Path(root).expanduser().resolve(strict=False)
    try:
        relative = candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path {candidate} is outside {base}") from exc
    return relative.as_posix() or "."


def _parser_path(value: str) -> str:
    if not value or "\x00" in value:
        raise argparse.ArgumentTypeError("path must be nonempty and cannot contain NUL")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-factor-mining")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a reproducible quarterly ML experiment")
    run.add_argument("--config", required=True, type=_parser_path)
    run.add_argument("--factor-dir", "--factors", dest="factor_dir", required=True, type=_parser_path)
    run.add_argument("--h5", "--h5-path", dest="h5_path", required=True, type=_parser_path)
    run.add_argument("--output", "--output-dir", dest="output", required=True, type=_parser_path)
    run.add_argument("--code-root", default=".", type=_parser_path)
    run.add_argument("--evidence-end", default=None)
    run.add_argument("--no-evaluate", action="store_true", help="only produce model predictions")

    verify = commands.add_parser("verify", help="verify a run snapshot and cutoff replays")
    verify.add_argument("run_dir", type=_parser_path)
    verify.add_argument("--cutoff", action="append", default=[])
    verify.add_argument("--atol", type=float, default=1e-12)
    verify.add_argument("--code-root", default=None, type=_parser_path)
    return parser


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def _state(path: Path, status: str, **values: Any) -> None:
    payload = {"status": status, **values}
    write_json(path / "state.json", payload)


def _load_config(path: Path) -> Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must contain a JSON object")
    return Config.from_dict(payload)


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _code_provenance(root: Path) -> dict[str, Any]:
    files = []
    for directory in (root / "ML_factor_mining", root / "factor_common"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files.append({"path": canonical_relative(path, root), "sha256": sha256(path)})
    return {"root": ".", "git_revision": _git_revision(root), "files": files}


def _snapshot_run(args: argparse.Namespace, config: Config, run_dir: Path, code_root: Path) -> list[dict[str, Any]]:
    staging = run_dir.parent / f".{run_dir.name}.snapshot-{uuid.uuid4().hex}"
    try:
        snapshot_inputs(args.factor_dir, args.h5_path, staging, config)
        shutil.move(str(staging / "inputs"), str(run_dir / "inputs"))
        shutil.move(str(staging / "manifest.json"), str(run_dir / "manifest.json"))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"] = _json_safe(asdict(config))
    canonical_config = json.dumps(manifest["config"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["config_sha256"] = sha256_bytes(canonical_config.encode())
    manifest["code"] = _code_provenance(code_root)
    manifest["paths"] = {
        "inputs": "inputs",
        "factors": "inputs/factors",
        "h5": "inputs/crypto_quant.h5",
        "state": "state.json",
    }
    write_json(manifest_path, manifest)
    return discover_catalog(run_dir / "inputs" / "factors", config)


def _run(args: argparse.Namespace) -> int:
    repo_root = _repository_root()
    run_dir = _resolve_path(args.output, repo_root).resolve(strict=False)
    if run_dir.exists():
        raise FileExistsError(f"output run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    _state(run_dir, "running", command="run")
    try:
        code_root = _resolve_path(args.code_root, repo_root).resolve(strict=True)
        config_path = _resolve_path(args.config, repo_root).resolve(strict=True)
        args.factor_dir = _resolve_path(args.factor_dir, repo_root).resolve(strict=True)
        args.h5_path = _resolve_path(args.h5_path, repo_root).resolve(strict=True)
        config = _load_config(config_path)
        require_backends(config.models)
        catalog = _snapshot_run(args, config, run_dir, code_root)
        evidence_end = args.evidence_end or config.end
        folds = quarter_folds(config, evidence_end)
        if not folds:
            raise ValueError("no quarterly folds available through evidence_end")
        from .training import predict_quarter, prepare_fold, save_quarter, select_and_refit

        all_predictions: dict[str, list[pd.DataFrame]] = {name: [] for name in config.models}
        all_coverage: dict[str, list[pd.DataFrame]] = {name: [] for name in config.models}
        failed_candidates: dict[str, Any] = {}
        h5_path = run_dir / "inputs" / "crypto_quant.h5"
        for name in config.models:
            model_root = run_dir / "models" / name
            for fold in folds:
                if fold.retrain_at > pd.Timestamp(evidence_end):
                    continue
                try:
                    prepared = prepare_fold(catalog, h5_path, fold, config)
                    model, candidates, audit = select_and_refit(prepared, fold, config, name)
                    predictions, coverage = predict_quarter(model, prepared.features, catalog, h5_path, fold, audit)
                    save_quarter(model_root / fold.quarter, model, candidates, audit, predictions, coverage)
                    all_predictions[name].append(predictions)
                    all_coverage[name].append(coverage)
                except Exception as exc:
                    diagnostics = getattr(exc, "records", None)
                    failed_candidates[f"{name}/{fold.quarter}"] = {
                        "error": _error_payload(exc),
                        "candidates": diagnostics or [],
                    }
                    write_json(run_dir / "failed_candidates.json", failed_candidates)
                    raise
        outputs = {}
        for name, frames in all_predictions.items():
            if not frames:
                continue
            predictions = pd.concat(frames, ignore_index=True)
            coverage = pd.concat(all_coverage[name], ignore_index=True)
            model_dir = run_dir / "models" / name
            output = model_dir / "predictions.parquet"
            predictions.to_parquet(output, index=False)
            coverage.to_parquet(model_dir / "coverage.parquet", index=False)
            outputs[name] = {"predictions": str(output.relative_to(run_dir)), "quarters": len(frames)}
            if not args.no_evaluate:
                from .evaluation import evaluate_oos

                evaluate_oos(predictions, model_dir, factor_name=f"ml_{name}", config=config, h5_path=h5_path, evidence_end=evidence_end, coverage=coverage, plot=False)
        _state(run_dir, "complete", folds=len(folds), outputs=outputs)
        return 0
    except Exception as exc:
        _state(run_dir, "failed", error=_error_payload(exc))
        return 1


def _verify(args: argparse.Namespace) -> int:
    repo_root = _repository_root()
    run_dir = _resolve_path(args.run_dir, repo_root).resolve(strict=True)
    payload: dict[str, Any] = {"status": "running", "run_dir": str(run_dir)}
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if args.code_root:
            code_root = _resolve_path(args.code_root, repo_root).resolve(strict=True)
        else:
            code_root = None
            code_files = manifest.get("code", {}).get("files", [])
            for ancestor in (run_dir, *run_dir.parents):
                if all((ancestor / item["path"]).is_file() for item in code_files):
                    code_root = ancestor
                    break
            code_root = code_root or run_dir.parent
        hashes = verify_hashes(run_dir, code_root=code_root)
        source_files = _runtime_source_files(manifest, code_root)
        scan = static_scan(source_files) if source_files else {"status": "verified", "findings": [], "scanned": []}
        payload.update(hashes=hashes, static_scan=scan)
        full_path = run_dir / "models"
        cutoffs = _validate_cutoffs(args.cutoff or [manifest["config"]["oos_start"]], manifest["config"]["oos_start"])
        replay_results = []
        for model_dir in sorted(path for path in full_path.iterdir() if path.is_dir()) if full_path.is_dir() else []:
            full_predictions_path = model_dir / "predictions.parquet"
            if not full_predictions_path.is_file():
                continue
            full_predictions = pd.read_parquet(full_predictions_path)
            for cutoff in cutoffs:
                replay = replay_model(run_dir, cutoff, model_name=model_dir.name)
                replay_results.append(compare_cutoff(full_predictions, replay, cutoff, atol=args.atol))
        if not replay_results:
            raise ValueError("no cutoff replay results")
        payload["cutoff"] = replay_results
        payload["status"] = "verified" if hashes["status"] == "verified" and scan["status"] == "verified" and all(item["status"] == "verified" for item in replay_results) else "failed"
        write_json(run_dir / "verification.json", payload)
        return 0 if payload["status"] == "verified" else 1
    except Exception as exc:
        payload.update(status="failed", error=_error_payload(exc))
        write_json(run_dir / "verification.json", payload)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run(args) if args.command == "run" else _verify(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "canonical_relative", "main"]
