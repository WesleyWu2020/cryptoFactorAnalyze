"""Explicit, audit-first command line for the daily GP workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import freeze_candidates, read_verified_manifest, write_artifact
from .config import STAGES, load_config
from .data import load_stage, run_training_audit
from .evolution import SearchFormationError, search as evolution_search
from .evaluator import evaluate_tree
from .export import export_factor
from .expression import Node
from .features import RAW_FIELDS
from .replay import ReplayEvaluationError, replay, test_manifest
from .search import run_search
from .selection import select_validation


def _runtime_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = ("config", "data", "evolution", "evaluator", "expression", "features", "operators", "selection", "replay", "export")
    return {f"Genetic_Algorithm/{name}.py": hashlib.sha256((package / f"{name}.py").read_bytes()).hexdigest() for name in names}


def _tree(payload: dict[str, Any]) -> Node:
    return Node(payload["op"], tuple(_tree(child) for child in payload["children"]), payload["field"], payload["window"])


def _archive_candidates(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "training_candidates.json"
    if not path.is_file():
        raise FileNotFoundError(f"training candidate archive is missing: {path}")
    document = read_verified_manifest(path)
    entries = document.get("candidates")
    if not isinstance(entries, list):
        raise ValueError("training candidate archive is malformed")
    candidates = []
    for item in entries:
        direction = item.get("training_direction")
        if direction not in (-1, 1):
            raise ValueError(f"training candidate {item.get('expression_id')!r} lacks a valid training direction")
        candidates.append({"expression_id": item["expression_id"], "tree": _tree(item["ast"]), "direction": direction, "training_direction": direction, "complexity": 0})
    return candidates


def _write_json(path: Path, payload: dict[str, Any], *, immutable: bool = False) -> Path:
    return write_artifact(path, payload, immutable=immutable)


def _verified(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return read_verified_manifest(path)


def _load_run_config(path: Path) -> Any:
    document = _verified(path, "configuration")
    from .config import SearchConfig
    return SearchConfig(**{key: value for key, value in document.items() if key != "sha256"})


def _validation_evidence(candidate: dict[str, Any], stage_data: Any, config: Any) -> dict[str, Any]:
    """Compute gate evidence from the bounded validation panel, never defaults."""
    from factor_common.labels import make_labels
    from factor_common.metrics import _daily_ic

    values = evaluate_tree(candidate["tree"], stage_data.features, stage_data.eligible)
    values = values.loc[stage_data.opens.index]
    labels = make_labels(stage_data.opens, config.hold_days)
    quality = stage_data.quality_eligible.fillna(False).astype(bool)
    daily = _daily_ic(values.where(quality), labels.where(quality))
    daily = daily.loc[(daily["n_pairs"] >= config.min_pairs) & pd.to_numeric(daily["rank_ic"], errors="coerce").notna()].copy()
    direction = candidate["direction"]
    daily["directed_ic"] = daily["rank_ic"] * direction
    eligible_days = int(quality.any(axis=1).sum())
    valid_days = int(len(daily))
    denominator = int(quality.to_numpy(dtype=bool).sum())
    observed = int(values.where(quality).notna().to_numpy().sum())
    quarters = pd.to_datetime(daily["date"]).dt.quarter if valid_days else pd.Series(dtype="int64")
    return {
        "direction": direction,
        "day_coverage": valid_days / eligible_days if eligible_days else 0.0,
        "cell_coverage": observed / denominator if denominator else 0.0,
        "quarter_valid_days": {f"Q{quarter}": int((quarters == quarter).sum()) for quarter in range(1, 5)},
        "mean_ic": float(daily["directed_ic"].mean()) if valid_days else None,
        "quarter_means": {f"Q{quarter}": (float(daily.loc[quarters == quarter, "directed_ic"].mean()) if bool((quarters == quarter).any()) else None) for quarter in range(1, 5)},
        "values_fingerprint": stage_data.fingerprint,
    }


def _audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage != "train":
        raise ValueError("audit supports only the frozen train stage")
    output = Path(args.output)
    run_training_audit(args.h5, output, stage=STAGES["train"], warmup_days=args.warmup_days, fields=sorted(RAW_FIELDS))
    return {"status": "complete", "artifact": str(output)}


def _search(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)  # validate before touching data or the run directory
    run_dir = Path(args.run_dir)
    raw_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != raw_config:
        raise ValueError("refusing to reuse run directory with a different config")
    fields = sorted(RAW_FIELDS)
    audit_path = run_dir / "audit_train.json"
    result = run_search(
        args.h5, audit_path, stage=STAGES["train"], warmup_days=config.max_history,
        fields=fields,
        search_stage=lambda _audit: evolution_search(load_stage(args.h5, STAGES["train"], config.max_history, fields), config),
        artifact_dir=run_dir, config=asdict(config), repository_root=Path.cwd(),
        selected_code_paths=(), seed=config.seed, experiment_id=run_dir.name,
    )
    _write_json(config_path, raw_config, immutable=True)
    (run_dir / "generations.jsonl").write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in result.generation_log), encoding="utf-8")
    return {"status": "complete" if result.candidates else "no_candidates", "run_dir": str(run_dir)}


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    archive = _verified(run_dir / "training_candidates.json", "training candidate archive")
    provenance = _verified(run_dir / "provenance.json", "provenance")
    config_document = _verified(run_dir / "config.json", "configuration")
    candidates = _archive_candidates(run_dir)
    if not candidates:
        _write_json(run_dir / "validation.json", {"status": "no_candidates", "accepted": [], "rejected": [], "results": {}, "training_archive_sha256": archive["sha256"], "provenance_sha256": provenance["sha256"], "config_sha256": config_document["sha256"], "validation_fingerprint": None}, immutable=True)
        return {"status": "no_candidates", "artifact": str(run_dir / "validation.json")}
    config = _load_run_config(run_dir / "config.json")
    stage_data = load_stage(args.h5, STAGES["validation"], config.max_history, sorted(RAW_FIELDS))
    outcomes: dict[str, Any] = {}
    for candidate in candidates[:config.validation_limit]:
        result = replay(candidate, STAGES["validation"], args.h5, run_dir, direction=candidate["direction"])
        metrics = result["metrics"]
        evidence = _validation_evidence(candidate, stage_data, config)
        outcomes[candidate["expression_id"]] = {**evidence, "all_costs_cumulative_return": metrics.get("cumulative_return", 0.0), "net_sharpe": metrics.get("sharpe"), "turnover": metrics.get("turnover", 0.0), "replay": result["artifact_path"]}
    selected = select_validation(candidates, outcomes, config)
    _write_json(run_dir / "validation.json", {"status": "complete" if selected.accepted else "no_candidates", "accepted": [item["expression_id"] for item in selected.accepted], "rejected": [item["expression_id"] for item in selected.rejected], "rejection_reasons": selected.rejection_reasons, "results": outcomes, "training_archive_sha256": archive["sha256"], "provenance_sha256": provenance["sha256"], "config_sha256": config_document["sha256"], "validation_fingerprint": stage_data.fingerprint}, immutable=True)
    return {"status": "complete" if selected.accepted else "no_candidates", "artifact": str(run_dir / "validation.json")}


def _freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    validation = _verified(run_dir / "validation.json", "validation result")
    archive = _verified(run_dir / "training_candidates.json", "training candidate archive")
    provenance = _verified(run_dir / "provenance.json", "provenance")
    config_document = _verified(run_dir / "config.json", "configuration")
    if (validation.get("training_archive_sha256"), validation.get("provenance_sha256"), validation.get("config_sha256")) != (archive["sha256"], provenance["sha256"], config_document["sha256"]):
        raise ValueError("validation result is not bound to the current verified training inputs")
    if not isinstance(validation.get("validation_fingerprint"), str) or not validation["validation_fingerprint"]:
        raise ValueError("validation result has no validation data fingerprint")
    accepted = set(validation.get("accepted", []))
    candidates = [candidate for candidate in _archive_candidates(run_dir) if candidate["expression_id"] in accepted]
    frozen = freeze_candidates(candidates, training_fingerprint=provenance["stage_content_hashes"]["train"], validation_fingerprint=validation["validation_fingerprint"], config={key: value for key, value in config_document.items() if key != "sha256"}, profile=provenance["backtest_profile"], runtime_source_hashes=_runtime_hashes(), selection_results={"accepted": list(accepted)}, path=run_dir / "frozen.json")
    return {"status": "complete" if candidates else "no_candidates", "artifact": str(frozen)}


def _test(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest is missing: {manifest}")
    frozen = read_verified_manifest(manifest)
    run_dir = manifest.parent
    config = _load_run_config(run_dir / "config.json")
    receipt = test_manifest(manifest, test_stage=STAGES["test"], run_dir=run_dir, training_fingerprint=frozen["stage_fingerprints"]["training"], validation_fingerprint=frozen["stage_fingerprints"]["validation"], load_test_data=lambda stage: {"fingerprint": load_stage(args.h5, stage, config.max_history, sorted(RAW_FIELDS)).fingerprint, "history_start": stage.start - __import__("pandas").Timedelta(days=config.max_history), "end": stage.end}, evaluate=lambda candidate, stage, _data, direction: replay({**candidate, "tree": _tree(candidate["ast"])}, stage, args.h5, run_dir, direction=direction), runtime_source_hashes=_runtime_hashes, permitted_history_start=STAGES["test"].start - __import__("pandas").Timedelta(days=config.max_history))
    outcomes = json.loads(receipt.read_text(encoding="utf-8"))["outcomes"]
    complete = sum(item["status"] == "complete" for item in outcomes)
    return {"status": "complete" if complete == len(outcomes) else ("partial" if complete else "failed"), "receipt": str(receipt)}


def _export(args: argparse.Namespace) -> dict[str, Any]:
    frozen = read_verified_manifest(args.manifest)
    exports = [export_factor({**candidate, "tree": _tree(candidate["ast"])}, args.output_dir, direction=candidate["direction"]) for candidate in frozen.get("candidates", [])]
    return {"status": "complete" if exports else "no_candidates", "exports": [str(item.path) for item in exports]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit"); audit.add_argument("--stage", required=True, choices=sorted(STAGES)); audit.add_argument("--h5", required=True); audit.add_argument("--output", required=True); audit.add_argument("--warmup-days", type=int, default=180); audit.set_defaults(handler=_audit)
    search = commands.add_parser("search"); search.add_argument("--config", required=True); search.add_argument("--h5", required=True); search.add_argument("--run-dir", required=True); search.set_defaults(handler=_search)
    validate = commands.add_parser("validate"); validate.add_argument("--run-dir", required=True); validate.add_argument("--h5", required=True); validate.set_defaults(handler=_validate)
    freeze = commands.add_parser("freeze"); freeze.add_argument("--run-dir", required=True); freeze.set_defaults(handler=_freeze)
    test = commands.add_parser("test"); test.add_argument("--manifest", required=True); test.add_argument("--h5", required=True); test.set_defaults(handler=_test)
    export = commands.add_parser("export"); export.add_argument("--manifest", required=True); export.add_argument("--output-dir", required=True); export.set_defaults(handler=_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        outcome = args.handler(args)
        print(json.dumps(outcome, sort_keys=True))
        return 2 if args.command == "test" and outcome.get("status") in {"partial", "failed"} else 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, ReplayEvaluationError, SearchFormationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
