"""Retrain, commit, then evaluate the next quarter with explicit test access."""
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import pandas as pd

from factor_common.metrics import summarize_returns
from .artifacts import read_verified_manifest, write_artifact
from .config import Stage, STAGES
from .cost_fitness import evaluate_cost_window
from .data import load_stage
from .evaluator import evaluate_tree
from .evolution import search
from .features import RAW_FIELDS
from .selection import deduplicate_training, trading_similarity


def fold_schedule(first_test_start, last_test_end, *, allow_2026_test=False):
    """Four train quarters, one independent validation quarter, then test."""
    start, end = pd.Timestamp(first_test_start), pd.Timestamp(last_test_end)
    if start != start.to_period("Q").start_time or end != end.to_period("Q").end_time.normalize():
        raise ValueError("walk-forward boundaries must be complete calendar quarters")
    if start > end:
        raise ValueError("walk-forward windows must be ordered")
    if end >= STAGES["test"].start:
        if not allow_2026_test:
            raise ValueError("walk-forward research must end before 2026 unless explicit test access is enabled")
        # The configured 2026 test stage ends Sep 1, so Q3 is incomplete.
        if end > pd.Timestamp("2026-06-30"):
            raise ValueError("only complete 2026 Q1 and Q2 test quarters are available")
    folds = []
    for quarter in pd.period_range(start, end, freq="Q"):
        test_start = quarter.start_time
        folds.append((
            Stage("walk_forward_train", (quarter - 5).start_time, (quarter - 1).start_time - pd.Timedelta(days=1)),
            Stage("walk_forward_validation", (quarter - 1).start_time, test_start - pd.Timedelta(days=1)),
            Stage("walk_forward_test", test_start, quarter.end_time.normalize()),
        ))
    return folds


def runtime_hashes():
    root = Path(__file__).resolve().parent.parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ("Genetic_Algorithm", "factor_common")
            for path in sorted((root / directory).glob("*.py"))}


def run_walk_forward(h5_path, run_dir, config, *, first_test_start="2025-01-01", last_test_end="2025-12-31", allow_2026_test=False):
    if config.validation_stability or config.family_diversity or config.reference_factor or config.validation_parameter_stability:
        raise ValueError("family/stability selection currently requires the fixed-year search/validate/freeze/test workflow")
    if config.fitness_mode != "all_costs_sharpe":
        raise ValueError("walk-forward requires fitness_mode=all_costs_sharpe")
    folds = fold_schedule(first_test_start, last_test_end, allow_2026_test=allow_2026_test)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=False)
    source_hashes = runtime_hashes()
    uses_2026 = any(test.start >= STAGES["test"].start for _, _, test in folds)
    if uses_2026 != bool(allow_2026_test):
        raise ValueError("2026 test access flag does not match requested walk-forward windows")
    experiment = write_artifact(root / "experiment.json", {
        "kind": "walk_forward_research", "config": asdict(config),
        "first_test_start": str(first_test_start), "last_test_end": str(last_test_end),
        "runtime_source_hashes": source_hashes,
        "selection": "freeze shortlist before independent validation; strict net gates; training return/holding diversity; at most frozen_limit equal capital sleeves",
        "holdout_access": {
            "uses_2026": uses_2026,
            "explicit": bool(allow_2026_test),
            "disclosure": "2026 was previously inspected; this is repeat research evidence, not unseen holdout evidence" if uses_2026 else "no 2026 access",
        },
    }, immutable=True)
    experiment_hash = read_verified_manifest(experiment)["sha256"]
    # This immutable commitment is written before the first 2026 data read.
    if uses_2026:
        write_artifact(root / "holdout_access_commitment.json", {
            "experiment_sha256": experiment_hash,
            "test_stage": {"start": "2026-01-01", "end": "2026-06-30"},
            "purpose": "one explicit walk-forward repeat evaluation",
            "unseen_claim_permitted": False,
        }, immutable=True)
    outcomes, portfolio_periods = [], []
    for number, (train, validation, test) in enumerate(folds, 1):
        folder = root / f"fold_{number:02d}"
        folder.mkdir()
        print(f"fold {number}/{len(folds)}: train {train.start.date()}..{train.end.date()}", flush=True)
        # No test loader is invoked until the selected trees are committed.
        training = load_stage(h5_path, train, config.max_history, sorted(RAW_FIELDS), include_accounting=True)
        result = search(training, config)
        training_values = {
            identifier: {**value, "training_fingerprint": training.fingerprint,
                         "operator_version": source_hashes["Genetic_Algorithm/operators.py"]}
            for identifier, value in result.values_by_id.items()
        }
        selection = deduplicate_training(result.candidates, training_values, (), config)
        shortlist = tuple(selection)[:config.validation_limit]
        write_artifact(folder / "training.json", {
            "stage": asdict(train), "audit": training.audit,
            "generations": result.generation_log, "evaluations": result.evaluations,
            "cost_fitness": result.fitness_diagnostics,
            "selected": [c.expression_id for c in shortlist],
            "deduplication_reasons": getattr(selection, "rejection_reasons", {}),
        }, immutable=True)
        shortlist_path = write_artifact(folder / "shortlist.json", {
            "experiment_sha256": experiment_hash,
            "train": asdict(train), "validation": asdict(validation),
            "candidates": [{"expression_id": c.expression_id, "ast": asdict(c.tree),
                            "direction": c.direction, "score": list(c.score)} for c in shortlist],
        }, immutable=True)
        shortlist_hash = read_verified_manifest(shortlist_path)["sha256"]
        # Training evidence is fixed before validation access. No validation
        # score feeds evolution, direction, or ranking.
        evidence, evidence_errors = {}, {}
        for candidate in shortlist:
            try:
                values = evaluate_tree(candidate.tree, training.features, training.eligible).loc[training.opens.index]
                _, returns, positions = evaluate_cost_window(
                    values, training, config, candidate.direction, train.start, train.end,
                    include_positions=True,
                )
                evidence[candidate.expression_id] = (returns, positions)
            except (ValueError, KeyError) as exc:
                evidence_errors[candidate.expression_id] = str(exc)
        print(f"fold {number}: validate {len(shortlist)} frozen candidates {validation.start.date()}..{validation.end.date()}", flush=True)
        validating = load_stage(h5_path, validation, config.max_history, sorted(RAW_FIELDS), include_accounting=True)
        selected, validation_results = [], []
        for candidate in shortlist:
            record = {"expression_id": candidate.expression_id, "comparisons": []}
            try:
                values = evaluate_tree(candidate.tree, validating.features, validating.eligible).loc[validating.opens.index]
                metrics, returns = evaluate_cost_window(values, validating, config, candidate.direction, validation.start, validation.end)
                record["metrics"] = metrics
                if len(returns) < config.min_quarter_days:
                    raise ValueError("insufficient validation accounting days")
                if not all(metrics.get(k) is not None and math.isfinite(metrics[k]) for k in ("total_return", "sharpe")):
                    raise ValueError("non-finite validation metrics")
                if not (metrics["total_return"] > 0 and metrics["sharpe"] is not None
                        and metrics["sharpe"] > config.min_all_costs_sharpe):
                    record.update(status="rejected", reason="validation net return/Sharpe gate")
                else:
                    if candidate.expression_id in evidence_errors:
                        raise ValueError(evidence_errors[candidate.expression_id])
                    r, p = evidence[candidate.expression_id]
                    for reference in selected:
                        rr, pp = evidence[reference.expression_id]
                        comparison = trading_similarity(r, rr, p, pp, config)
                        record["comparisons"].append({"reference_id": reference.expression_id, **comparison})
                    if any(c["too_similar"] for c in record["comparisons"]):
                        record.update(status="rejected", reason="training trading similarity")
                    elif len(selected) >= config.frozen_limit:
                        record.update(status="rejected", reason="portfolio capacity")
                    else:
                        selected.append(candidate)
                        record["status"] = "accepted"
            except (ValueError, KeyError, TypeError) as exc:
                record.update(status="failed", reason=str(exc))
            validation_results.append(record)
        validation_failed = any(r["status"] == "failed" for r in validation_results)
        validation_path = write_artifact(folder / "validation.json", {
            "stage": asdict(validation), "shortlist_sha256": shortlist_hash,
            "fingerprint": validating.fingerprint, "audit": validating.audit,
            "candidates": validation_results,
            "status": "failed" if validation_failed else "complete",
        }, immutable=True)
        del validating, evidence
        manifest_path = write_artifact(folder / "frozen.json", {
            "kind": "walk_forward_fold", "experiment_sha256": experiment_hash,
            "train": asdict(train), "validation": asdict(validation), "test": asdict(test),
            "validation_sha256": read_verified_manifest(validation_path)["sha256"],
            "training_fingerprint": training.fingerprint,
            "accounting_fingerprint": training.audit["accounting_fingerprint"],
            "candidates": [{"expression_id": c.expression_id, "ast": asdict(c.tree),
                            "direction": c.direction, "score": list(c.score)} for c in selected],
        }, immutable=True)
        committed = read_verified_manifest(manifest_path)
        del training, result, training_values
        print(f"fold {number}: frozen {len(selected)}; test {test.start.date()}..{test.end.date()}", flush=True)
        evaluation = load_stage(h5_path, test, config.max_history, sorted(RAW_FIELDS), include_accounting=True)
        daily_index = evaluation.opens.index
        sleeves, results = [], []
        for candidate in selected:
            try:
                values = evaluate_tree(candidate.tree, evaluation.features, evaluation.eligible).loc[daily_index]
                metrics, returns = evaluate_cost_window(values, evaluation, config, candidate.direction, test.start, test.end)
                if len(returns) < config.min_quarter_days:
                    raise ValueError("insufficient out-of-sample accounting days")
                sleeves.append(returns.reindex(daily_index, fill_value=0.0))
                results.append({"expression_id": candidate.expression_id, "status": "complete", "metrics": metrics})
            except (ValueError, KeyError) as exc:
                results.append({"expression_id": candidate.expression_id, "status": "failed", "error": str(exc)})
        failed = validation_failed or any(r["status"] == "failed" for r in results)
        # Equal initial capital per frozen sleeve; no daily capital rebalancing.
        portfolio = None
        if not failed:
            if sleeves:
                nav = pd.concat([(1 + s).cumprod() for s in sleeves], axis=1).mean(axis=1)
                portfolio = nav.pct_change()
                portfolio.iloc[0] = nav.iloc[0] - 1
            else:
                portfolio = pd.Series(0.0, index=daily_index)  # no candidate = cash, recorded
            portfolio_periods.append(portfolio)
        outcome = {
            "fold": number, "train": asdict(train), "validation": asdict(validation), "test": asdict(test),
            "validation_failed": validation_failed,
            "manifest_sha256": committed["sha256"], "status": "failed" if failed else ("complete" if selected else "no_candidates"),
            "test_fingerprint": evaluation.fingerprint,
            "accounting_fingerprint": evaluation.audit["accounting_fingerprint"],
            "candidates": results,
            "portfolio_metrics": None if portfolio is None else summarize_returns(portfolio),
            "daily_returns": None if portfolio is None else {str(k.date()): float(v) for k, v in portfolio.items()},
        }
        write_artifact(folder / "test.json", outcome, immutable=True)
        outcomes.append(outcome)
    failures = sum(o["status"] == "failed" for o in outcomes)
    positive = sum(o["portfolio_metrics"] is not None and o["portfolio_metrics"]["total_return"] > 0 for o in outcomes)
    combined = None if failures else summarize_returns(pd.concat(portfolio_periods).sort_index())
    summary = {
        "status": "partial" if failures else "complete", "experiment_sha256": experiment_hash,
        "fold_count": len(folds), "failed_fold_count": failures,
        "cash_fold_count": sum(o["status"] == "no_candidates" for o in outcomes),
        "positive_fold_count": positive, "positive_fold_fraction": positive / len(folds),
        "all_costs_portfolio": combined,
        "passed": bool(combined and combined["sharpe"] is not None
                       and combined["sharpe"] > config.min_all_costs_sharpe
                       and combined["total_return"] > 0 and positive > len(folds) / 2),
        "worst_fold_return": min((o["portfolio_metrics"]["total_return"] for o in outcomes if o["portfolio_metrics"] is not None), default=None),
        "folds": [{k: v for k, v in o.items() if k != "daily_returns"} for o in outcomes],
    }
    write_artifact(root / "summary.json", summary, immutable=True)
    return {"status": summary["status"], "artifact": str(root / "summary.json"), "passed": summary["passed"]}
