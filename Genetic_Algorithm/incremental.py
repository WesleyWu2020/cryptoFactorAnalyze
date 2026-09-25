"""Fixed-reference research on training/validation data, never test data."""
from dataclasses import asdict, replace
import math

import numpy as np
import pandas as pd

from .config import STAGES
from .expression import Node, expression_hash, history_days
from .evaluator import evaluate_tree
from .operators import MIN_OPERATOR_WINDOW
from .selection import trading_similarity
from factor_common.metrics import summarize_returns


def reference_tree():
    # Frozen 06418510; its original direction is -1, never recalibrated here.
    tree = Node("range_relative")
    for op, window in (("rolling_max", 40), ("rolling_std", 60),
                       ("rolling_mean", 10), ("rolling_mean", 10)):
        tree = Node(op, (tree,), window=window)
    return tree


def second_reference_tree():
    # Frozen a37c60cf; its original direction is -1, never recalibrated here.
    tree = Node("body_relative")
    for op, window in (("rolling_max", 10), ("rolling_min", 60),
                       ("rolling_max", 60), ("rolling_min", 40)):
        tree = Node(op, (tree,), window=window)
    return tree


def third_reference_tree():
    # Frozen 0006a47b; its original direction is -1, never recalibrated here.
    tree = Node("return_1d")
    for op, window in (("rolling_max", 5), ("rolling_min", 20),
                       ("rolling_max", 60), ("rolling_mean", 60)):
        tree = Node(op, (tree,), window=window)
    return tree


def reference_specs(config):
    specs = [("064185107a8f267e", reference_tree(), -1)]
    if config.reference_factor in {
        "064185107a8f267e+a37c60cf4492e9f1",
        "064185107a8f267e+a37c60cf4492e9f1+0006a47b612eaf9b",
    }:
        specs.append(("a37c60cf4492e9f1", second_reference_tree(), -1))
    if config.reference_factor == "064185107a8f267e+a37c60cf4492e9f1+0006a47b612eaf9b":
        specs.append(("0006a47b612eaf9b", third_reference_tree(), -1))
    elif config.reference_factor != "064185107a8f267e":
        if config.reference_factor != "064185107a8f267e+a37c60cf4492e9f1":
            raise ValueError("reference factor is not configured")
    return specs


def reference_evidence(data, config, *, start=None, end=None):
    from .cost_fitness import evaluate_cost_window
    if data.stage not in (STAGES["train"], STAGES["validation"]):
        raise ValueError("reference research only permits fixed train/validation stages")
    specs = reference_specs(config)
    if config.max_history < max(history_days(tree) for _, tree, _ in specs):
        raise ValueError("max_history cannot warm up reference factors")
    start = pd.Timestamp(start) if start is not None else data.stage.start
    end = pd.Timestamp(end) if end is not None else data.stage.end
    if start < data.stage.start or end > data.stage.end or start > end:
        raise ValueError("reference evidence window exceeds bounded stage")
    key = (config.reference_factor, config.n_groups, data.stage, start, end,
           data.fingerprint, data.audit.get("accounting_fingerprint"))
    if key not in data.reference_evidence:
        records = []
        for identifier, tree, direction in specs:
            values = evaluate_tree(tree, data.features, data.eligible).loc[data.opens.index]
            metrics, returns, positions = evaluate_cost_window(
                values, data, config, direction, start, end,
                include_positions=True)
            records.append({"reference_id": identifier, "metrics": metrics,
                            "returns": returns, "positions": positions})
        data.reference_evidence[key] = records
    return data.reference_evidence[key]


def net_summary(returns):
    values = np.asarray(returns, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("invalid net return series")
    summary = summarize_returns(pd.Series(values), periods_per_year=365)
    if summary["sharpe"] is None or summary["total_return"] is None:
        raise ValueError("undefined net Sharpe")
    return {k: summary[k] for k in ("total_return", "sharpe")}


def incremental_evidence(
    returns, positions, data, config, *, start=None, end=None, exclude_period=None,
):
    references = (
        reference_evidence(data, config, start=start, end=end)
        if start is not None or end is not None
        else reference_evidence(data, config)
    )
    comparisons = []
    reference_returns = []
    for record in references:
        baseline = record["returns"]
        if not returns.index.equals(baseline.index):
            raise ValueError("reference/candidate accounting calendars differ")
        comparisons.append({"reference_id": record["reference_id"], **trading_similarity(
            returns, baseline, positions, record["positions"], config)})
        reference_returns.append(baseline.rename(record["reference_id"]))
    # All sleeves receive equal initial capital; no weight fitting or transfers.
    reference_wealth = (1 + pd.concat(reference_returns, axis=1)).cumprod().mean(axis=1)
    baseline = reference_wealth / reference_wealth.shift(1, fill_value=1.0) - 1
    all_returns = [*reference_returns, returns.rename("candidate")]
    wealth = (1 + pd.concat(all_returns, axis=1)).cumprod().mean(axis=1)
    combination = wealth / wealth.shift(1, fill_value=1.0) - 1
    base_metrics = net_summary(baseline)
    combined_metrics = net_summary(combination)
    delta = combined_metrics["sharpe"] - base_metrics["sharpe"]
    reasons = []
    if any(comparison["too_similar"] for comparison in comparisons):
        reasons.append("candidate too similar to fixed reference")
    if delta <= config.validation_min_incremental_sharpe or combined_metrics["total_return"] <= 0:
        reasons.append("fixed equal-weight net portfolio does not improve reference Sharpe")
    leave_out = None
    if exclude_period is not None:
        period = pd.Period(exclude_period, freq="Q")
        keep = returns.index.to_period("Q") != period
        if int(keep.sum()) < 2:
            raise ValueError("insufficient portfolio returns after removing best quarter")
        leave_base = net_summary(baseline.loc[keep])
        leave_combined = net_summary(combination.loc[keep])
        leave_out = {
            "excluded_period": str(period),
            "reference": leave_base,
            "combination": leave_combined,
            "sharpe_increment": leave_combined["sharpe"] - leave_base["sharpe"],
            "days": int(keep.sum()),
        }
    return {"comparisons": comparisons, "reference_id": config.reference_factor,
            "reference": base_metrics, "combination": combined_metrics,
            "combination_rule": "equal initial capital per independently funded sleeve; no transfers",
            "sharpe_increment": delta, "leave_best_quarter_out": leave_out,
            "passed": not reasons, "reasons": reasons}


def neighboring_trees(tree, fraction):
    """Change one window at a time, never choose a best-performing variant."""
    variants = {}
    def visit(node):
        if node.window is not None:
            for multiplier in (1 - fraction, 1 + fraction):
                minimum = MIN_OPERATOR_WINDOW.get(node.op, 1)
                window = max(minimum, int(math.floor(node.window * multiplier + .5)))
                if window != node.window:
                    yield replace(node, window=window)
        for i, child in enumerate(node.children):
            for changed in visit(child):
                children = list(node.children)
                children[i] = changed
                yield replace(node, children=tuple(children))
    for variant in visit(tree):
        variants[expression_hash(variant)] = variant
    return list(variants.values())


def parameter_stability(tree, direction, data, config):
    from .cost_fitness import evaluate_cost_window
    if data.stage != STAGES["validation"]:
        raise ValueError("parameter stability only permits fixed 2025 validation")
    records = []
    for variant in neighboring_trees(tree, config.parameter_perturbation):
        record = {"expression_id": expression_hash(variant), "tree": asdict(variant), "passed": False}
        try:
            if history_days(variant) > config.max_history:
                raise ValueError("neighbor exceeds available warmup budget")
            values = evaluate_tree(variant, data.features, data.eligible).loc[data.opens.index]
            metrics, returns = evaluate_cost_window(values, data, config, direction,
                                                    data.stage.start, data.stage.end)
            if len(returns) < config.min_overlap_days:
                raise ValueError("insufficient neighbor accounting days")
            record.update(metrics=metrics, passed=bool(all(
                metrics.get(k) is not None and math.isfinite(metrics[k]) and metrics[k] > 0
                for k in ("total_return", "sharpe"))))
        except ValueError as exc:
            record["error"] = str(exc)
        records.append(record)
    # Window-free trees have no tunable windows; this check is not applicable.
    fraction = sum(r["passed"] for r in records) / len(records) if records else None
    passed = fraction is None or fraction >= config.parameter_min_positive_fraction
    return {"passed": passed, "applicable": bool(records), "variants": records,
            "positive_fraction": fraction, "perturbation": config.parameter_perturbation,
            "required_fraction": config.parameter_min_positive_fraction,
            "reasons": [] if passed else ["neighboring parameters lack positive net stability"]}
