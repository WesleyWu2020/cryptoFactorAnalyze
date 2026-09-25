"""Training-only objectives from the same certified ledger used by replay.

Funding is evaluation evidence, never a feature. Each scoring quarter is
liquidated inside its boundary; direction is calibrated in the first quarter.
"""
from dataclasses import asdict
import math

import numpy as np
import pandas as pd

from factor_common.backtest import run_backtest
from factor_common.metrics import _daily_ic, _ledger_summary, summarize_returns
from factor_common.profiles import resolve_profile
from .replay import _REPLAY_OVERRIDES


def cost_profile(config, direction, *, cost_multiplier=1.0, rebalance_days=1):
    if not math.isfinite(cost_multiplier) or cost_multiplier < 1:
        raise ValueError("cost multiplier must be finite and >= 1")
    if type(rebalance_days) is not int or rebalance_days < 1:
        raise ValueError("rebalance_days must be a positive integer")
    return resolve_profile("perp_1d", {
        **_REPLAY_OVERRIDES, "n_groups": config.n_groups,
        "rebalance_days": rebalance_days,
        "factor_direction": direction,
        "fee_rate": _REPLAY_OVERRIDES["fee_rate"] * cost_multiplier,
        "slippage": _REPLAY_OVERRIDES["slippage"] * cost_multiplier,
    })


def evaluate_cost_window(values, data, config, direction, start, end, *, include_positions=False, cost_multiplier=1.0, rebalance_days=1):
    """Return certified metrics/returns; reject partial or unliquidated ledgers."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start < data.stage.start or end > data.stage.end or start > end:
        raise ValueError("accounting window exceeds its bounded stage")
    if data.funding_events is None or data.accounting_quality is None:
        raise ValueError("all_costs fitness requires bounded funding and quality data")
    panel = values.loc[start:end].copy()
    signal_end = end - pd.Timedelta(days=1 + rebalance_days)
    panel.loc[panel.index > signal_end] = np.nan
    result = run_backtest(
        panel, data.opens.loc[panel.index], data.funding_events,
        data.accounting_quality, cost_profile(config, direction, cost_multiplier=cost_multiplier,
                                              rebalance_days=rebalance_days),
        signal_start=start, signal_end=signal_end,
    )["scenarios"]["all_costs"]
    diag = result["diagnostics"]
    if (result["status"] != "complete" or result["ledger"].empty
            or not diag.get("liquidation_reached") or diag.get("final_quantities")
            or pd.Timestamp(diag["liquidation_date"]) > end):
        raise ValueError(f"uncertified all_costs accounting: {diag.get('halt_reason')}")
    if not result["positions"].empty and (result["positions"].iloc[-1] != 0).any():
        raise ValueError("all_costs accounting retains positions")
    metrics = _ledger_summary(result["ledger"], periods_per_year=365)
    if include_positions:
        return metrics, result["ledger"]["return"], result["positions"]
    return metrics, result["ledger"]["return"]


def summarize_net_stability(returns, *, min_quarter_days):
    """Summarize one continuous net-return ledger and remove its best quarter.

    Quarter profit contribution is measured in wealth units, so the four
    contributions reconcile exactly to the full-period wealth change.
    """
    series = pd.Series(returns, copy=True).astype(float)
    if series.empty or not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("continuous stability requires dated net returns")
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError("continuous stability returns must have a unique sorted calendar")
    if not np.isfinite(series.to_numpy()).all() or (series <= -1).any():
        raise ValueError("continuous stability returns are invalid")
    wealth = (1.0 + series).cumprod()
    start_wealth = wealth.shift(1, fill_value=1.0)
    periods = series.index.to_period("Q")
    quarters = {}
    contributions = {}
    for quarter in periods.unique().sort_values():
        mask = periods == quarter
        quarter_returns = series.loc[mask]
        if len(quarter_returns) < min_quarter_days:
            raise ValueError("insufficient continuous stability quarter days")
        metrics = summarize_returns(quarter_returns, periods_per_year=365)
        if not all(metrics.get(key) is not None and math.isfinite(metrics[key])
                   for key in ("total_return", "sharpe")):
            raise ValueError("non-finite continuous stability quarter metrics")
        name = str(quarter)
        quarters[name] = metrics
        contributions[name] = float(wealth.loc[mask].iloc[-1] - start_wealth.loc[mask].iloc[0])
    positive_sum = sum(max(0.0, value) for value in contributions.values())
    best_quarter = max(contributions, key=lambda name: (contributions[name], name))
    concentration = (
        max(0.0, contributions[best_quarter]) / positive_sum if positive_sum > 0 else None
    )
    remaining = series.loc[periods != pd.Period(best_quarter, freq="Q")]
    if len(remaining) < 2:
        raise ValueError("insufficient returns after removing best quarter")
    remaining_metrics = summarize_returns(remaining, periods_per_year=365)
    if not all(remaining_metrics.get(key) is not None and math.isfinite(remaining_metrics[key])
               for key in ("total_return", "sharpe")):
        raise ValueError("non-finite leave-best-quarter-out metrics")
    return {
        "quarters": quarters,
        "quarter_profit_contributions": contributions,
        "profit_contribution_reconciliation": float(
            sum(contributions.values()) - (wealth.iloc[-1] - 1.0)
        ),
        "positive_quarters": sum(value > 0 for value in contributions.values()),
        "worst_quarter_return": min(item["total_return"] for item in quarters.values()),
        "positive_quarter_profit_concentration": concentration,
        "best_quarter": best_quarter,
        "leave_best_quarter_out": remaining_metrics,
        "leave_best_quarter_out_days": int(len(remaining)),
    }


def validation_cost_stability(values, data, config, direction):
    """Four separately liquidated validation quarters and a stressed full year.

    Funding is unchanged; only fees and slippage are stressed. Quarter returns
    are diagnostics of independent flat-to-flat windows, not annual attribution.
    """
    from .config import STAGES
    if data.stage != STAGES["validation"]:
        raise ValueError("cost stability must use the fixed 2025 validation stage")
    reasons = []
    if config.stability_mode == "continuous_leave_best_out":
        _, base_returns = evaluate_cost_window(
            values, data, config, direction, data.stage.start, data.stage.end,
        )
        stability = summarize_net_stability(
            base_returns, min_quarter_days=config.min_quarter_days,
        )
        quarters = stability["quarters"]
        positive = stability["positive_quarters"]
        worst = stability["worst_quarter_return"]
        concentration = stability["positive_quarter_profit_concentration"]
        remaining = stability["leave_best_quarter_out"]
        if positive < config.validation_min_positive_quarters:
            reasons.append("insufficient positive net validation quarters")
        if worst < -config.validation_max_quarter_loss:
            reasons.append("worst validation quarter loss exceeds limit")
        if remaining["total_return"] <= config.validation_min_remaining_return:
            reasons.append("leave-best-quarter-out net return does not exceed threshold")
        if remaining["sharpe"] <= config.validation_min_remaining_sharpe:
            reasons.append("leave-best-quarter-out Sharpe does not exceed threshold")
    else:
        quarters = {}
        stability = {}
        for q in pd.period_range(data.stage.start, data.stage.end, freq="Q"):
            metrics, returns = evaluate_cost_window(values, data, config, direction,
                                                  q.start_time, q.end_time.normalize())
            if len(returns) < config.min_quarter_days:
                raise ValueError("insufficient validation quarter accounting days")
            if not all(metrics.get(k) is not None and math.isfinite(metrics[k]) for k in ("total_return", "sharpe")):
                raise ValueError("non-finite validation quarter metrics")
            quarters[str(q)] = metrics
        profits = [m["total_return"] for m in quarters.values()]
        positive = sum(r > 0 for r in profits)
        worst = min(profits)
        positive_sum = sum(max(0, r) for r in profits)
        concentration = max(profits) / positive_sum if positive_sum > 0 else None
        if positive < config.validation_min_positive_quarters:
            reasons.append("insufficient positive net validation quarters")
        if worst < -config.validation_max_quarter_loss:
            reasons.append("worst validation quarter loss exceeds limit")
        if concentration is None or concentration > config.validation_max_profit_concentration:
            reasons.append("validation profit too concentrated in one quarter")
    stressed, returns = evaluate_cost_window(
        values, data, config, direction, data.stage.start, data.stage.end,
        cost_multiplier=config.validation_cost_multiplier,
    )
    if len(returns) < config.min_quarter_days or not all(
        stressed.get(k) is not None and math.isfinite(stressed[k]) for k in ("total_return", "sharpe")
    ):
        raise ValueError("invalid stressed validation accounting")
    if stressed["total_return"] <= 0 or stressed["sharpe"] <= 0:
        reasons.append("validation net advantage disappears under cost stress")
    return {"passed": not reasons, "reasons": reasons, "quarters": quarters,
            "positive_quarters": positive, "worst_quarter_return": worst,
            "positive_quarter_profit_concentration": concentration,
            "stability_mode": config.stability_mode, **stability,
            "cost_multiplier": config.validation_cost_multiplier,
            "stressed_full_year": stressed,
            "accounting_fingerprint": data.audit["accounting_fingerprint"]}


def training_parameter_stability(tree, direction, data, config, baseline_metrics):
    """Score ±window perturbations on the bounded 2024 training ledger.

    Direction is frozen to the original candidate calibration. Neighboring
    parameters may lower, but never improve, the candidate's base fitness.
    """
    from .config import STAGES
    from .evaluator import evaluate_tree
    from .expression import history_days, expression_hash
    from .incremental import neighboring_trees

    if data.stage != STAGES["train"]:
        raise ValueError("training parameter stability only permits fixed 2024 training")
    if direction not in (-1, 1):
        raise ValueError("training parameter stability requires a fixed direction")
    base_return = baseline_metrics.get("total_return")
    base_sharpe = baseline_metrics.get("sharpe")
    if not all(value is not None and math.isfinite(float(value))
               for value in (base_return, base_sharpe)):
        raise ValueError("training parameter stability requires finite baseline metrics")

    records = []
    backtests = 0
    for variant in neighboring_trees(tree, config.parameter_perturbation):
        record = {"expression_id": expression_hash(variant), "passed": False}
        try:
            if history_days(variant) > config.max_history:
                raise ValueError("neighbor exceeds available warmup budget")
            values = evaluate_tree(
                variant, data.features, data.eligible,
                cache=getattr(config, "cache", None), cache_bytes=config.cache_bytes,
            ).loc[data.opens.index]
            backtests += 1
            metrics, returns = evaluate_cost_window(
                values, data, config, direction, data.stage.start, data.stage.end,
            )
            if len(returns) < config.min_overlap_days:
                raise ValueError("insufficient neighbor accounting days")
            passed = bool(all(
                metrics.get(name) is not None
                and math.isfinite(float(metrics[name]))
                and float(metrics[name]) > 0
                for name in ("total_return", "sharpe")
            ))
            record.update(metrics=metrics, passed=passed)
        except ValueError as exc:
            record["error"] = str(exc)
        records.append(record)

    valid_metrics = [record["metrics"] for record in records if "metrics" in record]
    if not records:
        return {
            "applicable": False, "variants": [], "variant_count": 0,
            "backtest_count": 0, "positive_fraction": None,
            "median_neighbor_return": None, "median_neighbor_sharpe": None,
            "return_degradation": 0.0, "sharpe_degradation": 0.0,
            "positive_fraction_shortfall": 0.0, "penalty_units": 0.0,
            "perturbation": config.parameter_perturbation,
        }

    positive_fraction = sum(record["passed"] for record in records) / len(records)
    if valid_metrics:
        median_return = float(np.median([item["total_return"] for item in valid_metrics]))
        median_sharpe = float(np.median([item["sharpe"] for item in valid_metrics]))
    else:
        median_return = float("-inf")
        median_sharpe = float("-inf")
    return_degradation = (
        max(0.0, float(base_return) - median_return)
        if math.isfinite(median_return) else max(1.0, abs(float(base_return)))
    )
    sharpe_degradation = (
        max(0.0, float(base_sharpe) - median_sharpe)
        if math.isfinite(median_sharpe) else max(1.0, abs(float(base_sharpe)))
    )
    fraction_shortfall = max(
        0.0, config.parameter_min_positive_fraction - positive_fraction,
    )
    return {
        "applicable": True, "variants": records, "variant_count": len(records),
        "backtest_count": backtests, "positive_fraction": positive_fraction,
        "median_neighbor_return": median_return if math.isfinite(median_return) else None,
        "median_neighbor_sharpe": median_sharpe if math.isfinite(median_sharpe) else None,
        "return_degradation": return_degradation,
        "sharpe_degradation": sharpe_degradation,
        "positive_fraction_shortfall": fraction_shortfall,
        "penalty_units": return_degradation + sharpe_degradation + fraction_shortfall,
        "perturbation": config.parameter_perturbation,
    }


def aggregate_cost_score(segments, full, config, nodes, *, stability=None):
    """Primary objective is net Sharpe with stability/risk regularization."""
    for metrics in [full, *segments]:
        for name in ("sharpe", "total_return", "max_drawdown"):
            value = metrics.get(name)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"non-finite all_costs metric: {name}")
    sharpes = [m["sharpe"] for m in segments]
    worst = min(sharpes)
    primary = float(
        np.median(sharpes) - config.stability_penalty * np.std(sharpes)
        - config.worst_quarter_penalty * max(0.0, -worst)
        - config.drawdown_penalty * max(m["max_drawdown"] for m in segments)
        - config.complexity_penalty * nodes
    )
    if config.stability_mode == "continuous_leave_best_out":
        if not isinstance(stability, dict):
            raise ValueError("continuous stability evidence is missing")
        concentration = stability.get("positive_quarter_profit_concentration")
        remaining = stability.get("leave_best_quarter_out", {})
        if concentration is None or not math.isfinite(float(concentration)):
            raise ValueError("continuous profit concentration is undefined")
        remaining_sharpe = remaining.get("sharpe")
        if remaining_sharpe is None or not math.isfinite(float(remaining_sharpe)):
            raise ValueError("leave-best-quarter-out Sharpe is undefined")
        primary -= config.concentration_penalty * max(
            0.0, float(concentration) - config.validation_max_profit_concentration
        )
        primary -= config.leave_best_out_penalty * max(0.0, -float(remaining_sharpe))
    reasons = []
    if full["total_return"] <= 0:
        reasons.append("training all_costs return is not positive")
    if full["sharpe"] <= config.min_all_costs_sharpe:
        reasons.append("training all_costs Sharpe does not exceed threshold")
    if sum(m["total_return"] > 0 for m in segments) <= len(segments) / 2:
        reasons.append("not a majority of positive all_costs quarters")
    if primary <= 0:
        reasons.append("risk-adjusted all_costs objective is not positive")
    return (primary, float(worst), -nodes), tuple(reasons)


def _training_reference_score(objective, similarities, incremental, config, nodes):
    """Apply the selectable training objective while preserving legacy behavior."""
    reasons = []
    if any(item["too_similar"] for item in similarities):
        reasons.append("training candidate too similar to fixed reference")
    if config.training_reference_objective == "similarity_penalty":
        penalty = config.reference_similarity_penalty * max(
            max(0.0, item["return_correlation"], item["position_overlap"])
            for item in similarities
        )
        score = (objective[0] - penalty, *objective[1:])
        if score[0] <= 0:
            reasons.append("reference-adjusted objective is not positive")
        return score, tuple(reasons)
    if incremental is None:
        raise ValueError("portfolio incremental training evidence is missing")
    delta = incremental.get("sharpe_increment")
    if delta is None or not math.isfinite(float(delta)):
        raise ValueError("portfolio incremental training Sharpe is undefined")
    score = (float(delta) - config.complexity_penalty * nodes, *objective[1:])
    if float(delta) <= 0:
        reasons.append(
            "training fixed equal-weight portfolio Sharpe does not improve reference"
        )
    return score, tuple(reasons)


def score_all_costs(values, labels, data, config, nodes):
    """IC screens coverage/calibrates direction only; cost Sharpe ranks trees."""
    if config.behavior_diversity or config.dsr_diagnostics or config.layered_elites:
        from .config import STAGES
        if data.stage != STAGES["train"]:
            raise ValueError("research search evidence requires fixed 2024 training data")
    periods = data.opens.index.to_period("Q")
    quarters = periods.unique().sort_values()
    if len(quarters) != 4 or any(
        data.stage.start > q.start_time or data.stage.end < q.end_time.normalize()
        for q in quarters
    ):
        raise ValueError("all_costs fitness requires four complete consecutive quarters")
    usable = data.opens.index + pd.Timedelta(days=2) <= periods.end_time.normalize()
    purged = labels.copy()
    purged.loc[~usable] = np.nan
    quality = data.quality_eligible.fillna(False).astype(bool)
    signal = values.replace([np.inf, -np.inf], np.nan).where(quality)
    daily = _daily_ic(signal, purged)
    daily = daily.loc[(daily.n_pairs >= config.min_pairs) & daily.rank_ic.notna()]
    daily_periods = pd.to_datetime(daily.date).dt.to_period("Q")
    counts = {str(q): int((daily_periods == q).sum()) for q in quarters}
    calibration = daily.loc[daily_periods == quarters[0], "rank_ic"].mean()
    direction = 1 if calibration > 0 else -1
    reasons = []
    if not np.isfinite(calibration) or calibration == 0:
        reasons.append("direction calibration unavailable")
    if any(n < config.min_quarter_days for n in counts.values()):
        reasons.append("insufficient quarterly IC coverage")
    day_coverage = len(daily) / max(1, int(quality.any(axis=1).sum()))
    cell_coverage = int(signal.notna().to_numpy().sum()) / max(1, int(quality.to_numpy().sum()))
    if day_coverage < config.min_day_coverage or cell_coverage < config.min_cell_coverage:
        reasons.append("insufficient signal coverage")
    diagnostics = {
        "objective": "all_costs_sharpe", "direction": direction,
        "signal_semantics": {
            "median_unique_values": float(signal.nunique(axis=1).median()),
            "informative_days": int(signal.nunique(axis=1).ge(2).sum()),
            "constant_days": int(signal.nunique(axis=1).eq(1).sum()),
            "group_tie_policy": cost_profile(config, direction).group_tie_policy,
        },
        "calibration_quarter": str(quarters[0]), "quarter_valid_days": counts,
        "day_coverage": day_coverage, "cell_coverage": cell_coverage,
        "accounting_fingerprint": data.audit.get("accounting_fingerprint"),
        "profile": asdict(cost_profile(config, direction)),
    }
    if reasons:
        diagnostics["reasons"] = reasons
        return (None, None, -nodes), False, tuple(reasons), direction, diagnostics
    try:
        segments = {}
        continuous_stability = None
        if config.stability_mode == "continuous_leave_best_out":
            scoring_start = quarters[1].start_time
            scoring_result = evaluate_cost_window(
                values, data, config, direction, scoring_start, data.stage.end,
                **({"include_positions": True} if config.reference_factor else {}),
            )
            scoring_metrics, scoring_returns = scoring_result[:2]
            continuous_stability = summarize_net_stability(
                scoring_returns, min_quarter_days=config.min_quarter_days,
            )
            segments = continuous_stability["quarters"]
            scoring_full = scoring_metrics
        else:
            for q in quarters[1:]:
                metrics, returns = evaluate_cost_window(
                    values, data, config, direction, q.start_time, q.end_time.normalize(),
                )
                if len(returns) < config.min_quarter_days:
                    raise ValueError("insufficient all_costs accounting days")
                segments[str(q)] = metrics
            scoring_full = None
        full_result = evaluate_cost_window(
            values, data, config, direction, data.stage.start, data.stage.end,
            **({"include_positions": True} if config.reference_factor else {}),
        )
        full = full_result[0]
        objective, reasons = aggregate_cost_score(
            list(segments.values()), scoring_full or full, config, nodes,
            stability=continuous_stability,
        )
        research_result = scoring_result if config.stability_mode == "continuous_leave_best_out" else full_result
        if config.dsr_diagnostics:
            from .research import return_moments
            diagnostics["return_moments"] = return_moments(research_result[1])
        similarities = []
        if config.reference_factor:
            from .incremental import incremental_evidence, reference_evidence
            reference_result = (
                scoring_result
                if config.stability_mode == "continuous_leave_best_out"
                else full_result
            )
            _, returns, positions = reference_result
            incremental = None
            if config.training_reference_objective == "portfolio_incremental_sharpe":
                incremental = incremental_evidence(
                    returns, positions, data, config,
                    start=(scoring_start if config.stability_mode == "continuous_leave_best_out" else data.stage.start),
                    end=data.stage.end,
                )
                similarities = incremental["comparisons"]
            else:
                from .selection import trading_similarity
                similarities = [{"reference_id": record["reference_id"], **trading_similarity(
                    returns, record["returns"], positions, record["positions"], config)}
                    for record in reference_evidence(
                        data, config,
                        start=(scoring_start if config.stability_mode == "continuous_leave_best_out" else data.stage.start),
                        end=data.stage.end,
                    )]
            objective, reference_reasons = _training_reference_score(
                objective, similarities, incremental, config, nodes
            )
            reasons += reference_reasons
            diagnostics["reference_comparison"] = {"comparisons": similarities,
                "penalty": (
                    config.reference_similarity_penalty * max(
                        max(0.0, item["return_correlation"], item["position_overlap"])
                        for item in similarities
                    ) if config.training_reference_objective == "similarity_penalty" else 0.0
                ),
                "reference_id": config.reference_factor,
                "training_objective": config.training_reference_objective}
            if incremental is not None:
                diagnostics["training_incremental"] = incremental
        if config.behavior_diversity:
            from .research import behavior_evidence
            diagnostics["behavior"] = behavior_evidence(
                research_result[0], research_result[1], data, similarities,
            )
        diagnostics.update(
            segments=segments, full_training_period=full,
            scoring_period=(scoring_full or full),
            continuous_stability=continuous_stability,
            score=list(objective), reasons=list(reasons),
        )
        return objective, not reasons, reasons, direction, diagnostics
    except ValueError as exc:
        diagnostics["reasons"] = [str(exc)]
        return (None, None, -nodes), False, (str(exc),), direction, diagnostics
