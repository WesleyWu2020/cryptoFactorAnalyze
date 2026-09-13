"""Training-only objectives from the same certified ledger used by replay.

Funding is evaluation evidence, never a feature. Each scoring quarter is
liquidated inside its boundary; direction is calibrated in the first quarter.
"""
from dataclasses import asdict
import math

import numpy as np
import pandas as pd

from factor_common.backtest import run_backtest
from factor_common.metrics import _daily_ic, _ledger_summary
from factor_common.profiles import resolve_profile
from .replay import _REPLAY_OVERRIDES


def cost_profile(config, direction, *, cost_multiplier=1.0):
    if not math.isfinite(cost_multiplier) or cost_multiplier < 1:
        raise ValueError("cost multiplier must be finite and >= 1")
    return resolve_profile("perp_1d", {
        **_REPLAY_OVERRIDES, "n_groups": config.n_groups,
        "factor_direction": direction,
        "fee_rate": _REPLAY_OVERRIDES["fee_rate"] * cost_multiplier,
        "slippage": _REPLAY_OVERRIDES["slippage"] * cost_multiplier,
    })


def evaluate_cost_window(values, data, config, direction, start, end, *, include_positions=False, cost_multiplier=1.0):
    """Return certified metrics/returns; reject partial or unliquidated ledgers."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start < data.stage.start or end > data.stage.end or start > end:
        raise ValueError("accounting window exceeds its bounded stage")
    if data.funding_events is None or data.accounting_quality is None:
        raise ValueError("all_costs fitness requires bounded funding and quality data")
    panel = values.loc[start:end].copy()
    signal_end = end - pd.Timedelta(days=2)
    panel.loc[panel.index > signal_end] = np.nan
    result = run_backtest(
        panel, data.opens.loc[panel.index], data.funding_events,
        data.accounting_quality, cost_profile(config, direction, cost_multiplier=cost_multiplier),
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


def validation_cost_stability(values, data, config, direction):
    """Four separately liquidated validation quarters and a stressed full year.

    Funding is unchanged; only fees and slippage are stressed. Quarter returns
    are diagnostics of independent flat-to-flat windows, not annual attribution.
    """
    from .config import STAGES
    if data.stage != STAGES["validation"]:
        raise ValueError("cost stability must use the fixed 2025 validation stage")
    quarters = {}
    reasons = []
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
            "cost_multiplier": config.validation_cost_multiplier,
            "stressed_full_year": stressed,
            "accounting_fingerprint": data.audit["accounting_fingerprint"]}


def aggregate_cost_score(segments, full, config, nodes):
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


def score_all_costs(values, labels, data, config, nodes):
    """IC screens coverage/calibrates direction only; cost Sharpe ranks trees."""
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
        for q in quarters[1:]:
            metrics, returns = evaluate_cost_window(
                values, data, config, direction, q.start_time, q.end_time.normalize(),
            )
            if len(returns) < config.min_quarter_days:
                raise ValueError("insufficient all_costs accounting days")
            segments[str(q)] = metrics
        full_result = evaluate_cost_window(
            values, data, config, direction, data.stage.start, data.stage.end,
            **({"include_positions": True} if config.reference_factor else {}),
        )
        full = full_result[0]
        objective, reasons = aggregate_cost_score(list(segments.values()), full, config, nodes)
        if config.reference_factor:
            from .incremental import reference_evidence
            from .selection import trading_similarity
            _, returns, positions = full_result
            similarities = [{"reference_id": record["reference_id"], **trading_similarity(
                returns, record["returns"], positions, record["positions"], config)}
                for record in reference_evidence(data, config)]
            penalty = config.reference_similarity_penalty * max(
                max(0.0, item["return_correlation"], item["position_overlap"])
                for item in similarities)
            objective = (objective[0] - penalty, *objective[1:])
            if any(item["too_similar"] for item in similarities):
                reasons += ("training candidate too similar to fixed reference",)
            if objective[0] <= 0:
                reasons += ("reference-adjusted objective is not positive",)
            diagnostics["reference_comparison"] = {"comparisons": similarities,
                "penalty": penalty, "reference_id": config.reference_factor}
        diagnostics.update(segments=segments, full_training_period=full, score=list(objective), reasons=list(reasons))
        return objective, not reasons, reasons, direction, diagnostics
    except ValueError as exc:
        diagnostics["reasons"] = [str(exc)]
        return (None, None, -nodes), False, (str(exc),), direction, diagnostics
