"""Point-in-time style exposure and matched-sample residual evidence for GP validation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from barra.crypto_barra_exposure import BarraConfig, analyze_exposures
from factor_common.labels import make_labels
from factor_common.metrics import _daily_ic

from .cost_fitness import evaluate_cost_window


def _directed_ic(values: pd.DataFrame, labels: pd.DataFrame, direction: int, min_pairs: int) -> dict:
    daily = _daily_ic(values, labels)
    valid = daily.loc[
        (daily["n_pairs"] >= min_pairs)
        & pd.to_numeric(daily["rank_ic"], errors="coerce").notna()
    ]
    return {
        "days": int(len(valid)),
        "mean": float(valid["rank_ic"].mean() * direction) if len(valid) else None,
    }


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def assess_exposure_residual(values, data, config, direction, styles):
    """Compare raw and Barra residual on exactly the same validation cells.

    Exposures and residuals use signal-day information only. Forward labels
    enter the IC calculation alone; both net returns use certified accounting.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be frozen to +1 or -1")
    if data.funding_events is None or data.accounting_quality is None:
        raise ValueError("residual comparison requires bounded accounting data")
    signal = values.reindex(index=data.opens.index, columns=data.opens.columns)
    signal = signal.replace([np.inf, -np.inf], np.nan).where(data.quality_eligible.fillna(False))
    cfg = BarraConfig(min_count=max(2, config.min_pairs))
    analysis = analyze_exposures(signal, styles, cfg=cfg)
    regression = analysis["daily_barra_regression"]
    residual = analysis["alpha_barra_residual"].reindex_like(signal)
    # Each day's input is standardized to unit variance by analyze_exposures.
    # Residuals below this numerical scale are regression roundoff, not ranks.
    numerical_zero = residual.std(axis=1, ddof=0).le(1e-8)
    residual.loc[numerical_zero] = np.nan
    common = signal.notna() & residual.notna()
    raw_matched = signal.where(common)
    residual = residual.where(common)
    usable_days = common.sum(axis=1).ge(config.n_groups)
    valid_days = int(usable_days.sum())
    quarter_counts = {
        str(quarter): int(usable_days.loc[common.index.to_period("Q") == quarter].sum())
        for quarter in common.index.to_period("Q").unique().sort_values()
    }
    eligible_days = int(data.quality_eligible.any(axis=1).sum())
    coverage = valid_days / eligible_days if eligible_days else 0.0
    valid_regression = regression.loc[regression["status"] == "ok"]
    style_summary = [
        {
            "style": str(row.style), "days": int(row.days),
            "pearson_mean": _finite(row.pearson_mean),
            "pearson_abs_mean": _finite(row.pearson_abs_mean),
            "spearman_mean": _finite(row.spearman_mean),
            "beta_mean": _finite(row.beta_mean),
        }
        for row in analysis["style_summary"].itertuples(index=False)
    ]
    evidence = {
        "status": "insufficient_data", "passed": False,
        "styles": list(styles),
        "style_summary": style_summary,
        "regression_status_counts": {str(k): int(v) for k, v in regression["status"].value_counts().items()},
        "valid_regression_days": int(len(valid_regression)),
        "numerically_zero_residual_days": int(numerical_zero.sum()),
        "mean_r2": _finite(valid_regression["r2"].mean()),
        "matched_cells": int(common.to_numpy().sum()),
        "matched_days": valid_days,
        "quarter_valid_days": quarter_counts,
        "day_coverage": coverage,
        "reasons": [],
    }
    if coverage < config.min_day_coverage:
        evidence["reasons"].append("Barra residual coverage below validation threshold")
    if valid_days < config.min_quarter_days:
        evidence["reasons"].append("too few valid Barra residual days")
    for quarter, count in quarter_counts.items():
        if count < config.min_quarter_days:
            evidence["reasons"].append(f"{quarter} has too few valid Barra residual days")
    if evidence["reasons"]:
        return evidence

    labels = make_labels(data.opens, config.hold_days)
    labels.loc[labels.index > data.stage.signal_end] = np.nan
    labels = labels.where(common)
    evidence["raw_matched_ic"] = _directed_ic(raw_matched, labels, direction, config.min_pairs)
    evidence["residual_ic"] = _directed_ic(residual, labels, direction, config.min_pairs)
    try:
        raw_metrics, raw_returns = evaluate_cost_window(
            raw_matched, data, config, direction, data.stage.start, data.stage.end,
        )
        residual_metrics, residual_returns = evaluate_cost_window(
            residual, data, config, direction, data.stage.start, data.stage.end,
        )
    except ValueError as exc:
        evidence["reasons"].append(f"matched Barra residual accounting unavailable: {exc}")
        return evidence
    if not raw_returns.index.equals(residual_returns.index):
        evidence["reasons"].append("matched Barra ledgers have different date axes")
        return evidence
    evidence.update(
        status="complete",
        raw_matched_net={"sharpe": _finite(raw_metrics.get("sharpe")),
                         "total_return": _finite(raw_metrics.get("total_return"))},
        residual_net={"sharpe": _finite(residual_metrics.get("sharpe")),
                      "total_return": _finite(residual_metrics.get("total_return"))},
        ledger_days=int(len(raw_returns)),
    )
    if not (evidence["residual_net"]["sharpe"] is not None
            and evidence["residual_net"]["sharpe"] > 0
            and evidence["residual_net"]["total_return"] is not None
            and evidence["residual_net"]["total_return"] > 0):
        evidence["reasons"].append("Barra residual net return or Sharpe is not positive")
    evidence["passed"] = not evidence["reasons"]
    return evidence
