"""Training-only behavior niches and explicitly conditional DSR diagnostics.

Neither descriptors nor DSR are factor inputs. Bin edges are fixed before
search, never fitted on validation/test performance.
"""
from collections import defaultdict
import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


BEHAVIOR_BINS = {
    "turnover": (0.05, 0.15, 0.30),
    "reference_correlation": (-0.2, 0.2, 0.6),
    "position_overlap": (0.2, 0.5, 0.8),
    "market_beta": (-0.5, 0.0, 0.5),
}


def return_moments(returns):
    values = np.asarray(returns, dtype=float)
    if len(values) < 4 or not np.isfinite(values).all():
        return {"status": "unavailable", "reason": "short or nonfinite returns"}
    std = float(np.std(values, ddof=1))
    if std <= 1e-12:
        return {"status": "unavailable", "reason": "constant returns"}
    return {
        "status": "complete", "n": len(values),
        "daily_sharpe": float(np.mean(values) / std),
        "skew": float(skew(values, bias=False)),
        "kurtosis": float(kurtosis(values, fisher=False, bias=False)),
    }


def behavior_evidence(metrics, returns, data, comparisons):
    # Point-in-time equal-weight open-to-open market proxy. Missing prices
    # stay missing; membership comes from the preceding observation.
    market = data.opens.pct_change(fill_method=None).where(
        data.eligible.reindex_like(data.opens).shift(1).eq(True)
    ).mean(axis=1)
    pair = pd.concat([returns.rename("candidate"), market.rename("market")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    beta = correlation = None
    if len(pair) >= 4 and pair.market.var() > 1e-16:
        beta = float(pair.candidate.cov(pair.market) / pair.market.var())
        if pair.candidate.std() > 1e-12:
            correlation = float(pair.candidate.corr(pair.market))
    def maximum(name):
        values = [float(c[name]) for c in comparisons if c.get(name) is not None and math.isfinite(c[name])]
        return max(values) if values else None
    return {
        "turnover": metrics.get("turnover"), "market_beta": beta,
        "market_correlation": correlation,
        "reference_correlation": maximum("return_correlation"),
        "position_overlap": maximum("position_overlap"),
        "market_proxy": "point-in-time equal-weight open-to-open",
        "period_start": str(returns.index.min().date()),
        "period_end": str(returns.index.max().date()),
    }


def behavior_cell(evidence):
    return tuple(
        -1 if evidence.get(name) is None or not math.isfinite(evidence[name])
        else int(np.searchsorted(edges, evidence[name], side="right"))
        for name, edges in BEHAVIOR_BINS.items()
    )


def behavior_buckets(candidates, diagnostics, key):
    buckets = defaultdict(list)
    for candidate in sorted(candidates, key=key):
        evidence = diagnostics.get(candidate.expression_id, {}).get("behavior")
        # Coverage-invalid candidates can explore, but never masquerade as
        # a measured zero-exposure strategy.
        cell = behavior_cell(evidence) if evidence else (-1,) * len(BEHAVIOR_BINS)
        buckets[cell].append(candidate)
    return buckets


def select_behavior(candidates, limit, diagnostics, key, capacity):
    if limit <= 0:
        return []
    buckets = behavior_buckets(candidates, diagnostics, key)
    cells = sorted(buckets, key=lambda cell: (key(buckets[cell][0]), cell))
    selected = []
    for depth in range(capacity):
        for cell in cells:
            if depth < len(buckets[cell]):
                selected.append(buckets[cell][depth])
                if len(selected) == limit:
                    return selected
    return selected


def dsr_report(diagnostics, effective_trials, attempted_evaluations):
    """Sensitivity analysis, not an estimate of the independent trial count.

    Uses per-observation Sharpe/moments, a zero-mean null and the dispersion
    of ALL available trial Sharpes (including rejected trials). Serial
    dependence and adaptive GP selection are NOT fully corrected here.
    """
    records = {
        key: value["return_moments"] for key, value in diagnostics.items()
        if value.get("return_moments", {}).get("status") == "complete"
    }
    sharpes = [record["daily_sharpe"] for record in records.values()]
    dispersion = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else None
    scenarios = []
    gamma = 0.5772156649015329
    for trials in effective_trials:
        row = {"assumed_effective_trials": trials}
        if trials > len(records) or (trials > 1 and (dispersion is None or dispersion <= 1e-12)):
            row.update(status="unavailable", reason="insufficient usable trials or undefined trial Sharpe dispersion")
            scenarios.append(row)
            continue
        benchmark = 0.0 if trials == 1 else dispersion * (
            (1 - gamma) * norm.ppf(1 - 1 / trials)
            + gamma * norm.ppf(1 - 1 / (trials * math.e))
        )
        probabilities = {}
        for identifier, record in records.items():
            sr = record["daily_sharpe"]
            variance = 1 - record["skew"] * sr + (record["kurtosis"] - 1) * sr * sr / 4
            probabilities[identifier] = (
                float(norm.cdf((sr - benchmark) * math.sqrt((record["n"] - 1) / variance)))
                if math.isfinite(variance) and variance > 0 else None
            )
        row.update(status="complete", benchmark_daily_sharpe=float(benchmark),
                   conditional_dsr=probabilities)
        scenarios.append(row)
    return {
        "diagnostic_only": True, "selection_gate": False,
        "attempted_evaluations": attempted_evaluations, "usable_trials": len(records),
        "daily_sharpe_dispersion": dispersion, "scenarios": scenarios,
        "limitations": [
            "Effective trial counts are user assumptions, not formula counts or estimated independence.",
            "Zero-mean null; dispersion from observed, adaptively selected trial Sharpes.",
            "Coverage-invalid trials lack returns and are excluded; survivor bias remains.",
            "Serial dependence and the complete adaptive GP search are not corrected.",
            "Not a probability of future profitability and not unseen holdout evidence.",
        ],
    }
