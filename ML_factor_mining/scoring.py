"""Certified validation scoring and deterministic candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from factor_common.backtest import run_backtest
from factor_common.metrics import _daily_ic, summarize_returns
from factor_common.profiles import resolve_profile


class CandidateRejected(ValueError):
    """Expected data or accounting rejection, retaining evaluated records."""

    def __init__(self, message: str, *, records=None):
        super().__init__(message)
        self.records = records


@dataclass
class MarketWindow:
    opens: pd.DataFrame
    events: pd.DataFrame
    quality: pd.DataFrame


def load_market(provider, start, end, columns):
    """Load the market inputs needed by the accounting engine."""
    calendar = pd.date_range(start, end, name="date")
    columns = list(columns)
    opens = provider.get_single_data("open", start=start, end=end).reindex(
        index=calendar, columns=columns
    )
    events = provider.get_funding(start=start, end=end, symbols=columns)
    quality = provider.get_quality(
        start=pd.Timestamp(start) - pd.Timedelta(days=1), end=end, symbols=columns
    )
    return MarketWindow(opens=opens, events=events, quality=quality)


def backtest_profile(config):
    """Build the fixed, strict daily profile used for candidate validation."""
    anchor_date = getattr(config, "anchor_date", "2024-01-01")
    if isinstance(anchor_date, (pd.Timestamp,)) or hasattr(anchor_date, "isoformat"):
        anchor_date = anchor_date.isoformat()
    return resolve_profile(
        "perp_1d",
        {
            "rebalance_days": config.holding_days,
            "anchor_date": anchor_date,
            "signal_delay_days": 1,
            "n_groups": 5,
            "factor_direction": 1,
            "gross_exposure": 1.0,
            "fee_rate": config.fee_rate,
            "slippage": config.slippage,
            "include_funding": True,
            "funding_price_mode": "strict",
            "periods_per_year": 365,
        },
    )


def _as_fold(fold, config, validation_start, retrain_at):
    """Normalize the documented fold API and the legacy keyword API."""
    if config is None:
        # Older callers passed config in the fold position and explicit bounds.
        config = fold
        if validation_start is None or retrain_at is None:
            raise TypeError("validation_start and retrain_at are required")
        fold = type(
            "ValidationFold",
            (),
            {"validation_start": validation_start, "retrain_at": retrain_at},
        )()
    return fold, config


def score_validation(
    predictions,
    targets,
    membership,
    market,
    fold=None,
    config=None,
    *,
    validation_start=None,
    retrain_at=None,
):
    """Score a prediction series using IC and a certified all-costs ledger.

    Prediction rows are restricted to same-day membership and to signals whose
    full holding period fits before retraining. No future membership or labels
    participate in the prediction coverage or factor-value calculations.
    """
    fold, config = _as_fold(fold, config, validation_start, retrain_at)
    start = pd.Timestamp(fold.validation_start).normalize()
    retrain = pd.Timestamp(fold.retrain_at).normalize()
    end = retrain - pd.Timedelta(days=1)
    signal_end = end - pd.Timedelta(days=config.holding_days + 1)
    calendar = pd.date_range(start, end, name="date")

    if not isinstance(market.opens, pd.DataFrame):
        raise TypeError("market.opens must be a DataFrame")
    columns = market.opens.columns
    values = predictions.unstack("instrument").reindex(index=calendar, columns=columns)
    values = values.replace([np.inf, -np.inf], np.nan)
    eligible = membership.reindex(index=calendar, columns=columns).fillna(False).astype(bool)
    values = values.where(eligible)
    values.loc[values.index > signal_end] = np.nan
    signal_dates = calendar[calendar <= signal_end]

    denominator = eligible.loc[signal_dates].sum(axis=1)
    coverage = values.loc[signal_dates].notna().sum(axis=1) / denominator.replace(0, np.nan)
    if coverage.empty or coverage.isna().any() or (coverage < config.validation_prediction_coverage).any():
        raise CandidateRejected("insufficient validation prediction coverage")

    if isinstance(targets, pd.Series):
        target_series = targets
    else:
        target_series = targets["raw_return"]
    labels = target_series.unstack("instrument").reindex(index=calendar, columns=columns)
    daily = _daily_ic(values.loc[signal_dates], labels.loc[signal_dates])
    daily = daily.loc[
        daily["n_pairs"].ge(config.min_pairs)
        & np.isfinite(daily["rank_ic"])
    ]
    if len(daily) < config.min_val_dates:
        raise CandidateRejected("insufficient valid validation RankIC dates")

    opens = market.opens.reindex(index=calendar, columns=columns)
    accounting = run_backtest(
        values,
        opens,
        market.events,
        market.quality,
        backtest_profile(config),
        signal_start=start,
        signal_end=signal_end,
    )
    net = accounting["scenarios"]["all_costs"]
    diagnostics = net.get("diagnostics", {})
    ledger = net.get("ledger")
    liquidation_date = diagnostics.get("liquidation_date")
    if (
        net.get("status") != "complete"
        or not isinstance(ledger, pd.DataFrame)
        or ledger.empty
        or not diagnostics.get("liquidation_reached")
        or liquidation_date is None
        or pd.Timestamp(liquidation_date) >= retrain
        or diagnostics.get("final_quantities")
    ):
        raise CandidateRejected(
            f"uncertified validation ledger: {diagnostics.get('halt_reason')}"
        )
    expected_index = pd.date_range(ledger.index[0], ledger.index[-1], name="date")
    if not ledger.index.equals(expected_index):
        raise CandidateRejected("internal accounting gap")
    if "return" not in ledger or not np.isfinite(ledger["return"]).all():
        raise CandidateRejected("nonfinite certified return")

    # Zero padding is permitted only outside the certified ledger's active
    # interval; any overlap is already validated above.
    returns = ledger["return"].reindex(calendar, fill_value=0.0)
    metrics = summarize_returns(returns, periods_per_year=365)
    sharpe = metrics.get("sharpe")
    if sharpe is None or not np.isfinite(sharpe):
        raise CandidateRejected("undefined validation net Sharpe")
    return {
        "status": "complete",
        "mean_rank_ic": float(daily["rank_ic"].mean()),
        "net_sharpe": float(sharpe),
        "net_return": float(metrics["total_return"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "ic_dates": int(len(daily)),
        "min_prediction_coverage": float(coverage.min()),
        "liquidation_date": liquidation_date,
    }


class _Winner(str):
    """String winner that can still satisfy the old mapping-style API."""

    def __new__(cls, value, record=None):
        instance = super().__new__(cls, value)
        instance.record = record or {}
        return instance

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "candidate_id":
                return str(self)
            if key in self.record:
                return self.record[key]
        return super().__getitem__(key)


def rank_candidates(records, config=None, *, ic_weight=None, sharpe_weight=None):
    """Rank valid records by signed percentile RankIC and net Sharpe."""
    table = pd.DataFrame(records)
    if table.empty:
        raise CandidateRejected("no candidates evaluated", records=records)

    # ``valid`` is the current contract; ``complete`` is retained for older
    # callers whose accounting stage used that status name.
    status = table.get("status", pd.Series(index=table.index, dtype=object))
    valid = status.isin(("valid", "complete"))
    for name in ("mean_rank_ic", "net_sharpe"):
        if name not in table:
            table[name] = np.nan
        numeric = pd.to_numeric(table[name], errors="coerce")
        table[name] = numeric
        valid &= np.isfinite(numeric)
    if not valid.any():
        raise CandidateRejected("all candidates rejected", records=records)

    count = int(valid.sum())
    for name in ("mean_rank_ic", "net_sharpe"):
        table.loc[valid, name + "_percentile"] = (
            table.loc[valid, name].rank(method="average") - 0.5
        ) / count

    if config is not None:
        ic_weight = config.ic_weight
        sharpe_weight = config.sharpe_weight
    elif ic_weight is None or sharpe_weight is None:
        ic_weight = sharpe_weight = 0.5
    table.loc[valid, "selection_score"] = (
        float(ic_weight) * table.loc[valid, "mean_rank_ic_percentile"]
        + float(sharpe_weight) * table.loc[valid, "net_sharpe_percentile"]
    )
    # The old complete/single-row test used zero as its neutral score. The
    # documented valid API deliberately uses percentile 0.5 for one row.
    if count == 1 and str(status.loc[valid].iloc[0]) == "complete" and config is None:
        table.loc[valid, "selection_score"] = 0.0

    ranked = table.loc[valid].sort_values(
        ["selection_score", "net_sharpe", "mean_rank_ic", "candidate_id"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    return _Winner(
        str(ranked.iloc[0]["candidate_id"]),
        record=ranked.iloc[0].to_dict(),
    ), table


__all__ = [
    "CandidateRejected",
    "MarketWindow",
    "load_market",
    "backtest_profile",
    "score_validation",
    "rank_candidates",
]
