"""Separate forward prediction horizons from executable rebalance intervals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from factor_common.labels import make_labels
from factor_common.metrics import _daily_ic, summarize_returns

from .cost_fitness import cost_profile, evaluate_cost_window


def _finite(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def assess_horizon_execution(values, data, config, direction):
    """2025-only diagnostics, with the training direction and one-day entry delay.

    Predictive IC uses every eligible daily signal and labels at each horizon.
    Executable ledgers use a fixed calendar anchor and non-overlapping
    rebalance intervals, including fees, slippage and funding. No horizon is
    selected or substituted into the frozen one-day trading rule here.
    """
    if data.stage.name != "validation":
        raise ValueError("horizon diagnostics require validation-stage data")
    if direction not in (-1, 1):
        raise ValueError("direction must be frozen to +1 or -1")
    if data.funding_events is None or data.accounting_quality is None:
        raise ValueError("horizon execution requires bounded accounting data")
    signal = values.reindex(index=data.opens.index, columns=data.opens.columns)
    signal = signal.replace([np.inf, -np.inf], np.nan)
    prediction_signal = signal.where(data.quality_eligible.fillna(False))
    prediction = {}
    for horizon in config.prediction_horizons:
        last_signal = data.stage.end - pd.Timedelta(days=1 + horizon)
        labels = make_labels(data.opens, horizon)
        labels.loc[labels.index > last_signal] = np.nan
        daily = _daily_ic(prediction_signal, labels)
        daily = daily.loc[
            (daily["n_pairs"] >= config.min_pairs)
            & pd.to_numeric(daily["rank_ic"], errors="coerce").notna()
        ]
        prediction[str(horizon)] = {
            "hold_days": horizon,
            "last_eligible_signal": last_signal.date().isoformat(),
            "valid_days": int(len(daily)),
            "directed_mean_rank_ic": _finite(daily["rank_ic"].mean() * direction),
            "quarter_means": {
                str(quarter): _finite(
                    daily.loc[pd.to_datetime(daily["date"]).dt.to_period("Q") == quarter,
                              "rank_ic"].mean() * direction
                )
                for quarter in data.opens.index.to_period("Q").unique().sort_values()
            },
        }

    calendar = data.opens.index
    execution = {}
    for interval in config.execution_intervals:
        profile = cost_profile(config, direction, rebalance_days=interval)
        last_allowed = data.stage.end - pd.Timedelta(days=1 + interval)
        signal_dates = calendar - pd.Timedelta(days=profile.signal_delay_days)
        scheduled = calendar[
            ((calendar - pd.Timestamp(profile.anchor_date)).days % interval == 0)
            & (signal_dates >= data.stage.start)
            & (signal_dates <= last_allowed)
        ]
        entry = {"rebalance_days": interval, "status": "unavailable",
                 "signal_delay_days": profile.signal_delay_days,
                 "anchor_date": profile.anchor_date,
                 "fee_rate": profile.fee_rate, "slippage": profile.slippage,
                 "include_funding": profile.include_funding,
                 "group_tie_policy": profile.group_tie_policy,
                 "last_eligible_signal": last_allowed.date().isoformat(),
                 "scheduled_entries": int(len(scheduled)),
                 "last_scheduled_signal": (
                     (scheduled[-1] - pd.Timedelta(days=profile.signal_delay_days)).date().isoformat()
                     if len(scheduled) else None
                 )}
        try:
            ledger_metrics, returns = evaluate_cost_window(
                signal, data, config, direction, data.stage.start, data.stage.end,
                rebalance_days=interval,
            )
            if not returns.index.isin(calendar).all():
                raise ValueError("execution ledger exceeds validation calendar")
            common_calendar = returns.reindex(calendar, fill_value=0.0)
            comparable = summarize_returns(common_calendar, periods_per_year=365)
            entry.update(
                status="complete",
                net_total_return=_finite(comparable.get("total_return")),
                calendar_net_sharpe=_finite(comparable.get("sharpe")),
                ledger_net_sharpe=_finite(ledger_metrics.get("sharpe")),
                turnover=_finite(ledger_metrics.get("turnover")),
                ledger_days=int(len(returns)),
                calendar_days=int(len(calendar)),
            )
        except ValueError as exc:
            entry["reason"] = str(exc)
        execution[str(interval)] = entry
    return {
        "status": "diagnostic",
        "direction_source": "frozen_training",
        "prediction": prediction,
        "execution": execution,
    }
