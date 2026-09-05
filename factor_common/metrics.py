"""Shared factor metrics and full/in/out-of-sample evaluation.

``summarize_returns(returns, *, periods_per_year=365)`` summarizes a daily
return series; ``evaluate_metrics(values, labels, accounting, profile)``
combines factor values, evaluation-only forward-return labels and a single
``run_backtest`` accounting into per-sample and per-scenario metrics. Every
sample slice reuses the same functions and the same accounting — the
portfolio is never rerun per slice; scenario slices are plain date slices of
the certified ledger, so out-of-sample continues the in-sample equity path
without any reset trade.

Result structure (consumed by the manager and storage layers)::

    {
        "profile_id": str,
        "split": {
            "split_date": "YYYY-MM-DD",
            "out_of_sample_days": int,
            "hold_days": int,                    # = profile.rebalance_days
            "in_sample_signal_dates": int,
            "out_of_sample_signal_dates": int,
            "purged_signal_dates": ["YYYY-MM-DD", ...],
        },
        "samples": {
            "full":        {"ic": {...}, "ic_decay": [...],
                            "rank_ic_autocorr": [...], "rank_ic_half_life": ...,
                            "coverage": {...}},
            "in_sample":   {...},
            "out_of_sample": {...},
        },
        "scenarios": {
            "gross":       {"status": "complete", "full": {...},
                            "in_sample": {...}, "out_of_sample": {...}},
            "trading_net": {...},
            "all_costs":   {"status": "incomplete", "full": None,
                            "in_sample": None, "out_of_sample": None,
                            "known_segment": {...}},
        },
    }

Missing metrics are always ``None`` (JSON null) — never NaN or inf literals.
When a scenario accounting is incomplete (an unresolved cash flow halted
certified valuation), its per-sample metric blocks are null; the separately
labeled ``known_segment`` summarizes only the certified ledger prefix.

Sample boundaries: the split date is ``profile.split_date`` when given,
otherwise ``last_date - profile.out_of_sample_days`` (180 natural days by
default). Labels are assumed built with ``hold_days = profile.rebalance_days``
so entry/exit execution dates follow ``label_dates``. A signal whose holding
period crosses the split (entry on or before the split, exit after it) is
purged from both slices but kept in the full sample; purging compares the
concrete exit date against the split — never label NaN-ness, since exits
remain concrete past the data tail. In-sample keeps every historical
observation whose exit lies inside the sample.

Legacy compatibility map (for the Task 13 adapter):

- ``performance`` column ``return`` (summed daily returns) -> ``total_return``
  (now compounded ``prod(1+r)-1``).
- ``annual_return`` -> ``annual_return`` (now compounded over the actual
  daily count; legacy scaled linearly by ``365/n``).
- ``sharp`` -> ``sharpe`` (annualized, zero risk-free, ddof=1; zero variance
  now yields null instead of legacy inf/0 artifacts).
- ``volatility`` / ``annual_volatility`` -> same names (daily std ddof=1 and
  ``std*sqrt(periods_per_year)``).
- ``max_drawback`` -> ``max_drawdown`` (fraction, initial NAV 1 included;
  legacy reported a rounded percentage).
- ``md_period_days`` / ``recovery_period_days`` -> not retained.
- ``win_percent`` -> ``win_rate`` (fraction of nonmissing returns > 0).
- ``profit-loss ratio`` -> ``profit_loss_ratio`` (mean gain / |mean loss|;
  legacy divided sums).
- ``turnover`` -> ``turnover`` (mean daily traded notional over pretrade
  equity; legacy measured group membership churn).
- ``IC()`` tuple: ``ic`` -> ``ic_mean``, ``rank_ic`` -> ``rank_ic_mean``,
  ``ic_ir`` -> ``icir`` (unannualized mean/std, ddof=1), ``t_stat`` /
  ``p_value`` -> same names (null without at least two dates and positive
  IC variance), ``acc_ic`` -> derivable from ``ic.daily``. New:
  ``annualized_icir`` and the rank-IC twins ``rank_icir`` /
  ``annualized_rank_icir``.
- ``calculate_rank_ic_decay`` rows ``{lag, rank_ic}`` -> ``ic_decay`` rows
  ``{horizon, rank_ic}``; pooled Spearman over entry-aligned horizons 1..10
  retained; legacy 0-fills are now null.
- ``calculate_rank_ic_autocorr`` rows ``{lag, autocorr}`` ->
  ``rank_ic_autocorr`` rows ``{lag, autocorr}`` over the daily RankIC series
  at exact calendar lags 1..20; legacy 0-fills are now null.
- ``calc_rankic_halflife`` (exponential fit) -> ``rank_ic_half_life`` (first
  horizon whose |RankIC| <= half the |horizon-1| magnitude; null without a
  crossing).
- ``portfolio/backtester/metrics.py`` (annual_return/annual_volatility/
  sharpe/max_drawdown/calmar, 252-day convention) is superseded by
  ``summarize_returns``; its zero-variance ``sharpe`` 0.0 is now null and
  ``calmar`` is not retained.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .labels import label_dates
from .profiles import BacktestProfile
from .value_engine import _validate_axes

SAMPLES = ("full", "in_sample", "out_of_sample")
SCENARIOS = ("gross", "trading_net", "all_costs")
MIN_CROSS_SECTION = 3
IC_DECAY_HORIZONS = tuple(range(1, 11))
AUTOCORR_LAGS = tuple(range(1, 21))

# Standard deviations at or below this floor are treated as zero variance, so
# floating-point noise on a constant series cannot manufacture a huge ICIR or
# Sharpe; the metric is reported missing instead.
_STD_FLOOR = 1e-12

_RETURN_KEYS = (
    "total_return",
    "annual_return",
    "volatility",
    "annual_volatility",
    "sharpe",
    "max_drawdown",
    "win_rate",
    "profit_loss_ratio",
)


def _finite_or_none(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize_returns(returns, *, periods_per_year: int = 365) -> dict:
    """Summarize a daily return series; missing metrics are None.

    Total return compounds as ``prod(1+r)-1``; the annualized return compounds
    over the actual count of nonmissing days; volatility uses the sample
    standard deviation (``ddof=1``); the Sharpe is annualized with a zero
    risk-free rate and is None under zero variance. The drawdown path starts
    at the initial NAV 1.
    """
    if not isinstance(periods_per_year, int) or isinstance(periods_per_year, bool) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")
    series = pd.Series(returns, dtype="float64").dropna()
    n = int(series.shape[0])
    summary = {key: None for key in _RETURN_KEYS}
    summary["n_periods"] = n
    if n == 0:
        return summary

    total = float((1.0 + series).prod() - 1.0)
    summary["total_return"] = _finite_or_none(total)
    if 1.0 + total > 0.0:
        summary["annual_return"] = _finite_or_none((1.0 + total) ** (periods_per_year / n) - 1.0)
    else:
        summary["annual_return"] = -1.0

    if n >= 2:
        volatility = float(series.std(ddof=1))
        summary["volatility"] = _finite_or_none(volatility)
        summary["annual_volatility"] = _finite_or_none(volatility * math.sqrt(periods_per_year))
        if volatility > _STD_FLOOR:
            mean = float(series.mean())
            summary["sharpe"] = _finite_or_none(mean / volatility * math.sqrt(periods_per_year))

    nav = np.concatenate(([1.0], np.cumprod(1.0 + series.to_numpy())))
    running_max = np.maximum.accumulate(nav)
    drawdown = (running_max - nav) / running_max
    summary["max_drawdown"] = _finite_or_none(drawdown.max())

    summary["win_rate"] = float((series > 0).mean())
    gains = series[series > 0]
    losses = series[series < 0]
    if len(gains) and len(losses):
        summary["profit_loss_ratio"] = _finite_or_none(gains.mean() / abs(losses.mean()))
    return summary


def _mean_turnover(ledger: pd.DataFrame):
    """Mean daily traded notional over pretrade equity (no 1/2 factor).

    Pretrade equity is the recorded post-fee equity plus the day's fees — the
    exact sizing equity when no intraday funding cash flow intervenes.
    """
    if ledger.empty:
        return None
    pretrade = ledger["equity"] + ledger["fee"]
    valid = pretrade > 0.0
    if not valid.any():
        return None
    daily = ledger.loc[valid, "trade_notional"] / pretrade[valid]
    return _finite_or_none(daily.mean())


def _ledger_summary(ledger: pd.DataFrame, *, periods_per_year: int) -> dict:
    summary = summarize_returns(ledger["return"], periods_per_year=periods_per_year)
    summary["turnover"] = _mean_turnover(ledger)
    return summary


def _daily_ic(values: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Per-date Pearson/Spearman IC over valid cross-sections.

    A cross-section is valid with at least ``MIN_CROSS_SECTION`` paired
    observations and nonconstant factor and label values.
    """
    rows = []
    for date in values.index:
        pairs = pd.concat(
            [values.loc[date], labels.loc[date]], axis=1, keys=["value", "label"]
        ).dropna()
        if len(pairs) < MIN_CROSS_SECTION:
            continue
        if pairs["value"].nunique() < 2 or pairs["label"].nunique() < 2:
            continue
        rows.append(
            {
                "date": date,
                "ic": float(pairs["value"].corr(pairs["label"])),
                "rank_ic": float(pairs["value"].corr(pairs["label"], method="spearman")),
                "n_pairs": int(len(pairs)),
            }
        )
    return pd.DataFrame(rows, columns=["date", "ic", "rank_ic", "n_pairs"])


def _ic_summary(daily: pd.DataFrame, *, periods_per_year: int) -> dict:
    """Aggregate a daily IC series; t/p require >= 2 dates and variance."""
    n = int(len(daily))
    summary = {
        "ic_mean": None,
        "rank_ic_mean": None,
        "icir": None,
        "annualized_icir": None,
        "rank_icir": None,
        "annualized_rank_icir": None,
        "t_stat": None,
        "p_value": None,
        "n_dates": n,
        "daily": [],
    }
    if n:
        summary["daily"] = [
            {
                "date": row.date.date().isoformat(),
                "ic": _finite_or_none(row.ic),
                "rank_ic": _finite_or_none(row.rank_ic),
            }
            for row in daily.itertuples()
        ]
    if n == 0:
        return summary

    ic_mean = float(daily["ic"].mean())
    rank_ic_mean = float(daily["rank_ic"].mean())
    summary["ic_mean"] = _finite_or_none(ic_mean)
    summary["rank_ic_mean"] = _finite_or_none(rank_ic_mean)

    if n >= 2:
        ic_std = float(daily["ic"].std(ddof=1))
        rank_ic_std = float(daily["rank_ic"].std(ddof=1))
        if ic_std > _STD_FLOOR:
            icir = ic_mean / ic_std
            summary["icir"] = _finite_or_none(icir)
            summary["annualized_icir"] = _finite_or_none(icir * math.sqrt(periods_per_year))
            t_stat = ic_mean / (ic_std / math.sqrt(n))
            summary["t_stat"] = _finite_or_none(t_stat)
            summary["p_value"] = _finite_or_none(
                2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1))
            )
        if rank_ic_std > _STD_FLOOR:
            rank_icir = rank_ic_mean / rank_ic_std
            summary["rank_icir"] = _finite_or_none(rank_icir)
            summary["annualized_rank_icir"] = _finite_or_none(
                rank_icir * math.sqrt(periods_per_year)
            )
    return summary


def _ic_decay(values: pd.DataFrame, labels: pd.DataFrame) -> list:
    """Pooled Spearman RankIC per entry-aligned horizon 1..10.

    Horizon ``h`` pairs the factor at signal date ``t`` with the label ``h``
    signal days ahead (its entry is ``t+h+1`` on the execution calendar),
    matching the legacy ``calculate_rank_ic_decay`` lag semantics. Purged or
    unobserved labels are NaN and drop out pairwise.
    """
    stacked_values = values.stack()
    decay = []
    for horizon in IC_DECAY_HORIZONS:
        shifted = labels.shift(-horizon).stack()
        pairs = pd.concat([stacked_values, shifted], axis=1, keys=["value", "label"]).dropna()
        rank_ic = None
        if (
            len(pairs) >= MIN_CROSS_SECTION
            and pairs["value"].nunique() >= 2
            and pairs["label"].nunique() >= 2
        ):
            rank_ic = _finite_or_none(pairs["value"].corr(pairs["label"], method="spearman"))
        decay.append({"horizon": horizon, "rank_ic": rank_ic})
    return decay


def _rank_ic_autocorr(daily: pd.DataFrame, calendar: pd.DatetimeIndex) -> list:
    """Autocorrelation of the daily RankIC series at exact calendar lags 1..20.

    The series is reindexed onto the full daily calendar — NaN where a date has
    no valid cross-section — so lag ``k`` always spans ``k`` calendar days,
    never ``k`` surviving rows. Each lag correlates the series with its
    calendar-shifted self pairwise, dropping NaN pairs.
    """
    if daily.empty:
        series = pd.Series(np.nan, index=calendar, dtype="float64")
    else:
        series = daily.set_index("date")["rank_ic"].reindex(calendar)
    autocorr = []
    for lag in AUTOCORR_LAGS:
        pairs = pd.concat([series, series.shift(lag)], axis=1).dropna()
        value = None
        if len(pairs) >= 2:
            value = _finite_or_none(pairs.iloc[:, 0].corr(pairs.iloc[:, 1]))
        autocorr.append({"lag": lag, "autocorr": value})
    return autocorr


def _half_life(decay: list):
    """First horizon whose |RankIC| <= half the |horizon-1| magnitude.

    None when the horizon-1 value is missing/zero-crossing-free or no horizon
    crosses.
    """
    if not decay:
        return None
    base = decay[0].get("rank_ic")
    if base is None:
        return None
    threshold = abs(base) / 2.0
    for entry in decay[1:]:
        rank_ic = entry.get("rank_ic")
        if rank_ic is not None and abs(rank_ic) <= threshold:
            return int(entry["horizon"])
    return None


def _coverage(values: pd.DataFrame, labels: pd.DataFrame, date_mask) -> dict:
    """Label coverage against the point-in-time eligible universe.

    The denominator is the number of instruments with a factor value on each
    date — not the count of surviving labels.
    """
    eligible = values.notna().sum(axis=1)[date_mask]
    surviving = labels.notna().sum(axis=1)[date_mask]
    valid = eligible > 0
    if not valid.any():
        return {"mean_label_coverage": None, "n_dates": 0}
    ratio = surviving[valid] / eligible[valid]
    return {
        "mean_label_coverage": _finite_or_none(ratio.mean()),
        "n_dates": int(valid.sum()),
    }


def _resolve_split(values: pd.DataFrame, profile: BacktestProfile) -> pd.Timestamp:
    if profile.split_date is not None:
        return pd.Timestamp(profile.split_date)
    return values.index[-1] - pd.Timedelta(days=profile.out_of_sample_days)


def _iso(day) -> str:
    return pd.Timestamp(day).date().isoformat()


def _scenario_block(scenario: dict, split: pd.Timestamp, *, periods_per_year: int) -> dict:
    status = scenario["status"]
    ledger = scenario["ledger"]
    if status != "complete":
        known_segment = None
        if not ledger.empty:
            known_segment = _ledger_summary(ledger, periods_per_year=periods_per_year)
            known_segment["through"] = _iso(ledger.index[-1])
        return {
            "status": status,
            "full": None,
            "in_sample": None,
            "out_of_sample": None,
            "known_segment": known_segment,
        }
    return {
        "status": status,
        "full": _ledger_summary(ledger, periods_per_year=periods_per_year),
        "in_sample": _ledger_summary(
            ledger[ledger.index <= split], periods_per_year=periods_per_year
        ),
        "out_of_sample": _ledger_summary(
            ledger[ledger.index > split], periods_per_year=periods_per_year
        ),
    }


def evaluate_metrics(values, labels, accounting, profile) -> dict:
    """Evaluate factor values, labels and one accounting over sample slices.

    See the module docstring for the exact result structure. ``labels`` must
    be the evaluation-only forward-return matrix built with
    ``hold_days = profile.rebalance_days``; ``accounting`` is a single
    ``run_backtest`` result whose scenario ledgers are sliced by date — the
    portfolio is never rerun per slice.
    """
    _validate_axes(values, name="values")
    _validate_axes(labels, name="labels", expected=(values.index, values.columns))
    if len(values.index) == 0:
        raise ValueError("values must contain at least one date")
    if not isinstance(profile, BacktestProfile):
        raise TypeError("profile must be a BacktestProfile")
    if not isinstance(accounting, dict) or not isinstance(accounting.get("scenarios"), dict):
        raise TypeError("accounting must be a run_backtest result dict")
    scenarios = accounting["scenarios"]
    missing = [name for name in SCENARIOS if name not in scenarios]
    if missing:
        raise ValueError(f"accounting missing scenarios: {missing}")
    for name in SCENARIOS:
        scenario = scenarios[name]
        if "status" not in scenario or "ledger" not in scenario:
            raise ValueError(f"scenario {name!r} must carry status and ledger")
        if not isinstance(scenario["ledger"], pd.DataFrame):
            raise TypeError(f"scenario {name!r} ledger must be a DataFrame")

    hold_days = profile.rebalance_days
    entry_dates, exit_dates = label_dates(values.index, hold_days)
    split = _resolve_split(values, profile)

    in_mask = pd.Series(exit_dates <= split, index=values.index)
    out_mask = pd.Series(entry_dates > split, index=values.index)
    purged = [_iso(day) for day in values.index if not in_mask[day] and not out_mask[day]]

    labels_in = labels.copy()
    labels_in.loc[~in_mask] = np.nan
    labels_out = labels.copy()
    labels_out.loc[~out_mask] = np.nan

    ppy = profile.periods_per_year
    sample_inputs = {
        "full": (labels, pd.Series(True, index=values.index)),
        "in_sample": (labels_in, in_mask),
        "out_of_sample": (labels_out, out_mask),
    }
    samples = {}
    for name in SAMPLES:
        sliced_labels, date_mask = sample_inputs[name]
        daily = _daily_ic(values, sliced_labels)
        decay = _ic_decay(values, sliced_labels)
        samples[name] = {
            "ic": _ic_summary(daily, periods_per_year=ppy),
            "ic_decay": decay,
            "rank_ic_autocorr": _rank_ic_autocorr(daily, values.index),
            "rank_ic_half_life": _half_life(decay),
            "coverage": _coverage(values, sliced_labels, date_mask),
        }

    scenario_blocks = {
        name: _scenario_block(scenarios[name], split, periods_per_year=ppy)
        for name in SCENARIOS
    }

    return {
        "profile_id": profile.profile_id,
        "split": {
            "split_date": _iso(split),
            "out_of_sample_days": profile.out_of_sample_days,
            "hold_days": hold_days,
            "in_sample_signal_dates": int(in_mask.sum()),
            "out_of_sample_signal_dates": int(out_mask.sum()),
            "purged_signal_dates": purged,
        },
        "samples": samples,
        "scenarios": scenario_blocks,
    }


__all__ = [
    "AUTOCORR_LAGS",
    "IC_DECAY_HORIZONS",
    "MIN_CROSS_SECTION",
    "SAMPLES",
    "SCENARIOS",
    "evaluate_metrics",
    "summarize_returns",
]
