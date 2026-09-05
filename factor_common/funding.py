"""Signed funding-event cash flows and funding coverage policy.

Funding timestamps are timezone-aware UTC; daily price axes are tz-naive dates.
Missing funding rates are never approximated. Strict mode turns an invalid
mark price into an unresolved cash flow (never zero). Observed events alone
never prove complete coverage; only ``complete``/``not_applicable`` quality
statuses backed by independent settlement schedules are accepted.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from factor_common.profiles import BacktestProfile

ACCEPTED_COVERAGE_STATUSES = frozenset({"complete", "not_applicable"})

SETTLEMENT_COLUMNS = (
    "funding_time",
    "instrument",
    "quantity",
    "funding_rate",
    "mark_price",
    "settlement_price",
    "price_approximated",
    "resolved",
    "unresolved_reason",
    "cashflow",
)

COVERAGE_COLUMNS = (
    "date",
    "instrument",
    "observed_status",
    "status",
    "partial_day",
    "accepted",
)

_EVENT_REQUIRED_COLUMNS = {"funding_time", "instrument", "funding_rate", "mark_price"}


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def funding_cashflow(quantity, mark_price, rate) -> float:
    """Return ``-(quantity * mark_price * rate)``; longs pay positive rates."""

    if not _is_finite_number(quantity):
        raise ValueError("quantity must be finite")
    if not _is_finite_number(mark_price) or mark_price <= 0:
        raise ValueError("mark_price must be finite and positive")
    if not _is_finite_number(rate):
        raise ValueError("rate must be finite")
    return -(quantity * mark_price * rate)


def _open_lookup(opens: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], float]:
    if not isinstance(opens, pd.DataFrame):
        raise TypeError("opens must be a DataFrame of daily opening prices")
    if opens.empty:
        return {}
    index = pd.DatetimeIndex(opens.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    index = index.normalize()
    lookup: dict[tuple[pd.Timestamp, str], float] = {}
    for day, row in zip(index, opens.itertuples(index=False, name=None)):
        for instrument, value in zip(opens.columns, row):
            if _is_finite_number(value):
                lookup[(day, str(instrument))] = float(value)
    return lookup


def settle_funding(
    quantities: Mapping[str, float],
    events: pd.DataFrame,
    *,
    profile: BacktestProfile,
    opens: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Settle funding events against fixed held quantities.

    Returns one output row per event whose instrument is actually held
    (nonzero quantity) plus a diagnostics mapping. Raw event records are
    never edited: original ``mark_price`` values are preserved, replacement
    prices go to ``settlement_price`` with ``price_approximated`` set.
    Unresolved events carry ``cashflow=NaN`` and an ``unresolved_reason``;
    they are excluded from ``funding_total`` rather than treated as zero.
    """

    if not isinstance(profile, BacktestProfile):
        raise TypeError("profile must be a BacktestProfile")
    if not isinstance(quantities, Mapping):
        raise TypeError("quantities must be a mapping of instrument to quantity")
    for instrument, quantity in quantities.items():
        if not _is_finite_number(quantity):
            raise ValueError(f"quantity for {instrument} must be finite")
    missing = _EVENT_REQUIRED_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")

    open_prices = _open_lookup(opens)
    approx = profile.funding_price_mode == "daily_open_approx"

    records = events.copy(deep=True)
    times = pd.to_datetime(records["funding_time"], utc=True)
    rates = pd.to_numeric(records["funding_rate"], errors="coerce")
    marks = pd.to_numeric(records["mark_price"], errors="coerce")
    if "mark_price_valid" in records.columns:
        valid_mark = records["mark_price_valid"].fillna(False).astype(bool)
    else:
        valid_mark = marks.gt(0) & np.isfinite(marks)

    rows = []
    for position, (instrument, raw_mark) in enumerate(
        zip(records["instrument"].astype(str), records["mark_price"])
    ):
        quantity = float(quantities.get(instrument, 0.0))
        if quantity == 0.0:
            continue
        event_time = times.iloc[position]
        rate = rates.iloc[position]
        mark = marks.iloc[position]
        settlement_price = np.nan
        price_approximated = False
        unresolved_reason = None
        if not math.isfinite(rate):
            unresolved_reason = "missing_rate"
        elif not valid_mark.iloc[position]:
            replacement = None
            if approx:
                event_naive = event_time.tz_convert("UTC").tz_localize(None)
                day_open = event_naive.normalize()
                if day_open <= event_naive:
                    replacement = open_prices.get((day_open, instrument))
            if replacement is not None and replacement > 0:
                settlement_price = replacement
                price_approximated = True
            else:
                unresolved_reason = "invalid_mark_price"
        else:
            settlement_price = float(mark)
        resolved = unresolved_reason is None
        rows.append({
            "funding_time": event_time,
            "instrument": instrument,
            "quantity": quantity,
            "funding_rate": float(rate) if math.isfinite(rate) else np.nan,
            "mark_price": raw_mark,
            "settlement_price": settlement_price,
            "price_approximated": price_approximated,
            "resolved": resolved,
            "unresolved_reason": unresolved_reason,
            "cashflow": (
                funding_cashflow(quantity, settlement_price, rate) if resolved else np.nan
            ),
        })

    result = pd.DataFrame(rows, columns=list(SETTLEMENT_COLUMNS))
    resolved_rows = result[result["resolved"]] if not result.empty else result
    unresolved_reasons = (
        result["unresolved_reason"].value_counts().to_dict() if not result.empty else {}
    )
    diagnostics = {
        "status": "incomplete" if unresolved_reasons else "complete",
        "events_total": int(len(events)),
        "events_held": int(len(result)),
        "events_resolved": int(len(resolved_rows)),
        "events_unresolved": int(len(result) - len(resolved_rows)),
        "approximated_events": int(result["price_approximated"].sum()) if not result.empty else 0,
        "funding_total": float(resolved_rows["cashflow"].sum()) if not result.empty else 0.0,
        "unresolved_reasons": {str(reason): int(count) for reason, count in unresolved_reasons.items()},
    }
    return result, diagnostics


def _as_utc_naive(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def check_funding_coverage(
    held_intervals: Iterable[tuple[str, object, object]],
    quality: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Check funding coverage for held ``(instrument, start, end)`` intervals.

    Intervals cover funding event times ``start <= t < end``; callers holding
    through an exit-boundary settlement must extend ``end`` past it. Coverage
    comes from point-in-time quality rows keyed by ``(date, instrument)``;
    holding intervals are queried as given, including after membership exit.
    Only ``complete`` and ``not_applicable`` are accepted. An absent quality
    row is ``unknown``. A partially covered day cannot claim completeness
    from daily observations and is conservatively reported ``unknown``,
    unless an independent schedule proves zero events (``not_applicable``).
    """

    if "funding_coverage_status" not in quality.columns:
        raise ValueError("quality must carry funding_coverage_status")
    if not isinstance(quality.index, pd.MultiIndex) or quality.index.nlevels != 2:
        raise ValueError("quality must be indexed by (date, instrument)")

    status_by_key: dict[tuple[pd.Timestamp, str], object] = {}
    for (day, instrument), status in zip(quality.index, quality["funding_coverage_status"]):
        status_by_key[(_as_utc_naive(day).normalize(), str(instrument))] = status

    rows = []
    for interval in held_intervals:
        instrument, start, end = interval
        start_ts = _as_utc_naive(start)
        end_ts = _as_utc_naive(end)
        if not end_ts > start_ts:
            raise ValueError("held interval end must be after start")
        day = start_ts.normalize()
        while day < end_ts:
            day_end = day + pd.Timedelta(days=1)
            partial_day = day < start_ts or day_end > end_ts
            observed = status_by_key.get((day, str(instrument)))
            if observed is None or pd.isna(observed):
                observed = None
                effective = "unknown"
            else:
                effective = str(observed)
            if partial_day and effective == "complete":
                # A daily complete flag cannot prove the partial window.
                effective = "unknown"
            rows.append({
                "date": day,
                "instrument": str(instrument),
                "observed_status": observed,
                "status": effective,
                "partial_day": partial_day,
                "accepted": effective in ACCEPTED_COVERAGE_STATUSES,
            })
            day = day_end

    result = pd.DataFrame(rows, columns=list(COVERAGE_COLUMNS))
    unresolved = result[~result["accepted"]] if not result.empty else result
    status_counts = (
        result["status"].value_counts().to_dict() if not result.empty else {}
    )
    diagnostics = {
        "status": "incomplete" if len(unresolved) else "complete",
        "days_total": int(len(result)),
        "days_accepted": int(len(result) - len(unresolved)),
        "days_unresolved": int(len(unresolved)),
        "status_counts": {str(status): int(count) for status, count in status_counts.items()},
        "unresolved": [
            {
                "date": row["date"].date().isoformat(),
                "instrument": row["instrument"],
                "status": row["status"],
            }
            for _, row in unresolved.iterrows()
        ],
    }
    return result, diagnostics


__all__ = [
    "ACCEPTED_COVERAGE_STATUSES",
    "COVERAGE_COLUMNS",
    "SETTLEMENT_COLUMNS",
    "check_funding_coverage",
    "funding_cashflow",
    "settle_funding",
]
