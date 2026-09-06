"""Daily membership, kline, and funding research panel construction."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .universe import placeholder_kline_mask


FUNDING_COLUMNS = [
    "date",
    "symbol",
    "funding_rate_sum",
    "funding_rate_mean",
    "funding_event_count",
    "funding_rate_last",
    "funding_invalid_price_count",
]

KLINE_COLUMNS = [
    "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_base_volume",
    "taker_buy_quote_volume",
]
REQUIRED_KLINE_COLUMNS = ["open", "high", "low", "close", "volume"]
RESEARCH_PANEL_COLUMNS = [
    "date", "binance_symbol", "universe_effective_date", "cmc_weight_at_decision",
    *KLINE_COLUMNS,
    "funding_rate_sum", "funding_rate_mean", "funding_event_count", "funding_rate_last",
    "has_complete_kline", "has_complete_funding",
    "has_placeholder_kline", "funding_coverage_status", "funding_invalid_price_count",
]


def _utc_naive(frame: pd.Series) -> pd.Series:
    parsed = frame if pd.api.types.is_datetime64_any_dtype(frame) else frame.map(lambda value: pd.to_datetime(value, errors="coerce", utc=True))
    return pd.to_datetime(parsed, errors="coerce", utc=True).dt.tz_localize(None)


def annotate_funding_prices(events: pd.DataFrame) -> pd.DataFrame:
    """Annotate usable settlement prices without altering source values or rates."""
    result = events.copy()
    price = pd.to_numeric(result.get("mark_price", pd.Series(float("nan"), index=result.index)), errors="coerce")
    result["mark_price_valid"] = price.gt(0) & np.isfinite(price)
    return result


def aggregate_funding_daily(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate funding events by natural UTC day and symbol."""
    if events.empty:
        return pd.DataFrame(columns=FUNDING_COLUMNS)

    data = annotate_funding_prices(events)
    data["_invalid_price"] = ~data["mark_price_valid"]
    event_symbol = _symbol_column(data)
    if event_symbol != "symbol":
        data = data.rename(columns={event_symbol: "symbol"})
    data["funding_time"] = _utc_naive(data["funding_time"])
    data["funding_rate"] = pd.to_numeric(data["funding_rate"], errors="raise").astype("float64")
    data["_invalid_rate"] = ~np.isfinite(data["funding_rate"])
    data = data.dropna(subset=["funding_time", "symbol"])
    data["date"] = data["funding_time"].dt.normalize()
    data = data.sort_values(["date", "symbol", "funding_time"], kind="stable")
    grouped = data.groupby(["date", "symbol"], sort=True, as_index=False)
    out = grouped.agg(
        funding_rate_sum=("funding_rate", "sum"),
        funding_rate_mean=("funding_rate", "mean"),
        funding_event_count=("funding_rate", "size"),
        funding_rate_last=("funding_rate", "last"),
        funding_invalid_price_count=("_invalid_price", "sum"),
        _invalid_rate=("_invalid_rate", "any"),
    )
    out.loc[out["_invalid_rate"], ["funding_rate_sum", "funding_rate_mean", "funding_rate_last"]] = float("nan")
    return out[FUNDING_COLUMNS]


def build_funding_schedule(
    universe: pd.DataFrame,
    funding: pd.DataFrame,
    panel_end: date,
) -> pd.DataFrame:
    """Build an auditable expected-time grid from Binance event cadences.

    Binance contracts normally settle on regular UTC grids. The raw history
    contains symbols that temporarily use 4h/2h/1h intervals, so one global
    8h assumption would incorrectly mark valid rows as missing. We learn a
    per-symbol template for each observed daily event count, then use the
    modal template to test days with missing or partial observations. This
    detects dropped events while preserving interval changes visible in the
    raw history; it is still a data-derived schedule and should be replaced
    by an exchange-published historical schedule when one is available.
    """
    if universe.empty or funding.empty:
        return pd.DataFrame(columns=["date", "binance_symbol", "expected_times"])
    members = universe.copy()
    members["_start"] = _utc_naive(members["effective_date"]).dt.normalize()
    members["_end"] = _utc_naive(members["effective_end_date"]).dt.normalize()
    end = pd.Timestamp(panel_end).normalize()
    events = funding.copy()
    symbol = _symbol_column(events)
    events["_time"] = _utc_naive(events["funding_time"])
    events["_day"] = events["_time"].dt.normalize()
    events["_symbol"] = events[symbol].astype(str)
    events = events.dropna(subset=["_time", "_day", "_symbol"])
    grouped = {
        key: sorted(group["_time"].tolist())
        for key, group in events.groupby(["_symbol", "_day"], sort=False)
    }
    templates: dict[str, dict[int, list[pd.Timedelta]]] = {}
    modal: dict[str, int] = {}
    for (instrument, _day), times in grouped.items():
        count = len(times)
        day_template = [time - pd.Timestamp(time).normalize() for time in times]
        templates.setdefault(instrument, {})
        templates[instrument].setdefault(count, day_template)
    # Contract intervals can change during a UTC day.  For example, a symbol
    # may produce hourly settlements for the first few hours and then switch
    # to a four-hour cadence, yielding 5, 9, or another non-standard daily
    # count.  Counts below three are retained as incomplete-day evidence and
    # fall back to the symbol's modal complete template.
    valid_counts = set(range(3, 25))
    for instrument, by_count in templates.items():
        valid = {count: value for count, value in by_count.items() if count in valid_counts}
        if not valid:
            continue
        modal[instrument] = max(
            valid,
            key=lambda count: sum(
                len(times) == count
                for (symbol_value, _), times in grouped.items()
                if symbol_value == instrument
            ),
        )

    rows = []
    for _, row in members.iterrows():
        instrument = str(row["binance_symbol"])
        start = row["_start"]
        stop = row["_end"]
        stop = min(stop if pd.notna(stop) else end, end)
        if pd.isna(start) or start > stop:
            continue
        for day in pd.date_range(start, stop, freq="D"):
            actual = grouped.get((instrument, day), [])
            count = len(actual)
            template = templates.get(instrument, {}).get(count) if count in valid_counts else None
            if template is None:
                template = templates.get(instrument, {}).get(modal.get(instrument, 0), [])
            expected = [day + offset for offset in template]
            rows.append({
                "date": day,
                "binance_symbol": instrument,
                "expected_times": expected,
            })
    return pd.DataFrame(rows, columns=["date", "binance_symbol", "expected_times"])


def _symbol_column(frame: pd.DataFrame) -> str:
    if "binance_symbol" in frame.columns:
        return "binance_symbol"
    if "symbol" in frame.columns:
        return "symbol"
    raise ValueError("frame must contain symbol or binance_symbol")


def build_research_panel(
    universe: pd.DataFrame,
    klines: pd.DataFrame,
    funding: pd.DataFrame,
    panel_end: date,
    funding_complete_through: date | None = None,
    *,
    funding_schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expand memberships without deleting holdings with missing observations.

    funding_complete_through is retained for caller compatibility only: a fetch
    watermark cannot establish event completeness. funding_schedule, if supplied,
    has one row per date/symbol and an exhaustive expected_times list for that day.
    An empty list explicitly means no settlement was applicable. Missing schedule
    rows mean unknown. Only use historically verified schedules here.
    """
    end = pd.Timestamp(panel_end).normalize()
    memberships = universe.copy()
    universe_symbol = _symbol_column(memberships)
    memberships["_start"] = _utc_naive(memberships["effective_date"]).dt.normalize()
    memberships["_end"] = _utc_naive(memberships["effective_end_date"]).dt.normalize()
    memberships["_end"] = memberships["_end"].fillna(end)

    expanded = []
    for _, row in memberships.iterrows():
        start = row["_start"]
        stop = min(row["_end"], end)
        if pd.notna(start) and start <= stop:
            item = {
                "date": None,
                "binance_symbol": row[universe_symbol],
                "universe_effective_date": start,
                "cmc_weight_at_decision": row.get(
                    "cmc_weight_at_decision", row.get("weight", float("nan"))
                ),
            }
            for timestamp in pd.date_range(start, stop, freq="D"):
                item["date"] = timestamp
                expanded.append(item.copy())
    panel = pd.DataFrame(expanded)
    if panel.empty:
        return pd.DataFrame(columns=RESEARCH_PANEL_COLUMNS)

    kline = klines.copy()
    kline["date"] = _utc_naive(kline["date"]).dt.normalize()
    kline_symbol = _symbol_column(kline)
    kline = kline.rename(columns={kline_symbol: "binance_symbol"})
    kline = kline.drop_duplicates(["date", "binance_symbol"], keep="last")
    kline_columns = [column for column in KLINE_COLUMNS if column in kline.columns]
    kline = kline[["date", "binance_symbol", *kline_columns, *(["completed"] if "completed" in kline else [])]]
    panel = panel.merge(kline, on=["date", "binance_symbol"], how="left", indicator=True)
    for column in KLINE_COLUMNS:
        if column not in panel:
            panel[column] = pd.NA
    complete = panel["_merge"].eq("both")
    complete &= panel[REQUIRED_KLINE_COLUMNS].notna().all(axis=1)
    if "completed" in panel.columns:
        complete &= panel["completed"].map(lambda value: type(value) is bool and value)
    panel["has_placeholder_kline"] = placeholder_kline_mask(panel)
    panel["has_complete_kline"] = complete & ~panel["has_placeholder_kline"]
    panel = panel.drop(columns=["_merge", "completed"], errors="ignore")
    funding_daily = aggregate_funding_daily(funding)
    panel = panel.merge(funding_daily.rename(columns={"symbol": "binance_symbol"}), on=["date", "binance_symbol"], how="left")
    panel["funding_event_count"] = panel["funding_event_count"].fillna(0).astype("int64")
    panel["funding_invalid_price_count"] = panel["funding_invalid_price_count"].fillna(0).astype("int64")
    panel["funding_coverage_status"] = "unknown"
    panel.loc[panel["funding_event_count"].eq(0), "funding_coverage_status"] = "no_events"
    for column in ("funding_rate_sum", "funding_rate_mean", "funding_rate_last"):
        panel[column] = pd.to_numeric(panel[column], errors="raise").astype("float64")
    invalid_rate = panel["funding_event_count"].gt(0) & ~np.isfinite(panel["funding_rate_sum"])
    if funding_schedule is not None and not funding_schedule.empty:
        schedule = funding_schedule.rename(columns={_symbol_column(funding_schedule): "binance_symbol"}).copy()
        schedule["date"] = _utc_naive(schedule["date"]).dt.normalize()
        if schedule.duplicated(["date", "binance_symbol"]).any():
            raise ValueError("duplicate funding schedule date/symbol")
        observed = {}
        if not funding.empty:
            event_data = funding.copy()
            event_data["_time"] = _utc_naive(event_data["funding_time"])
            event_data["_date"] = event_data["_time"].dt.normalize()
            for key, group in event_data.groupby(["_date", _symbol_column(event_data)]):
                observed[key] = sorted(group["_time"].tolist())
        statuses = {}
        for row in schedule.itertuples(index=False):
            expected = sorted(_utc_naive(pd.Series(row.expected_times, dtype=object)).tolist())
            if any(pd.isna(t) or t.normalize() != row.date for t in expected) or len(set(expected)) != len(expected):
                raise ValueError("invalid or duplicate expected funding timestamp")
            actual = observed.get((row.date, row.binance_symbol), [])
            # Binance historical timestamps can drift by milliseconds. One-to-one
            # matching within one second preserves missing/duplicate event detection.
            matches = len(actual) == len(expected) and all(abs(a - b) <= pd.Timedelta(seconds=1) for a, b in zip(actual, expected))
            if matches:
                status = "complete" if expected else "not_applicable"
            elif len(actual) < len(expected):
                status = "missing"
            else:
                status = "schedule_mismatch"
            statuses[(row.date, row.binance_symbol)] = status
        panel["funding_coverage_status"] = [statuses.get((row.date, row.binance_symbol), row.funding_coverage_status) for row in panel.itertuples(index=False)]
    panel.loc[invalid_rate, "funding_coverage_status"] = "invalid_rate"
    panel["has_complete_funding"] = panel["funding_coverage_status"].eq("complete")
    return panel[RESEARCH_PANEL_COLUMNS].sort_values(["date", "binance_symbol"], kind="stable").reset_index(drop=True)


def last_complete_panel_date(panel: pd.DataFrame) -> pd.Timestamp | None:
    """Return the end of the contiguous 50-member, complete-kline prefix."""
    if panel.empty:
        return None
    dates = sorted(pd.to_datetime(panel["date"]).dt.normalize().unique())
    expected = dates[0]
    last = None
    for timestamp in dates:
        timestamp = pd.Timestamp(timestamp)
        if timestamp != expected:
            break
        rows = panel[panel["date"] == timestamp]
        if rows["binance_symbol"].nunique() != 50 or not rows["has_complete_kline"].fillna(False).all():
            break
        last = timestamp
        expected += pd.Timedelta(days=1)
    return last
