"""Daily membership, kline, and funding research panel construction."""
from __future__ import annotations

from datetime import date

import pandas as pd


FUNDING_COLUMNS = [
    "date",
    "symbol",
    "funding_rate_sum",
    "funding_event_count",
    "funding_rate_last",
]


def aggregate_funding_daily(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate funding events by natural UTC day and symbol."""
    if events.empty:
        return pd.DataFrame(columns=FUNDING_COLUMNS)

    data = events.copy()
    data["funding_time"] = pd.to_datetime(data["funding_time"], errors="coerce")
    data = data.dropna(subset=["funding_time", "symbol"])
    data["date"] = data["funding_time"].dt.normalize()
    data = data.sort_values(["date", "symbol", "funding_time"], kind="stable")
    grouped = data.groupby(["date", "symbol"], sort=True, as_index=False)
    out = grouped.agg(
        funding_rate_sum=("funding_rate", "sum"),
        funding_event_count=("funding_rate", "size"),
        funding_rate_last=("funding_rate", "last"),
    )
    return out[FUNDING_COLUMNS]


def _symbol_column(frame: pd.DataFrame) -> str:
    if "symbol" in frame.columns:
        return "symbol"
    if "binance_symbol" in frame.columns:
        return "binance_symbol"
    raise ValueError("frame must contain symbol or binance_symbol")


def build_research_panel(
    universe: pd.DataFrame,
    klines: pd.DataFrame,
    funding: pd.DataFrame,
    panel_end: date,
    funding_complete_through: date,
) -> pd.DataFrame:
    """Expand accepted memberships to daily rows and left-join observations."""
    end = pd.Timestamp(panel_end).normalize()
    memberships = universe.copy()
    universe_symbol = _symbol_column(memberships)
    memberships["_start"] = pd.to_datetime(memberships["effective_date"]).dt.normalize()
    memberships["_end"] = pd.to_datetime(memberships["effective_end_date"], errors="coerce").dt.normalize()
    memberships["_end"] = memberships["_end"].fillna(end)

    expanded = []
    for _, row in memberships.iterrows():
        start = row["_start"]
        stop = min(row["_end"], end)
        if pd.notna(start) and start <= stop:
            item = {column: row[column] for column in universe.columns}
            item["symbol"] = row[universe_symbol]
            for timestamp in pd.date_range(start, stop, freq="D"):
                item["date"] = timestamp
                expanded.append(item.copy())
    panel = pd.DataFrame(expanded)
    if panel.empty:
        return pd.DataFrame(columns=["date", "symbol", "has_complete_kline", "has_complete_funding"])

    panel = panel.drop(columns=[column for column in ["effective_date", "effective_end_date"] if column in panel], errors="ignore")
    kline = klines.copy()
    kline["date"] = pd.to_datetime(kline["date"]).dt.normalize()
    kline_symbol = _symbol_column(kline)
    kline = kline.rename(columns={kline_symbol: "symbol"})
    kline = kline.drop_duplicates(["date", "symbol"], keep="last")
    panel = panel.merge(kline, on=["date", "symbol"], how="left", suffixes=("", "_kline"), indicator=True)
    complete = panel["_merge"].eq("both")
    if "completed" in panel.columns:
        complete &= panel["completed"].map(lambda value: type(value) is bool and value)
    panel["has_complete_kline"] = complete
    panel = panel.drop(columns="_merge")

    funding_daily = aggregate_funding_daily(funding)
    panel = panel.merge(funding_daily, on=["date", "symbol"], how="left")
    panel["funding_event_count"] = panel["funding_event_count"].fillna(0).astype("int64")
    panel["has_complete_funding"] = panel["date"] <= pd.Timestamp(funding_complete_through).normalize()
    return panel.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


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
        if rows["symbol"].nunique() != 50 or not rows["has_complete_kline"].fillna(False).all():
            break
        last = timestamp
        expected += pd.Timedelta(days=1)
    return last
