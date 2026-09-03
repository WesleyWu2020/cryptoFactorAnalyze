"""Daily membership, kline, and funding research panel construction."""
from __future__ import annotations

from datetime import date

import pandas as pd


FUNDING_COLUMNS = [
    "date",
    "symbol",
    "funding_rate_sum",
    "funding_rate_mean",
    "funding_event_count",
    "funding_rate_last",
]

KLINE_COLUMNS = [
    "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "trade_count", "taker_buy_base_volume",
    "taker_buy_quote_volume",
]
REQUIRED_KLINE_COLUMNS = ["open", "high", "low", "close", "volume"]
RESEARCH_PANEL_COLUMNS = [
    "date", "binance_symbol", "universe_effective_date", "cmc_weight_at_decision",
    *KLINE_COLUMNS,
    "funding_rate_sum", "funding_rate_mean", "funding_event_count", "funding_rate_last",
    "has_complete_kline", "has_complete_funding",
]


def _utc_naive(frame: pd.Series) -> pd.Series:
    return pd.to_datetime(frame, errors="coerce", utc=True).dt.tz_localize(None)


def aggregate_funding_daily(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate funding events by natural UTC day and symbol."""
    if events.empty:
        return pd.DataFrame(columns=FUNDING_COLUMNS)

    data = events.copy()
    event_symbol = _symbol_column(data)
    if event_symbol != "symbol":
        data = data.rename(columns={event_symbol: "symbol"})
    data["funding_time"] = _utc_naive(data["funding_time"])
    data = data.dropna(subset=["funding_time", "symbol"])
    data["date"] = data["funding_time"].dt.normalize()
    data = data.sort_values(["date", "symbol", "funding_time"], kind="stable")
    grouped = data.groupby(["date", "symbol"], sort=True, as_index=False)
    out = grouped.agg(
        funding_rate_sum=("funding_rate", "sum"),
        funding_rate_mean=("funding_rate", "mean"),
        funding_event_count=("funding_rate", "size"),
        funding_rate_last=("funding_rate", "last"),
    )
    return out[FUNDING_COLUMNS]


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
    funding_complete_through: date,
) -> pd.DataFrame:
    """Expand accepted memberships to daily rows and left-join observations."""
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
    panel["has_complete_kline"] = complete
    panel = panel.drop(columns=["_merge", "completed"], errors="ignore")
    funding_daily = aggregate_funding_daily(funding)
    panel = panel.merge(funding_daily.rename(columns={"symbol": "binance_symbol"}), on=["date", "binance_symbol"], how="left")
    panel["funding_event_count"] = panel["funding_event_count"].fillna(0).astype("int64")
    panel["has_complete_funding"] = panel["date"] <= pd.Timestamp(funding_complete_through).normalize()
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
