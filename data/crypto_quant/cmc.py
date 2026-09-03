"""CoinMarketCap CMC100 historical index adapter."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from datetime import date, timedelta

import pandas as pd

from .http import JsonHttpClient

CMC100_HISTORY_URL = "https://pro-api.coinmarketcap.com/public-api/v3/index/cmc100-historical"


class CmcSchemaError(ValueError):
    """Raised when a CMC response is malformed or internally inconsistent."""


def iter_cmc_windows(
    start: date, end: date, page_days: int = 10
) -> Iterator[tuple[date, date]]:
    if page_days < 1:
        raise ValueError("page_days must be positive")
    current = start
    while current <= end:
        window_end = min(end, current + timedelta(days=page_days - 1))
        yield current, window_end
        current = window_end + timedelta(days=1)


def _utc_timestamp(value: object, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.to_datetime(value, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise CmcSchemaError(f"invalid {field}") from exc
    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.Timestamp(timestamp)
    return timestamp


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CmcSchemaError(f"{label} must be an object")
    return value


def _float(value: object, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CmcSchemaError(f"invalid {field}") from exc


def _int(value: object, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CmcSchemaError(f"invalid {field}") from exc


def _deduplicate(rows: list[dict[str, object]], keys: list[str], label: str) -> list[dict[str, object]]:
    result: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row[column] for column in keys)
        previous = result.get(key)
        if previous is not None:
            if previous != row:
                raise CmcSchemaError(f"conflicting duplicate {label}: {key}")
            continue
        result[key] = row
    return list(result.values())


def normalize_cmc_payload(
    payload: object, fetched_at: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = _as_mapping(payload, "payload")
    status = root.get("status", {})
    status_mapping = _as_mapping(status, "status")
    error_code = status_mapping.get("error_code")
    if error_code not in (0, None):
        raise CmcSchemaError(f"CMC status error_code={error_code}")

    data = root.get("data")
    if not isinstance(data, list):
        raise CmcSchemaError("data must be a list")
    fetched_at_utc = _utc_timestamp(fetched_at, "fetched_at")
    daily_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for point in data:
        point_mapping = _as_mapping(point, "data point")
        constituents = point_mapping.get("constituents")
        if not isinstance(constituents, list):
            raise CmcSchemaError("missing or invalid constituents")
        point_time = point_mapping.get("update_time")
        point_value = point_mapping.get("value")
        for constituent in constituents:
            item = _as_mapping(constituent, "constituent")
            source_time = _utc_timestamp(item.get("update_time", point_time), "update_time")
            cmc_id = _int(item.get("id"), "id")
            if point_time is None:
                point_time = item.get("update_time")
            if point_value is None:
                point_value = item.get("value")
            member_rows.append({
                "date": source_time.date(),
                "cmc_id": cmc_id,
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "weight": _float(item.get("weight"), "weight"),
            })
        if point_time is None or point_value is None:
            raise CmcSchemaError("missing update_time or value")
        daily_time = _utc_timestamp(point_time, "update_time")
        daily_rows.append({
            "date": daily_time.date(),
            "index_value": _float(point_value, "value"),
            "source_update_time": daily_time,
            "fetched_at_utc": fetched_at_utc,
        })

    daily_rows = _deduplicate(daily_rows, ["date"], "date")
    member_rows = _deduplicate(member_rows, ["date", "cmc_id"], "date + cmc_id")
    daily_rows.sort(key=lambda row: row["date"])
    member_rows.sort(key=lambda row: (row["date"], row["cmc_id"]))
    daily = pd.DataFrame(daily_rows, columns=["date", "index_value", "source_update_time", "fetched_at_utc"])
    members = pd.DataFrame(member_rows, columns=["date", "cmc_id", "symbol", "name", "weight"])
    return daily, members


def fetch_cmc_history(
    client: JsonHttpClient,
    start: date,
    end: date,
    on_page: Callable[[pd.DataFrame, pd.DataFrame, date], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_pages: list[pd.DataFrame] = []
    member_pages: list[pd.DataFrame] = []
    for window_start, window_end in iter_cmc_windows(start, end):
        payload = client.get_json(
            CMC100_HISTORY_URL,
            params={
                "time_start": window_start.isoformat(),
                "time_end": window_end.isoformat(),
                "count": 10,
                "interval": "daily",
            },
        )
        daily, members = normalize_cmc_payload(payload, pd.Timestamp.now(tz="UTC"))
        daily_pages.append(daily)
        member_pages.append(members)
        if on_page is not None:
            on_page(daily, members, window_end)

    daily_columns = ["date", "index_value", "source_update_time", "fetched_at_utc"]
    member_columns = ["date", "cmc_id", "symbol", "name", "weight"]
    if not daily_pages:
        return pd.DataFrame(columns=daily_columns), pd.DataFrame(columns=member_columns)
    daily = pd.concat(daily_pages, ignore_index=True)
    members = pd.concat(member_pages, ignore_index=True)
    daily_rows = daily.to_dict("records")
    member_rows = members.to_dict("records")
    daily_rows = _deduplicate(daily_rows, ["date"], "date")
    member_rows = _deduplicate(member_rows, ["date", "cmc_id"], "date + cmc_id")
    daily = pd.DataFrame(sorted(daily_rows, key=lambda row: row["date"]), columns=daily_columns)
    members = pd.DataFrame(sorted(member_rows, key=lambda row: (row["date"], row["cmc_id"])), columns=member_columns)
    return daily, members
