"""CoinMarketCap CMC100 historical index adapter."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from datetime import date, datetime, time, timedelta, timezone
from math import isfinite
from numbers import Integral

import pandas as pd
from pandas.api.types import is_scalar

from .http import JsonHttpClient

CMC100_HISTORY_URL = "https://pro-api.coinmarketcap.com/public-api/v3/index/cmc100-historical"


def _cmc_day_timestamp(value: date, *, end_of_day: bool = False) -> str:
    day_time = time(23, 59, 59) if end_of_day else time.min
    return datetime.combine(value, day_time, tzinfo=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


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
    if not is_scalar(value):
        raise CmcSchemaError(f"invalid {field}: timestamp must be scalar")
    if value is None or (isinstance(value, str) and not value.strip()):
        raise CmcSchemaError(f"invalid {field}: timestamp is required")
    try:
        timestamp = pd.to_datetime(value, utc=True, errors="raise")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CmcSchemaError(f"invalid {field}") from exc
    if pd.isna(timestamp):
        raise CmcSchemaError(f"invalid {field}: timestamp is required")
    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.Timestamp(timestamp)
    return timestamp


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CmcSchemaError(f"{label} must be an object")
    return value


def _float(value: object, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise CmcSchemaError(f"invalid {field}") from exc
    if not isfinite(converted):
        raise CmcSchemaError(f"invalid {field}: must be finite")
    return converted


def _int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CmcSchemaError(f"invalid {field}: must be an integer")
    return int(value)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CmcSchemaError(f"invalid {field}: must be a non-empty string")
    return value


def _deduplicate(
    rows: list[dict[str, object]],
    keys: list[str],
    label: str,
    compare_columns: list[str] | None = None,
) -> list[dict[str, object]]:
    result: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row[column] for column in keys)
        previous = result.get(key)
        if previous is not None:
            columns = compare_columns or list(row)
            if any(previous[column] != row[column] for column in columns):
                raise CmcSchemaError(f"conflicting duplicate {label}: {key}")
            continue
        result[key] = row
    return list(result.values())


def normalize_cmc_payload(
    payload: object, fetched_at: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = _as_mapping(payload, "payload")
    if "status" not in root:
        raise CmcSchemaError("missing status")
    status_mapping = _as_mapping(root["status"], "status")
    if not status_mapping:
        raise CmcSchemaError("status must not be empty")
    error_code = status_mapping.get("error_code")
    valid_error_code = (type(error_code) is int and error_code == 0) or (
        type(error_code) is str and error_code.strip() == "0"
    )
    if not valid_error_code:
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
            symbol = _text(item.get("symbol"), "symbol")
            name = _text(item.get("name"), "name")
            weight = _float(item.get("weight"), "weight")
            constituent_value = _float(item.get("value"), "value")
            if point_time is None:
                point_time = item.get("update_time")
            if point_value is None:
                point_value = constituent_value
            member_rows.append({
                "date": source_time.date(),
                "cmc_id": cmc_id,
                "symbol": symbol,
                "name": name,
                "weight": weight,
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

    daily_rows = _deduplicate(
        daily_rows,
        ["date"],
        "date",
        ["date", "index_value", "source_update_time"],
    )
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
    daily_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for window_start, window_end in iter_cmc_windows(start, end):
        payload = client.get_json(
            CMC100_HISTORY_URL,
            params={
                "time_start": _cmc_day_timestamp(window_start),
                "time_end": _cmc_day_timestamp(window_end, end_of_day=True),
                "count": 10,
                "interval": "daily",
            },
        )
        daily, members = normalize_cmc_payload(payload, pd.Timestamp.now(tz="UTC"))
        if not daily.empty and (
            (daily["date"] < window_start) | (daily["date"] > window_end)
        ).any():
            raise CmcSchemaError(
                f"CMC response date outside request window [{window_start}, {window_end}]"
            )
        if not members.empty and (
            (members["date"] < window_start) | (members["date"] > window_end)
        ).any():
            raise CmcSchemaError(
                f"CMC response date outside request window [{window_start}, {window_end}]"
            )
        candidate_daily = daily_rows + daily.to_dict("records")
        candidate_members = member_rows + members.to_dict("records")
        daily_rows = _deduplicate(
            candidate_daily,
            ["date"],
            "date",
            ["date", "index_value", "source_update_time"],
        )
        member_rows = _deduplicate(
            candidate_members, ["date", "cmc_id"], "date + cmc_id"
        )
        if on_page is not None:
            on_page(daily, members, window_end)

    daily_columns = ["date", "index_value", "source_update_time", "fetched_at_utc"]
    member_columns = ["date", "cmc_id", "symbol", "name", "weight"]
    if not daily_rows:
        return pd.DataFrame(columns=daily_columns), pd.DataFrame(columns=member_columns)
    daily = pd.DataFrame(sorted(daily_rows, key=lambda row: row["date"]), columns=daily_columns)
    members = pd.DataFrame(sorted(member_rows, key=lambda row: (row["date"], row["cmc_id"])), columns=member_columns)
    return daily, members
