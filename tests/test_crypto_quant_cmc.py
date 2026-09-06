import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from data.crypto_quant.cmc import (
    CmcSchemaError,
    fetch_cmc_history,
    iter_cmc_windows,
    normalize_cmc_payload,
)
from data.crypto_quant.cmc import _utc_timestamp


@pytest.fixture
def fixture_payload():
    path = Path(__file__).parent / "fixtures" / "cmc100_page.json"
    return json.loads(path.read_text())


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get_json(self, url, *, params=None):
        self.calls.append((url, params))
        return self.payload


class SequentialClient(FakeClient):
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = []

    def get_json(self, url, *, params=None):
        self.calls.append((url, params))
        return next(self.payloads)


@pytest.fixture
def fake_client(fixture_payload):
    return FakeClient(fixture_payload)


def test_iter_cmc_windows_is_inclusive_and_ten_days():
    assert list(iter_cmc_windows(date(2024, 1, 1), date(2024, 1, 23))) == [
        (date(2024, 1, 1), date(2024, 1, 10)),
        (date(2024, 1, 11), date(2024, 1, 20)),
        (date(2024, 1, 21), date(2024, 1, 23)),
    ]


def test_normalize_cmc_payload_uses_update_date_and_cmc_id(fixture_payload):
    daily, members = normalize_cmc_payload(
        fixture_payload, fetched_at=pd.Timestamp("2026-09-03T00:20:00")
    )
    assert list(daily.columns) == [
        "date", "index_value", "source_update_time", "fetched_at_utc"
    ]
    assert {"date", "cmc_id", "symbol", "name", "weight"} <= set(members.columns)
    assert not members.duplicated(["date", "cmc_id"]).any()
    assert list(members.loc[members.date == date(2024, 1, 1), "cmc_id"]) == [1, 2, 3]
    assert daily.loc[0, "date"] == date(2024, 1, 1)
    assert daily.loc[0, "source_update_time"] == pd.Timestamp("2024-01-01", tz="UTC")


def test_normalize_cmc_payload_accepts_keyless_cmc100_shape():
    payload = {
        "status": {"error_code": "0"},
        "data": [
            {
                "value": "100.00",
                "update_time": "2024-01-01T00:00:00.000Z",
                "constituents": [
                    {
                        "id": cmc_id,
                        "name": f"Coin {cmc_id}",
                        "symbol": f"C{cmc_id}",
                        "url": [f"https://coinmarketcap.com/currencies/c{cmc_id}/"],
                        "weight": 0.01,
                    }
                    for cmc_id in range(1, 101)
                ],
            }
        ],
    }

    daily, members = normalize_cmc_payload(
        payload, fetched_at=pd.Timestamp("2026-09-03T00:20:00")
    )

    assert daily.loc[0, "index_value"] == 100.0
    assert len(members) == 100
    assert set(members["cmc_id"]) == set(range(1, 101))


@pytest.mark.parametrize("field, value", [("symbol", None), ("name", None), ("symbol", ""), ("name", "")])
def test_normalize_cmc_payload_preserves_real_cmc100_missing_text_shape(field, value):
    payload = {
        "status": {"error_code": 0},
        "data": [
            {
                "value": 100.0,
                "update_time": "2024-01-01T00:00:00.000Z",
                "constituents": [
                    {
                        "id": 28683,
                        "name": "Coin 28683",
                        "symbol": "C28683",
                        "weight": 0.01,
                    }
                ],
            }
        ],
    }
    payload["data"][0]["constituents"][0][field] = value

    _, members = normalize_cmc_payload(payload, pd.Timestamp("2026-09-03"))

    assert len(members) == 1
    assert members.loc[0, "cmc_id"] == 28683
    assert members.loc[0, field] == ""


def test_normalize_cmc_payload_accepts_string_zero_status_code(fixture_payload):
    payload = json.loads(json.dumps(fixture_payload))
    payload["status"]["error_code"] = " 0 "

    daily, members = normalize_cmc_payload(
        payload, fetched_at=pd.Timestamp("2026-09-03T00:20:00")
    )

    assert not daily.empty
    assert not members.empty


def test_fetch_history_calls_page_callback_after_each_valid_page(fake_client):
    calls = []
    daily, members = fetch_cmc_history(
        fake_client,
        date(2024, 1, 1),
        date(2024, 1, 2),
        on_page=lambda d, m, end: calls.append((len(d), len(m), end)),
    )
    assert calls == [(2, 6, date(2024, 1, 2))]
    assert len(daily) == 2
    assert len(members) == 6
    assert fake_client.calls[0][1] == {
        "time_start": "2024-01-01T00:00:00Z",
        "time_end": "2024-01-02T23:59:59Z",
        "count": 10,
        "interval": "daily",
    }


def test_fetch_history_sends_full_utc_timestamps_for_daily_closed_window(fake_client):
    fetch_cmc_history(fake_client, date(2024, 1, 1), date(2024, 1, 2))

    assert fake_client.calls[0][1]["time_start"] == "2024-01-01T00:00:00Z"
    assert fake_client.calls[0][1]["time_end"] == "2024-01-02T23:59:59Z"


def test_fetch_history_caps_current_day_end_at_current_utc_time(monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 5, 2, 55, tzinfo=tz or timezone.utc)

    monkeypatch.setattr("data.crypto_quant.cmc.datetime", FrozenDateTime)
    client = FakeClient({"status": {"error_code": 0}, "data": []})

    fetch_cmc_history(client, date(2026, 9, 5), date(2026, 9, 5))

    assert client.calls[0][1]["time_end"] == "2026-09-05T02:55:00Z"


def test_normalize_rejects_malformed_status(fixture_payload):
    payload = {**fixture_payload, "status": {"error_code": 100}}
    with pytest.raises(CmcSchemaError, match="error_code"):
        normalize_cmc_payload(payload, pd.Timestamp("2026-09-03"))


@pytest.mark.parametrize("status", [None, {}, []])
def test_normalize_requires_nonempty_status(fixture_payload, status):
    payload = {**fixture_payload, "status": status}
    if status is None:
        payload.pop("status")
    with pytest.raises(CmcSchemaError, match="status"):
        normalize_cmc_payload(payload, pd.Timestamp("2026-09-03"))


@pytest.mark.parametrize("status", [{"error_code": None}, {"message": "ok"}])
def test_normalize_requires_status_error_code_zero(fixture_payload, status):
    payload = {**fixture_payload, "status": status}
    with pytest.raises(CmcSchemaError, match="error_code"):
        normalize_cmc_payload(payload, pd.Timestamp("2026-09-03"))


@pytest.mark.parametrize("error_code", [False, 0.0, "1", " 1 ", "not-a-code"])
def test_normalize_requires_builtin_integer_zero_status_code(
    fixture_payload, error_code
):
    payload = {**fixture_payload, "status": {"error_code": error_code}}
    with pytest.raises(CmcSchemaError, match="error_code"):
        normalize_cmc_payload(payload, pd.Timestamp("2026-09-03"))


@pytest.mark.parametrize(
    "value", [None, "", "not-a-timestamp", pd.NaT, [], ["2024-01-01T00:00:00Z"]]
)
def test_utc_timestamp_rejects_missing_or_invalid_values(value):
    with pytest.raises(CmcSchemaError, match="timestamp"):
        _utc_timestamp(value, "timestamp")


def test_normalize_rejects_non_integer_id(fixture_payload):
    payload = json.loads(json.dumps(fixture_payload))
    payload["data"][0]["constituents"][0]["id"] = 1.5
    with pytest.raises(CmcSchemaError, match="id"):
        normalize_cmc_payload(payload, pd.Timestamp("2026-09-03"))


@pytest.mark.parametrize("field, value", [("weight", float("nan")), ("weight", float("inf"))])
def test_normalize_rejects_nonfinite_weight(fixture_payload, field, value):
    payload = json.loads(json.dumps(fixture_payload))
    payload["data"][0]["constituents"][0][field] = value
    with pytest.raises(CmcSchemaError, match=field):
        normalize_cmc_payload(payload, pd.Timestamp("2026-09-03"))


def test_normalize_rejects_nonfinite_point_value(fixture_payload):
    payload = json.loads(json.dumps(fixture_payload))
    payload["data"][0]["value"] = float("nan")
    with pytest.raises(CmcSchemaError, match="value"):
        normalize_cmc_payload(payload, pd.Timestamp("2026-09-03"))


@pytest.mark.parametrize("field, value", [("symbol", 123), ("name", 123)])
def test_normalize_rejects_non_string_text_fields(fixture_payload, field, value):
    payload = json.loads(json.dumps(fixture_payload))
    payload["data"][0]["constituents"][0][field] = value
    with pytest.raises(CmcSchemaError, match=field):
        normalize_cmc_payload(payload, pd.Timestamp("2026-09-03"))


def test_fetch_deduplicates_cross_page_member_by_business_fields(
    fixture_payload, monkeypatch
):
    monkeypatch.setattr(
        "data.crypto_quant.cmc.iter_cmc_windows",
        lambda start, end: iter([(date(2024, 1, 1), date(2024, 1, 2))] * 3),
    )
    client = SequentialClient([fixture_payload] * 3)
    daily, members = fetch_cmc_history(client, date(2024, 1, 1), date(2024, 1, 23))
    assert len(daily) == 2
    assert len(members) == 6


def test_fetch_rejects_cross_page_conflict_before_second_callback(
    fixture_payload, monkeypatch
):
    monkeypatch.setattr(
        "data.crypto_quant.cmc.iter_cmc_windows",
        lambda start, end: iter([(date(2024, 1, 1), date(2024, 1, 2))] * 3),
    )
    conflicting = json.loads(json.dumps(fixture_payload))
    conflicting["data"][0]["constituents"][0]["weight"] = 0.99
    client = SequentialClient([fixture_payload, conflicting, fixture_payload])
    callbacks = []
    with pytest.raises(CmcSchemaError, match="duplicate"):
        fetch_cmc_history(
            client,
            date(2024, 1, 1),
            date(2024, 1, 23),
            on_page=lambda d, m, end: callbacks.append(end),
        )
    assert callbacks == [date(2024, 1, 2)]


def test_normalize_rejects_missing_constituents(fixture_payload):
    payload = {**fixture_payload, "data": [{"value": 1, "update_time": "2024-01-01T00:00:00Z"}]}
    with pytest.raises(CmcSchemaError, match="constituents"):
        normalize_cmc_payload(payload, pd.Timestamp("2026-09-03"))


def test_normalize_rejects_missing_point_value(fixture_payload):
    payload = json.loads(json.dumps(fixture_payload))
    payload["data"][0].pop("value")
    with pytest.raises(CmcSchemaError, match="value"):
        normalize_cmc_payload(payload, pd.Timestamp("2026-09-03"))


def test_normalize_rejects_conflicting_duplicate_date_and_cmc_id(fixture_payload):
    first = fixture_payload["data"][0]["constituents"][0]
    duplicate = {**first, "weight": 0.9}
    payload = {
        **fixture_payload,
        "data": [
            {
                "value": fixture_payload["data"][0]["value"],
                "update_time": fixture_payload["data"][0]["update_time"],
                "constituents": [first, duplicate],
            }
        ],
    }
    with pytest.raises(CmcSchemaError, match="duplicate"):
        normalize_cmc_payload(payload, pd.Timestamp("2026-09-03"))


def test_fetch_history_empty_2023_interval_does_not_request(fake_client):
    fake_client.payload = {"status": {"error_code": 0}, "data": []}
    daily, members = fetch_cmc_history(
        fake_client, date(2023, 1, 1), date(2023, 1, 2)
    )
    assert daily.empty
    assert members.empty
    assert len(fake_client.calls) == 1


def test_fetch_rejects_page_dates_outside_requested_window(fixture_payload):
    out_of_window = json.loads(json.dumps(fixture_payload))
    for point in out_of_window["data"]:
        for constituent in point["constituents"]:
            constituent["update_time"] = "2024-01-03T00:00:00Z"
    out_of_window["data"] = out_of_window["data"][:1]
    client = FakeClient(out_of_window)
    callbacks = []
    with pytest.raises(CmcSchemaError, match="window"):
        fetch_cmc_history(
            client,
            date(2024, 1, 1),
            date(2024, 1, 2),
            on_page=lambda d, m, end: callbacks.append(end),
        )
    assert callbacks == []
