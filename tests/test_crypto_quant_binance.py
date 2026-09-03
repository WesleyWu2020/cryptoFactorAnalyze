import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.crypto_quant.binance import (
    BINANCE_EXCHANGE_INFO_URL,
    BINANCE_FUNDING_URL,
    BINANCE_KLINES_URL,
    fetch_daily_klines,
    fetch_exchange_info,
    fetch_funding_events,
    has_completed_daily_kline,
    normalize_exchange_info,
)


FIXTURES = Path(__file__).parent / "fixtures"


class SequentialClient:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = []

    def get_json(self, url, *, params=None):
        self.calls.append((url, params))
        return next(self.payloads, [])


@pytest.fixture
def exchange_payload():
    return json.loads((FIXTURES / "binance_exchange_info.json").read_text())


@pytest.fixture
def kline_payload():
    return json.loads((FIXTURES / "binance_klines.json").read_text())


@pytest.fixture
def funding_payload():
    return json.loads((FIXTURES / "binance_funding.json").read_text())


@pytest.fixture
def fake_client(kline_payload, funding_payload):
    return SequentialClient([kline_payload, [], funding_payload])


def test_exchange_info_keeps_only_usdt_perpetuals(exchange_payload):
    out = normalize_exchange_info(exchange_payload, fetched_at=pd.Timestamp("2026-09-03"))
    assert set(out["binance_symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert set(out["contract_type"]) == {"PERPETUAL"}
    assert list(out.columns) == ["binance_symbol", "base_asset", "quote_asset", "contract_type", "onboard_date", "status", "fetched_at_utc"]
    assert pd.api.types.is_datetime64_dtype(out["onboard_date"])
    assert pd.api.types.is_datetime64_dtype(out["fetched_at_utc"])


def test_fetch_exchange_info_uses_endpoint_and_normalizes(exchange_payload):
    client = SequentialClient([exchange_payload])
    out = fetch_exchange_info(client, fetched_at=pd.Timestamp("2026-09-03"))
    assert len(out) == 2
    assert client.calls == [(BINANCE_EXCHANGE_INFO_URL, None)]


def test_kline_pagination_advances_by_last_open_time_plus_one_day(kline_payload):
    client = SequentialClient([kline_payload, []])
    out = fetch_daily_klines(client, "BTCUSDT", date(2024, 1, 1), date(2024, 1, 3))
    assert not out.duplicated(["date", "symbol"]).any()
    assert out["date"].max() <= pd.Timestamp("2024-01-03")
    assert client.calls[0][1]["limit"] == 1500
    assert client.calls[0][1]["interval"] == "1d"
    assert client.calls[1][1]["startTime"] == 1704240000000


def test_klines_discard_candle_after_completed_day_boundary(kline_payload):
    client = SequentialClient([kline_payload])
    out = fetch_daily_klines(client, "BTCUSDT", date(2024, 1, 1), date(2024, 1, 2))
    assert list(out["date"]) == [pd.Timestamp("2024-01-01")]


def test_klines_normalize_ohlc_trade_count_and_taker_fields(kline_payload):
    client = SequentialClient([kline_payload, []])
    out = fetch_daily_klines(client, "BTCUSDT", date(2024, 1, 1), date(2024, 1, 3))
    assert list(out.columns) == ["date", "symbol", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume"]
    assert out.iloc[0][["open", "high", "low", "close"]].tolist() == [42000.0, 43000.0, 41000.0, 42500.0]
    assert out.iloc[0]["trade_count"] == 1200
    assert out.iloc[0]["taker_buy_base_volume"] == 50.25
    assert out.iloc[0]["taker_buy_quote_volume"] == 2125000.0
    for column in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        assert pd.api.types.is_float_dtype(out[column])
    assert pd.api.types.is_integer_dtype(out["trade_count"])


def test_funding_pagination_advances_one_millisecond_and_normalizes_type(funding_payload):
    full_page = funding_payload * 500
    client = SequentialClient([full_page, []])
    out = fetch_funding_events(client, "BTCUSDT", 1704067200000, 1704153600000)
    assert set(out["rate_type"]) >= {"Regular"}
    assert not out.duplicated(["funding_time", "symbol", "rate_type"]).any()
    assert client.calls[0][1]["limit"] == 1000
    assert client.calls[1][1]["startTime"] == 1704096000001
    assert pd.api.types.is_float_dtype(out["funding_rate"])
    assert pd.api.types.is_float_dtype(out["mark_price"])
    assert pd.isna(out.loc[out["rate_type"] == "Regular", "mark_price"]).all()


def test_historical_probe_requires_a_completed_t_minus_one_kline(kline_payload):
    client = SequentialClient([[kline_payload[0]]])
    assert has_completed_daily_kline(client, "BTCUSDT", date(2024, 1, 2)) is True
    assert client.calls[0][1]["limit"] == 1500
