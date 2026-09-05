from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.crypto_quant.panel import build_research_panel
from data.crypto_quant.store import CryptoQuantStore


SYMBOLS = [f"{letter}USDT" for letter in "ABCDEFGHIJKL"]
INITIAL_SYMBOLS = SYMBOLS[:6]
LATER_SYMBOLS = SYMBOLS[6:]
CALENDAR = pd.date_range("2024-01-01", "2024-01-12", freq="D")


def _kline_row(day: pd.Timestamp, symbol: str, *, close: float, placeholder: bool = False) -> dict:
    if placeholder:
        return {
            "date": day,
            "symbol": symbol,
            "close_time": day + pd.Timedelta(hours=23, minutes=59),
            "open": 0.248,
            "high": 0.248,
            "low": 0.248,
            "close": 0.248,
            "volume": 0.0,
            "quote_volume": 0.0,
            "trade_count": 0,
            "taker_buy_base_volume": 0.0,
            "taker_buy_quote_volume": 0.0,
        }
    return {
        "date": day,
        "symbol": symbol,
        "close_time": day + pd.Timedelta(hours=23, minutes=59),
        "open": close - 1.0,
        "high": close + 1.0,
        "low": close - 2.0,
        "close": close,
        "volume": 1_000.0,
        "quote_volume": close * 1_000.0,
        "trade_count": 10,
        "taker_buy_base_volume": 500.0,
        "taker_buy_quote_volume": close * 500.0,
    }


def _universe() -> pd.DataFrame:
    rows = []
    for cmc_id, symbol in enumerate(INITIAL_SYMBOLS, start=1):
        rows.append({
            "decision_date": "2024-01-01",
            "effective_date": "2024-01-02",
            "effective_end_date": "2024-01-06",
            "cmc_id": cmc_id,
            "cmc_symbol": symbol[:-4],
            "binance_symbol": symbol,
            "market_cap_rank": cmc_id,
            "cmc_weight": 1.0 / len(SYMBOLS),
        })
    for cmc_id, symbol in enumerate(LATER_SYMBOLS, start=7):
        rows.append({
            "decision_date": "2024-01-06",
            "effective_date": "2024-01-07",
            "effective_end_date": pd.NaT,
            "cmc_id": cmc_id,
            "cmc_symbol": symbol[:-4],
            "binance_symbol": symbol,
            "market_cap_rank": cmc_id,
            "cmc_weight": 1.0 / len(SYMBOLS),
        })
    return pd.DataFrame(rows)


def _klines() -> pd.DataFrame:
    rows = []
    for day in CALENDAR:
        for index, symbol in enumerate(SYMBOLS, start=1):
            if symbol == "LUSDT":
                continue
            if symbol == "EUSDT" and day == pd.Timestamp("2024-01-05"):
                continue
            rows.append(_kline_row(
                day,
                symbol,
                close=100.0 + index + (day - CALENDAR[0]).days,
                placeholder=symbol == "BUSDT" and day == pd.Timestamp("2024-01-04"),
            ))
    return pd.DataFrame(rows)


def _funding_events() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "funding_time": "2024-01-03 00:00:00",
            "symbol": "AUSDT",
            "funding_rate": 0.001,
            "mark_price": 101.0,
            "rate_type": "Regular",
        },
        {
            "funding_time": "2024-01-03 08:00:00",
            "symbol": "AUSDT",
            "funding_rate": 0.002,
            "mark_price": 102.0,
            "rate_type": "Regular",
        },
        {
            "funding_time": "2024-01-03 00:00:00",
            "symbol": "BUSDT",
            "funding_rate": -0.001,
            "mark_price": 101.0,
            "rate_type": "Regular",
        },
        {
            "funding_time": "2024-01-03 08:00:00",
            "symbol": "BUSDT",
            "funding_rate": -0.002,
            "mark_price": 102.0,
            "rate_type": "Regular",
        },
        {
            "funding_time": "2024-01-07 00:00:00",
            "symbol": "GUSDT",
            "funding_rate": 0.003,
            "mark_price": 107.0,
            "rate_type": "Regular",
        },
    ])


def _funding_schedule() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": "2024-01-03", "symbol": "AUSDT", "expected_times": [
            "2024-01-03 00:00:00", "2024-01-03 08:00:00", "2024-01-03 16:00:00",
        ]},
        {"date": "2024-01-03", "symbol": "BUSDT", "expected_times": [
            "2024-01-03 00:00:00", "2024-01-03 08:00:00",
        ]},
        {"date": "2024-01-03", "symbol": "CUSDT", "expected_times": []},
    ])


@pytest.fixture
def h5_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "crypto_quant_fixture.h5"
    store = CryptoQuantStore(path)
    universe = _universe()
    klines = _klines()
    funding = _funding_events()
    panel = build_research_panel(
        universe,
        klines,
        funding,
        pd.Timestamp("2024-01-12"),
        funding_schedule=_funding_schedule(),
    )
    panel = panel.merge(
        universe[["binance_symbol", "decision_date", "market_cap_rank"]],
        on="binance_symbol",
        how="left",
    )
    store.replace("universe_monthly", universe)
    store.replace("klines_daily", klines)
    store.replace("funding_events", funding)
    store.replace("research_panel_daily", panel)
    return path
