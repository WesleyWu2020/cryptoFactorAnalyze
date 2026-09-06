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
EXAMPLE_CALENDAR = pd.date_range("2024-01-01", "2024-02-19", freq="D")
EXAMPLE_PARAMS = {
    "start": "2024-01-22",
    "end": "2024-01-23",
    "n_groups": 3,
    "include_funding": True,
    # Pin trading costs so fee reconciliation is independent of the profile
    # defaults.
    "fee_rate": 0.0003,
    "slippage": 0.0,
}


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


def _klines(calendar: pd.DatetimeIndex = CALENDAR) -> pd.DataFrame:
    rows = []
    for day in calendar:
        for index, symbol in enumerate(SYMBOLS, start=1):
            if symbol == "LUSDT":
                continue
            if symbol == "EUSDT" and day == pd.Timestamp("2024-01-05"):
                continue
            rows.append(_kline_row(
                day,
                symbol,
                close=100.0 + index + (day - calendar[0]).days,
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


def _example_funding_events() -> pd.DataFrame:
    return pd.concat(
        [
            _funding_events(),
            pd.DataFrame([
                {
                    "funding_time": "2024-01-23 08:00:00",
                    "symbol": "GUSDT",
                    "funding_rate": 0.001,
                    "mark_price": 123.0,
                    "rate_type": "Regular",
                },
            ]),
        ],
        ignore_index=True,
    )


def _complete_funding_schedule(
    calendar: pd.DatetimeIndex = EXAMPLE_CALENDAR,
) -> pd.DataFrame:
    expected_by_key = {
        (pd.Timestamp("2024-01-03"), "AUSDT"): [
            "2024-01-03 00:00:00", "2024-01-03 08:00:00",
        ],
        (pd.Timestamp("2024-01-03"), "BUSDT"): [
            "2024-01-03 00:00:00", "2024-01-03 08:00:00",
        ],
        (pd.Timestamp("2024-01-07"), "GUSDT"): ["2024-01-07 00:00:00"],
        (pd.Timestamp("2024-01-23"), "GUSDT"): ["2024-01-23 08:00:00"],
    }
    rows = []
    for day in calendar:
        for symbol in SYMBOLS:
            rows.append({
                "date": day,
                "symbol": symbol,
                "expected_times": expected_by_key.get((day, symbol), []),
            })
    return pd.DataFrame(rows)


def write_h5_fixture(
    path: Path,
    *,
    calendar: pd.DatetimeIndex = CALENDAR,
    funding_schedule: pd.DataFrame | None,
    funding_events: pd.DataFrame | None = None,
) -> Path:
    """Write the shared 12-name fixture with caller-supplied coverage evidence."""
    store = CryptoQuantStore(path)
    universe = _universe()
    klines = _klines(calendar)
    funding = _funding_events() if funding_events is None else funding_events
    panel = build_research_panel(
        universe,
        klines,
        funding,
        calendar[-1],
        funding_schedule=funding_schedule,
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


@pytest.fixture
def h5_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "crypto_quant_fixture.h5"
    return write_h5_fixture(
        path,
        funding_schedule=_funding_schedule(),
    )
