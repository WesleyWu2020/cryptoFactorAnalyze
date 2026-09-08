"""Deterministic, offline HDF5 data for GP causality and workflow tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.crypto_quant.schemas import TABLE_SPECS
from data.crypto_quant.store import CryptoQuantStore


SYMBOLS = tuple(f"S{number:02d}USDT" for number in range(30))
CALENDAR = pd.date_range("2023-06-30", "2026-09-01", freq="D")


def _fixture_tables() -> dict[str, pd.DataFrame]:
    """Create point-in-time membership plus complete price/funding evidence.

    One future listing starts on 2025-01-01 and intentional non-member gaps
    exercise the provider's NaN/quality handling without weakening the 20-pair
    stage eligibility requirement.
    """
    klines, panel, funding = [], [], []
    start = CALENDAR[0]
    for day_index, day in enumerate(CALENDAR):
        for symbol_index, symbol in enumerate(SYMBOLS):
            listed = not (symbol == SYMBOLS[-1] and day < pd.Timestamp("2025-01-01"))
            if not listed:
                continue
            # Intentional raw gap for an inactive/new listing boundary only.
            gap = symbol == SYMBOLS[-1] and day == pd.Timestamp("2025-01-02")
            close = 100.0 + symbol_index + day_index * (0.03 + symbol_index * 0.0001)
            row = {
                "date": day, "symbol": symbol, "close_time": day + pd.Timedelta(hours=23, minutes=59),
                "open": close - 0.2, "high": close + 0.4, "low": close - 0.5,
                "close": close, "volume": 1000.0 + symbol_index, "quote_volume": close * 1000.0,
                "trade_count": 100 + symbol_index, "taker_buy_base_volume": 500.0,
                "taker_buy_quote_volume": close * 500.0,
            }
            if not gap:
                klines.append(row)
            panel.append({
                **row, "binance_symbol": symbol, "decision_date": pd.Timestamp("2023-06-30") if symbol != SYMBOLS[-1] else pd.Timestamp("2024-12-31"),
                "universe_effective_date": pd.Timestamp("2023-07-01") if symbol != SYMBOLS[-1] else pd.Timestamp("2025-01-01"),
                "market_cap_rank": symbol_index + 1, "cmc_weight_at_decision": 1 / len(SYMBOLS),
                "funding_rate_sum": 0.0001, "funding_rate_mean": 0.0001, "funding_rate_last": 0.0001,
                "funding_event_count": 1, "has_complete_kline": not gap, "has_complete_funding": True,
                "has_placeholder_kline": False, "funding_coverage_status": "complete", "funding_invalid_price_count": 0,
            })
            funding.append({"funding_time": day + pd.Timedelta(hours=8), "symbol": symbol, "funding_rate": 0.0001, "mark_price": close, "rate_type": "Regular"})
    universe = []
    for index, symbol in enumerate(SYMBOLS[:-1], 1):
        universe.append({"decision_date": "2023-06-30", "effective_date": "2023-07-01", "effective_end_date": pd.NaT, "cmc_id": index, "cmc_symbol": symbol[:-4], "binance_symbol": symbol, "market_cap_rank": index, "cmc_weight": 1 / len(SYMBOLS)})
    universe.append({"decision_date": "2024-12-31", "effective_date": "2025-01-01", "effective_end_date": pd.NaT, "cmc_id": 30, "cmc_symbol": SYMBOLS[-1][:-4], "binance_symbol": SYMBOLS[-1], "market_cap_rank": 30, "cmc_weight": 1 / len(SYMBOLS)})
    return {"klines_daily": pd.DataFrame(klines), "funding_events": pd.DataFrame(funding), "universe_monthly": pd.DataFrame(universe), "research_panel_daily": pd.DataFrame(panel)}


def write_gp_h5(path: Path) -> Path:
    tables = _fixture_tables()
    store = CryptoQuantStore(path)
    for name, frame in tables.items():
        assert set(TABLE_SPECS[name].columns).issubset(frame.columns)
        store.replace(name, frame)
    return path


@pytest.fixture
def gp_h5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No fixture construction or test path may rely on network access."""
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network disabled")))
    return write_gp_h5(tmp_path / "daily_gp.h5")
