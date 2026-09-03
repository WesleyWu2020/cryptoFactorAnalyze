from datetime import date

import pandas as pd
import pytest

from data.crypto_quant.panel import (
    aggregate_funding_daily,
    build_research_panel,
    last_complete_panel_date,
)


def test_aggregate_funding_preserves_zero_event_semantics():
    events = pd.DataFrame({
        "funding_time": pd.to_datetime(["2024-01-02 00:00", "2024-01-02 08:00"]),
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "funding_rate": [0.0, 0.0001],
        "mark_price": [42000.0, 42100.0],
        "rate_type": ["Regular", "Regular"],
    })
    out = aggregate_funding_daily(events)
    row = out.iloc[0]
    assert row["funding_rate_sum"] == pytest.approx(0.0001)
    assert row["funding_event_count"] == 2
    assert row["funding_rate_last"] == pytest.approx(0.0001)


def test_aggregate_funding_last_uses_latest_timestamp_not_input_order():
    events = pd.DataFrame({
        "funding_time": pd.to_datetime(["2024-01-02 08:00", "2024-01-02 00:00"]),
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "funding_rate": [0.0001, 0.0],
    })
    out = aggregate_funding_daily(events)
    assert out.iloc[0]["funding_rate_last"] == pytest.approx(0.0001)


def _panel_inputs():
    symbols = [f"C{index:02d}USDT" for index in range(50)]
    universe = pd.DataFrame({
        "effective_date": pd.Timestamp("2024-01-02"),
        "effective_end_date": pd.NaT,
        "binance_symbol": symbols,
    })
    klines = pd.DataFrame([
        {"date": day, "symbol": symbol, "open": 100.0, "close": 101.0}
        for day in pd.to_datetime(["2024-01-02", "2024-01-03"])
        for symbol in symbols
        if not (day == pd.Timestamp("2024-01-03") and symbol == symbols[0])
    ])
    funding = pd.DataFrame({
        "funding_time": pd.to_datetime(["2024-01-02 08:00"]),
        "symbol": [symbols[1]],
        "funding_rate": [0.0001],
        "mark_price": [42000.0],
    })
    return universe, klines, funding, symbols


def test_panel_expands_membership_and_keeps_missing_kline_as_nan():
    universe, klines, funding, symbols = _panel_inputs()
    panel = build_research_panel(universe, klines, funding, date(2024, 1, 3), date(2024, 1, 2))

    second_day = panel[panel["date"] == pd.Timestamp("2024-01-03")]
    assert len(second_day) == 50
    missing = second_day[second_day["symbol"] == symbols[0]].iloc[0]
    assert pd.isna(missing["open"])
    assert pd.isna(missing["close"])
    assert missing["has_complete_kline"] == False


def test_panel_distinguishes_complete_empty_funding_from_incomplete_coverage():
    universe, klines, funding, symbols = _panel_inputs()
    panel = build_research_panel(universe, klines, funding, date(2024, 1, 3), date(2024, 1, 2))

    empty_day = panel[(panel["date"] == pd.Timestamp("2024-01-02")) & (panel["symbol"] == symbols[0])].iloc[0]
    assert empty_day["funding_event_count"] == 0
    assert pd.isna(empty_day["funding_rate_sum"])
    assert pd.isna(empty_day["funding_rate_last"])
    assert empty_day["has_complete_funding"] == True

    incomplete_day = panel[panel["date"] == pd.Timestamp("2024-01-03")]
    assert incomplete_day["has_complete_funding"].eq(False).all()


def test_panel_starts_on_first_effective_date_and_complete_date_requires_all_klines():
    universe, klines, funding, _ = _panel_inputs()
    universe["decision_date"] = pd.Timestamp("2024-01-01")
    panel = build_research_panel(universe, klines, funding, date(2024, 1, 3), date(2024, 1, 2))

    assert panel["date"].min() == pd.Timestamp("2024-01-02")
    assert last_complete_panel_date(panel) == pd.Timestamp("2024-01-02")
