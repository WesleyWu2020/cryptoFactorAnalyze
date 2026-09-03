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


def test_aggregate_funding_groups_by_utc_natural_day_and_calculates_mean():
    events = pd.DataFrame({
        "funding_time": ["2024-01-02 00:30:00+01:00", "2024-01-02 08:00:00Z"],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "funding_rate": [0.0002, 0.0004],
    })
    out = aggregate_funding_daily(events)
    assert out["date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert out.loc[out["date"] == pd.Timestamp("2024-01-01"), "funding_rate_mean"].iloc[0] == pytest.approx(0.0002)


def _panel_inputs():
    symbols = [f"C{index:02d}USDT" for index in range(50)]
    universe = pd.DataFrame({
        "effective_date": pd.Timestamp("2024-01-02"),
        "effective_end_date": pd.NaT,
        "binance_symbol": symbols,
    })
    klines = pd.DataFrame([
        {
            "date": day, "symbol": symbol, "open": 100.0, "high": 102.0,
            "low": 99.0, "close": 101.0, "volume": 1000.0,
            "close_time": day + pd.Timedelta(hours=23, minutes=59),
            "quote_asset_volume": 101000.0, "trade_count": 10,
            "taker_buy_base_volume": 500.0, "taker_buy_quote_volume": 50500.0,
        }
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
    missing = second_day[second_day["binance_symbol"] == symbols[0]].iloc[0]
    assert pd.isna(missing["open"])
    assert pd.isna(missing["close"])
    assert missing["has_complete_kline"] == False


def test_panel_marks_kline_with_missing_required_close_incomplete():
    universe, klines, funding, symbols = _panel_inputs()
    klines.loc[(klines["date"] == pd.Timestamp("2024-01-02")) & (klines["symbol"] == symbols[0]), "close"] = float("nan")
    panel = build_research_panel(universe, klines, funding, date(2024, 1, 2), date(2024, 1, 2))
    row = panel[panel["binance_symbol"] == symbols[0]].iloc[0]
    assert pd.isna(row["close"])
    assert row["has_complete_kline"] == False


def test_panel_distinguishes_complete_empty_funding_from_incomplete_coverage():
    universe, klines, funding, symbols = _panel_inputs()
    panel = build_research_panel(universe, klines, funding, date(2024, 1, 3), date(2024, 1, 2))

    empty_day = panel[(panel["date"] == pd.Timestamp("2024-01-02")) & (panel["binance_symbol"] == symbols[0])].iloc[0]
    assert empty_day["funding_event_count"] == 0
    assert pd.isna(empty_day["funding_rate_sum"])
    assert pd.isna(empty_day["funding_rate_last"])
    assert empty_day["has_complete_funding"] == True

    incomplete_day = panel[panel["date"] == pd.Timestamp("2024-01-03")]
    assert incomplete_day["has_complete_funding"].eq(False).all()


def test_aggregate_funding_daily_normalizes_string_rates():
    events = pd.DataFrame([{
        "funding_time": pd.Timestamp("2024-01-02 08:00", tz="UTC"),
        "symbol": "BTCUSDT", "funding_rate": "0.001", "mark_price": "100", "rate_type": "Regular",
    }])
    daily = aggregate_funding_daily(events)
    assert daily.loc[0, "funding_rate_mean"] == 0.001
    assert daily["funding_rate_mean"].dtype.kind == "f"


def test_panel_starts_on_first_effective_date_and_complete_date_requires_all_klines():
    universe, klines, funding, _ = _panel_inputs()
    universe["decision_date"] = pd.Timestamp("2024-01-01")
    panel = build_research_panel(universe, klines, funding, date(2024, 1, 3), date(2024, 1, 2))

    assert panel["date"].min() == pd.Timestamp("2024-01-02")
    assert last_complete_panel_date(panel) == pd.Timestamp("2024-01-02")


def test_panel_has_exact_research_panel_daily_columns_and_normalized_universe_fields():
    universe, klines, funding, _ = _panel_inputs()
    universe["weight"] = 0.25
    panel = build_research_panel(universe, klines, funding, date(2024, 1, 2), date(2024, 1, 2))
    assert list(panel.columns) == [
        "date", "binance_symbol", "universe_effective_date", "cmc_weight_at_decision",
        "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
        "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume",
        "funding_rate_sum", "funding_rate_mean", "funding_event_count", "funding_rate_last",
        "has_complete_kline", "has_complete_funding",
    ]
    assert "symbol" not in panel.columns
    assert "weight" not in panel.columns
    assert panel["universe_effective_date"].eq(pd.Timestamp("2024-01-02")).all()
    assert panel["cmc_weight_at_decision"].eq(0.25).all()
