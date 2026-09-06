from datetime import date

import pandas as pd
import pytest

from data.crypto_quant.panel import (
    aggregate_funding_daily,
    build_funding_schedule,
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
            "quote_volume": 101000.0, "trade_count": 10,
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


def test_panel_keeps_empty_funding_unknown_despite_fetch_watermark():
    universe, klines, funding, symbols = _panel_inputs()
    panel = build_research_panel(universe, klines, funding, date(2024, 1, 3), date(2024, 1, 2))

    empty_day = panel[(panel["date"] == pd.Timestamp("2024-01-02")) & (panel["binance_symbol"] == symbols[0])].iloc[0]
    assert empty_day["funding_event_count"] == 0
    assert pd.isna(empty_day["funding_rate_sum"])
    assert pd.isna(empty_day["funding_rate_last"])
    assert empty_day["has_complete_funding"] == False
    assert empty_day["funding_coverage_status"] == "no_events"

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
        "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume",
        "funding_rate_sum", "funding_rate_mean", "funding_event_count", "funding_rate_last",
        "has_complete_kline", "has_complete_funding",
        "has_placeholder_kline", "funding_coverage_status", "funding_invalid_price_count",
    ]
    assert "symbol" not in panel.columns
    assert "weight" not in panel.columns
    assert panel["universe_effective_date"].eq(pd.Timestamp("2024-01-02")).all()
    assert panel["cmc_weight_at_decision"].eq(0.25).all()


def test_global_watermark_does_not_prove_symbol_day_complete():
    universe, klines, funding, symbols = _panel_inputs()
    panel = build_research_panel(universe, klines, funding, date(2024, 1, 2), date(2024, 1, 2))
    assert not panel.has_complete_funding.any()
    assert panel.loc[panel.binance_symbol == symbols[1], "funding_coverage_status"].iloc[0] == "unknown"
    assert panel.loc[panel.binance_symbol == symbols[0], "funding_coverage_status"].iloc[0] == "no_events"


def test_price_quality_preserves_raw_values_and_rates():
    from data.crypto_quant.panel import annotate_funding_prices
    events = pd.DataFrame({
        "funding_time": pd.date_range("2024-01-02", periods=4, freq="4h"),
        "symbol": ["BTCUSDT"] * 4,
        "funding_rate": [0.0, -0.001, 0.002, 0.003],
        "mark_price": [float("nan"), 0, float("inf"), 100],
    })
    original = events.copy(deep=True)
    flagged = annotate_funding_prices(events)
    assert flagged.mark_price_valid.tolist() == [False, False, False, True]
    pd.testing.assert_frame_equal(events, original)
    daily = aggregate_funding_daily(events)
    assert daily.funding_invalid_price_count.iloc[0] == 3
    assert daily.funding_event_count.iloc[0] == 4
    assert daily.funding_rate_sum.iloc[0] == pytest.approx(0.004)


def test_explicit_daily_schedule_checks_times_and_isolates_symbols():
    universe, klines, funding, symbols = _panel_inputs()
    schedules = pd.DataFrame([
        {"date": "2024-01-02", "symbol": symbols[1], "expected_times": ["2024-01-02 08:00"]},
        {"date": "2024-01-02", "symbol": symbols[2], "expected_times": ["2024-01-02 00:00", "2024-01-02 04:00"]},
        {"date": "2024-01-02", "symbol": symbols[3], "expected_times": []},
    ])
    panel = build_research_panel(universe, klines, funding, date(2024, 1, 2), None, funding_schedule=schedules).set_index("binance_symbol")
    assert panel.loc[symbols[1], "has_complete_funding"]
    assert panel.loc[symbols[2], "funding_coverage_status"] == "missing"
    assert panel.loc[symbols[3], "funding_coverage_status"] == "not_applicable"
    assert not panel.loc[symbols[3], "has_complete_funding"]


def test_funding_schedule_accepts_intraday_cadence_transition():
    universe, klines, _, symbols = _panel_inputs()
    times = [
        "2024-01-02 00:00:00.001",
        "2024-01-02 01:00:00",
        "2024-01-02 02:00:00",
        "2024-01-02 03:00:00.001",
        "2024-01-02 04:00:00.003",
        "2024-01-02 08:00:00",
        "2024-01-02 12:00:00.002",
        "2024-01-02 16:00:00",
        "2024-01-02 20:00:00.010",
    ]
    funding = pd.DataFrame({
        "funding_time": times,
        "symbol": [symbols[0]] * len(times),
        "funding_rate": [0.0001] * len(times),
        "mark_price": [100.0] * len(times),
    })
    schedule = build_funding_schedule(universe, funding, date(2024, 1, 2))
    row = schedule[
        (schedule["date"] == pd.Timestamp("2024-01-02"))
        & schedule["binance_symbol"].eq(symbols[0])
    ].iloc[0]
    assert len(row["expected_times"]) == 9
    panel = build_research_panel(
        universe, klines, funding, date(2024, 1, 2), funding_schedule=schedule
    )
    result = panel.loc[panel["binance_symbol"].eq(symbols[0]), "funding_coverage_status"].iloc[0]
    assert result == "complete"


def test_placeholder_bar_is_retained_but_flagged_in_panel():
    universe, klines, funding, symbols = _panel_inputs()
    mask = klines.symbol == symbols[0]
    klines.loc[mask, ["open", "high", "low", "close"]] = 0.248
    klines.loc[mask, "volume"] = 0
    panel = build_research_panel(universe, klines, funding, date(2024, 1, 2), None)
    row = panel[panel.binance_symbol == symbols[0]].iloc[0]
    assert row.has_placeholder_kline
    assert not row.has_complete_kline
    assert row.close == 0.248


@pytest.mark.parametrize("times, rates, status", [
    (["2024-01-02 00:00", "2024-01-02 08:00", "2024-01-02 12:00"], [0.1]*3, "schedule_mismatch"),
    (["2024-01-02 00:00", "2024-01-02 08:00"], [0.1]*2, "missing"),
    (["2024-01-02 00:00", "2024-01-02 08:00", "2024-01-02 16:00"], [0.1, float("nan"), 0.1], "invalid_rate"),
    (["2024-01-02 00:00:00.003", "2024-01-02 08:00", "2024-01-02 16:00"], [0.1]*3, "complete"),
])
def test_coverage_requires_correct_timestamps_and_valid_rates(times, rates, status):
    universe, klines, _, symbols = _panel_inputs()
    events = pd.DataFrame({"symbol": symbols[0], "funding_time": times, "funding_rate": rates})
    schedule = pd.DataFrame([{"date": "2024-01-02", "symbol": symbols[0], "expected_times": ["2024-01-02 00:00", "2024-01-02 08:00", "2024-01-02 16:00"]}])
    panel = build_research_panel(universe, klines, events, date(2024, 1, 2), funding_schedule=schedule)
    assert panel.iloc[0].funding_coverage_status == status


def test_unknown_schedule_and_invalid_rate_never_produce_partial_sum():
    universe, klines, funding, symbols = _panel_inputs()
    funding.loc[0, "funding_rate"] = float("nan")
    panel = build_research_panel(universe, klines, funding, date(2024, 1, 2))
    row = panel[panel.binance_symbol == symbols[1]].iloc[0]
    assert row.funding_coverage_status == "invalid_rate"
    assert pd.isna(row.funding_rate_sum)
    assert row.funding_event_count == 1


def test_schedule_duplicate_or_out_of_day_times_are_rejected():
    universe, klines, funding, symbols = _panel_inputs()
    for times in [["2024-01-03 00:00"], ["2024-01-02 00:00"]*2]:
        schedule = pd.DataFrame([{"date": "2024-01-02", "symbol": symbols[0], "expected_times": times}])
        with pytest.raises(ValueError, match="expected funding timestamp"):
            build_research_panel(universe, klines, funding, date(2024, 1, 2), funding_schedule=schedule)


def test_entirely_empty_funding_table_preserves_missing_rates():
    universe, klines, funding, _ = _panel_inputs()
    panel = build_research_panel(universe, klines, funding.iloc[:0], date(2024, 1, 2))
    assert panel.funding_coverage_status.eq("no_events").all()
    assert panel.funding_rate_sum.isna().all()
    assert panel.funding_event_count.eq(0).all()
    assert not panel.has_complete_funding.any()
