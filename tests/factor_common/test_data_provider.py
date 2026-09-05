from __future__ import annotations

import pandas as pd
import pytest


def test_provider_exposes_calendar_safe_market_and_universe_axes(h5_fixture):
    from factor_common.data_provider import DataProvider

    dp = DataProvider(h5_fixture)
    close = dp.get_single_data("close", start="2024-01-03", end="2024-01-08")
    mask = dp.get_universe(start="2024-01-03", end="2024-01-08")

    assert close.index.equals(mask.index)
    assert close.index.equals(pd.date_range("2024-01-03", "2024-01-08", freq="D"))
    assert mask.loc["2024-01-03", "AUSDT"]
    assert not mask.loc["2024-01-03", "GUSDT"]
    assert mask.loc["2024-01-07", "GUSDT"]
    assert len(close.index) == 6


def test_provider_retains_missing_calendar_rows_and_raw_prices_after_membership_end(h5_fixture):
    from factor_common.data_provider import DataProvider

    dp = DataProvider(h5_fixture)
    close = dp.get_single_data("close", start="2024-01-03", end="2024-01-08")

    assert pd.isna(close.loc["2024-01-05", "EUSDT"])
    assert close.loc["2024-01-08", "AUSDT"] == pytest.approx(108.0)


def test_provider_uses_current_quote_volume_field_name(h5_fixture):
    from factor_common.data_provider import DataProvider

    dp = DataProvider(h5_fixture)
    quote_volume = dp.get_single_data("quote_volume", start="2024-01-03", end="2024-01-03")

    assert quote_volume.loc["2024-01-03", "AUSDT"] == pytest.approx(103_000.0)
    with pytest.raises(ValueError, match="unknown market field"):
        dp.get_single_data("quoteVolume", start="2024-01-03", end="2024-01-03")


def test_provider_returns_raw_funding_events_with_positive_and_negative_rates(h5_fixture):
    from factor_common.data_provider import DataProvider

    dp = DataProvider(h5_fixture)
    events = dp.get_funding(start="2024-01-03", end="2024-01-03", symbols=["AUSDT", "BUSDT"])

    assert list(events.columns) == [
        "funding_time", "instrument", "funding_rate", "mark_price", "rate_type", "mark_price_valid",
    ]
    ordered = events.sort_values(["instrument", "funding_time"]).reset_index(drop=True)
    assert ordered.instrument.tolist() == ["AUSDT", "AUSDT", "BUSDT", "BUSDT"]
    assert ordered.funding_rate.tolist() == [0.001, 0.002, -0.001, -0.002]
    assert events.mark_price_valid.all()


def test_provider_returns_panel_quality_without_inference(h5_fixture):
    from factor_common.data_provider import DataProvider

    dp = DataProvider(h5_fixture)
    quality = dp.get_quality(start="2024-01-03", end="2024-01-04", symbols=["AUSDT", "BUSDT"])

    assert quality.index.names == ["date", "instrument"]
    assert quality.loc[(pd.Timestamp("2024-01-04"), "BUSDT"), "has_placeholder_kline"]
    assert quality.loc[(pd.Timestamp("2024-01-03"), "AUSDT"), "funding_coverage_status"] == "missing"
    assert quality.loc[(pd.Timestamp("2024-01-03"), "BUSDT"), "funding_coverage_status"] == "complete"

    close = dp.get_single_data("close", start="2024-01-03", end="2024-01-03")
    assert pd.notna(close.loc["2024-01-03", "AUSDT"])


def test_provider_keeps_quality_unknown_for_legacy_panel_flags(h5_fixture):
    from data.crypto_quant.store import CryptoQuantStore
    from factor_common.data_provider import DataProvider

    store = CryptoQuantStore(h5_fixture)
    panel = store.read("research_panel_daily").drop(
        columns=["has_placeholder_kline", "funding_coverage_status", "funding_invalid_price_count"]
    )
    with pd.HDFStore(h5_fixture, mode="a") as hdf:
        hdf.put("research_panel_daily", panel, format="table", data_columns=["date", "binance_symbol"], index=False)

    quality = DataProvider(h5_fixture).get_quality(
        start="2024-01-04", end="2024-01-04", symbols=["BUSDT"]
    )
    row = quality.iloc[0]
    assert pd.isna(row["has_placeholder_kline"])
    assert row["funding_coverage_status"] == "unknown"
    assert pd.isna(row["funding_invalid_price_count"])


def test_provider_metadata_lists_fields_symbols_and_time_range(h5_fixture):
    from factor_common.data_provider import DataProvider

    dp = DataProvider(h5_fixture)

    assert "close" in dp.list_datas()
    assert "quote_volume" in dp.list_datas()
    assert dp.symbols == tuple(f"{letter}USDT" for letter in "ABCDEFGHIJKL")
    assert dp.get_time_range() == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-12"))


def test_provider_cutoff_truncates_market_and_membership_knowledge_together(h5_fixture):
    from factor_common.data_provider import DataProvider

    dp = DataProvider(h5_fixture, as_of="2024-01-05")
    close = dp.get_single_data("close", start="2024-01-03", end="2024-01-08")
    mask = dp.get_universe(start="2024-01-03", end="2024-01-08")

    assert close.index.equals(pd.date_range("2024-01-03", "2024-01-05", freq="D"))
    assert mask.index.equals(close.index)
    assert not mask.loc["2024-01-03", "GUSDT"]
