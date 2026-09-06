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


def test_provider_includes_member_without_any_kline_rows(h5_fixture):
    from factor_common.data_provider import DataProvider

    dp = DataProvider(h5_fixture)
    close = dp.get_single_data("close", start="2024-01-07", end="2024-01-08")
    mask = dp.get_universe(start="2024-01-07", end="2024-01-08")

    assert "LUSDT" in dp.symbols
    assert mask.loc["2024-01-07", "LUSDT"]
    assert close["LUSDT"].isna().all()


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


def test_exit_day_fallback_accepts_cadence_superset_rejects_missing(tmp_path):
    # AUSDT/BUSDT membership ends 2024-01-06; 2024-01-07 is an exit day with
    # no panel row. The fallback certifies it from the prior complete day when
    # every prior event time reappears (extra events are observed data), and
    # refuses when a prior-cadence event is unobserved.
    import pandas as pd
    from factor_common.data_provider import DataProvider
    from tests.factor_common.conftest import write_h5_fixture

    def event(day, hour, symbol):
        return {
            "funding_time": f"2024-01-0{day} {hour:02d}:00:00",
            "symbol": symbol,
            "funding_rate": 0.001,
            "mark_price": 100.0,
            "rate_type": "Regular",
        }

    events = pd.DataFrame([
        event(6, 0, "AUSDT"), event(6, 8, "AUSDT"),
        event(6, 0, "BUSDT"), event(6, 8, "BUSDT"),
        # Superset: the 04:00 event is extra; prior cadence is preserved.
        event(7, 0, "AUSDT"), event(7, 4, "AUSDT"), event(7, 8, "AUSDT"),
        # Missing: the 08:00 event of the prior cadence is unobserved.
        event(7, 0, "BUSDT"),
    ])
    schedule = pd.DataFrame([
        {"date": "2024-01-06", "symbol": "AUSDT",
         "expected_times": ["2024-01-06 00:00:00", "2024-01-06 08:00:00"]},
        {"date": "2024-01-06", "symbol": "BUSDT",
         "expected_times": ["2024-01-06 00:00:00", "2024-01-06 08:00:00"]},
    ])
    path = write_h5_fixture(
        tmp_path / "fixture.h5", funding_schedule=schedule, funding_events=events
    )

    quality = DataProvider(path).get_quality(
        start="2024-01-06", end="2024-01-08", symbols=["AUSDT", "BUSDT"]
    )
    statuses = quality["funding_coverage_status"]
    assert statuses.loc[(pd.Timestamp("2024-01-06"), "AUSDT")] == "complete"
    assert statuses.loc[(pd.Timestamp("2024-01-07"), "AUSDT")] == "complete"
    assert statuses.loc[(pd.Timestamp("2024-01-07"), "BUSDT")] == "unknown"


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


def test_provider_rejects_duplicate_quality_keys(h5_fixture):
    from data.crypto_quant.store import CryptoQuantStore
    from factor_common.data_provider import DataProvider

    store = CryptoQuantStore(h5_fixture)
    panel = store.read("research_panel_daily")
    duplicate = panel.loc[
        (panel["date"] == pd.Timestamp("2024-01-03"))
        & (panel["binance_symbol"] == "AUSDT")
    ].iloc[[0]]
    with pd.HDFStore(h5_fixture, mode="a") as hdf:
        hdf.put(
            "research_panel_daily",
            pd.concat([panel, duplicate], ignore_index=True),
            format="table",
            data_columns=["date", "binance_symbol"],
            index=False,
        )

    with pytest.raises(ValueError, match="duplicate quality key"):
        DataProvider(h5_fixture).get_quality(
            start="2024-01-03", end="2024-01-03", symbols=["AUSDT"]
        )


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
