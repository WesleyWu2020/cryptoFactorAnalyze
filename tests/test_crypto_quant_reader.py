from datetime import date
from typing import Mapping, get_type_hints

import pandas as pd
import pytest

from data.crypto_quant.reader import (
    filter_factor_output,
    load_daily_universe,
    load_market_history,
    load_membership_history,
)
from data.crypto_quant.store import CryptoQuantStore


MARKET_COLUMNS = [
    "date", "instrument", "close_time", "open", "high", "low", "close",
    "volume", "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume",
]


def _market_row(day, symbol, close=100.0):
    return {
        "date": day,
        "symbol": symbol,
        "close_time": pd.Timestamp(day) + pd.Timedelta(hours=23, minutes=59),
        "open": close - 1,
        "high": close + 1,
        "low": close - 2,
        "close": close,
        "volume": 1000.0,
        "quote_volume": 100000.0,
        "trade_count": 10,
        "taker_buy_base_volume": 500.0,
        "taker_buy_quote_volume": 50000.0,
    }


def _store_fixture(tmp_path):
    path = tmp_path / "crypto_quant.h5"
    store = CryptoQuantStore(path)
    universe = pd.DataFrame([
        {"decision_date": "2024-02-29", "effective_date": "2024-03-01", "effective_end_date": "2024-03-10", "cmc_id": 1, "cmc_symbol": "A", "binance_symbol": "AUSDT", "market_cap_rank": 1, "cmc_weight": 0.5},
        {"decision_date": "2024-02-29", "effective_date": "2024-03-01", "effective_end_date": "2024-03-10", "cmc_id": 2, "cmc_symbol": "B", "binance_symbol": "BUSDT", "market_cap_rank": 2, "cmc_weight": 0.5},
        {"decision_date": "2024-03-10", "effective_date": "2024-03-11", "effective_end_date": pd.NaT, "cmc_id": 3, "cmc_symbol": "C", "binance_symbol": "CUSDT", "market_cap_rank": 1, "cmc_weight": 1.0},
    ])
    store.replace("universe_monthly", universe)

    klines = pd.DataFrame([
        _market_row(day, symbol, close=100 + index)
        for day in pd.date_range("2024-03-05", "2024-03-12")
        for index, symbol in enumerate(["AUSDT", "BUSDT", "CUSDT", "XUSDT"])
    ])
    store.replace("klines_daily", klines)

    panel_rows = []
    for day in pd.date_range("2024-03-09", "2024-03-11"):
        for symbol in ["AUSDT", "BUSDT"] if day <= pd.Timestamp("2024-03-10") else ["CUSDT"]:
            panel_rows.append({
                "date": day, "binance_symbol": symbol, "close_time": day,
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
                "quote_volume": 1.0, "trade_count": 1, "taker_buy_base_volume": 1.0,
                "taker_buy_quote_volume": 1.0, "decision_date": day - pd.Timedelta(days=1),
                "universe_effective_date": day, "market_cap_rank": 1, "cmc_weight_at_decision": 1.0,
                "funding_rate_sum": 0.0, "funding_rate_mean": 0.0, "funding_rate_last": 0.0,
                "funding_event_count": 0, "has_complete_kline": not (day == pd.Timestamp("2024-03-10") and symbol == "BUSDT"),
                "has_complete_funding": True,
            })
    store.replace("research_panel_daily", pd.DataFrame(panel_rows))
    return path


def test_load_market_history_uses_overlapping_memberships_and_keeps_warmup(tmp_path):
    path = _store_fixture(tmp_path)

    out = load_market_history(path, date(2024, 3, 10), date(2024, 3, 12), lookback_days=5)

    assert list(out.columns) == MARKET_COLUMNS
    assert out["instrument"].unique().tolist() == ["AUSDT", "BUSDT", "CUSDT"]
    assert out["date"].min() == pd.Timestamp("2024-03-05")
    assert out["date"].max() == pd.Timestamp("2024-03-12")
    assert "XUSDT" not in out["instrument"].tolist()
    assert out[["date", "instrument"]].duplicated().sum() == 0


def test_load_market_history_reads_explicit_symbol_after_membership_end(tmp_path):
    path = _store_fixture(tmp_path)

    out = load_market_history(
        path,
        date(2024, 3, 11),
        date(2024, 3, 12),
        lookback_days=0,
        symbols={"AUSDT"},
    )

    assert out["instrument"].unique().tolist() == ["AUSDT"]
    assert out["date"].tolist() == [pd.Timestamp("2024-03-11"), pd.Timestamp("2024-03-12")]


def test_load_market_history_as_of_truncates_market_and_membership_knowledge(tmp_path):
    path = _store_fixture(tmp_path)

    out = load_market_history(
        path,
        date(2024, 3, 9),
        date(2024, 3, 12),
        lookback_days=0,
        as_of=date(2024, 3, 10),
    )

    assert out["date"].max() == pd.Timestamp("2024-03-10")
    assert set(out["instrument"]) == {"AUSDT", "BUSDT"}


def test_load_market_history_normalizes_timezone_aware_bounds_and_as_of(tmp_path):
    path = _store_fixture(tmp_path)

    out = load_market_history(
        path,
        pd.Timestamp("2024-03-09 12:00", tz="Asia/Shanghai"),
        pd.Timestamp("2024-03-12 12:00", tz="Asia/Shanghai"),
        lookback_days=0,
        as_of=pd.Timestamp("2024-03-10 12:00", tz="Asia/Shanghai"),
    )

    assert out["date"].min() == pd.Timestamp("2024-03-09")
    assert out["date"].max() == pd.Timestamp("2024-03-10")


def test_membership_cutoff_excludes_effective_interval_decided_after_cutoff(tmp_path):
    path = _store_fixture(tmp_path)
    store = CryptoQuantStore(path)
    universe = store.read("universe_monthly")
    future_decision = universe.loc[universe["binance_symbol"] == "CUSDT"].iloc[[0]].copy()
    future_decision["decision_date"] = pd.Timestamp("2024-03-11")
    future_decision["effective_date"] = pd.Timestamp("2024-03-09")
    future_decision["effective_end_date"] = pd.Timestamp("2024-03-10")
    store.replace("universe_monthly", pd.concat([universe, future_decision], ignore_index=True))

    memberships = load_membership_history(
        path,
        date(2024, 3, 9),
        date(2024, 3, 10),
        as_of=date(2024, 3, 10),
    )

    assert "CUSDT" not in memberships[pd.Timestamp("2024-03-09")]


def test_load_market_history_pushes_date_range_into_hdf_read(tmp_path, monkeypatch):
    path = _store_fixture(tmp_path)
    calls = []
    original_read = CryptoQuantStore.read

    def read_with_spy(self, name, where=None):
        calls.append((name, where))
        return original_read(self, name, where=where)

    monkeypatch.setattr(CryptoQuantStore, "read", read_with_spy)
    load_market_history(path, date(2024, 3, 10), date(2024, 3, 12), lookback_days=5)

    kline_where = next(where for name, where in calls if name == "klines_daily")
    assert kline_where is not None
    assert "2024-03-05" in kline_where
    assert "2024-03-12" in kline_where


def test_load_market_history_empty_result_keeps_full_market_schema(tmp_path):
    out = load_market_history(_store_fixture(tmp_path), date(2024, 2, 1), date(2024, 2, 2))

    assert out.empty
    assert list(out.columns) == MARKET_COLUMNS


@pytest.mark.parametrize("missing", ["date", "quote_volume"])
def test_load_market_history_reports_malformed_hdf_before_where(tmp_path, missing):
    path = _store_fixture(tmp_path)
    malformed = pd.DataFrame([_market_row("2024-03-10", "AUSDT")]).drop(columns=[missing])
    with pd.HDFStore(path, mode="a") as hdf:
        hdf.put("klines_daily", malformed, format="table", data_columns=["symbol"], index=False)

    with pytest.raises(ValueError, match=f"klines_daily missing columns.*{missing}"):
        load_market_history(path, date(2024, 3, 10), date(2024, 3, 12))


def test_load_daily_universe_excludes_incomplete_panel_date(tmp_path):
    path = _store_fixture(tmp_path)

    out = load_daily_universe(path, date(2024, 3, 9), date(2024, 3, 11))

    assert list(out) == [pd.Timestamp("2024-03-09"), pd.Timestamp("2024-03-11")]
    assert pd.Timestamp("2024-03-10") not in out
    assert out[pd.Timestamp("2024-03-09")] == {"AUSDT", "BUSDT"}
    assert out[pd.Timestamp("2024-03-11")] == {"CUSDT"}


def test_load_daily_universe_requires_complete_kline_and_funding(tmp_path):
    path = _store_fixture(tmp_path)
    store = CryptoQuantStore(path)
    panel = store.read("research_panel_daily")
    panel.loc[panel["date"] == pd.Timestamp("2024-03-09"), "has_complete_funding"] = False
    store.replace("research_panel_daily", panel)

    out = load_daily_universe(path, date(2024, 3, 9), date(2024, 3, 11))

    assert pd.Timestamp("2024-03-09") not in out


def test_load_daily_universe_reports_missing_panel_columns(tmp_path, monkeypatch):
    path = _store_fixture(tmp_path)
    original_read = CryptoQuantStore.read

    def read_missing_panel(self, name, where=None):
        frame = original_read(self, name, where=where)
        return frame.drop(columns=["has_complete_funding"]) if name == "research_panel_daily" else frame

    monkeypatch.setattr(CryptoQuantStore, "read", read_missing_panel)
    with pytest.raises(ValueError, match="research_panel_daily missing columns.*has_complete_funding"):
        load_daily_universe(path, date(2024, 3, 9), date(2024, 3, 11))


def test_load_daily_universe_rejects_empty_malformed_panel(tmp_path, monkeypatch):
    path = _store_fixture(tmp_path)
    original_read = CryptoQuantStore.read

    def read_empty_panel(self, name, where=None):
        return pd.DataFrame() if name == "research_panel_daily" else original_read(self, name, where=where)

    monkeypatch.setattr(CryptoQuantStore, "read", read_empty_panel)
    with pytest.raises(ValueError, match="research_panel_daily missing columns"):
        load_daily_universe(path, date(2024, 3, 9), date(2024, 3, 11))


def test_load_daily_universe_reports_missing_date_before_hdf_where(tmp_path):
    path = _store_fixture(tmp_path)
    with pd.HDFStore(path, mode="a") as hdf:
        hdf.put(
            "research_panel_daily",
            pd.DataFrame({
                "binance_symbol": ["AUSDT"],
                "has_complete_kline": [True],
                "has_complete_funding": [True],
            }),
            format="table",
            data_columns=["binance_symbol"],
            index=False,
        )

    with pytest.raises(ValueError, match="research_panel_daily missing columns.*date"):
        load_daily_universe(path, date(2024, 3, 9), date(2024, 3, 11))


def test_filter_factor_output_requires_same_day_membership_and_exact_schema():
    factors = pd.DataFrame({
        "date": ["2024-03-09 12:00", "2024-03-09", "2024-03-10", "2024-03-11"],
        "instrument": ["AUSDT", "XUSDT", "BUSDT", "CUSDT"],
        "factor": [1.0, 2.0, 3.0, 4.0],
        "extra": [9, 9, 9, 9],
    })
    universe = {
        pd.Timestamp("2024-03-09"): {"AUSDT"},
        pd.Timestamp("2024-03-10"): {"AUSDT"},
    }

    out = filter_factor_output(factors, universe)

    assert list(out.columns) == ["date", "instrument", "factor"]
    assert out.to_dict("records") == [{"date": pd.Timestamp("2024-03-09"), "instrument": "AUSDT", "factor": 1.0}]


def test_filter_factor_output_uses_mapping_type_contract():
    annotation = get_type_hints(filter_factor_output)["universe_by_date"]

    assert annotation == Mapping[pd.Timestamp, set[str]]


@pytest.mark.parametrize("bad", [pd.DataFrame(), pd.DataFrame({"date": [], "instrument": []})])
def test_filter_factor_output_rejects_missing_factor_columns(bad):
    with pytest.raises(ValueError, match="date.*instrument.*factor"):
        filter_factor_output(bad, {})


def test_reader_validates_date_range_and_nonnegative_lookback(tmp_path):
    path = _store_fixture(tmp_path)
    with pytest.raises(ValueError, match="start must be on or before end"):
        load_market_history(path, date(2024, 3, 12), date(2024, 3, 10))
    with pytest.raises(ValueError, match="lookback_days"):
        load_market_history(path, date(2024, 3, 10), date(2024, 3, 12), lookback_days=-1)
