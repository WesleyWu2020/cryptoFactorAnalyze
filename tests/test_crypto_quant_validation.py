from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.crypto_quant.validation import (
    StoreValidationError,
    ValidationIssue,
    validate_frames,
    validate_store,
)
from data.crypto_quant.panel import build_research_panel
from data.crypto_quant.universe import build_monthly_universe


def _valid_store_frames():
    symbols = [f"C{i:02d}USDT" for i in range(50)]
    members = pd.DataFrame({
        "date": [date(2024, 1, 1)] * 50,
        "cmc_id": range(1, 51),
        "symbol": [f"C{i:02d}" for i in range(50)],
        "name": [f"Coin {i}" for i in range(50)],
        "weight": [1 / 50] * 50,
    })
    mappings = pd.DataFrame({
        "cmc_id": range(1, 51),
        "cmc_symbol": members["symbol"],
        "binance_symbol": symbols,
        "valid_from": [pd.Timestamp("2020-01-01")] * 50,
        "valid_to": [pd.NaT] * 50,
    })
    universe = pd.DataFrame({
        "decision_date": [pd.Timestamp("2024-01-01")] * 50,
        "effective_date": [pd.Timestamp("2024-01-02")] * 50,
        "effective_end_date": [pd.NaT] * 50,
        "cmc_id": range(1, 51),
        "cmc_symbol": members["symbol"],
        "binance_symbol": symbols,
        "weight": [1 / 50] * 50,
        "market_cap_rank": range(1, 51),
    })
    klines = pd.DataFrame({
        "date": [pd.Timestamp("2023-12-31")] * 50,
        "symbol": symbols,
        "open": [100.0] * 50,
        "high": [102.0] * 50,
        "low": [99.0] * 50,
        "close": [101.0] * 50,
        "volume": [1000.0] * 50,
    })
    funding = pd.DataFrame({
        "funding_time": pd.date_range("2023-12-31", periods=50, freq="8h"),
        "symbol": ["BTCUSDT"] * 50,
        "funding_rate": [0.0001] * 50,
        "mark_price": [100.0] * 50,
        "rate_type": ["Regular"] * 50,
    })
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-02")] * 50,
        "binance_symbol": symbols,
        "open": [100.0] * 50,
        "high": [102.0] * 50,
        "low": [99.0] * 50,
        "close": [101.0] * 50,
        "volume": [1000.0] * 50,
        "has_complete_kline": [True] * 50,
        "has_complete_funding": [True] * 50,
    })
    return {
        "cmc100_daily": pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")], "index_value": [100.0],
        }),
        "cmc100_constituents": members,
        "futures_contracts": mappings,
        "klines_daily": klines,
        "funding_events": funding,
        "universe_monthly": universe,
        "research_panel_daily": panel,
    }


def _issue_codes(report):
    return {issue.code for issue in report.issues}


@pytest.mark.parametrize(
    ("name", "mutate", "code"),
    [
        ("duplicate source primary key", lambda f: f["klines_daily"].loc[1].to_dict(), "duplicate_key"),
        ("negative volume", lambda f: f["klines_daily"].loc.__setitem__((0, "volume"), -1), "negative_volume"),
        ("invalid OHLC", lambda f: f["klines_daily"].loc.__setitem__((0, "high"), 90.0), "invalid_ohlc"),
        ("universe size", lambda f: f["universe_monthly"].drop(f["universe_monthly"].index[-1], inplace=True), "universe_size"),
        ("excluded asset", lambda f: f["universe_monthly"].loc.__setitem__((0, "cmc_symbol"), "USDT"), "excluded_asset"),
        ("absent CMC snapshot", lambda f: f["universe_monthly"].loc.__setitem__((0, "cmc_id"), 999), "missing_cmc_snapshot"),
        ("missing T-1 kline", lambda f: f["klines_daily"].drop(f["klines_daily"].index[0], inplace=True), "missing_t_minus_one"),
        ("early effective date", lambda f: f["universe_monthly"].loc.__setitem__((0, "effective_date"), pd.Timestamp("2023-12-31")), "early_effective_date"),
        ("non-monotonic funding", lambda f: f["funding_events"].loc.__setitem__((slice(None), "funding_time"), f["funding_events"]["funding_time"].iloc[::-1].to_numpy()), "funding_not_monotonic"),
        ("panel membership count", lambda f: f["research_panel_daily"].drop(f["research_panel_daily"].index[-1], inplace=True), "panel_universe_size"),
    ],
)
def test_corruptions_report_specific_error(name, mutate, code):
    frames = _valid_store_frames()
    if name == "duplicate source primary key":
        frames["klines_daily"] = pd.concat([frames["klines_daily"], frames["klines_daily"].iloc[[0]]], ignore_index=True)
    else:
        mutate(frames)
    report = validate_frames(frames, {"stablecoin_symbols": ["USDT"]})
    assert code in _issue_codes(report), report.issues
    assert any(issue.level == "error" and issue.code == code and issue.detail for issue in report.issues)


def test_valid_store_is_ok_and_errors_can_be_raised():
    report = validate_frames(_valid_store_frames(), {"stablecoin_symbols": ["USDT"]})
    assert report.ok
    assert all(isinstance(issue, ValidationIssue) for issue in report.issues)
    with pytest.raises(StoreValidationError):
        bad = _valid_store_frames()
        bad["klines_daily"].loc[0, "volume"] = -1
        validate_frames(bad, {}).raise_for_errors()


def test_incomplete_observations_are_warnings():
    frames = _valid_store_frames()
    frames["research_panel_daily"].loc[0, "has_complete_funding"] = False
    frames["research_panel_daily"].loc[0, "has_complete_kline"] = False
    report = validate_frames(frames, {})
    assert report.ok
    assert {issue.level for issue in report.issues} == {"warning"}
    assert "2024-01-02" in " ".join(issue.detail for issue in report.issues)


def test_full_vs_cutoff_derived_builders_have_no_future_leak(capsys):
    base = _valid_store_frames()
    members = base["cmc100_constituents"].copy()
    february = members.copy()
    february["date"] = pd.Timestamp("2024-02-01")
    february["weight"] = february["weight"].iloc[::-1].to_numpy()
    march = february.copy()
    march["date"] = pd.Timestamp("2024-03-01")
    march["weight"] = [100.0 - i for i in range(50)]
    march["cmc_id"] = range(100, 150)
    march["symbol"] = [f"FUTURE{i:02d}" for i in range(50)]
    constituents = pd.concat([members, february, march], ignore_index=True)

    mappings = base["futures_contracts"]
    symbols = mappings["binance_symbol"].tolist()
    dates = pd.date_range("2023-12-31", "2024-03-05", freq="D")
    klines = pd.DataFrame([
        {"date": day, "symbol": symbol, "completed": True}
        for day in dates for symbol in symbols
    ])
    future_symbols = [f"FUTURE{i:02d}USDT" for i in range(50)]
    klines = pd.concat([
        klines,
        pd.DataFrame([
            {"date": day, "symbol": symbol, "completed": True}
            for day in pd.date_range("2024-03-01", "2024-03-05") for symbol in future_symbols
        ]),
    ], ignore_index=True)
    funding = pd.DataFrame({
        "funding_time": pd.to_datetime(["2024-02-15 08:00", "2024-03-01 08:00"]),
        "symbol": [symbols[0], future_symbols[0]],
        "funding_rate": [0.0001, 0.25],
    })
    cutoff = pd.Timestamp("2024-02-15")
    assert pd.to_datetime(constituents["date"]).max() > cutoff
    assert pd.to_datetime(klines["date"]).max() > cutoff
    assert pd.to_datetime(funding["funding_time"]).max() > cutoff
    full_universe = build_monthly_universe(constituents, mappings, klines, date(2024, 1, 1), date(2024, 2, 15))
    cut_universe = build_monthly_universe(
        constituents[pd.to_datetime(constituents["date"]) <= cutoff], mappings,
        klines[pd.to_datetime(klines["date"]) <= cutoff], date(2024, 1, 1), date(2024, 2, 15),
    )
    full_panel = build_research_panel(full_universe, klines, funding, date(2024, 2, 15), date(2024, 2, 15))
    cut_panel = build_research_panel(cut_universe, klines[pd.to_datetime(klines["date"]) <= cutoff], funding, date(2024, 2, 15), date(2024, 2, 15))

    full_u = full_universe[full_universe["effective_date"] <= cutoff].reset_index(drop=True)
    full_p = full_panel[full_panel["date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_u, cut_universe.reset_index(drop=True), check_exact=False, atol=1e-12, rtol=0)
    pd.testing.assert_frame_equal(full_p, cut_panel.reset_index(drop=True), check_exact=False, atol=1e-12, rtol=0)
    def max_numeric_diff(left, right):
        left_values = left.select_dtypes(include="number").to_numpy(dtype=float)
        right_values = right.select_dtypes(include="number").to_numpy(dtype=float)
        delta = np.nan_to_num(left_values, nan=0.0) - np.nan_to_num(right_values, nan=0.0)
        return float(np.abs(delta).max(initial=0.0))

    max_abs_diff = max(max_numeric_diff(full_u, cut_universe), max_numeric_diff(full_p, cut_panel))
    print(f"max_abs_diff={max_abs_diff}")
    assert max_abs_diff <= 1e-12
    assert f"max_abs_diff={max_abs_diff}" in capsys.readouterr().out


def test_validate_store_rejects_missing_and_empty_stores(tmp_path):
    missing = validate_store(tmp_path / "missing.h5")
    assert not missing.ok
    assert "missing_store" in _issue_codes(missing)

    empty_path = tmp_path / "empty.h5"
    empty_path.touch()
    empty = validate_store(empty_path)
    assert not empty.ok
    assert {"empty_store", "missing_table"}.issubset(_issue_codes(empty))


def test_panel_members_must_be_active_in_universe_interval():
    frames = _valid_store_frames()
    frames["universe_monthly"]["effective_end_date"] = pd.Timestamp("2024-01-02")
    frames["research_panel_daily"].loc[:, "date"] = pd.Timestamp("2024-01-03")
    report = validate_frames(frames, {})
    assert "panel_outside_universe" in _issue_codes(report)


def test_universe_intervals_must_be_ordered_and_non_overlapping():
    frames = _valid_store_frames()
    universe = frames["universe_monthly"]
    universe.loc[0, "effective_end_date"] = pd.Timestamp("2024-01-01")
    universe.loc[1, "effective_date"] = pd.Timestamp("2024-01-02")
    universe.loc[1, "effective_end_date"] = pd.Timestamp("2024-01-03")
    universe.loc[2, "effective_end_date"] = pd.Timestamp("2024-01-03")
    universe.loc[3, "effective_date"] = pd.Timestamp("2024-01-02")
    universe.loc[3, "effective_end_date"] = pd.Timestamp("2024-01-04")
    universe.loc[3, "binance_symbol"] = universe.loc[1, "binance_symbol"]
    report = validate_frames(frames, {})
    assert "invalid_effective_interval" in _issue_codes(report)
    assert "overlapping_universe" in _issue_codes(report)


def test_open_universe_interval_overlaps_any_later_same_symbol_interval():
    frames = _valid_store_frames()
    universe = frames["universe_monthly"]
    universe.loc[1, "binance_symbol"] = universe.loc[0, "binance_symbol"]
    universe.loc[1, "effective_date"] = pd.Timestamp("2024-01-03")
    universe.loc[1, "effective_end_date"] = pd.Timestamp("2024-01-04")
    report = validate_frames(frames, {})
    assert "overlapping_universe" in _issue_codes(report)


def test_invalid_funding_timestamps_are_errors():
    frames = _valid_store_frames()
    funding = frames["funding_events"].astype({"funding_time": object})
    funding.loc[0, "funding_time"] = "not-a-timestamp"
    frames["funding_events"] = funding
    report = validate_frames(frames, {})
    assert "invalid_funding_timestamp" in _issue_codes(report)


@pytest.mark.parametrize("column", ["decision_date", "effective_date", "effective_end_date"])
def test_invalid_universe_timestamps_are_structured_issues(column):
    frames = _valid_store_frames()
    universe = frames["universe_monthly"].astype({column: object})
    universe.loc[0, column] = "not-a-timestamp"
    frames["universe_monthly"] = universe
    report = validate_frames(frames, {})
    assert report.ok is False
    assert "invalid_timestamp" in _issue_codes(report)
    assert any(column in issue.detail and "index=0" in issue.detail for issue in report.issues)


def test_validation_date_parsing_does_not_require_pandas_mixed_format(monkeypatch):
    original = pd.to_datetime

    def pandas_13_to_datetime(*args, **kwargs):
        assert kwargs.get("format") != "mixed"
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "to_datetime", pandas_13_to_datetime)
    report = validate_frames(_valid_store_frames(), {})
    assert report.ok
