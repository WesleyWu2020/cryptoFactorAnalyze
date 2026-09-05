"""End-to-end synthetic validation of the common factor workflow.

Two chains over the shared 12-name H5 fixture shape:

1. Independent settlement coverage: every panel date/symbol carries an
   explicit synthetic schedule (literal expected timestamps for the synthetic
   funding events, empty lists elsewhere), so coverage is provably
   complete/not_applicable and the full chain — compute the example factor,
   write/reload Parquet, run all cost scenarios, inspect cost cash flows,
   render — must reach ``status="complete"``.
2. Unknown coverage (the shared ``h5_fixture`` without a full schedule): the
   same chain must report ``status="incomplete"``, null all-costs full
   metrics, preserve the factor output, and show the reason in the report.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from data.crypto_quant.panel import build_research_panel
from data.crypto_quant.store import CryptoQuantStore
from factor_common.manager import FactorManager
from scripts import verify_factor_common

# Same 12-name universe/calendar as the shared conftest fixture.
SYMBOLS = [f"{letter}USDT" for letter in "ABCDEFGHIJKL"]
INITIAL_SYMBOLS = SYMBOLS[:6]
LATER_SYMBOLS = SYMBOLS[6:]
CALENDAR = pd.date_range("2024-01-01", "2024-01-12", freq="D")

FEE_RATE = 0.0003
# Six names in the early window and n_groups=6 put exactly one name per
# group: the long-short portfolio holds AUSDT (long) and FUSDT (short), so it
# never touches the intentional EUSDT kline gap or the BUSDT placeholder bar.
PARAMS = {
    "start": "2024-01-02",
    "end": "2024-01-03",
    "n_groups": 6,
    "include_funding": True,
}

FACTOR_SOURCE = '''
import numpy as np

TYPE = "regular"

META = {{"factor_name": "{name}", "author": "test", "level": "daily",
        "category": "momentum", "description": "1-day log momentum"}}

SETTING = {{"data_needed": ["close"], "universe": "historical_top50",
          "warmup_bars": 1, "preprocessing": "none",
          "params": {{"window": 1}}, "factor_direction": 1}}


def calc_factor(data_ctx):
    close = data_ctx["close"]
    window = SETTING["params"]["window"]
    return np.log(close / close.shift(window))
'''


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


def _full_funding_schedule() -> pd.DataFrame:
    # Independent settlement evidence for the synthetic world: literal
    # expected timestamps where the synthetic market settles, explicit empty
    # lists everywhere else (proving no settlement was applicable).
    expected_by_key = {
        (pd.Timestamp("2024-01-03"), "AUSDT"): [
            "2024-01-03 00:00:00", "2024-01-03 08:00:00",
        ],
        (pd.Timestamp("2024-01-03"), "BUSDT"): [
            "2024-01-03 00:00:00", "2024-01-03 08:00:00",
        ],
        (pd.Timestamp("2024-01-07"), "GUSDT"): ["2024-01-07 00:00:00"],
    }
    rows = []
    for day in CALENDAR:
        for symbol in SYMBOLS:
            rows.append({
                "date": day,
                "symbol": symbol,
                "expected_times": expected_by_key.get((day, symbol), []),
            })
    return pd.DataFrame(rows)


def _write_store(path, funding_schedule) -> None:
    store = CryptoQuantStore(path)
    universe = _universe()
    klines = _klines()
    funding = _funding_events()
    panel = build_research_panel(
        universe, klines, funding, CALENDAR[-1], funding_schedule=funding_schedule
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


@pytest.fixture
def scheduled_h5(tmp_path):
    path = tmp_path / "crypto_quant_scheduled.h5"
    _write_store(path, _full_funding_schedule())
    return path


@pytest.fixture
def scheduled_manager(tmp_path, scheduled_h5):
    return FactorManager(
        h5_path=scheduled_h5,
        base_dir=tmp_path / "factor_results",
        reports_dir=tmp_path / "reports",
    )


@pytest.fixture
def unknown_manager(tmp_path, h5_fixture):
    return FactorManager(
        h5_path=h5_fixture,
        base_dir=tmp_path / "factor_results",
        reports_dir=tmp_path / "reports",
    )


@pytest.fixture
def unknown_schedule_manager(tmp_path):
    path = tmp_path / "crypto_quant_unknown_schedule.h5"
    _write_store(path, None)
    return FactorManager(
        h5_path=path,
        base_dir=tmp_path / "factor_results",
        reports_dir=tmp_path / "reports",
    )


@pytest.fixture
def factor_file(tmp_path):
    path = tmp_path / "e2e_mom.py"
    path.write_text(FACTOR_SOURCE.format(name="e2e_mom"), encoding="utf-8")
    return path


def test_e2e_complete_chain(scheduled_manager, factor_file, tmp_path):
    result = scheduled_manager.evaluate(
        str(factor_file), params=dict(PARAMS), plot=True
    )
    assert result["status"] == "complete"

    # All three cost scenarios certified, including funding accounting.
    accounting = result["factor_result"]
    assert accounting["status"] == "complete"
    scenarios = accounting["scenarios"]
    for name in ("gross", "trading_net", "all_costs"):
        assert scenarios[name]["status"] == "complete"
        assert not scenarios[name]["ledger"].empty

    # Parquet write/reload round-trips the factor matrix exactly.
    factor_path = result["paths"]["factor_path"]
    table = pd.read_parquet(factor_path)
    assert list(table.columns) == ["date", "instrument", "factor"]
    reloaded = scheduled_manager.get_value("e2e_mom", run_id=result["run_id"])
    pd.testing.assert_frame_equal(reloaded, result["factor_value"])

    # The saved evaluation reloads by id with the same certified content.
    loaded = scheduled_manager.get_performance(
        "e2e_mom", evaluation_id=result["evaluation_id"]
    )
    assert loaded["status"] == "complete"
    pd.testing.assert_frame_equal(loaded["factor_value"], result["factor_value"])

    # Cost cash flows are auditable against the order/funding tables.
    gross_ledger = scenarios["gross"]["ledger"]
    assert gross_ledger["fee"].sum() == 0.0
    assert gross_ledger["funding_cashflow"].sum() == 0.0

    trading = scenarios["trading_net"]
    filled = trading["orders"][trading["orders"]["status"] == "filled"]
    assert not filled.empty
    assert (filled["fee"] == filled["notional"] * FEE_RATE).all()
    daily_fee = filled.groupby("date")["fee"].sum()
    ledger = trading["ledger"]
    for day, fee in daily_fee.items():
        assert ledger.loc[day, "fee"] == pytest.approx(fee)
    assert ledger["fee"].sum() > 0.0

    all_costs = scenarios["all_costs"]
    funding = all_costs["funding"]
    # The AUSDT 08:00 event settles on the post-trade long quantity; the
    # 00:00 boundary events predate any holding and produce no rows.
    assert len(funding) == 1
    row = funding.iloc[0]
    assert row["instrument"] == "AUSDT"
    assert row["funding_time"] == pd.Timestamp("2024-01-03 08:00:00", tz="UTC")
    assert row["resolved"] and not row["price_approximated"]
    assert row["quantity"] > 0.0 and row["funding_rate"] > 0.0
    # Sign convention: longs pay positive rates.
    assert row["cashflow"] == pytest.approx(
        -(row["quantity"] * row["settlement_price"] * row["funding_rate"])
    )
    assert row["cashflow"] < 0.0
    ac_ledger = all_costs["ledger"]
    assert ac_ledger.loc[pd.Timestamp("2024-01-03"), "funding_cashflow"] == pytest.approx(
        row["cashflow"]
    )
    assert ac_ledger["funding_cashflow"].sum() == pytest.approx(row["cashflow"])

    coverage = all_costs["funding_coverage"]
    assert not coverage.empty
    assert coverage["accepted"].all()
    assert set(coverage["status"]) <= {"complete", "not_applicable"}

    # Fees and funding both reduce the certified equity path.
    assert gross_ledger["equity"].iloc[-1] >= ledger["equity"].iloc[-1]
    assert ledger["equity"].iloc[-1] >= ac_ledger["equity"].iloc[-1]

    # Full-sample metrics exist for every scenario.
    performance = result["factor_performance"]
    for name in ("gross", "trading_net", "all_costs"):
        block = performance["scenarios"][name]
        assert block["status"] == "complete"
        assert block["full"] is not None
        assert block["full"]["n_periods"] > 0

    # The rendered report labels costs and coverage explicitly.
    report_path = result["paths"]["report_path"]
    html = pd.io.common.stringify_path(report_path)
    text = open(html, encoding="utf-8").read()
    assert "Status: complete" in text
    assert "all_costs" in text
    assert "Total fees (from ledger)" in text
    assert "Funding coverage (all_costs)" in text

    # Rendering from reloaded artifacts alone reproduces the report.
    offline = tmp_path / "offline_report.html"
    scheduled_manager.plot_result(loaded, output_path=offline)
    assert "Status: complete" in offline.read_text(encoding="utf-8")


def test_e2e_unknown_coverage_chain(unknown_manager, factor_file):
    result = unknown_manager.evaluate(
        str(factor_file), params=dict(PARAMS), plot=True
    )
    assert result["status"] == "incomplete"

    scenarios = result["factor_result"]["scenarios"]
    assert scenarios["gross"]["status"] == "complete"
    assert scenarios["trading_net"]["status"] == "complete"
    all_costs = scenarios["all_costs"]
    assert all_costs["status"] == "incomplete"
    assert all_costs["diagnostics"]["halt_reason"] == "unresolved_funding_coverage"

    # All-cost full-sample metrics are null, never zero-filled.
    block = result["factor_performance"]["scenarios"]["all_costs"]
    assert block["full"] is None
    assert block["in_sample"] is None
    assert block["out_of_sample"] is None

    # The factor value was persisted before funding data was touched.
    reloaded = unknown_manager.get_value("e2e_mom", run_id=result["run_id"])
    pd.testing.assert_frame_equal(reloaded, result["factor_value"])
    assert reloaded.notna().any().any()

    # The report shows the reason instead of a silent zero curve.
    report_path = result["paths"]["report_path"]
    text = open(pd.io.common.stringify_path(report_path), encoding="utf-8").read()
    assert "Status: incomplete" in text
    assert "unresolved_funding_coverage" in text


def test_e2e_unknown_schedule_is_explicitly_unverified(
    unknown_schedule_manager, factor_file
):
    result = unknown_schedule_manager.evaluate(
        str(factor_file), params=dict(PARAMS), plot=True
    )

    assert result["status"] == "incomplete"
    coverage = result["factor_result"]["scenarios"]["all_costs"]["funding_coverage"]
    assert "unknown" in set(coverage["status"])
    assert not coverage["accepted"].any()
    assert result["factor_performance"]["scenarios"]["all_costs"]["full"] is None


def test_acceptance_synthetic_summary_reconciles_artifacts_and_costs(tmp_path):
    failures = []
    summary = verify_factor_common._run_synthetic(tmp_path, failures)

    assert failures == []
    assert summary["parquet_reload_equal"] is True
    assert summary["report_labels_ok"] is True
    assert summary["cash_flow_checks"]["fee_total"] > 0.0
    assert summary["cash_flow_checks"]["funding_total"] < 0.0


def test_acceptance_kernel_spec_uses_current_interpreter(tmp_path):
    assert callable(getattr(verify_factor_common, "_write_kernel_spec", None))
    spec_path = verify_factor_common._write_kernel_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    assert spec_path == tmp_path / "kernels" / "factor-common-verify" / "kernel.json"
    assert spec["argv"] == [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
