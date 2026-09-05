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
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from factor_common.manager import FactorManager
from scripts import verify_factor_common
from tests.factor_common.conftest import (
    EXAMPLE_CALENDAR,
    EXAMPLE_PARAMS,
    _complete_funding_schedule,
    _example_funding_events,
    _funding_schedule,
    write_h5_fixture,
)

EXAMPLE_FACTOR = Path(__file__).resolve().parents[2] / "factor_analyse" / "factor_mining" / "example_momentum.py"
PARAMS = EXAMPLE_PARAMS
FEE_RATE = 0.0003


@pytest.fixture
def complete_h5(h5_fixture):
    return write_h5_fixture(
        h5_fixture,
        calendar=EXAMPLE_CALENDAR,
        funding_schedule=_complete_funding_schedule(),
        funding_events=_example_funding_events(),
    )


@pytest.fixture
def scheduled_manager(tmp_path, complete_h5):
    return FactorManager(
        h5_path=complete_h5,
        base_dir=tmp_path / "factor_results",
        reports_dir=tmp_path / "reports",
    )


@pytest.fixture
def partial_h5(h5_fixture):
    return write_h5_fixture(
        h5_fixture,
        calendar=EXAMPLE_CALENDAR,
        funding_schedule=_funding_schedule(),
    )


@pytest.fixture
def unknown_manager(tmp_path, partial_h5):
    return FactorManager(
        h5_path=partial_h5,
        base_dir=tmp_path / "factor_results",
        reports_dir=tmp_path / "reports",
    )


@pytest.fixture
def unknown_schedule_manager(tmp_path, h5_fixture):
    path = write_h5_fixture(
        h5_fixture,
        calendar=EXAMPLE_CALENDAR,
        funding_schedule=None,
        funding_events=_example_funding_events(),
    )
    return FactorManager(
        h5_path=path,
        base_dir=tmp_path / "factor_results",
        reports_dir=tmp_path / "reports",
    )


def test_e2e_complete_chain(scheduled_manager, tmp_path):
    result = scheduled_manager.evaluate(
        str(EXAMPLE_FACTOR), params=dict(PARAMS), plot=True
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
    reloaded = scheduled_manager.get_value("example_momentum", run_id=result["run_id"])
    pd.testing.assert_frame_equal(reloaded, result["factor_value"])

    # The saved evaluation reloads by id with the same certified content.
    loaded = scheduled_manager.get_performance(
        "example_momentum", evaluation_id=result["evaluation_id"]
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
    assert filled["fee"].tolist() == pytest.approx(
        (filled["notional"] * FEE_RATE).tolist()
    )
    daily_fee = filled.groupby("date")["fee"].sum()
    ledger = trading["ledger"]
    for day, fee in daily_fee.items():
        assert ledger.loc[day, "fee"] == pytest.approx(fee)
    assert ledger["fee"].sum() > 0.0

    all_costs = scenarios["all_costs"]
    funding = all_costs["funding"]
    # The GUSDT 08:00 event settles on the post-trade long quantity; the
    # earlier boundary events predate any holding and produce no rows.
    assert len(funding) == 1
    row = funding.iloc[0]
    assert row["instrument"] == "GUSDT"
    assert row["funding_time"] == pd.Timestamp("2024-01-23 08:00:00", tz="UTC")
    assert row["resolved"] and not row["price_approximated"]
    assert row["quantity"] > 0.0 and row["funding_rate"] > 0.0
    # Sign convention: longs pay positive rates.
    assert row["cashflow"] == pytest.approx(
        -(row["quantity"] * row["settlement_price"] * row["funding_rate"])
    )
    assert row["cashflow"] < 0.0
    ac_ledger = all_costs["ledger"]
    assert ac_ledger.loc[pd.Timestamp("2024-01-23"), "funding_cashflow"] == pytest.approx(
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


def test_e2e_unknown_coverage_chain(unknown_manager):
    result = unknown_manager.evaluate(
        str(EXAMPLE_FACTOR), params=dict(PARAMS), plot=True
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
    reloaded = unknown_manager.get_value("example_momentum", run_id=result["run_id"])
    pd.testing.assert_frame_equal(reloaded, result["factor_value"])
    assert reloaded.notna().any().any()

    # The report shows the reason instead of a silent zero curve.
    report_path = result["paths"]["report_path"]
    text = open(pd.io.common.stringify_path(report_path), encoding="utf-8").read()
    assert "Status: incomplete" in text
    assert "unresolved_funding_coverage" in text


def test_e2e_unknown_schedule_chain_is_incomplete_and_preserves_values(
    unknown_schedule_manager,
):
    result = unknown_schedule_manager.evaluate(
        str(EXAMPLE_FACTOR), params=dict(PARAMS), plot=True
    )

    assert result["status"] == "incomplete"
    all_costs = result["factor_result"]["scenarios"]["all_costs"]
    assert all_costs["status"] == "incomplete"
    assert all_costs["diagnostics"]["halt_reason"] == "unresolved_funding_coverage"
    coverage = all_costs["funding_coverage"]
    assert "unknown" in set(coverage["status"])
    assert not coverage["accepted"].any()
    metrics = result["factor_performance"]["scenarios"]["all_costs"]
    assert metrics["full"] is None
    assert metrics["in_sample"] is None
    assert metrics["out_of_sample"] is None

    reloaded = unknown_schedule_manager.get_value(
        "example_momentum", run_id=result["run_id"]
    )
    pd.testing.assert_frame_equal(reloaded, result["factor_value"])
    assert reloaded.notna().any().any()

    report = Path(result["paths"]["report_path"]).read_text(encoding="utf-8")
    assert "Status: incomplete" in report
    assert "unresolved_funding_coverage" in report


def test_acceptance_synthetic_summary_reconciles_artifacts_and_costs(tmp_path):
    failures = []
    summary = verify_factor_common._run_synthetic(tmp_path, failures)

    assert failures == []
    assert summary["factor_name"] == "example_momentum"
    assert summary["factor_source"] == str(
        verify_factor_common.EXAMPLE_FACTOR
    )
    assert summary["parquet_reload_equal"] is True
    assert summary["report_labels_ok"] is True
    assert summary["cash_flow_checks"]["fee_total"] > 0.0
    assert summary["cash_flow_checks"]["funding_total"] < 0.0


def test_real_contract_rejects_non_null_incomplete_net_metrics():
    result = {
        "status": "incomplete",
        "factor_performance": {
            "scenarios": {"all_costs": {"status": "incomplete", "full": {"total_return": 0.1}}}
        },
        "factor_result": {
            "scenarios": {"all_costs": {"status": "incomplete", "funding_coverage": pd.DataFrame()}}
        },
    }

    failures, verified = verify_factor_common._validate_real_contract(result)

    assert verified is False
    assert any("all_costs full metrics must be null" in failure for failure in failures)


def test_real_contract_requires_incomplete_halt_and_unaccepted_coverage():
    result = {
        "status": "incomplete",
        "factor_performance": {
            "scenarios": {"all_costs": {"status": "incomplete", "full": None}}
        },
        "factor_result": {
            "scenarios": {
                "all_costs": {
                    "status": "complete",
                    "diagnostics": {"halt_reason": None},
                    "funding_coverage": pd.DataFrame(
                        [{"status": "complete", "accepted": True}]
                    ),
                }
            }
        },
    }

    failures, verified = verify_factor_common._validate_real_contract(result)

    assert verified is False
    assert any("all_costs status must be incomplete" in failure for failure in failures)
    assert any("unaccepted unresolved row" in failure for failure in failures)
    assert any("halt reason" in failure for failure in failures)


@pytest.mark.parametrize(
    "coverage_status",
    ["missing", "no_events", "invalid_rate", "schedule_mismatch", "rejected", "unknown"],
)
@pytest.mark.parametrize("halt_reason", ["unresolved_funding", "unresolved_funding_coverage"])
def test_real_contract_accepts_documented_unresolved_coverage_statuses(
    coverage_status, halt_reason
):
    result = {
        "status": "incomplete",
        "factor_performance": {
            "scenarios": {"all_costs": {"status": "incomplete", "full": None}}
        },
        "factor_result": {
            "scenarios": {
                "all_costs": {
                    "status": "incomplete",
                    "diagnostics": {"halt_reason": halt_reason},
                    "funding_coverage": pd.DataFrame(
                        [{"status": coverage_status, "accepted": False}]
                    ),
                }
            }
        },
    }

    failures, verified = verify_factor_common._validate_real_contract(result)

    assert failures == []
    assert verified is False


def test_real_contract_rejects_complete_status_without_complete_coverage():
    result = {
        "status": "complete",
        "factor_performance": {
            "scenarios": {"all_costs": {"status": "complete", "full": {"total_return": 0.1}}}
        },
        "factor_result": {
            "scenarios": {
                "all_costs": {
                    "status": "complete",
                    "funding_coverage": pd.DataFrame(
                        [{"status": "unknown", "accepted": False}]
                    ),
                }
            }
        },
    }

    failures, verified = verify_factor_common._validate_real_contract(result)

    assert verified is False
    assert any("complete coverage evidence" in failure for failure in failures)


def test_acceptance_kernel_spec_uses_current_interpreter(tmp_path):
    assert callable(getattr(verify_factor_common, "_write_kernel_spec", None))
    spec_path = verify_factor_common._write_kernel_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    assert spec_path == tmp_path / "kernels" / "factor-common-verify" / "kernel.json"
    assert spec["argv"] == [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]


def test_cli_refuses_nonempty_output_dir(tmp_path):
    output_dir = tmp_path / "occupied"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(verify_factor_common.__file__),
            "--h5",
            str(tmp_path / "missing.h5"),
            "--start",
            "2024-02-01",
            "--end",
            "2024-03-15",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "output directory must be empty" in proc.stderr
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (output_dir / "acceptance.json").exists()


def test_cli_writes_strict_failure_acceptance_for_malformed_h5(tmp_path):
    h5_path = tmp_path / "malformed.h5"
    h5_path.write_text("not an HDF5 store", encoding="utf-8")
    output_dir = tmp_path / "acceptance"

    proc = subprocess.run(
        [
            sys.executable,
            str(verify_factor_common.__file__),
            "--h5",
            str(h5_path),
            "--start",
            "2024-02-01",
            "--end",
            "2024-03-15",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path("/tmp"),
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    acceptance_path = output_dir / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    assert acceptance["ok"] is False
    assert acceptance["real_run"]["status"] == "error"
    assert acceptance["contract_failures"]
    raw = acceptance_path.read_text(encoding="utf-8")
    assert all(token not in raw for token in ("NaN", "Infinity", "-Infinity"))
    assert "Traceback" not in proc.stderr


def test_write_acceptance_fallback_is_strict_and_reports_failure(tmp_path):
    acceptance_path = tmp_path / "acceptance.json"

    failure = verify_factor_common._write_acceptance(
        acceptance_path, {"ok": True, "non_finite": float("nan")}
    )

    saved = json.loads(acceptance_path.read_text(encoding="utf-8"))
    assert failure["component"] == "acceptance_serialization"
    assert saved["ok"] is False
    assert saved["contract_failures"][0]["component"] == "acceptance_serialization"
    assert all(
        token not in acceptance_path.read_text(encoding="utf-8")
        for token in ("NaN", "Infinity", "-Infinity")
    )


def test_main_propagates_acceptance_serialization_failure(tmp_path, monkeypatch, capsys):
    h5_path = tmp_path / "fixture.h5"
    h5_path.write_bytes(b"fixture")
    output_dir = tmp_path / "acceptance"

    monkeypatch.setattr(
        verify_factor_common,
        "_fingerprint",
        lambda path: {"size": 1, "mtime_ns": 1, "sha256": "hash"},
    )
    monkeypatch.setattr(
        verify_factor_common,
        "_run_real",
        lambda h5, start, end, out, failures: {
            "status": "incomplete",
            "real_data_status": "incomplete",
            "verified_complete_net_performance": False,
            "cutoff": {"max_abs_diff": 0.0},
        },
    )
    monkeypatch.setattr(
        verify_factor_common,
        "_run_synthetic",
        lambda out, failures: {"status": "complete"},
    )

    def fake_write(path, acceptance):
        path.write_text('{"ok": false}\n', encoding="utf-8")
        return {
            "component": "acceptance_serialization",
            "type": "ValueError",
            "message": "forced fallback",
        }

    monkeypatch.setattr(verify_factor_common, "_write_acceptance", fake_write)

    rc = verify_factor_common.main(
        [
            "--h5",
            str(h5_path),
            "--start",
            "2024-02-01",
            "--end",
            "2024-03-15",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert rc != 0
    assert "ok=true" not in captured.out
    assert "acceptance_serialization" in captured.err
