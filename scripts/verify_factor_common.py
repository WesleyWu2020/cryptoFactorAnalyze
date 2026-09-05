"""Executable acceptance for the H5 factor research workflow.

Runs the example factor and its cutoff replays against a production H5 store
(read-only: size/mtime/SHA-256 are fingerprinted before and after), reloads
the persisted factor Parquet, replays the synthetic fully-scheduled fixture
to prove complete net accounting, optionally executes the usage notebook with
a temporary kernel registered for the current interpreter, and writes
``acceptance.json`` under ``--output-dir`` together with every generated run
artifact and report.

A correctly diagnosed incomplete real-data evaluation is recorded as
``real_data_status="incomplete"`` with ``verified_complete_net_performance``
false — it is not a contract failure and is never reported as a complete
performance result. Contract failures (H5 mutation, reload mismatch, failed
synthetic accounting, cutoff drift beyond 1e-12, static-scan findings,
notebook errors, unexpected statuses) exit nonzero.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from factor_common.manager import FactorManager  # noqa: E402
from factor_common.validation import scan_future_leaks  # noqa: E402

EXAMPLE_FACTOR = _REPO_ROOT / "factor_analyse" / "factor_mining" / "example_momentum.py"
NOTEBOOK = _REPO_ROOT / "factor_analyse" / "factor_common_usage.ipynb"
E2E_MODULE = _REPO_ROOT / "tests" / "factor_common" / "test_e2e.py"
_CUTOFF_ATOL = 1e-12

KNOWN_DATA_LIMITATIONS = [
    "The production H5 carries no authoritative historical funding settlement "
    "schedules, so funding coverage is 'unknown' and the all_costs scenario "
    "cannot be certified complete on real data; observed events alone never "
    "prove completeness and no schedule is fabricated from observed "
    "timestamps.",
    "Strict funding_price_mode never approximates missing mark prices; "
    "unresolved cash flows halt certification instead of being zero-filled.",
]


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _load_e2e_module():
    spec = importlib.util.spec_from_file_location("_verify_e2e_fixture", E2E_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_kernel_spec(output_dir: Path) -> Path:
    """Write a private kernelspec that launches this interpreter."""
    spec_path = output_dir / "kernels" / "factor-common-verify" / "kernel.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "Python (factor_common verification)",
                "language": "python",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return spec_path


def _run_real(h5_path: Path, start: str, end: str, output_dir: Path, failures: list) -> dict:
    manager = FactorManager(
        h5_path=h5_path,
        base_dir=output_dir / "factor_results",
        reports_dir=output_dir / "reports",
    )
    result = manager.evaluate(
        str(EXAMPLE_FACTOR),
        params={"start": start, "end": end, "rebalance_days": 1},
        plot=True,
    )
    status = result["status"]
    if status not in ("complete", "incomplete"):
        failures.append(f"real run returned unexpected status {status!r}")

    reloaded = manager.get_value("example_momentum", run_id=result["run_id"])
    reload_equal = reloaded.equals(result["factor_value"])
    if not reload_equal:
        failures.append("factor Parquet reload does not match the computed matrix")

    cutoff = result["diagnostics"]["validation"]["cutoff"]
    cutoff_entries = cutoff.get("cutoffs", [])
    max_abs_diff = max(
        (float(entry["max_abs_diff"]) for entry in cutoff_entries), default=0.0
    )
    if cutoff.get("status") != "verified" or max_abs_diff > _CUTOFF_ATOL:
        failures.append(
            f"cutoff replay status={cutoff.get('status')} max_abs_diff={max_abs_diff}"
        )

    scan_findings = scan_future_leaks([EXAMPLE_FACTOR])
    if scan_findings:
        failures.append(f"static scan findings: {scan_findings}")

    report_path = result["paths"]["report_path"]
    report_labels_ok = False
    if report_path and Path(report_path).is_file():
        html = Path(report_path).read_text(encoding="utf-8")
        report_labels_ok = (
            "all_costs" in html
            and f"Status: {status}" in html
            and "Funding coverage (all_costs)" in html
            and (status == "complete" or "unresolved_funding_coverage" in html)
        )
    if not report_labels_ok:
        failures.append("report missing or does not label cost scenarios/status")

    coverage = result["diagnostics"]["coverage"]["funding"]["status_counts"]
    all_costs_diag = result["factor_result"]["scenarios"]["all_costs"]["diagnostics"]
    all_costs_status = result["factor_result"]["scenarios"]["all_costs"]["status"]
    if all_costs_status != "complete" and status != "incomplete":
        failures.append(
            f"real run status {status!r} hides incomplete all_costs accounting"
        )
    verified_complete = status == "complete" and all_costs_status == "complete"
    return {
        "factor_name": "example_momentum",
        "signal_start": start,
        "signal_end": end,
        "real_data_status": status,
        "verified_complete_net_performance": verified_complete,
        "run_id": result["run_id"],
        "evaluation_id": result["evaluation_id"],
        "factor_path": result["paths"]["factor_path"],
        "report_path": report_path,
        "parquet_reload_equal": reload_equal,
        "cutoff": {
            "status": cutoff.get("status"),
            "max_abs_diff": max_abs_diff,
            "entries": [
                {"cutoff": str(entry["cutoff"]), "max_abs_diff": entry["max_abs_diff"]}
                for entry in cutoff_entries
            ],
        },
        "static_scan": {"status": "clean" if not scan_findings else "findings",
                        "findings": scan_findings},
        "funding_coverage_status_counts": coverage,
        "all_costs_halt_reason": all_costs_diag.get("halt_reason"),
        "all_costs_halt_date": all_costs_diag.get("halt_date"),
        "report_labels_ok": report_labels_ok,
    }


def _run_synthetic(output_dir: Path, failures: list) -> dict:
    e2e = _load_e2e_module()
    synth_dir = output_dir / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)
    h5_path = synth_dir / "crypto_quant_synthetic.h5"
    e2e._write_store(h5_path, e2e._full_funding_schedule())
    factor_path = synth_dir / "e2e_mom.py"
    factor_path.write_text(e2e.FACTOR_SOURCE.format(name="e2e_mom"), encoding="utf-8")

    manager = FactorManager(
        h5_path=h5_path,
        base_dir=synth_dir / "factor_results",
        reports_dir=synth_dir / "reports",
    )
    result = manager.evaluate(str(factor_path), params=dict(e2e.PARAMS), plot=True)
    status = result["status"]
    scenarios = result["factor_result"]["scenarios"]
    scenario_status = {name: s["status"] for name, s in scenarios.items()}
    reloaded = manager.get_value("e2e_mom", run_id=result["run_id"])
    parquet_reload_equal = reloaded.equals(result["factor_value"])
    if not parquet_reload_equal:
        failures.append("synthetic factor Parquet reload does not match the matrix")

    trading_net = scenarios["trading_net"]
    all_costs = scenarios["all_costs"]
    trading_ledger = trading_net["ledger"]
    all_costs_ledger = all_costs["ledger"]
    fee_total = float(trading_ledger["fee"].sum())
    funding_total = float(all_costs_ledger["funding_cashflow"].sum())
    filled = trading_net["orders"]
    filled = filled[filled["status"] == "filled"]
    fee_reconciles = bool(
        not filled.empty
        and (filled["fee"] == filled["notional"] * e2e.FEE_RATE).all()
        and abs(fee_total - float(filled["fee"].sum())) <= 1e-12
    )
    funding = all_costs["funding"]
    funding_reconciles = bool(
        len(funding) == 1
        and funding["resolved"].all()
        and abs(float(funding["cashflow"].sum()) - funding_total) <= 1e-12
        and funding.iloc[0]["cashflow"] < 0.0
    )
    coverage = all_costs["funding_coverage"]
    coverage_complete = bool(
        not coverage.empty
        and coverage["accepted"].all()
        and set(coverage["status"]) <= {"complete", "not_applicable"}
    )
    report_path = result["paths"]["report_path"]
    report_labels_ok = False
    if report_path and Path(report_path).is_file():
        html = Path(report_path).read_text(encoding="utf-8")
        report_labels_ok = all(
            marker in html
            for marker in (
                "Status: complete",
                "all_costs",
                "Total fees (from ledger)",
                "Funding coverage (all_costs)",
            )
        )
    if not fee_reconciles:
        failures.append("synthetic fee cash flows do not reconcile to filled orders")
    if not funding_reconciles:
        failures.append("synthetic funding cash flows do not reconcile to ledger")
    if not coverage_complete:
        failures.append("synthetic funding coverage is not fully accepted")
    if not report_labels_ok:
        failures.append("synthetic report does not label costs and coverage")
    if status != "complete" or any(s != "complete" for s in scenario_status.values()):
        failures.append(
            f"synthetic net accounting not complete: status={status} "
            f"scenarios={scenario_status}"
        )
    return {
        "status": status,
        "scenario_status": scenario_status,
        "factor_path": result["paths"]["factor_path"],
        "report_path": report_path,
        "parquet_reload_equal": parquet_reload_equal,
        "funding_settlement_rows": int(len(funding)),
        "trading_net_fee_total": fee_total,
        "cash_flow_checks": {
            "fee_total": fee_total,
            "funding_total": funding_total,
            "fee_reconciles": fee_reconciles,
            "funding_reconciles": funding_reconciles,
            "coverage_complete": coverage_complete,
        },
        "run_id": result["run_id"],
        "evaluation_id": result["evaluation_id"],
        "report_labels_ok": report_labels_ok,
    }


def _execute_notebook(
    h5_path: Path, start: str, end: str, output_dir: Path, failures: list
) -> dict:
    import nbformat
    from nbclient import NotebookClient

    spec_path = _write_kernel_spec(output_dir)
    kernel_name = "factor-common-verify"
    os.environ["JUPYTER_PATH"] = (
        str(output_dir)
        + os.pathsep
        + os.environ.get("JUPYTER_PATH", "")
    ).rstrip(os.pathsep)
    os.environ["FACTOR_COMMON_H5"] = str(h5_path)
    os.environ["FACTOR_COMMON_OUTPUT_DIR"] = str(output_dir / "notebook_run")
    os.environ["FACTOR_COMMON_START"] = start
    os.environ["FACTOR_COMMON_END"] = end

    executed_path = output_dir / "factor_common_usage.executed.ipynb"
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(_REPO_ROOT)}},
    )
    try:
        client.execute()
    except Exception as exc:
        failures.append(f"notebook execution failed: {exc}")
    finally:
        nbformat.write(notebook, executed_path)
    return {
        "executed_path": str(executed_path),
        "kernel_name": kernel_name,
        "kernel_spec": str(spec_path),
        "kernel_python": sys.executable,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h5", required=True, help="path to the crypto_quant H5 store")
    parser.add_argument("--start", required=True, help="signal start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="signal end date YYYY-MM-DD")
    parser.add_argument("--output-dir", required=True, help="acceptance output directory")
    parser.add_argument(
        "--execute-notebook",
        action="store_true",
        help="execute factor_analyse/factor_common_usage.ipynb into the output dir",
    )
    args = parser.parse_args(argv)

    h5_path = Path(args.h5).resolve()
    if not h5_path.is_file():
        print(f"error: H5 store not found: {h5_path}", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    before = _fingerprint(h5_path)

    real = _run_real(h5_path, args.start, args.end, output_dir, failures)
    synthetic = _run_synthetic(output_dir, failures)
    notebook = None
    if args.execute_notebook:
        notebook = _execute_notebook(h5_path, args.start, args.end, output_dir, failures)

    after = _fingerprint(h5_path)
    unchanged = before == after
    if not unchanged:
        failures.append("H5 store changed during verification (size/mtime/sha256)")

    acceptance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "h5": {
            "path": str(h5_path),
            "before": before,
            "after": after,
            "unchanged": unchanged,
        },
        "real_run": real,
        "synthetic_run": synthetic,
        "notebook": notebook,
        "known_data_limitations": KNOWN_DATA_LIMITATIONS,
        "contract_failures": failures,
        "ok": not failures,
    }
    acceptance_path = output_dir / "acceptance.json"
    acceptance_path.write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"acceptance: {acceptance_path}")
    print(f"h5_unchanged={unchanged}")
    print(
        f"real_data_status={real['real_data_status']} "
        f"verified_complete_net_performance={real['verified_complete_net_performance']}"
    )
    print(f"cutoff_max_abs_diff={real['cutoff']['max_abs_diff']}")
    print(f"synthetic_status={synthetic['status']}")
    if notebook:
        print(f"notebook={notebook['executed_path']}")
    if failures:
        print("contract_failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("ok=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
