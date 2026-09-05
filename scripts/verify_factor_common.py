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
from tests.factor_common.conftest import (  # noqa: E402
    EXAMPLE_CALENDAR,
    EXAMPLE_PARAMS,
    _complete_funding_schedule,
    _example_funding_events,
    write_h5_fixture,
)

EXAMPLE_FACTOR = _REPO_ROOT / "factor_analyse" / "factor_mining" / "example_momentum.py"
NOTEBOOK = _REPO_ROOT / "factor_analyse" / "factor_common_usage.ipynb"
_CUTOFF_ATOL = 1e-12
_SYNTHETIC_FEE_RATE = 0.0003

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
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return spec_path


def _validate_real_contract(result: dict) -> tuple[list[str], bool]:
    """Validate status, all-cost metrics, and funding evidence for real data."""
    failures = []
    status = result["status"]
    performance = result["factor_performance"]["scenarios"]["all_costs"]
    accounting = result["factor_result"]["scenarios"]["all_costs"]
    coverage = accounting.get("funding_coverage")
    coverage_complete = bool(
        hasattr(coverage, "empty")
        and not coverage.empty
        and coverage["accepted"].all()
        and set(coverage["status"]) <= {"complete", "not_applicable"}
    )
    coverage_has_unresolved = bool(
        hasattr(coverage, "empty")
        and not coverage.empty
        and (~coverage["accepted"]).any()
    )
    halt_reason = accounting.get("diagnostics", {}).get("halt_reason")

    if status == "incomplete":
        if accounting.get("status") != "incomplete":
            failures.append("incomplete real data: all_costs status must be incomplete")
        if performance.get("full") is not None:
            failures.append("incomplete real data: all_costs full metrics must be null")
        if not coverage_has_unresolved:
            failures.append(
                "incomplete real data: funding coverage must include an unaccepted unresolved row"
            )
        if halt_reason not in {"unresolved_funding", "unresolved_funding_coverage"}:
            failures.append(
                "incomplete real data: all_costs halt reason must identify unresolved funding"
            )
        return failures, False

    if status != "complete":
        return failures, False

    if accounting.get("status") != "complete" or not coverage_complete:
        failures.append(
            "complete real data requires complete coverage evidence before verified net performance"
        )
    if performance.get("full") is None:
        failures.append("complete real data: all_costs full metrics cannot be null")
    return failures, not failures


def _exception_record(component: str, exc: BaseException) -> dict:
    return {
        "component": component,
        "type": type(exc).__name__,
        "message": str(exc) or repr(exc),
    }


def _write_acceptance(path: Path, acceptance: dict) -> dict | None:
    """Write strict JSON, falling back to a strict error document if needed."""
    serialization_failure = None
    try:
        encoded = json.dumps(
            acceptance, indent=2, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        serialization_failure = _exception_record("acceptance_serialization", exc)
        fallback = {
            "ok": False,
            "contract_failures": [serialization_failure],
            "serialization_error": {
                "type": type(exc).__name__,
                "message": str(exc) or repr(exc),
            },
        }
        encoded = json.dumps(fallback, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(encoded + "\n", encoding="utf-8")
    return serialization_failure


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

    all_costs_diag = result["factor_result"]["scenarios"]["all_costs"]["diagnostics"]
    halt_reason = all_costs_diag.get("halt_reason")
    report_path = result["paths"]["report_path"]
    report_labels_ok = False
    if report_path and Path(report_path).is_file():
        html = Path(report_path).read_text(encoding="utf-8")
        report_labels_ok = (
            "all_costs" in html
            and f"Status: {status}" in html
            and "Funding coverage (all_costs)" in html
            and (status == "complete" or bool(halt_reason and halt_reason in html))
        )
    if not report_labels_ok:
        failures.append("report missing or does not label cost scenarios/status")

    coverage = result["diagnostics"]["coverage"]["funding"]["status_counts"]
    contract_failures, verified_complete = _validate_real_contract(result)
    failures.extend(contract_failures)
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
    synth_dir = output_dir / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)
    h5_path = synth_dir / "crypto_quant_fixture.h5"
    write_h5_fixture(
        h5_path,
        calendar=EXAMPLE_CALENDAR,
        funding_schedule=_complete_funding_schedule(),
        funding_events=_example_funding_events(),
    )

    manager = FactorManager(
        h5_path=h5_path,
        base_dir=synth_dir / "factor_results",
        reports_dir=synth_dir / "reports",
    )
    result = manager.evaluate(str(EXAMPLE_FACTOR), params=dict(EXAMPLE_PARAMS), plot=True)
    status = result["status"]
    scenarios = result["factor_result"]["scenarios"]
    scenario_status = {name: s["status"] for name, s in scenarios.items()}
    reloaded = manager.get_value("example_momentum", run_id=result["run_id"])
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
        and (
            (filled["fee"] - filled["notional"] * _SYNTHETIC_FEE_RATE)
            .abs()
            .max()
            <= 1e-12
        )
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
        "factor_name": "example_momentum",
        "factor_source": str(EXAMPLE_FACTOR),
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
    os.environ["FACTOR_COMMON_REPO_ROOT"] = str(_REPO_ROOT)
    os.environ["FACTOR_COMMON_NOTEBOOK"] = str(NOTEBOOK)
    os.environ["IPYTHONDIR"] = str(output_dir / "ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(output_dir / "jupyter_runtime")
    (output_dir / "ipython").mkdir(parents=True, exist_ok=True)
    (output_dir / "jupyter_runtime").mkdir(parents=True, exist_ok=True)

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
    output_dir = Path(args.output_dir).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.iterdir()):
            print(
                f"error: output directory must be empty: {output_dir}",
                file=sys.stderr,
            )
            return 2
    except OSError as exc:
        print(
            f"error: output directory is not usable: {output_dir}: {exc}",
            file=sys.stderr,
        )
        return 2

    failures: list[object] = []
    before = _fingerprint(h5_path) if h5_path.is_file() else None
    if before is None:
        failures.append(
            {
                "component": "real_input",
                "type": "FileNotFoundError",
                "message": f"H5 store not found: {h5_path}",
            }
        )

    real = None
    if before is not None:
        try:
            real = _run_real(h5_path, args.start, args.end, output_dir, failures)
        except Exception as exc:
            failures.append(_exception_record("real_run", exc))
            real = {
                "status": "error",
                "real_data_status": "error",
                "verified_complete_net_performance": False,
                "error": _exception_record("real_run", exc),
            }
    else:
        real = {
            "status": "error",
            "real_data_status": "error",
            "verified_complete_net_performance": False,
            "error": failures[-1],
        }

    synthetic = None
    try:
        synthetic = _run_synthetic(output_dir, failures)
    except Exception as exc:
        failures.append(_exception_record("synthetic_run", exc))
        synthetic = {
            "status": "error",
            "error": _exception_record("synthetic_run", exc),
        }

    notebook = None
    if args.execute_notebook:
        try:
            notebook = _execute_notebook(
                h5_path, args.start, args.end, output_dir, failures
            )
        except Exception as exc:
            failures.append(_exception_record("notebook", exc))
            notebook = {
                "status": "error",
                "executed_path": str(output_dir / "factor_common_usage.executed.ipynb"),
                "error": _exception_record("notebook", exc),
            }

    after = _fingerprint(h5_path) if h5_path.is_file() else None
    unchanged = before is not None and before == after
    if before is not None and not unchanged:
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
    serialization_failure = _write_acceptance(acceptance_path, acceptance)
    if serialization_failure is not None:
        failures.append(serialization_failure)

    print(f"acceptance: {acceptance_path}")
    print(f"h5_unchanged={unchanged}")
    print(
        f"real_data_status={real.get('real_data_status', real.get('status'))} "
        f"verified_complete_net_performance={real.get('verified_complete_net_performance', False)}"
    )
    print(f"cutoff_max_abs_diff={real.get('cutoff', {}).get('max_abs_diff')}")
    print(f"synthetic_status={synthetic.get('status') if synthetic else 'error'}")
    if notebook:
        print(f"notebook={notebook.get('executed_path')}")
    if failures:
        print("contract_failures:", file=sys.stderr)
        for failure in failures:
            if isinstance(failure, dict):
                print(
                    "  - " + json.dumps(failure, sort_keys=True, allow_nan=False),
                    file=sys.stderr,
                )
            else:
                print(f"  - {failure}", file=sys.stderr)
        return 1
    print("ok=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
