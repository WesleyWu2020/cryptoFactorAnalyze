"""Command-line entry point for the immutable fixed portfolio workflow.

The module is intentionally usable only as ``python -m portfolio.main_fixed``
so each freeze/run/audit starts with a fresh interpreter and module state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from portfolio.factor_pool.module_loader import load_members
from portfolio.fixed_artifacts import write_result
from portfolio.fixed_audit import audit_fixed
from portfolio.fixed_config import FactorAllocation, FixedConfig
from portfolio.fixed_pipeline import run_fixed
from portfolio.fixed_provenance import code_snapshot


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Fixed daily perpetual portfolio research")
    commands = root.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze", help="Freeze modules and research configuration")
    freeze.add_argument("--h5", required=True)
    freeze.add_argument("--factor", action="append", required=True)
    freeze.add_argument("--allocation", action="append", type=float, required=True)
    freeze.add_argument("--start", required=True)
    freeze.add_argument("--end", required=True)
    freeze.add_argument("--as-of", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--rebalance-days", type=int, default=1)
    freeze.add_argument("--anchor-date", default="2024-01-01")
    freeze.add_argument("--n-groups", type=int, default=5)
    freeze.add_argument("--min-valid-instruments", type=int, default=10)
    freeze.add_argument("--fee-rate", type=float, default=0.0005)
    freeze.add_argument("--slippage", type=float, default=0.001)
    for name, default in (
        ("gross-limit", 0.5), ("long-limit", 0.25), ("short-limit", 0.25),
        ("single-limit", 0.05), ("net-limit", 0.05),
    ):
        freeze.add_argument("--" + name, type=float, default=default)

    run = commands.add_parser("run", help="Run a frozen research portfolio")
    run.add_argument("--config", required=True)
    run.add_argument("--output-root", default="reports/portfolio")

    audit = commands.add_parser("audit", help="Recompute historical prefixes from cutoff providers")
    audit.add_argument("--config", required=True)
    audit.add_argument("--cutoff", action="append", required=True)
    audit.add_argument("--output", required=True)
    return root


def write_new_json(path: str | Path, payload: object) -> None:
    """Write a JSON file exactly once; never replace a previous freeze/receipt."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_config(path: str | Path) -> FixedConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return FixedConfig.from_dict(json.load(handle))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "freeze":
            if len(args.factor) != len(args.allocation):
                raise ValueError("each factor requires one allocation")
            paths = [str(Path(path).resolve(strict=True)) for path in args.factor]
            config = FixedConfig(
                h5_path=str(Path(args.h5).resolve(strict=True)),
                signal_start=args.start, signal_end=args.end, as_of=args.as_of,
                factors=tuple(FactorAllocation(path, weight)
                              for path, weight in zip(paths, args.allocation)),
                code_hashes=code_snapshot(paths),
                frozen_at=datetime.now(timezone.utc).isoformat(),
                rebalance_days=args.rebalance_days, anchor_date=args.anchor_date,
                n_groups=args.n_groups, min_valid_instruments=args.min_valid_instruments,
                fee_rate=args.fee_rate, slippage=args.slippage,
                gross_limit=args.gross_limit, long_limit=args.long_limit,
                short_limit=args.short_limit, single_limit=args.single_limit,
                net_limit=args.net_limit,
            )
            # Validate source modules and future-leak policy before publishing.
            load_members(config)
            write_new_json(args.output, asdict(config))
            print(f"Frozen research config: {args.output}")
            return 0

        config = _load_config(args.config)
        if args.command == "audit":
            receipt = audit_fixed(config, args.cutoff)
            write_new_json(args.output, receipt)
            print(f"Cutoff audit verified: {args.output}")
            return 0

        result = run_fixed(config)
        output = write_result(result, args.output_root)
        print(f"Research status: {result['status']}")
        print(f"Report: {output / 'report.html'}")
        return 0 if result["status"] == "complete" else 2
    except (ValueError, TypeError, OSError, RuntimeError, AssertionError) as exc:
        print(f"Fixed portfolio failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
