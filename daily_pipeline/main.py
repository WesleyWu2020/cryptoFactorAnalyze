"""CLI for H5 update, daily signal production and historical replay."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from data.update_crypto_quant import build_pipeline

from .artifacts import publish_signal, write_failure, write_run
from .config import PROJECT_ROOT, load_strategy_config
from .pipeline import compute_daily_signal
from .readiness import check_readiness


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "portfolio_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "daily_pipeline"


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("run-at must include a timezone")
    return parsed.astimezone(timezone.utc)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Produce fixed daily portfolio signals")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("run", "replay"):
        command = sub.add_parser(name)
        command.add_argument("--config", default=str(DEFAULT_CONFIG))
        command.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
        command.add_argument("--signal-date")
        command.add_argument("--no-publish", action="store_true")
    run = sub.choices["run"]
    run.add_argument("--run-at", type=_instant)
    run.add_argument("--skip-update", action="store_true")
    run.add_argument("--cmc-keyless", action="store_true")
    replay = sub.choices["replay"]
    replay.add_argument("--reference-run")
    return root


def _compare(result: dict, reference: Path, signal_date: str) -> dict:
    day = pd.Timestamp(signal_date)
    checks = {}
    for factor in result["member_diagnostics"]:
        actual_values = result["factor_values"].query("factor_id == @factor").set_index("instrument")["factor_value"]
        actual_targets = result["member_targets"].query("factor_id == @factor").set_index("instrument")["member_weight"]
        expected_values = pd.read_parquet(reference / "members" / factor / "values.parquet").loc[day]
        expected_targets = pd.read_parquet(reference / "members" / factor / "targets.parquet").loc[day]
        checks[factor] = {
            "values_max_abs_diff": float(np.nanmax(np.abs(actual_values.reindex(expected_values.index) - expected_values))),
            "targets_max_abs_diff": float(np.nanmax(np.abs(actual_targets.reindex(expected_targets.index) - expected_targets))),
        }
    actual = result["combined_targets"].set_index("instrument")["target_weight"]
    expected = pd.read_parquet(reference / "targets.parquet").loc[day]
    checks["combined"] = {
        "max_abs_diff": float(np.nanmax(np.abs(actual.reindex(expected.index) - expected)))
    }
    passed = all(
        value <= 1e-12 for item in checks.values() for key, value in item.items()
        if key.endswith("diff")
    )
    return {"passed": passed, "tolerance": 1e-12, "checks": checks}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    signal_date = args.signal_date
    try:
        config = load_strategy_config(args.config)
        now = getattr(args, "run_at", None) or datetime.now(timezone.utc)
        signal_date = args.signal_date or (pd.Timestamp(now).tz_convert("UTC").normalize() - pd.Timedelta(days=1)).date().isoformat()
        if args.command == "run" and not args.skip_update:
            pipeline = build_pipeline(config.h5_path, cmc_keyless=args.cmc_keyless)
            try:
                update = pipeline.update(now)
            finally:
                pipeline.close()
        readiness = check_readiness(config.h5_path, signal_date,
                                    min_valid_instruments=config.min_valid_instruments)
        if args.command == "run" and not args.skip_update:
            readiness["update"] = {
                "as_of_utc": update.as_of_utc.isoformat(),
                "last_complete_kline_date": None if update.last_complete_kline_date is None else update.last_complete_kline_date.isoformat(),
                "last_complete_panel_date": None if update.last_complete_panel_date is None else update.last_complete_panel_date.isoformat(),
                "warnings": list(update.warnings),
            }
        result = compute_daily_signal(config, signal_date, readiness)
        run_dir = write_run(result, args.output_root)
        if args.command == "replay":
            if getattr(config, "combine_mode", "members") != "members":
                raise ValueError("replay comparison is only defined for combine_mode='members'")
            reference = Path(args.reference_run).resolve() if args.reference_run else config.source_report
            receipt = _compare(result, reference, signal_date)
            (run_dir / "replay.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not receipt["passed"]:
                print(f"Replay differs from reference: {run_dir / 'replay.json'}", file=sys.stderr)
                return 2
        if args.command == "run" and not args.no_publish:
            publish_signal(run_dir, args.output_root)
        print(f"Daily signal: {run_dir / 'signal.json'}")
        return 0
    except Exception as exc:
        if signal_date:
            try:
                failed = write_failure(args.output_root, str(signal_date), exc)
                print(f"Failure receipt: {failed / 'manifest.json'}", file=sys.stderr)
            except Exception as artifact_exc:
                print(f"failure receipt could not be written: {artifact_exc}", file=sys.stderr)
        print(f"daily_pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
