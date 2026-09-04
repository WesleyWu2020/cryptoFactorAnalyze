"""Command-line maintenance entry point for the CryptoQuant HDF5 store."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import requests
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.crypto_quant.binance import fetch_daily_klines, fetch_exchange_info, fetch_funding_events
from data.crypto_quant.cmc import fetch_cmc_history
from data.crypto_quant.config import PipelineConfig
from data.crypto_quant.http import JsonHttpClient
from data.crypto_quant.pipeline import CryptoQuantPipeline
from data.crypto_quant.validation import StoreValidationError, ValidationReport, validate_store


class _CmcSource:
    def __init__(self, client: JsonHttpClient):
        self.client = client

    def fetch_history(self, start, end, on_page=None):
        return fetch_cmc_history(self.client, start, end, on_page=on_page)


class _BinanceSource:
    def __init__(self, client: JsonHttpClient):
        self.client = client

    def fetch_exchange_info(self):
        return fetch_exchange_info(self.client)

    def fetch_klines(self, symbol, start, end):
        return fetch_daily_klines(self.client, symbol, start, end)

    def fetch_funding(self, symbol, start_ms, end_ms):
        return fetch_funding_events(self.client, symbol, start_ms, end_ms)


def _store_config(store_path: Path) -> PipelineConfig:
    base = PipelineConfig.default()
    store_path = store_path.expanduser().resolve()
    return replace(
        base,
        store_path=store_path,
        staging_path=store_path.with_name(f".{store_path.stem}.staging{store_path.suffix}"),
        lock_path=store_path.with_name(f".{store_path.stem}.lock"),
    )


def build_pipeline(store_path: Path) -> CryptoQuantPipeline:
    """Build all concrete sources around one shared HTTP session."""
    config = _store_config(store_path)
    session = requests.Session()
    api_key = os.environ.get("CMC_PRO_API_KEY") or os.environ.get("CMC_API_KEY")
    if api_key:
        session.headers.update({"X-CMC_PRO_API_KEY": api_key})
    client = JsonHttpClient(
        session,
        timeout=config.request_timeout_seconds,
        max_attempts=config.max_attempts,
        base_backoff=config.base_backoff_seconds,
        max_backoff=config.max_backoff_seconds,
    )
    try:
        pipeline = CryptoQuantPipeline(config, _CmcSource(client), _BinanceSource(client))
    except Exception:
        session.close()
        raise
    pipeline.close = session.close  # type: ignore[attr-defined]
    return pipeline


def _parse_as_of(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("as-of must be ISO-8601 with timezone, e.g. 2026-09-03T00:20:00Z") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("as-of must include a timezone")
    return parsed.astimezone(timezone.utc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="update_crypto_quant.py")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("backfill", "update", "rebuild-derived"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--as-of", type=_parse_as_of, default=None)
        sub.add_argument("--store", type=Path, default=Path("data/crypto_quant.h5"))
        sub.add_argument("--reset-staging", action="store_true")
    for command in ("validate", "inspect"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--store", type=Path, default=Path("data/crypto_quant.h5"))
    return parser


def _print_summary(summary) -> None:
    print(json.dumps({
        "mode": summary.mode,
        "as_of_utc": summary.as_of_utc.isoformat(),
        "store": str(summary.published_path),
        "row_counts": summary.row_counts,
        "last_complete_date": summary.last_complete_panel_date.isoformat() if summary.last_complete_panel_date else None,
    }, sort_keys=True, separators=(",", ":")))


def _run_pipeline(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args.store)
    try:
        as_of = args.as_of or datetime.now(timezone.utc)
        method = getattr(pipeline, args.command.replace("-", "_"))
        _print_summary(method(as_of, reset_staging=args.reset_staging))
        return 0
    finally:
        close = getattr(pipeline, "close", None)
        if close is not None:
            close()


def _run_validate(store_path: Path) -> int:
    report: ValidationReport = validate_store(store_path)
    if report.issues:
        print("validation=" + ("ok" if report.ok else "failed") + " issues=" + str(len(report.issues)))
        for issue in report.issues:
            print(f"{issue.level}:{issue.code}:{issue.detail}")
    else:
        print("validation=ok issues=0")
    return 0 if report.ok else 2


def _run_inspect(store_path: Path) -> int:
    store_path = Path(store_path)
    if not store_path.is_file():
        raise FileNotFoundError(f"store path does not exist: {store_path}")
    with pd.HDFStore(store_path, mode="r") as hdf:
        if not hdf.keys():
            raise ValueError(f"store has no tables: {store_path}")
        if "/_metadata" not in hdf.keys():
            metadata = {}
        else:
            metadata_frame = hdf.select("_metadata")
            metadata = {row.key: json.loads(row.value) for row in metadata_frame.itertuples()}
    print(json.dumps({
        "row_counts": metadata.get("table_row_counts", {}),
        "date_ranges": metadata.get("table_date_ranges", {}),
        "last_complete_date": metadata.get("last_complete_panel_date"),
    }, sort_keys=True, separators=(",", ":")))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            return _run_validate(args.store)
        if args.command == "inspect":
            return _run_inspect(args.store)
        return _run_pipeline(args)
    except (StoreValidationError,):
        print("validation=failed", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
