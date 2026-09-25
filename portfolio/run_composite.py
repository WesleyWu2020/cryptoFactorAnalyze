"""Composite-score (先加权后分组) variant of the fixed portfolio pipeline.

Reuses the frozen FixedConfig from an existing portfolio report, the shared
factor value engine, and the shared accounting core. The only difference vs.
portfolio.fixed_pipeline.run_fixed is portfolio construction:

- baseline (run_fixed): each factor -> quintile long/short unit-gross targets
  -> allocation-weighted sum of the 10 sub-portfolios (union of holdings).
- composite (this script): each factor value matrix -> per-day cross-sectional
  z-score -> multiplied by factor_direction -> allocation-weighted sum into a
  single composite score -> quintile long/short on that score.

Cross-sectional standardization uses same-day values only; no future data.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from factor_common.backtest import run_target_backtest
from factor_common.data_provider import DataProvider
from factor_common.grouping import target_weights
from factor_common.profiles import resolve_profile
from factor_common.value_engine import compute_factor
from data.crypto_quant.universe import HISTORICAL_CONTRACT_END_DATES
from factor_common.loader import load_factor
from factor_common.validation import scan_future_leaks
from portfolio.fixed_config import FixedConfig, daily_date
from portfolio.fixed_pipeline import summarize_accounting
from portfolio.portfolio_builder.composite import composite_score
from portfolio.portfolio_builder.fixed_weights import combine_targets

CACHE_DIR = Path("portfolio/output/composite_cache")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", help="frozen portfolio report directory")
    parser.add_argument("--out", default="portfolio/output/composite_result.json")
    parser.add_argument("--keep", nargs="*", default=None,
                        help="factor_ids to keep; allocations are renormalized")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    cfg = FixedConfig.from_dict(json.loads((report_dir / "config.json").read_text()))

    cutoff = daily_date(cfg.as_of)
    signal_end = min(daily_date(cfg.signal_end), cutoff)
    tail_end = min(daily_date(cfg.signal_end) + pd.Timedelta(days=1 + cfg.rebalance_days), cutoff)

    paths = [member.path for member in cfg.factors]
    findings = scan_future_leaks(paths)
    if findings:
        raise ValueError(f"factor source future-leak findings: {findings}")
    specs = [load_factor(path) for path in paths]
    provider = DataProvider(cfg.h5_path, as_of=cutoff)
    profile = resolve_profile(
        "perp_1d",
        {
            "n_groups": cfg.n_groups,
            "rebalance_days": cfg.rebalance_days,
            "anchor_date": cfg.anchor_date,
            "fee_rate": cfg.fee_rate,
            "slippage": cfg.slippage,
            "gross_exposure": cfg.gross_limit,
            "initial_equity": cfg.initial_equity,
            "include_funding": True,
            "funding_price_mode": "strict",
        },
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    values: dict[str, pd.DataFrame] = {}
    allocations: dict[str, float] = {}
    for allocation, spec in zip(cfg.factors, specs):
        cache = CACHE_DIR / f"{spec.factor_id}.parquet"
        if cache.exists():
            matrix = pd.read_parquet(cache)
        else:
            matrix, _ = compute_factor(spec, provider, start=cfg.signal_start, end=signal_end)
            matrix.to_parquet(cache)
        values[spec.factor_id] = matrix
        allocations[spec.factor_id] = allocation.allocation
        print(f"factor {spec.factor_id}: {matrix.shape}", flush=True)

    if args.keep is not None:
        keep = set(args.keep)
        missing = keep - set(values)
        if missing:
            raise ValueError(f"unknown factor_ids in --keep: {sorted(missing)}")
        values = {k: v for k, v in values.items() if k in keep}
        specs = [s for s in specs if s.factor_id in keep]
        total = sum(allocations[k] for k in values)
        allocations = {k: v / total for k, v in allocations.items() if k in keep}
        print(f"subset mode: {len(keep)} factors kept, allocations renormalized", flush=True)

    composite = composite_score(
        values,
        {s.factor_id: int(s.setting["factor_direction"]) for s in specs},
        allocations,
    )

    # Validity gate mirrors member_targets: enough valid names and real spread.
    count = composite.notna().sum(axis=1)
    spread = composite.max(axis=1) - composite.min(axis=1)
    valid = (count >= cfg.min_valid_instruments) & (spread > 0.0)

    unit_profile = replace(profile, gross_exposure=1.0, factor_direction=1)
    weights = target_weights(composite, unit_profile)["long_short"]
    weights.loc[~valid, :] = 0.0
    weights = weights.fillna(0.0)

    combined, risk = combine_targets({"composite": weights}, {"composite": 1.0}, cfg)

    calendar = pd.date_range(cfg.signal_start, tail_end, freq="D", name="date")
    targets = combined.reindex(calendar)
    post_signal = daily_date(cfg.signal_end) + pd.Timedelta(days=1)
    if post_signal in targets.index:
        targets.loc[post_signal] = 0.0

    opens = provider.get_single_data("open", start=calendar[0], end=calendar[-1])
    opens = opens.reindex(index=calendar, columns=targets.columns)
    closes = provider.get_single_data("close", start=calendar[0], end=calendar[-1])
    closes = closes.reindex(index=calendar, columns=targets.columns)
    settlement_prices = pd.DataFrame(float("nan"), index=calendar, columns=targets.columns)
    for instrument, settlement_date in HISTORICAL_CONTRACT_END_DATES.items():
        day = daily_date(settlement_date)
        if day in calendar and instrument in settlement_prices.columns:
            settlement_prices.loc[day, instrument] = closes.loc[day, instrument]
    events = provider.get_funding(start=calendar[0], end=calendar[-1], symbols=targets.columns)
    quality = provider.get_quality(start=calendar[0], end=calendar[-1], symbols=targets.columns)

    accounting = run_target_backtest(
        targets, opens, events, quality, profile,
        signal_start=cfg.signal_start, signal_end=cfg.signal_end,
        settlement_prices=settlement_prices,
    )
    metrics = summarize_accounting(accounting)

    nz = (targets.abs() > 1e-15).sum(axis=1)
    lng = (targets > 1e-15).sum(axis=1)
    sht = (targets < -1e-15).sum(axis=1)
    holdings = {
        "total_median": float(nz.median()),
        "long_median": float(lng.median()),
        "short_median": float(sht.median()),
        "valid_instrument_median": float(count.median()),
        "invalid_days": int((~valid).sum()),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metrics": metrics, "holdings": holdings}, indent=1))
    targets.to_parquet(out.with_suffix(".targets.parquet"))
    composite.to_parquet(out.with_suffix(".composite.parquet"))
    print(json.dumps({"metrics": metrics, "holdings": holdings}, indent=1))


if __name__ == "__main__":
    main()
