"""Backtest the composite strategy before/after dropping GP_7d96363e567c2813.

The degenerate factor's legacy wrapper can no longer be loaded (semantics
gate), so both scenarios are evaluated from the cached factor matrices in
``portfolio/output/composite_cache`` (built 2026-09-20 with the strategy's
frozen config). The six re-exported GP factors were verified numerically
identical to the legacy runtime (max_abs_diff <= 6.4e-14), so cached values
are exactly what the current runtime produces.

Scenario A ("before"): 8 members x 0.125, including GP_7d96363e567c2813.
Scenario B ("after"):  7 members x 1/7, the composition now live in
daily_pipeline/configs/portfolio_composite_daily1.json.

Both use the frozen report config (window 2024-01-01..2026-08-25, fees,
slippage, funding, 5-day rebalance) so the only difference is membership.

Usage: ./.venv/bin/python scripts/backtest_composite_7vs8.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factor_common.backtest import run_target_backtest  # noqa: E402
from factor_common.data_provider import DataProvider  # noqa: E402
from factor_common.grouping import target_weights  # noqa: E402
from factor_common.profiles import resolve_profile  # noqa: E402
from data.crypto_quant.universe import HISTORICAL_CONTRACT_END_DATES  # noqa: E402
from portfolio.fixed_config import FixedConfig, daily_date  # noqa: E402
from portfolio.fixed_pipeline import summarize_accounting  # noqa: E402
from portfolio.portfolio_builder.composite import composite_score  # noqa: E402
from portfolio.portfolio_builder.fixed_weights import combine_targets  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "portfolio" / "20260920T080140Z_756193dec1b9"
CACHE_DIR = PROJECT_ROOT / "portfolio" / "output" / "composite_cache"
DROPPED = "GP_7d96363e567c2813"
# Live composite membership (portfolio_composite_daily1.json), directions from
# the export manifests (unchanged by the re-export).
MEMBERS_8 = {
    "GP_0006a47b612eaf9b": -1,
    "GP_064185107a8f267e": -1,
    "GP_3eaff470cdb222f5": 1,
    "GP_77282e814dac81a8": -1,
    "GP_7d96363e567c2813": 1,
    "GP_a37c60cf4492e9f1": -1,
    "GP_ed2c09f43439ff39": 1,
    "Retail_Friction_Illiquidity_Factor": -1,
}


def run_scenario(name, members, cfg, provider, profile):
    values = {fid: pd.read_parquet(CACHE_DIR / f"{fid}.parquet") for fid in members}
    allocations = {fid: 1.0 / len(members) for fid in members}
    composite = composite_score(values, {k: int(v) for k, v in members.items()}, allocations)

    count = composite.notna().sum(axis=1)
    spread = composite.max(axis=1) - composite.min(axis=1)
    valid = (count >= cfg.min_valid_instruments) & (spread > 0.0)

    unit_profile = replace(profile, gross_exposure=1.0, factor_direction=1)
    weights = target_weights(composite, unit_profile)["long_short"]
    weights.loc[~valid, :] = 0.0
    weights = weights.fillna(0.0)
    combined, risk = combine_targets({"composite": weights}, {"composite": 1.0}, cfg)

    cutoff = daily_date(cfg.as_of)
    tail_end = min(daily_date(cfg.signal_end) + pd.Timedelta(days=1 + cfg.rebalance_days), cutoff)
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
    holdings = {
        "total_median": float(nz.median()),
        "invalid_days": int((~valid).sum()),
    }
    print(f"scenario {name} done", flush=True)
    return {"metrics": metrics, "holdings": holdings}, targets, accounting


def main() -> int:
    cfg = FixedConfig.from_dict(json.loads((REPORT_DIR / "config.json").read_text()))
    cutoff = daily_date(cfg.as_of)
    provider = DataProvider(cfg.h5_path, as_of=cutoff)
    profile = resolve_profile("perp_1d", {
        "n_groups": cfg.n_groups,
        "rebalance_days": cfg.rebalance_days,
        "anchor_date": cfg.anchor_date,
        "fee_rate": cfg.fee_rate,
        "slippage": cfg.slippage,
        "gross_exposure": cfg.gross_limit,
        "initial_equity": cfg.initial_equity,
        "include_funding": True,
        "funding_price_mode": "strict",
    })

    members_7 = {k: v for k, v in MEMBERS_8.items() if k != DROPPED}
    result_a, targets_a, acc_a = run_scenario("before_8f", MEMBERS_8, cfg, provider, profile)
    result_b, targets_b, acc_b = run_scenario("after_7f", members_7, cfg, provider, profile)

    overlap = (targets_a.abs() > 1e-15) & (targets_b.abs() > 1e-15)
    union = (targets_a.abs() > 1e-15) | (targets_b.abs() > 1e-15)
    overlap_ratio = float((overlap.sum(axis=1) / union.sum(axis=1).replace(0, np.nan)).median())

    out = {
        "window": {"signal_start": cfg.signal_start, "signal_end": cfg.signal_end,
                   "as_of": cfg.as_of, "rebalance_days": cfg.rebalance_days},
        "before_8f": result_a,
        "after_7f": result_b,
        "daily_holding_overlap_median": overlap_ratio,
    }
    out_path = PROJECT_ROOT / "portfolio" / "output" / "composite_7vs8_20260924.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
