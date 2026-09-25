"""Reproducible single-factor fixed portfolio orchestration."""
from __future__ import annotations

from dataclasses import asdict, replace

import pandas as pd

from factor_common.backtest import run_target_backtest
from factor_common.data_provider import DataProvider
from factor_common.metrics import summarize_returns
from factor_common.profiles import resolve_profile
from factor_common.storage import snapshot_source_stats
from factor_common.value_engine import compute_factor, value_pipeline_fingerprint
from data.crypto_quant.universe import HISTORICAL_CONTRACT_END_DATES
from portfolio.factor_pool.module_loader import load_members
from portfolio.fixed_config import FixedConfig, daily_date
from portfolio.fixed_provenance import (
    RecordingProvider,
    assert_source_unchanged,
    code_snapshot,
    environment_metadata,
)
from portfolio.portfolio_builder.fixed_weights import combine_targets, member_targets


def summarize_accounting(accounting: dict) -> dict:
    """Return compact, scenario-level metrics from certified ledgers."""
    summaries = {}
    for name, scenario in accounting["scenarios"].items():
        ledger = scenario["ledger"]
        metrics = summarize_returns(ledger["return"], periods_per_year=365)
        complete = scenario["status"] == "complete"
        summaries[name] = {
            "status": scenario["status"],
            "full": metrics if complete else None,
            "known_segment": None if complete else metrics,
            "fee_total": float(ledger["fee"].sum()),
            "slippage_total": float(ledger["slippage"].sum()),
            "funding_total": float(ledger["funding_cashflow"].sum()),
        }
    return summaries


def run_fixed(config: FixedConfig, *, as_of=None) -> dict:
    """Run the fixed single-factor pipeline through the shared accounting core."""
    if not isinstance(config, FixedConfig):
        raise TypeError("config must be a FixedConfig")
    # Round-trip through the public constructor to detach mutable caller data.
    cfg = FixedConfig.from_dict(asdict(config))
    cutoff = daily_date(cfg.as_of if as_of is None else as_of)
    configured_as_of = daily_date(cfg.as_of)
    if cutoff > configured_as_of or cutoff < daily_date(cfg.signal_start):
        raise ValueError("audit cutoff outside configured knowledge interval")
    signal_end = min(daily_date(cfg.signal_end), cutoff)
    tail_end = min(
        daily_date(cfg.signal_end) + pd.Timedelta(days=1 + cfg.rebalance_days), cutoff
    )

    source_before = snapshot_source_stats([cfg.h5_path])
    specs = load_members(cfg)
    provider = RecordingProvider(DataProvider(cfg.h5_path, as_of=cutoff))
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

    values: dict[str, pd.DataFrame] = {}
    members: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, dict] = {}
    allocations: dict[str, float] = {}
    for allocation, spec in zip(cfg.factors, specs):
        matrix, compute_diag = compute_factor(
            spec, provider, start=cfg.signal_start, end=signal_end
        )
        target, member_diag = member_targets(
            matrix, spec, profile, cfg.min_valid_instruments
        )
        values[spec.factor_id] = matrix
        members[spec.factor_id] = target
        allocations[spec.factor_id] = allocation.allocation
        diagnostics[spec.factor_id] = {
            "compute": compute_diag,
            "daily": member_diag,
            "direction": spec.setting["factor_direction"],
            "source_sha256": spec.source_sha256,
        }

    combined, risk = combine_targets(members, allocations, cfg)
    calendar = pd.date_range(cfg.signal_start, tail_end, freq="D", name="date")
    targets = combined.reindex(calendar)
    # Make the first post-signal boundary an explicit cash target.  The
    # accounting layer still gates signal rows by cfg.signal_end; this row is
    # consumed only by its liquidation handling and prevents stale positions
    # from being carried beyond the requested signal window.
    post_signal = daily_date(cfg.signal_end) + pd.Timedelta(days=1)
    if post_signal in targets.index:
        targets.loc[post_signal] = 0.0
    opens = provider.get_single_data("open", start=calendar[0], end=calendar[-1])
    opens = opens.reindex(index=calendar, columns=targets.columns)
    closes = provider.get_single_data("close", start=calendar[0], end=calendar[-1])
    closes = closes.reindex(index=calendar, columns=targets.columns)
    settlement_prices = pd.DataFrame(
        float("nan"), index=calendar, columns=targets.columns
    )
    for instrument, settlement_date in HISTORICAL_CONTRACT_END_DATES.items():
        day = daily_date(settlement_date)
        if day in calendar and instrument in settlement_prices.columns:
            settlement_prices.loc[day, instrument] = closes.loc[day, instrument]
    events = provider.get_funding(
        start=calendar[0], end=calendar[-1], symbols=targets.columns
    )
    quality = provider.get_quality(
        start=calendar[0], end=calendar[-1], symbols=targets.columns
    )
    accounting = run_target_backtest(
        targets,
        opens,
        events,
        quality,
        profile,
        signal_start=cfg.signal_start,
        signal_end=cfg.signal_end,
        settlement_prices=settlement_prices,
    )

    baselines = {}
    for name, member in members.items():
        baseline_targets = member.reindex(calendar)
        if post_signal in baseline_targets.index:
            baseline_targets.loc[post_signal] = 0.0
        baselines[name] = run_target_backtest(
            baseline_targets,
            opens,
            events,
            quality,
            replace(profile, gross_exposure=1.0),
            signal_start=cfg.signal_start,
            signal_end=cfg.signal_end,
            settlement_prices=settlement_prices,
        )

    assert_source_unchanged([cfg.h5_path], source_before)
    if code_snapshot([item.path for item in cfg.factors]) != dict(cfg.code_hashes):
        raise RuntimeError("code changed during portfolio evaluation")
    any_signal = any(item["daily"]["valid_signal"].any() for item in diagnostics.values())
    status = accounting["status"] if any_signal else "no_valid_signals"
    return {
        "status": status,
        "configuration": asdict(cfg),
        "knowledge_cutoff": cutoff.isoformat(),
        "environment": environment_metadata(),
        "source_stat": source_before,
        "input_reads": provider.reads,
        "value_pipeline_fingerprint": value_pipeline_fingerprint(),
        "values": values,
        "member_targets": members,
        "member_diagnostics": diagnostics,
        "targets": targets,
        "risk": risk,
        "accounting": accounting,
        "metrics": summarize_accounting(accounting),
        "baselines": baselines,
    }


__all__ = ["run_fixed", "summarize_accounting"]
