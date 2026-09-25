"""Compute one traceable daily target-weight signal from a frozen strategy."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from factor_common.data_provider import DataProvider
from factor_common.grouping import assign_groups
from factor_common.loader import load_factor
from factor_common.profiles import resolve_profile
from factor_common.validation import scan_future_leaks
from factor_common.value_engine import compute_factor, value_pipeline_fingerprint
from portfolio.portfolio_builder.composite import composite_score, composite_targets
from portfolio.portfolio_builder.fixed_weights import combine_targets, member_targets
from portfolio.fixed_provenance import environment_metadata

from .config import DailyStrategyConfig


@dataclass(frozen=True)
class _RiskLimits:
    gross_limit: float
    long_limit: float
    short_limit: float
    single_limit: float
    net_limit: float


def is_rebalance_signal_day(config: DailyStrategyConfig, signal_date) -> bool:
    signal_day = pd.Timestamp(signal_date).normalize()
    execution_day = signal_day + pd.Timedelta(days=config.signal_delay_days)
    return (execution_day - pd.Timestamp(config.anchor_date)).days % config.rebalance_days == 0


def _long(frame: pd.DataFrame, name: str, signal_date) -> pd.DataFrame:
    row = frame.loc[[pd.Timestamp(signal_date)]].copy()
    result = row.T.rename_axis("instrument").reset_index()
    result = result.rename(columns={row.index[0]: name})
    result.insert(0, "signal_date", row.index[0])
    return result


def _compute_factor_job(args):
    """Spawn-safe worker: compute one full factor matrix in a subprocess.

    Factor values are causal and deterministic, so computing members in
    parallel yields exactly the same matrices as the serial loop. When a
    cache_path is given, the worker persists the rebuilt matrix itself, so
    every completed rebuild is checkpointed on disk even if the run is
    killed (e.g. by an execution-window timeout) mid-way.
    """
    factor_path, h5_path, calc_start, day_iso, cache_path = args
    day = pd.Timestamp(day_iso)
    spec = load_factor(factor_path)
    provider = DataProvider(h5_path, as_of=day)
    matrix, compute_diag = compute_factor(spec, provider, start=calc_start, end=day)
    if cache_path is not None:
        _write_factor_cache(Path(cache_path), matrix)
        if isinstance(compute_diag, dict):
            compute_diag = {**compute_diag, "factor_cache": "rebuilt"}
    return str(factor_path), matrix, compute_diag


FACTOR_CACHE_ROOT = (
    Path(__file__).resolve().parents[1] / "outputs" / "daily_pipeline" / "factor_cache"
)
FACTOR_CACHE_OVERLAP_DAYS = 21


def _factor_cache_path(config: DailyStrategyConfig, member, spec) -> Path:
    return (
        FACTOR_CACHE_ROOT
        / spec.factor_id
        / f"{member.sha256[:16]}_{config.calculation_start}.parquet"
    )


def _frames_identical(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Exact equality treating NaN == NaN (same pipeline, same data => bitwise equal)."""
    right = right.reindex(index=left.index, columns=left.columns)
    if right.shape != left.shape:
        return False
    equal = (left == right) | (left.isna() & right.isna())
    return bool(equal.all().all())


def _write_factor_cache(path: Path, matrix: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    matrix.to_parquet(temporary)
    os.replace(temporary, path)


def _compute_one_factor(config: DailyStrategyConfig, member, day: pd.Timestamp, provider: DataProvider):
    """Serial compute of one member, with an incremental on-disk cache.

    A cached matrix is extended by recomputing only a trailing window; the
    overlap between cache and fresh tail must match exactly (NaN-safe). That
    check empirically proves on every run that the factor is causal with
    bounded warmup before any stitched value is used.
    """
    spec = load_factor(member.path)
    if getattr(config, "use_factor_cache", False):
        path = _factor_cache_path(config, member, spec)
        if path.is_file():
            cached = pd.read_parquet(path)
            cache_end = cached.index.max()
            if cache_end >= day:
                return spec, cached.loc[:day], {"factor_cache": "hit"}
            tail_start = max(
                pd.Timestamp(config.calculation_start),
                cache_end - pd.Timedelta(days=FACTOR_CACHE_OVERLAP_DAYS),
            )
            tail, diag = compute_factor(
                spec, provider, start=tail_start.date().isoformat(), end=day
            )
            overlap = cached.loc[tail_start:cache_end]
            if _frames_identical(overlap, tail.loc[tail_start:cache_end]):
                stitched = pd.concat(
                    [cached, tail.loc[cache_end + pd.Timedelta(days=1):]]
                )
                _write_factor_cache(path, stitched)
                if isinstance(diag, dict):
                    diag = {**diag, "factor_cache": "stitched"}
                return spec, stitched, diag
            # Overlap mismatch (e.g. revised upstream history): fall through
            # to a full recompute, which also rebuilds the cache.
    matrix, diag = compute_factor(spec, provider, start=config.calculation_start, end=day)
    if getattr(config, "use_factor_cache", False):
        _write_factor_cache(_factor_cache_path(config, member, spec), matrix)
        if isinstance(diag, dict):
            diag = {**diag, "factor_cache": "rebuilt"}
    return spec, matrix, diag


def _compute_factor_matrices(config: DailyStrategyConfig, day: pd.Timestamp, provider: DataProvider) -> dict:
    """Return {factor_path: (spec, matrix, compute_diag)} for every member.

    Cache hits and incremental trailing stitches stay serial: they are cheap
    and memory-light. Full rebuilds — cold cache or a factor-source sha
    change invalidating the cache key — are the expensive case (minutes per
    factor over the full history), so they fan out over a spawn pool when
    parallel_workers > 1. Each rebuild worker writes its own cache file, so
    per-factor checkpointing survives a killed run exactly like the serial
    path. Peak memory is bounded by parallel_workers, not by factor count.
    """
    workers = getattr(config, "parallel_workers", 1)
    use_cache = getattr(config, "use_factor_cache", False)
    rebuilds = []
    results = {}
    for member in config.factors:
        spec = load_factor(member.path)
        cache_path = _factor_cache_path(config, member, spec) if use_cache else None
        if cache_path is not None and cache_path.is_file():
            continue  # cache hit or cheap trailing stitch: serial loop below
        rebuilds.append((member, cache_path))
    if workers > 1 and len(rebuilds) > 1:
        from multiprocessing import get_context

        jobs = [
            (str(member.path), str(config.h5_path), config.calculation_start,
             day.isoformat(), None if cache_path is None else str(cache_path))
            for member, cache_path in rebuilds
        ]
        with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            for path, matrix, compute_diag in pool.map(_compute_factor_job, jobs):
                results[path] = (load_factor(path), matrix, compute_diag)
    else:
        for member, _cache_path in rebuilds:
            spec, matrix, compute_diag = _compute_one_factor(config, member, day, provider)
            results[str(member.path)] = (spec, matrix, compute_diag)
    for member in config.factors:
        if str(member.path) in results:
            continue
        spec, matrix, compute_diag = _compute_one_factor(config, member, day, provider)
        results[str(member.path)] = (spec, matrix, compute_diag)
    return results


def compute_daily_signal(config: DailyStrategyConfig, signal_date, readiness: dict) -> dict:
    """Calculate factors, five groups, member weights and the combined target."""
    day = pd.Timestamp(signal_date)
    if day.tzinfo is not None:
        day = day.tz_convert("UTC").tz_localize(None)
    day = day.normalize()
    if not readiness.get("ready"):
        raise ValueError("data is not ready for signal production")
    if day < pd.Timestamp(config.calculation_start):
        raise ValueError("signal date precedes calculation_start")

    findings = scan_future_leaks([item.path for item in config.factors])
    if findings:
        raise ValueError(f"factor source future-leak findings: {findings}")
    provider = DataProvider(config.h5_path, as_of=day)
    profile = resolve_profile("perp_1d", {
        "n_groups": config.n_groups,
        "rebalance_days": config.rebalance_days,
        "anchor_date": config.anchor_date,
        "gross_exposure": config.gross_limit,
    })
    values_by_factor = {}
    groups_by_factor = {}
    targets_by_factor = {}
    diagnostics = {}
    allocations = {}
    composite_mode = getattr(config, "combine_mode", "members") == "composite"
    factor_results = _compute_factor_matrices(config, day, provider)
    for member in config.factors:
        spec, matrix, compute_diag = factor_results[str(member.path)]
        values_by_factor[spec.factor_id] = matrix
        allocations[spec.factor_id] = member.allocation
        diagnostics[spec.factor_id] = {
            "compute": compute_diag,
            "direction": spec.setting["factor_direction"],
        }
        if composite_mode:
            continue
        groups, group_diag = assign_groups(matrix, config.n_groups)
        targets, target_diag = member_targets(
            matrix, spec, profile, config.min_valid_instruments
        )
        groups_by_factor[spec.factor_id] = groups
        targets_by_factor[spec.factor_id] = targets
        diagnostics[spec.factor_id].update({
            "valid_count": int(target_diag.loc[day, "valid_count"]),
            "valid_signal": bool(target_diag.loc[day, "valid_signal"]),
            "reason": str(target_diag.loc[day, "reason"]),
            "group_insufficient": day in group_diag["insufficient_dates"],
        })
    if composite_mode:
        from dataclasses import replace as _replace

        score = composite_score(
            values_by_factor,
            {name: item["direction"] for name, item in diagnostics.items()},
            allocations,
        )
        unit_profile = _replace(profile, gross_exposure=1.0, factor_direction=1)
        weights, score_diag = composite_targets(
            score, unit_profile, config.min_valid_instruments
        )
        targets_by_factor = {"composite": weights}
        diagnostics["composite"] = {
            "valid_count": int(score_diag.loc[day, "valid_count"]),
            "valid_signal": bool(score_diag.loc[day, "valid_signal"]),
            "reason": str(score_diag.loc[day, "reason"]),
        }
        allocations = {"composite": 1.0}
    if not all(item["valid_signal"] for item in diagnostics.values() if "valid_signal" in item):
        invalid = [name for name, item in diagnostics.items()
                   if "valid_signal" in item and not item["valid_signal"]]
        raise ValueError(f"invalid member signal(s): {', '.join(invalid)}")

    limits = _RiskLimits(config.gross_limit, config.long_limit, config.short_limit,
                         config.single_limit, config.net_limit)
    combined, risk = combine_targets(targets_by_factor, allocations, limits)
    target_row = combined.loc[day].fillna(0.0)
    action = "rebalance" if is_rebalance_signal_day(config, day) else "hold"
    positions = [
        {"instrument": str(instrument), "target_weight": float(weight)}
        for instrument, weight in target_row.items() if abs(float(weight)) > 1e-15
    ] if action == "rebalance" else []
    positions.sort(key=lambda item: item["instrument"])
    execution_day = day + pd.Timedelta(days=config.signal_delay_days)
    signal_core = {
        "schema_version": 1,
        "strategy_id": config.strategy_id,
        "strategy_version": config.strategy_version,
        "signal_date": day.date().isoformat(),
        "execution_date": execution_day.date().isoformat(),
        "execute_after": execution_day.strftime("%Y-%m-%dT00:00:00Z"),
        "expires_at": (execution_day + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
        "action": action,
        "targets": positions,
    }
    signal_id = hashlib.sha256(
        json.dumps(signal_core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    generated_at = datetime.now(timezone.utc).isoformat()
    signal = {**signal_core, "signal_id": signal_id, "generated_at": generated_at,
              "data_cutoff": day.date().isoformat()}
    return {
        "signal": signal,
        "readiness": readiness,
        "factor_values": pd.concat(
            [_long(frame, "factor_value", day).assign(factor_id=name)
             for name, frame in values_by_factor.items()], ignore_index=True
        ),
        "factor_groups": pd.concat(
            [_long(frame, "group", day).assign(factor_id=name)
             for name, frame in groups_by_factor.items()], ignore_index=True
        ) if groups_by_factor else pd.DataFrame(
            columns=["signal_date", "instrument", "group", "factor_id"]
        ),
        "member_targets": pd.concat(
            [_long(frame, "member_weight", day).assign(factor_id=name)
             for name, frame in targets_by_factor.items()], ignore_index=True
        ),
        "combined_targets": _long(combined, "target_weight", day),
        "risk": risk.loc[[day]],
        "member_diagnostics": diagnostics,
        "manifest": {
            "status": "complete",
            "strategy_id": config.strategy_id,
            "strategy_version": config.strategy_version,
            "source_report": str(config.source_report),
            "source_report_config_sha256": config.source_report_config_sha256,
            "signal_id": signal_id,
            "signal_date": day.date().isoformat(),
            "generated_at": generated_at,
            "value_pipeline_fingerprint": value_pipeline_fingerprint(),
            "environment": environment_metadata(),
        },
    }
