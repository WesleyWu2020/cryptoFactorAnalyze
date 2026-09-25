"""Evaluate GP-mined factors using the formal factor_system backtesting framework.

This variant batches factor reconstruction so the same minute data / indicator
cache is reused across multiple factors instead of being rebuilt once per
candidate.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, '/root/crypto-research/common')
# The active GP engine lives under users/wesleywu; common/gp is a stale copy
# (N_PARAMS=14) that cannot decode v3 composition genomes (15 params).
sys.path.insert(0, '/root/crypto-research/users/wesleywu')

from factor_system.data_provider import DataProvider
from factor_system.factor_hub.core.factor_performance_engine import FactorPerformanceEngine
from factor_system.factor_hub.core.factor_result_engine import FactorResultEngine
from gp.minute_gp_system.engines.unified_v2.config import CHUNK_PERIODS
from gp.minute_gp_system.registry import resolve_default_h5_path


DIRECT_REPLAY_CHUNK_PERIODS = 30


def _compact_date(value):
    if value is None:
        return None
    return pd.Timestamp(value).strftime('%Y%m%d')


def _load_pareto_results(pareto_path):
    with open(pareto_path) as f:
        return json.load(f)


def _prepare_loader_metadata(loader, h5_path, minutes_per_period):
    with h5py.File(h5_path, 'r') as f:
        timestamps = f['timestamp'][:]
        symbols = np.array([
            s.decode() if isinstance(s, bytes) else str(s)
            for s in f['symbols'][:]
        ])

    ts0 = timestamps[0]
    scale = 1000 if ts0 > 1e12 else 1
    period_indices = loader.active_period_indices
    minute_ends = np.minimum(
        (period_indices + 1).astype(np.int64) * minutes_per_period - 1,
        loader.n_usable_minutes - 1,
    )
    period_datetimes = pd.to_datetime(
        [datetime.utcfromtimestamp(timestamps[int(m)] / scale) for m in minute_ends]
    )
    coin_symbols = symbols[loader.coin_indices]
    return period_datetimes, coin_symbols


def _create_factor_loader(
    h5_path,
    minutes_per_period,
    start_date,
    end_date,
    backend,
    chunk_periods,
    tradable_mask_overlay_path=None,
    indicator_field_indices=None,
    fundamental_panel_path=None,
):
    from gp.minute_gp_system.engines.unified_v2.backend import set_backend
    from gp.minute_gp_system.engines.unified_v2.data_loader import CryptoDataLoader

    set_backend(backend, 0)
    return CryptoDataLoader(
        h5_path=h5_path,
        minutes_per_period=minutes_per_period,
        chunk_periods=chunk_periods,
        start_date=start_date,
        end_date=end_date,
        backend=backend,
        gpu_id=0,
        tradable_mask_overlay_path=tradable_mask_overlay_path,
        indicator_field_indices=indicator_field_indices,
        fundamental_panel_path=fundamental_panel_path,
    )


def _entry_params_array(entry):
    from gp.minute_gp_system.engines.unified_v2.config import N_PARAMS

    arr = np.asarray(entry['params'], dtype=np.int32)
    if arr.size < N_PARAMS:
        arr = np.concatenate([arr, np.zeros(N_PARAMS - arr.size, dtype=np.int32)])
    return arr


def _entries_field_indices(entries):
    from gp.minute_gp_system.engines.unified_v2.config import (
        CS_COMP_DEPENDENCY_FIELDS,
        CS_COMP_OPS,
        INDICATOR_INDEX,
    )

    rows = [_entry_params_array(entry) for entry in entries]
    if not rows:
        return np.empty(0, dtype=np.intp)

    pop = np.vstack(rows)
    field_indices = set(pop[:, (0, 1, 4)].astype(np.intp, copy=False).ravel())
    for cs_op_idx in np.unique(pop[:, 12]):
        cs_op_name = CS_COMP_OPS[int(cs_op_idx)]
        for field_name in CS_COMP_DEPENDENCY_FIELDS.get(cs_op_name, ()):
            field_indices.add(INDICATOR_INDEX[field_name])
    return np.asarray(sorted(field_indices), dtype=np.intp)


def _project_population_fields(pop, field_indices):
    fields = np.asarray(field_indices, dtype=np.int32)
    projected = np.asarray(pop, dtype=np.int32).copy()
    for col in (0, 1, 4):
        projected[:, col] = np.searchsorted(fields, projected[:, col])
    return projected


def _compute_factor_batch(entries, loader, backend, batch_size=32):
    n_factors = len(entries)
    n_periods = loader.n_active_periods
    n_coins = loader.n_tradable_coins
    factor_values = np.full((n_factors, n_periods, n_coins), np.nan, dtype=np.float32)

    from gp.minute_gp_system.engines.unified_v2.backend import free_gpu
    from gp.minute_gp_system.engines.unified_v2.evaluator import evaluate_population

    loader.precompute_indicators()
    field_indices = getattr(loader, 'indicator_field_indices', None)
    for batch_start in range(0, n_factors, batch_size):
        batch_end = min(batch_start + batch_size, n_factors)
        pop = np.array([_entry_params_array(entry) for entry in entries[batch_start:batch_end]], dtype=np.int32)
        if field_indices is not None:
            pop = _project_population_fields(pop, field_indices)
        day_offset = 0
        for ci in range(loader.n_chunks()):
            ind = loader.get_cached_chunk(ci)
            chunk_f = evaluate_population(pop, ind, indicator_field_indices=field_indices)
            cp = loader.get_chunk_size(ci)
            factor_values[batch_start:batch_end, day_offset:day_offset + cp, :] = chunk_f
            day_offset += cp
            del ind, chunk_f
            free_gpu()
    return factor_values


def _formula_sign(entry):
    formula = str(entry.get('formula') or '').strip()
    if formula.startswith('-('):
        return -1
    return 1


def _metadata_sign(entry):
    factor_direction = int(entry.get('factor_direction', entry.get('ic_sign', 1)))
    return -1 if factor_direction < 0 else 1


def _entry_sign(entry, direction_policy):
    if direction_policy == 'formula':
        return _formula_sign(entry)
    if direction_policy == 'metadata':
        return _metadata_sign(entry)
    if direction_policy == 'none':
        return 1
    raise ValueError(f"unknown direction_policy={direction_policy!r}")


def load_gp_entries_batch(
    entries,
    h5_path,
    start_date='20220102',
    end_date='20251230',
    minutes_per_period=240,
    backend='cpu',
    reconstruction_batch_size=32,
    chunk_periods=CHUNK_PERIODS,
    tradable_mask_overlay_path=None,
    fundamental_panel_path=None,
    direction_policy='formula',
):
    """Recompute multiple GP factors from in-memory entry payloads.

    ``params`` encode the raw GP expression. The saved ``formula`` is the
    canonical displayed factor definition, including a possible leading ``-``.
    Therefore the default policy applies only that formula sign. The historical
    metadata-based policy is kept explicit for legacy artifact audits.
    """
    entries = list(entries or [])
    if not entries:
        return []

    from gp.minute_gp_system.engines.unified_v2.config import N_PARAMS

    param_lengths = {len(np.asarray(entry.get('params', []), dtype=np.int32)) for entry in entries}
    valid_param_lengths = set(range(13, N_PARAMS + 1))
    if not param_lengths.issubset(valid_param_lengths):
        raise ValueError(
            f"Unexpected param length {param_lengths}; expected one of "
            f"{sorted(valid_param_lengths)}."
        )

    indicator_field_indices = _entries_field_indices(entries)
    loader = _create_factor_loader(
        h5_path=h5_path,
        minutes_per_period=minutes_per_period,
        start_date=start_date,
        end_date=end_date,
        backend=backend,
        chunk_periods=chunk_periods,
        tradable_mask_overlay_path=tradable_mask_overlay_path,
        indicator_field_indices=indicator_field_indices,
        fundamental_panel_path=fundamental_panel_path,
    )
    period_datetimes, coin_symbols = _prepare_loader_metadata(loader, h5_path, minutes_per_period)
    factor_values = _compute_factor_batch(
        entries,
        loader,
        backend=backend,
        batch_size=max(1, int(reconstruction_batch_size)),
    )

    tradable_mask = None
    if hasattr(loader, 'get_period_tradable_mask'):
        tradable_mask = loader.get_period_tradable_mask()

    frames = []
    for idx, entry in enumerate(entries):
        fv = factor_values[idx]
        if tradable_mask is not None:
            fv = np.where(tradable_mask, fv, np.nan)
        if _entry_sign(entry, direction_policy) < 0:
            fv = -fv
        df = pd.DataFrame(fv, index=period_datetimes, columns=coin_symbols)
        df.index.name = 'datetime'
        frames.append((df, entry))
    return frames


def load_gp_factors_batch(
    pareto_path,
    h5_path,
    factor_indices,
    start_date='20220102',
    end_date='20251230',
    minutes_per_period=240,
    backend='cpu',
    reconstruction_batch_size=32,
    chunk_periods=CHUNK_PERIODS,
    tradable_mask_overlay_path=None,
    fundamental_panel_path=None,
    direction_policy='formula',
):
    """Recompute multiple GP factors in one pass and return DataFrames."""
    results = _load_pareto_results(pareto_path)
    entries = []
    for factor_idx in factor_indices:
        if factor_idx >= len(results):
            raise ValueError(f"factor_idx {factor_idx} >= {len(results)}")
        entries.append(results[factor_idx])
    return load_gp_entries_batch(
        entries,
        h5_path,
        start_date=start_date,
        end_date=end_date,
        minutes_per_period=minutes_per_period,
        backend=backend,
        reconstruction_batch_size=reconstruction_batch_size,
        chunk_periods=chunk_periods,
        tradable_mask_overlay_path=tradable_mask_overlay_path,
        fundamental_panel_path=fundamental_panel_path,
        direction_policy=direction_policy,
    )


def load_gp_factor(pareto_path, h5_path, factor_idx=0,
                   start_date='20220102', end_date='20251230',
                   minutes_per_period=240, backend='cpu',
                   direction_policy='formula'):
    return load_gp_factors_batch(
        pareto_path,
        h5_path,
        [factor_idx],
        start_date=start_date,
        end_date=end_date,
        minutes_per_period=minutes_per_period,
        backend=backend,
        direction_policy=direction_policy,
    )[0]


def create_framework_context(h5_path, profile_id):
    dp = DataProvider(h5_path=h5_path)
    result_engine = FactorResultEngine(dp)
    perf_engine = FactorPerformanceEngine()

    from factor_system.factor_hub.core.factor_result_profiles import PROFILES

    profile = PROFILES[profile_id]
    return {
        'result_engine': result_engine,
        'perf_engine': perf_engine,
        'periods_per_year': profile['periods_per_year'],
    }


def evaluate_factor_df(factor_df, h5_path, profile_id='perp_4h',
                       start=None, end=None, cost=0.0007,
                       framework_context=None):
    """Run factor DataFrame through the formal backtesting framework."""
    if framework_context is None:
        framework_context = create_framework_context(h5_path, profile_id)

    result_engine = framework_context['result_engine']
    perf_engine = framework_context['perf_engine']
    periods_per_year = framework_context['periods_per_year']

    result = result_engine.calc_result(factor_df, profile_id=profile_id)
    params = {
        'start': pd.Timestamp(start) if start else None,
        'end': pd.Timestamp(end) if end else None,
        'cost': cost,
        'periods_per_year': periods_per_year,
    }
    perf = perf_engine.calc_basic_performance(result, params)
    return perf, result


def _write_factor_cache(cache_dir, rank, factor_df, result):
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    rank_tag = f"rank_{int(rank):03d}"
    values_path = cache_root / f"{rank_tag}_values.parquet"
    rankic_path = cache_root / f"{rank_tag}_rankic.parquet"
    factor_df.astype('float16').to_parquet(values_path, compression='snappy')
    rankics = result.get('rankics', pd.Series(dtype=float))
    if isinstance(rankics, pd.DataFrame):
        rankic_df = rankics
    elif isinstance(rankics, pd.Series):
        rankic_df = rankics.to_frame(name='rankic')
    else:
        rankic_df = pd.Series(rankics, name='rankic').to_frame()
    rankic_df.to_parquet(rankic_path, compression='snappy')
    return str(values_path), str(rankic_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pareto', type=str, required=True,
                        help='Path to pareto_results.json')
    parser.add_argument('--h5_path', type=str, default=resolve_default_h5_path())
    parser.add_argument('--top_n', type=int, default=5,
                        help='Evaluate top N factors')
    parser.add_argument('--profile', type=str, default='perp_1d')
    parser.add_argument('--start', type=str, default='2026-01-01',
                        help='OOS backtest start (must be after the GP train window)')
    parser.add_argument('--end', type=str, default='2026-05-01',
                        help='OOS backtest end')
    parser.add_argument('--cost', type=float, default=0.0015)
    parser.add_argument('--backend', type=str, default='cpu')
    parser.add_argument('--cpu_workers', type=int, default=None,
                        help='Override CPU worker count used by reconstruction/evaluation helpers')
    parser.add_argument('--reconstruction_batch_size', type=int, default=32,
                        help='How many factors to reconstruct together per batch')
    parser.add_argument('--chunk_periods', type=int, default=DIRECT_REPLAY_CHUNK_PERIODS,
                        help='Chunk size used during factor reconstruction; lower reduces peak memory')
    parser.add_argument('--minutes_per_period', type=int, default=1440,
                        help='Factor reconstruction bar size in minutes (must match the GP training bar)')
    parser.add_argument('--factor_indices', nargs='*', default=None,
                        help='Optional zero-based factor indices to replay. If omitted, top_n sequential ranks are used.')
    parser.add_argument('--json_output', type=str, default=None,
                        help='Optional path to write machine-readable replay results.')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Optional directory to persist reconstructed factor values/rankics for downstream review publish reuse.')
    parser.add_argument('--tradable_mask_overlay', type=str, default=None,
                        help='Optional .npy tradable mask overlay; use "none" to disable loader default.')
    parser.add_argument('--fundamental_panel', type=str, default=None,
                        help='Optional CoinMetrics daily panel .npz used by GP fundamental fields.')
    parser.add_argument('--factor_start_date', type=str, default='2022-01-02',
                        help='Earliest date used to reconstruct factor values; kept before OOS for TS warmup.')
    parser.add_argument('--factor_end_date', type=str, default=None,
                        help='Latest date used to reconstruct factor values. Defaults to OOS end to avoid holdout leakage.')
    parser.add_argument('--direction_policy', choices=('formula', 'none', 'metadata'), default='formula',
                        help='How to orient reconstructed GP params. Default formula uses the displayed formula sign.')
    args = parser.parse_args()

    if args.cpu_workers:
        os.environ['GP_CPU_WORKERS'] = str(args.cpu_workers)

    results = _load_pareto_results(args.pareto)
    if args.factor_indices:
        factor_indices: list[int] = []
        for token in args.factor_indices:
            for part in str(token).split(','):
                part = part.strip()
                if part:
                    factor_indices.append(int(part))
    else:
        factor_indices = list(range(min(args.top_n, len(results))))

    print(f"Evaluating {len(factor_indices)} factors from {args.pareto}")
    print(f"Period: {args.start} ~ {args.end}, Profile: {args.profile}, Cost: {args.cost}")
    print("=" * 120)

    factor_batches = load_gp_factors_batch(
        args.pareto,
        args.h5_path,
        factor_indices,
        backend=args.backend,
        reconstruction_batch_size=args.reconstruction_batch_size,
        chunk_periods=args.chunk_periods,
        minutes_per_period=args.minutes_per_period,
        tradable_mask_overlay_path=args.tradable_mask_overlay,
        fundamental_panel_path=args.fundamental_panel,
        start_date=_compact_date(args.factor_start_date),
        end_date=_compact_date(args.factor_end_date or args.end),
        direction_policy=args.direction_policy,
    )
    framework_context = create_framework_context(args.h5_path, args.profile)

    all_perf = []
    for i, (factor_df, entry) in enumerate(factor_batches):
        display_formula = entry.get('formula') or entry.get('raw_formula')
        factor_direction = _entry_sign(entry, args.direction_policy)
        print(f"\n--- Factor {i+1}/{len(factor_batches)} ---")
        print(f"Factor #{entry.get('rank', i+1)}: {display_formula} (applied_direction={factor_direction:+d}, policy={args.direction_policy})")
        try:
            perf, result = evaluate_factor_df(
                factor_df,
                args.h5_path,
                profile_id=args.profile,
                start=args.start,
                end=args.end,
                cost=args.cost,
                framework_context=framework_context,
            )

            keys = [
                'rankic', 'rankicir', 'ls_netret', 'ls_netir', 'ls_net_sharpe',
                'ls_maxdd', 'ls_calmar', 'ls_turnover', 'ls_win_rate', 'coverage',
                'long_netret', 'long_netir', 'long_net_sharpe', 'long_maxdd',
            ]
            print(f"  Formula: {display_formula}")
            for k in keys:
                v = perf.get(k, 'N/A')
                if isinstance(v, float):
                    print(f"  {k:20s}: {v:.4f}")
                else:
                    print(f"  {k:20s}: {v}")

            perf['formula'] = display_formula
            perf['rank'] = entry.get('rank', i + 1)
            perf['factor_index'] = factor_indices[i]
            perf['factor_direction'] = factor_direction
            if args.cache_dir:
                cache_values_path, cache_rankic_path = _write_factor_cache(
                    args.cache_dir,
                    perf['rank'],
                    factor_df,
                    result,
                )
                perf['cache_values_path'] = cache_values_path
                perf['cache_rankic_path'] = cache_rankic_path
            all_perf.append(perf)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if all_perf:
        print("\n" + "=" * 120)
        print(f"{'Rank':>4} | {'RankIC':>7} {'ICIR':>7} | {'LS_Net%':>8} {'LS_IR':>7} {'LS_Shp':>7} | "
              f"{'MaxDD%':>7} {'Calmar':>7} {'TO':>6} | Formula")
        print("-" * 120)
        for p in all_perf:
            print(f"{p.get('rank',0):4d} | "
                  f"{p.get('rankic',0):7.4f} {p.get('rankicir',0):7.2f} | "
                  f"{p.get('ls_netret',0)*100:7.2f}% {p.get('ls_netir',0):7.3f} "
                  f"{p.get('ls_net_sharpe',0):7.3f} | "
                  f"{p.get('ls_maxdd',0)*100:6.2f}% {p.get('ls_calmar',0):7.3f} "
                  f"{p.get('ls_turnover',0):6.3f} | {p.get('formula','')[:60]}")
    if args.json_output:
        os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
        output_payload = {
            'pareto': args.pareto,
            'profile': args.profile,
            'start': args.start,
            'end': args.end,
            'cost': args.cost,
            'results': all_perf,
        }
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)
        print(f"[JSON] {args.json_output}")


if __name__ == '__main__':
    main()
