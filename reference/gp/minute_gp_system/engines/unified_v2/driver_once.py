"""One-shot driver: pick next seed, run GP → OOS → publish, update state.

Design:
  - No loop. cron schedules periodic ticks. flock guarantees singleton.
  - On any step failure: log + exit non-zero. Next cron tick retries.
  - Persistent state tracks seed counter + last outcome, so restarts are safe.

Usage (invoked by cron):
    flock -n /tmp/minute_gp_driver.lock \
        python -m gp.minute_gp_system.engines.unified_v2.driver_once

Layout produced per run:
    {RUNS_ROOT}/auto_<ts>_seed<N>/raw/pareto_results.json
    {RUNS_ROOT}/auto_<ts>_seed<N>/formal_replay.json
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

from factor_platform.config import (
    DEFAULT_GP_DRIVER_LOG_PATH,
    DEFAULT_GP_DRIVER_STATE_PATH,
    DEFAULT_GP_RUNS_ROOT,
)

SYSTEM_ROOT = DEFAULT_GP_RUNS_ROOT.parent
RUNS_ROOT = DEFAULT_GP_RUNS_ROOT
STATE_FILE = DEFAULT_GP_DRIVER_STATE_PATH
LOG_FILE = DEFAULT_GP_DRIVER_LOG_PATH
PY = '/root/.pyenv/versions/3.9.0/bin/python'
PKG = 'gp.minute_gp_system.engines.unified_v2'
PRODUCTION_SEARCH_SPACE_PROFILE = 'production_stable_v2'
PRODUCTION_SEARCH_OBJECTIVE_PROFILE = 'annual_worst_v1'
DEFAULT_GP_RESEARCH_H5 = Path('/root/crypto-research/common/data/raw/h5/bybit_linear_1m_unified_gp_research.h5')
DEFAULT_GP_CANONICAL_H5 = Path('/root/crypto-research/common/data/raw/h5/bybit_linear_1m_unified.h5')
DEFAULT_GP_RESEARCH_MINING_MASK = Path('/root/crypto-research/common/data/raw/h5/bybit_tradable_mask_mining_gp_research.npy')
DEFAULT_PRODUCTION_MINING_MASK = Path('/root/crypto-research/common/data/raw/h5/bybit_tradable_mask_mining.npy')
RESOURCE_GUARD_EXIT_CODE = 137
DEFAULT_MINING_TRADABLE_MASK = str(
    DEFAULT_GP_RESEARCH_MINING_MASK
    if DEFAULT_GP_RESEARCH_MINING_MASK.exists()
    else DEFAULT_PRODUCTION_MINING_MASK
)
DEFAULT_FUNDAMENTAL_PANEL = (
    '/root/crypto-research/common/data/factor_platform/fundamentals/'
    'coinmetrics_cap_mrkt_est_daily.npz'
)
DEFAULT_BLOCKING_LOCKS = (
    Path('/root/crypto-research/common/logs/bybit_data/locks/bybit_linear_unified_h5_build.lock'),
    Path('/root/crypto-research/common/logs/bybit_data/locks/bybit_linear_1m_h5_build.lock'),
    Path('/root/crypto-research/common/logs/bybit_data/locks/bybit_tradable_mask_filtered.lock'),
    Path('/root/crypto-research/common/logs/bybit_data/locks/bybit_tradable_mask_mining.lock'),
    Path('/tmp/sg_trader_data_pull.lock'),
)
INTERNAL_DRIVER_LOCK = Path('/tmp/minute_gp_driver_internal.lock')
FULL_KEEP_FIELDS = (
    'funding_vol_30d', 'funding_oi_joint_5d', 'oi_change_5d',
    'amihud_illiq', 'volume_zscore_60d', 'rv_5d', 'rv_20d',
    'path_efficiency', 'intraday_reversal', 'volume_burst_share',
    'beta_btc_60d', 'relative_strength_btc_5d',
    'xbinance_taker_quote_ratio', 'xbinance_taker_pressure', 'xbinance_avg_trade_size',
)
BALANCED_KEEP_FIELDS = (
    'funding_vol_30d', 'funding_oi_joint_5d',
    'amihud_illiq', 'volume_zscore_60d', 'rv_20d',
    'path_efficiency', 'intraday_reversal', 'relative_strength_btc_5d',
    'xbinance_taker_quote_ratio', 'xbinance_taker_pressure', 'xbinance_avg_trade_size',
)
COMPACT_KEEP_FIELDS = (
    'funding_vol_30d', 'amihud_illiq', 'rv_20d',
    'path_efficiency', 'relative_strength_btc_5d', 'xbinance_taker_pressure',
)
TINY_KEEP_FIELDS = (
    'funding_vol_30d', 'rv_20d', 'path_efficiency', 'xbinance_taker_pressure',
)
KEEP_FIELDS_BY_RESOURCE_PROFILE = {
    'fixed': FULL_KEEP_FIELDS,
    'full': FULL_KEEP_FIELDS,
    'balanced': BALANCED_KEEP_FIELDS,
    'compact': COMPACT_KEEP_FIELDS,
    'tiny': TINY_KEEP_FIELDS,
}


@dataclass(frozen=True)
class ResourcePlan:
    name: str
    population: int
    generations: int
    chunk_periods: int
    eval_batch_size: int
    candidate_pool_size: int
    result_target_min: int
    result_target_max: int
    field_slate_size: int
    field_keep_fields: tuple[str, ...]
    child_rss_limit_gb: float


def _env_float(name, default):
    raw = os.environ.get(name, '').strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name, default):
    raw = os.environ.get(name, '').strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        'next_seed': 1,
        'run_count': 0,
        'success_count': 0,
        'fail_count': 0,
        'publish_total': 0,
    }


def _save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _log(line):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f'[{datetime.now(timezone.utc).isoformat()}] {line}\n')


def _try_acquire_internal_lock():
    INTERNAL_DRIVER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_f = open(INTERNAL_DRIVER_LOCK, 'a')
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_f.close()
        return None
    return lock_f


def _rss_gb(pid):
    try:
        statm = Path(f'/proc/{pid}/statm').read_text().split()
        if not statm:
            return 0.0
        page_size = os.sysconf('SC_PAGE_SIZE')
        return (int(statm[1]) * page_size) / 1024.0 / 1024.0 / 1024.0
    except (OSError, ValueError, IndexError):
        return 0.0


def _terminate_process_group(proc, log_f, reason):
    pid = proc.pid
    print(f'[DRIVER_RESOURCE] terminating pid={pid} reason={reason}', file=log_f, flush=True)
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        print(f'[DRIVER_RESOURCE] SIGTERM failed pid={pid}: {exc}', file=log_f, flush=True)
        return
    try:
        proc.wait(timeout=30.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError as exc:
        print(f'[DRIVER_RESOURCE] SIGKILL failed pid={pid}: {exc}', file=log_f, flush=True)
        return
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        print(f'[DRIVER_RESOURCE] SIGKILL timeout pid={pid}', file=log_f, flush=True)


def _run_subprocess(cmd, log_path, extra_env=None, rss_limit_gb=0.0, min_available_gb=0.0):
    env = os.environ.copy()
    existing_pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = (
        '/root/crypto-research/common'
        if not existing_pythonpath
        else f'/root/crypto-research/common{os.pathsep}{existing_pythonpath}'
    )
    env.setdefault('GP_ARCHIVE_SOURCE', 'active_approved')
    env.setdefault('GP_ARCHIVE_FEATURE_CACHE', '1')
    env.setdefault('GP_ENABLE_ADVANCED_PERIOD_FIELDS', '1')
    env.setdefault('GP_ENABLE_ORDER_FLOW', '0')
    env.setdefault('GP_ENABLE_XBINANCE_ORDER_FLOW', 'auto')
    if DEFAULT_GP_RESEARCH_H5.exists():
        env.setdefault('GP_H5_PATH', str(DEFAULT_GP_RESEARCH_H5))
    elif DEFAULT_GP_CANONICAL_H5.exists():
        env.setdefault('GP_H5_PATH', str(DEFAULT_GP_CANONICAL_H5))
    env.setdefault('GP_FIELD_SLATE_MODE', 'seeded')
    env.setdefault('GP_FIELD_SLATE_SIZE', '26')
    env.setdefault('GP_FIELD_SLATE_KEEP_FIELDS', ','.join(FULL_KEEP_FIELDS))
    env.setdefault(
        'GP_FIELD_SLATE_MIN_FAMILY_FIELDS',
        'xbinance_order_flow:3,intraday_path:2,btc_relative:2,'
        'period_funding:2,period_oi:1,realized_vol:2,period_liquidity:2',
    )
    env.setdefault('MALLOC_ARENA_MAX', '2')
    if extra_env:
        for key, value in extra_env.items():
            if value is None:
                env.pop(str(key), None)
            else:
                env[str(key)] = str(value)
    low_available_limit = max(1, _env_int('GP_DRIVER_CHILD_AVAILABLE_LOW_SAMPLES', 10))
    low_available_seen = 0
    with open(log_path, 'w') as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd='/root/crypto-research/common',
            env=env,
            start_new_session=True,
        )
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            child_rss = _rss_gb(proc.pid)
            available_gb = _available_memory_gb()
            if rss_limit_gb > 0 and child_rss > rss_limit_gb:
                _terminate_process_group(
                    proc,
                    log_f,
                    f'rss_gb={child_rss:.1f}>limit_gb={rss_limit_gb:.1f}',
                )
                return RESOURCE_GUARD_EXIT_CODE
            if (
                min_available_gb > 0
                and available_gb is not None
                and available_gb < min_available_gb
            ):
                low_available_seen += 1
                if low_available_seen >= low_available_limit:
                    _terminate_process_group(
                        proc,
                        log_f,
                        (
                            f'mem_available_gb={available_gb:.1f}<floor_gb={min_available_gb:.1f} '
                            f'samples={low_available_seen}/{low_available_limit}'
                        ),
                    )
                    return RESOURCE_GUARD_EXIT_CODE
            else:
                low_available_seen = 0
            time.sleep(3.0)


def _available_memory_gb():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return float(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        return None
    return None


def _locked_paths(paths):
    locked = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            with open(p, 'a') as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    locked.append(str(p))
                else:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            locked.append(f'{p} probe_error={exc}')
    return locked


def _resource_guard_reasons(min_available_gb, block_locks):
    reasons = []
    locked = _locked_paths(block_locks)
    if locked:
        reasons.append('blocking_locks=' + ','.join(locked))
    available_gb = _available_memory_gb()
    if min_available_gb > 0 and available_gb is not None and available_gb < min_available_gb:
        reasons.append(f'mem_available_gb={available_gb:.1f}<min_available_gb={min_available_gb:.1f}')
    return reasons, available_gb


def _resource_plan(args, available_gb):
    mode = os.environ.get('GP_DRIVER_RESOURCE_PROFILE', 'auto').strip().lower()
    explicit_field_slate = _env_int('GP_FIELD_SLATE_SIZE', 26)
    if mode in {'0', 'false', 'no', 'off', 'none', 'fixed'}:
        limit = _env_float('GP_DRIVER_CHILD_RSS_LIMIT_GB', 0.0)
        return ResourcePlan(
            name='fixed',
            population=args.population,
            generations=args.generations,
            chunk_periods=args.chunk_periods,
            eval_batch_size=args.eval_batch_size,
            candidate_pool_size=args.candidate_pool_size,
            result_target_min=args.result_target_min,
            result_target_max=args.result_target_max,
            field_slate_size=explicit_field_slate,
            field_keep_fields=KEEP_FIELDS_BY_RESOURCE_PROFILE['fixed'],
            child_rss_limit_gb=limit,
        )
    if mode in {'full', 'balanced', 'compact', 'tiny'}:
        bucket = mode
    elif available_gb is None or available_gb >= 110.0:
        bucket = 'full'
    elif available_gb >= 95.0:
        bucket = 'balanced'
    elif available_gb >= 82.0:
        bucket = 'compact'
    else:
        bucket = 'tiny'

    presets = {
        'full': (2000, 15, 30, 32, 160, 32, 48, 26, 62.0),
        'balanced': (1500, 12, 30, 28, 128, 24, 40, 24, 52.0),
        'compact': (1000, 10, 24, 24, 96, 16, 32, 22, 44.0),
        'tiny': (800, 8, 20, 16, 64, 8, 20, 20, 38.0),
    }
    pop, gen, chunk, batch, pool, target_min, target_max, slate, rss_limit = presets[bucket]
    if available_gb is not None:
        rss_limit = min(rss_limit, max(24.0, available_gb - 16.0))
    rss_override = _env_float('GP_DRIVER_CHILD_RSS_LIMIT_GB', 0.0)
    if rss_override > 0:
        rss_limit = rss_override
    final_target_max = min(args.result_target_max, target_max)
    final_target_min = min(args.result_target_min, target_min, final_target_max)
    return ResourcePlan(
        name=bucket,
        population=min(args.population, pop),
        generations=min(args.generations, gen),
        chunk_periods=min(args.chunk_periods, chunk),
        eval_batch_size=min(args.eval_batch_size, batch),
        candidate_pool_size=min(args.candidate_pool_size, pool),
        result_target_min=final_target_min,
        result_target_max=final_target_max,
        field_slate_size=min(explicit_field_slate, slate),
        field_keep_fields=KEEP_FIELDS_BY_RESOURCE_PROFILE[bucket],
        child_rss_limit_gb=rss_limit,
    )


def _parse_publish_count(publish_log):
    # Parse "  new appended to queue: N" from publish_pipeline.py output
    try:
        text = Path(publish_log).read_text()
    except Exception:
        return 0
    for line in text.splitlines():
        if 'new appended to queue' in line:
            parts = line.rsplit(':', 1)
            if len(parts) == 2:
                try:
                    return int(parts[1].strip())
                except ValueError:
                    return 0
    return 0


def _run_one(population, generations, top_quantile, trading_cost,
             minutes_per_period, start_date, end_date,
             oos_start, oos_end, oos_profile, chunk_periods,
             eval_batch_size, reconstruction_batch_size, replay_chunk_periods,
             candidate_pool_size, result_target_min, result_target_max,
             tradable_mask_overlay,
             fundamental_panel,
             perf_profile,
             runtime_env,
             child_rss_limit_gb,
             child_available_floor_gb,
             seed):
    seed = int(seed)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    experiment = f'auto_{ts}_seed{seed}'
    run_dir = RUNS_ROOT / experiment
    raw_dir = run_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    pareto_path = raw_dir / 'pareto_results.json'
    formal_path = run_dir / 'formal_replay.json'
    gp_log = run_dir / 'gp.log'
    oos_log = run_dir / 'oos.log'
    pub_log = run_dir / 'publish.log'
    gp_h5_path = (
        os.environ.get('GP_DRIVER_H5_PATH')
        or os.environ.get('GP_H5_PATH')
        or (str(DEFAULT_GP_RESEARCH_H5) if DEFAULT_GP_RESEARCH_H5.exists() else None)
        or (str(DEFAULT_GP_CANONICAL_H5) if DEFAULT_GP_CANONICAL_H5.exists() else None)
    )

    _log(
        f'{experiment} START  seed={seed} '
        f'search_space_profile={PRODUCTION_SEARCH_SPACE_PROFILE} '
        f'search_objective_profile={PRODUCTION_SEARCH_OBJECTIVE_PROFILE} '
        f'h5_path={gp_h5_path or "registry_default"} '
        f'tradable_mask_overlay={tradable_mask_overlay or "loader_default"} '
        f'fundamental_panel={fundamental_panel or "none"}'
    )

    # 1. GP search
    gp_cmd = [
        PY, '-u', '-m', f'{PKG}.run',
        '--start_date', start_date, '--end_date', end_date,
        '--period_minutes', str(minutes_per_period),
        '--chunk_periods', str(chunk_periods),
        '--population', str(population),
        '--generations', str(generations),
        '--backend', 'gpu', '--gpu_id', '0',
        '--cache_dtype', 'float16',
        '--eval_batch_size', str(eval_batch_size),
        '--result_target_min', str(result_target_min),
        '--result_target_max', str(result_target_max),
        '--candidate_pool_size', str(candidate_pool_size),
        '--top_quantile', str(top_quantile),
        '--trading_cost', str(trading_cost),
        '--seed', str(seed),
        '--output_dir', str(raw_dir),
    ]
    if gp_h5_path:
        gp_cmd.extend(['--h5_path', str(gp_h5_path)])
    if perf_profile:
        gp_cmd.append('--perf_profile')
    if tradable_mask_overlay:
        gp_cmd.extend(['--tradable_mask_overlay', str(tradable_mask_overlay)])
    if fundamental_panel:
        gp_cmd.extend(['--fundamental_panel', str(fundamental_panel)])
    rc = _run_subprocess(
        gp_cmd,
        gp_log,
        extra_env=runtime_env,
        rss_limit_gb=child_rss_limit_gb,
        min_available_gb=child_available_floor_gb,
    )
    if rc != 0:
        _log(f'{experiment} FAIL gp exit={rc}  see {gp_log}')
        raise SystemExit(rc)
    if not pareto_path.exists():
        _log(f'{experiment} FAIL gp no pareto_results.json')
        raise SystemExit(10)

    # 2. OOS formal replay
    oos_cmd = [
        PY, '-u', '-m', f'{PKG}.evaluate_with_framework',
        '--pareto', str(pareto_path),
        '--start', oos_start, '--end', oos_end,
        '--cost', str(trading_cost),
        '--profile', oos_profile,
        '--minutes_per_period', str(minutes_per_period),
        '--top_n', str(result_target_max),
        '--cpu_workers', '12',
        '--reconstruction_batch_size', str(reconstruction_batch_size),
        '--chunk_periods', str(replay_chunk_periods),
        '--factor_start_date', start_date,
        '--factor_end_date', oos_end,
        '--json_output', str(formal_path),
    ]
    if gp_h5_path:
        oos_cmd.extend(['--h5_path', str(gp_h5_path)])
    if tradable_mask_overlay:
        oos_cmd.extend(['--tradable_mask_overlay', str(tradable_mask_overlay)])
    if fundamental_panel:
        oos_cmd.extend(['--fundamental_panel', str(fundamental_panel)])
    rc = _run_subprocess(
        oos_cmd,
        oos_log,
        extra_env=runtime_env,
        rss_limit_gb=child_rss_limit_gb,
        min_available_gb=child_available_floor_gb,
    )
    if rc != 0:
        _log(f'{experiment} FAIL oos exit={rc}  see {oos_log}')
        raise SystemExit(rc)
    if not formal_path.exists():
        _log(f'{experiment} FAIL oos no formal_replay.json')
        raise SystemExit(11)

    # 3. Publish to review queue
    pub_cmd = [
        PY, '-u', '-m', f'{PKG}.publish_pipeline',
        '--pareto', str(pareto_path),
        '--formal', str(formal_path),
        '--experiment', experiment,
    ]
    rc = _run_subprocess(pub_cmd, pub_log, extra_env=runtime_env)
    if rc != 0:
        _log(f'{experiment} FAIL publish exit={rc}  see {pub_log}')
        raise SystemExit(rc)

    n_new = _parse_publish_count(pub_log)
    _log(f'{experiment} OK    seed={seed} published={n_new}')
    return experiment, n_new


def main():
    parser = argparse.ArgumentParser(description='Run one GP→OOS→publish cycle.')
    parser.add_argument('--population', type=int, default=2000)
    parser.add_argument('--generations', type=int, default=15)
    parser.add_argument('--minutes_per_period', type=int, default=1440)
    parser.add_argument('--top_quantile', type=float, default=0.2)
    parser.add_argument('--trading_cost', type=float, default=0.0015)
    parser.add_argument('--start_date', type=str, default='20220102')
    parser.add_argument('--end_date', type=str, default='20251231')
    parser.add_argument('--oos_start', type=str, default='2026-01-01')
    parser.add_argument('--oos_end', type=str, default='2026-05-01')
    parser.add_argument('--oos_profile', type=str, default='perp_1d')
    parser.add_argument('--chunk_periods', type=int, default=30)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--reconstruction_batch_size', type=int, default=8)
    parser.add_argument('--replay_chunk_periods', type=int, default=60)
    parser.add_argument('--candidate_pool_size', type=int, default=160)
    parser.add_argument('--result_target_min', type=int, default=32)
    parser.add_argument('--result_target_max', type=int, default=48)
    parser.add_argument('--tradable_mask_overlay', type=str, default=DEFAULT_MINING_TRADABLE_MASK)
    parser.add_argument('--fundamental_panel', type=str,
                        default=os.environ.get('GP_DRIVER_FUNDAMENTAL_PANEL') or os.environ.get('GP_FUNDAMENTAL_PANEL'),
                        help='Optional CoinMetrics daily panel .npz; pass none/off to disable.')
    parser.add_argument('--no_perf_profile', action='store_true')
    parser.add_argument('--min_available_gb', type=float,
                        default=_env_float('GP_DRIVER_MIN_AVAILABLE_GB', 55.0),
                        help='Skip this cron tick when MemAvailable is below this GiB threshold.')
    parser.add_argument('--block_lock', action='append', default=[],
                        help='Extra flock path that should block GP launch when locked.')
    parser.add_argument('--no_resource_guard', action='store_true',
                        help='Disable memory and external-build lock checks.')
    parser.add_argument('--child_available_floor_gb', type=float,
                        default=_env_float('GP_DRIVER_CHILD_AVAILABLE_FLOOR_GB', 12.0),
                        help='Terminate the GP/OOS child before host OOM when MemAvailable drops below this GiB floor.')
    args = parser.parse_args()

    internal_lock = _try_acquire_internal_lock()
    if internal_lock is None:
        msg = f'SKIP internal driver lock held: {INTERNAL_DRIVER_LOCK}'
        _log(msg)
        print(msg)
        return 0

    fundamental_panel = str(args.fundamental_panel or '').strip()
    if fundamental_panel.lower() in {'', '0', 'false', 'none', 'off'}:
        fundamental_panel = None
    elif not Path(fundamental_panel).exists():
        _log(f'fundamental panel missing, disabling fundamentals: {fundamental_panel}')
        fundamental_panel = None

    state = _load_state()
    seed_used = int(state['next_seed'])
    if not args.no_resource_guard:
        block_locks = list(DEFAULT_BLOCKING_LOCKS) + [Path(p) for p in args.block_lock]
        reasons, available_gb = _resource_guard_reasons(args.min_available_gb, block_locks)
        if reasons:
            state['skip_count'] = int(state.get('skip_count', 0)) + 1
            state['last_run_at'] = datetime.now(timezone.utc).isoformat()
            state['last_run_status'] = 'skipped_resource_guard'
            state['last_seed'] = seed_used
            state['last_skip_reason'] = '; '.join(reasons)
            _save_state(state)
            msg = (
                f'SKIP resource_guard seed={seed_used} '
                f'mem_available_gb={available_gb:.1f} '
                f'reason={state["last_skip_reason"]}'
                if available_gb is not None
                else f'SKIP resource_guard seed={seed_used} reason={state["last_skip_reason"]}'
            )
            _log(msg)
            print(msg)
            return 0
    else:
        available_gb = _available_memory_gb()

    plan = _resource_plan(args, available_gb)
    runtime_env = {
        'GP_FIELD_SLATE_SIZE': str(plan.field_slate_size),
        'GP_FIELD_SLATE_KEEP_FIELDS': ','.join(plan.field_keep_fields),
    }
    if plan.name == 'tiny':
        runtime_env.update({
            'GP_CHUNK_REUSE_BATCHES': '1',
            'GP_V2_GPU_EVAL_WORKERS': '1',
        })
    elif plan.name == 'compact':
        runtime_env.update({
            'GP_CHUNK_REUSE_BATCHES': '4',
            'GP_V2_GPU_EVAL_WORKERS': '2',
        })
    child_available_floor_gb = float(args.child_available_floor_gb)
    if 'GP_DRIVER_CHILD_AVAILABLE_FLOOR_GB' not in os.environ:
        if plan.name == 'tiny':
            child_available_floor_gb = min(child_available_floor_gb, 8.0)
        elif plan.name == 'compact':
            child_available_floor_gb = min(child_available_floor_gb, 10.0)
    _log(
        f'RESOURCE_PLAN seed={seed_used} profile={plan.name} '
        f'mem_available_gb={available_gb:.1f} '
        f'population={plan.population} generations={plan.generations} '
        f'field_slate_size={plan.field_slate_size} '
        f'candidate_pool_size={plan.candidate_pool_size} '
        f'result_target={plan.result_target_min}-{plan.result_target_max} '
        f'child_rss_limit_gb={plan.child_rss_limit_gb:.1f} '
        f'child_available_floor_gb={child_available_floor_gb:.1f}'
        if available_gb is not None
        else (
            f'RESOURCE_PLAN seed={seed_used} profile={plan.name} '
            f'population={plan.population} generations={plan.generations} '
            f'field_slate_size={plan.field_slate_size} '
            f'candidate_pool_size={plan.candidate_pool_size} '
            f'result_target={plan.result_target_min}-{plan.result_target_max} '
            f'child_rss_limit_gb={plan.child_rss_limit_gb:.1f} '
            f'child_available_floor_gb={child_available_floor_gb:.1f}'
        )
    )

    # Reserve the seed before the long GP/OOS/publish chain. If cron timeout,
    # host OOM, or manual kill terminates the process before final state save,
    # the next tick still advances instead of replaying the same seed.
    state['run_count'] = int(state.get('run_count', 0)) + 1
    state['next_seed'] = seed_used + 1
    state['last_run_at'] = datetime.now(timezone.utc).isoformat()
    state['last_run_status'] = 'running'
    state['last_seed'] = seed_used
    _save_state(state)
    try:
        experiment, n_new = _run_one(
            population=plan.population,
            generations=plan.generations,
            top_quantile=args.top_quantile,
            trading_cost=args.trading_cost,
            minutes_per_period=args.minutes_per_period,
            start_date=args.start_date,
            end_date=args.end_date,
            oos_start=args.oos_start,
            oos_end=args.oos_end,
            oos_profile=args.oos_profile,
            chunk_periods=plan.chunk_periods,
            eval_batch_size=plan.eval_batch_size,
            reconstruction_batch_size=args.reconstruction_batch_size,
            replay_chunk_periods=args.replay_chunk_periods,
            candidate_pool_size=plan.candidate_pool_size,
            result_target_min=plan.result_target_min,
            result_target_max=plan.result_target_max,
            tradable_mask_overlay=args.tradable_mask_overlay,
            fundamental_panel=fundamental_panel,
            perf_profile=not args.no_perf_profile,
            runtime_env=runtime_env,
            child_rss_limit_gb=plan.child_rss_limit_gb,
            child_available_floor_gb=child_available_floor_gb,
            seed=seed_used,
        )
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
        if exit_code == RESOURCE_GUARD_EXIT_CODE:
            state['skip_count'] = int(state.get('skip_count', 0)) + 1
            state['next_seed'] = seed_used
            state['last_run_at'] = datetime.now(timezone.utc).isoformat()
            state['last_run_status'] = 'skipped_runtime_resource_guard'
            state['last_seed'] = seed_used
            state['last_skip_reason'] = (
                f'child_exit={RESOURCE_GUARD_EXIT_CODE}; seed will be retried'
            )
            _save_state(state)
            msg = (
                f'SKIP runtime_resource_guard seed={seed_used} '
                f'retry_seed={seed_used}'
            )
            _log(msg)
            print(msg)
            return 0
        state['fail_count'] = int(state.get('fail_count', 0)) + 1
        state['last_run_at'] = datetime.now(timezone.utc).isoformat()
        state['last_run_status'] = 'failed'
        state['last_seed'] = seed_used
        _save_state(state)
        raise

    state['success_count'] = int(state.get('success_count', 0)) + 1
    state['publish_total'] = int(state.get('publish_total', 0)) + int(n_new)
    state['last_run_at'] = datetime.now(timezone.utc).isoformat()
    state['last_run_status'] = 'success'
    state['last_seed'] = seed_used
    state['last_experiment'] = experiment
    _save_state(state)
    print(f'DONE  experiment={experiment}  seed={seed_used}  published={n_new}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
