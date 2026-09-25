"""Crypto fitness v2: net-performance primitives + phenotype utilities.

Public API used by evolution.py:
  build_lagged_returns, prepare_fitness_context,
  compute_five_objectives_prepared, apply_shortcoming_penalty,
  sample_diversity_features, standardize_phenotypes,
  gpu_fitness_enabled, materialize_fitness_context,
  prepare_residual_basis_context, compute_residual_rankic, compute_rankic_novelty.

Objective slot 3 is intentionally left neutral here. Population-level
low-correlation / novelty is computed upstream, where cross-factor
phenotypes are available. Residual-IC (incremental RankIC after removing a
production-factor basis) is also computed upstream via the two
prepare_residual_basis_context / compute_residual_rankic helpers.
"""
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from threading import Lock

from .backend import get_backend, xp, to_xp, to_numpy
from .config import (
    TOP_QUANTILE, LAG_PERIODS, PERIODS_PER_YEAR,
    TRADING_COST,
    TURNOVER_HINGE_TARGET,
    ROLLING_IC_WINDOW, SHORTCOMING_PERCENTILE,
    DIVERSITY_MAX_PERIODS, MIN_SIGNAL_COVERAGE,
    MIN_EFFECTIVE_BARS, MIN_EFFECTIVE_BAR_RATIO,
    MIN_IC_VALID_BARS, MIN_RANKICIR,
    MIN_IC_POSITIVE_RATE, MAX_LS_TURNOVER,
    MAX_LONG_TURNOVER, MIN_LS_POSITIVE_RATE,
)


def _int_from_env(names, default):
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value >= 1:
            return value
    return default


N_WORKERS = _int_from_env(
    ("GP_V2_FITNESS_WORKERS", "GP_CPU_WORKERS"),
    max(1, os.cpu_count() or 4),
)
_EPS = 1e-10
_FITNESS_EXECUTOR = None
_FITNESS_EXECUTOR_LOCK = Lock()
_GPU_FITNESS_ENV = "GP_V2_GPU_FITNESS"
_GPU_ROW_PREFIX_MAX_KERNEL = None
_AUX_NAMES = (
    'ic_direction',
    'ic_mean',
    'rankicir',
    'positive_rate',
    'coverage',
    'n_valid',
    'effective_bars',
    'ls_turnover',
    'long_turnover',
    'ls_positive_rate',
    'ls_netret_ann',
)


def gpu_fitness_enabled():
    return os.environ.get(_GPU_FITNESS_ENV) == "1"


def _perf_add(perf_stats, key, value):
    if perf_stats is not None:
        perf_stats[key] = perf_stats.get(key, 0.0) + float(value)


def materialize_fitness_context(fitness_context):
    if fitness_context is None:
        return None

    idx, ret_slice, tradable_slice, ret_rank_cache = fitness_context
    idx_out = to_xp(np.asarray(idx, dtype=np.int32))
    ret_slice_out = to_xp(np.asarray(ret_slice, dtype=np.float32))
    tradable_slice_out = None
    if tradable_slice is not None:
        tradable_slice_out = to_xp(np.asarray(tradable_slice, dtype=bool))

    ret_rank_cache_out = None
    if ret_rank_cache is not None:
        ret_rank_cache_out = (
            to_xp(np.asarray(ret_rank_cache[0], dtype=np.float32)),
            to_xp(np.asarray(ret_rank_cache[1], dtype=bool)),
        )
    return idx_out, ret_slice_out, tradable_slice_out, ret_rank_cache_out


def _get_fitness_executor():
    global _FITNESS_EXECUTOR
    if _FITNESS_EXECUTOR is None:
        with _FITNESS_EXECUTOR_LOCK:
            if _FITNESS_EXECUTOR is None:
                _FITNESS_EXECUTOR = ThreadPoolExecutor(max_workers=N_WORKERS)
    return _FITNESS_EXECUTOR


def _row_nanmedian_2d(x):
    valid = np.isfinite(x)
    counts = valid.sum(axis=1)
    filled = np.where(valid, x, np.inf)
    filled.sort(axis=1)

    out = np.full((x.shape[0], 1), np.nan, dtype=np.float32)
    odd_rows = np.flatnonzero((counts % 2) == 1)
    if odd_rows.size:
        out[odd_rows, 0] = filled[odd_rows, counts[odd_rows] // 2]

    even_rows = np.flatnonzero((counts > 0) & ((counts % 2) == 0))
    if even_rows.size:
        hi = counts[even_rows] // 2
        lo = hi - 1
        out[even_rows, 0] = (
            filled[even_rows, lo] + filled[even_rows, hi]) * np.float32(0.5)
    return out


def _row_nanmean_2d(x):
    valid = np.isfinite(x)
    counts = valid.sum(axis=1)
    totals = np.sum(np.where(valid, x, 0.0), axis=1, dtype=np.float32)
    out = np.full((x.shape[0], 1), np.nan, dtype=np.float32)
    ok = counts > 0
    if np.any(ok):
        out[ok, 0] = totals[ok] / counts[ok].astype(np.float32)
    return out


def _row_nanstd_2d(x, mean=None):
    if mean is None:
        mean = _row_nanmean_2d(x)
    valid = np.isfinite(x)
    counts = valid.sum(axis=1)
    centered = np.where(valid, x - mean, 0.0)
    var_sum = np.sum(centered * centered, axis=1, dtype=np.float32)
    out = np.full((x.shape[0], 1), np.nan, dtype=np.float32)
    ok = counts > 0
    if np.any(ok):
        out[ok, 0] = var_sum[ok] / counts[ok].astype(np.float32)
    return np.sqrt(np.maximum(out, 0.0))


def _postprocess_single(factor_2d):
    med = _row_nanmedian_2d(factor_2d)
    mad = _row_nanmedian_2d(np.abs(factor_2d - med))
    ok = mad > _EPS
    lo, hi = med - 5 * mad, med + 5 * mad
    out = np.where(ok, np.clip(factor_2d, lo, hi), factor_2d)
    m = _row_nanmean_2d(out)
    s = _row_nanstd_2d(out, mean=m)
    s = np.where(s < _EPS, 1.0, s)
    return (out - m) / s


def _rank_ic_single(factor_pp, ret_slice, tradable_slice=None, ret_rank_cache=None):
    P, S = factor_pp.shape
    valid = np.isfinite(factor_pp) & np.isfinite(ret_slice)
    if tradable_slice is not None:
        valid &= tradable_slice
    n_valid = valid.sum(axis=1).astype(np.float32)

    f_fill = np.where(valid, factor_pp, 1e18)
    f_rank = np.argsort(np.argsort(f_fill, axis=1), axis=1).astype(np.float32)
    f_rank = np.where(valid, f_rank, 0.0)
    fm = np.sum(f_rank, axis=1, keepdims=True) / (n_valid[:, None] + _EPS)
    fd = np.where(valid, f_rank - fm, 0.0)

    if ret_rank_cache is not None:
        r_rank_full, r_valid_full = ret_rank_cache
        r_rank = np.where(valid, r_rank_full, 0.0)
        rm = np.sum(r_rank, axis=1, keepdims=True) / (n_valid[:, None] + _EPS)
        rd = np.where(valid, r_rank - rm, 0.0)
    else:
        r_fill = np.where(valid, ret_slice, 1e18)
        r_rank = np.argsort(np.argsort(r_fill, axis=1), axis=1).astype(np.float32)
        r_rank = np.where(valid, r_rank, 0.0)
        rm = np.sum(r_rank, axis=1, keepdims=True) / (n_valid[:, None] + _EPS)
        rd = np.where(valid, r_rank - rm, 0.0)

    cov = np.sum(fd * rd, axis=1) / (n_valid + _EPS)
    sf = np.sqrt(np.sum(fd**2, axis=1) / (n_valid + _EPS) + _EPS)
    sr = np.sqrt(np.sum(rd**2, axis=1) / (n_valid + _EPS) + _EPS)
    ic = cov / (sf * sr + _EPS)
    return np.where(n_valid >= 20, ic, np.nan)



def compute_rankic_novelty(candidate_rankic, library_rankic, min_valid=60, library_chunk=128):
    """Return ``1 - max(abs(correlation))`` against a production RankIC library.

    Both inputs are daily RankIC series: candidates are ``(C, T)`` and the
    library is ``(T, K)``. Correlations use pairwise finite observations and
    the same Pearson/variance guards as ``FactorComparator``; ``min_valid``
    adds a configurable reliability floor. A
    candidate with too few observations is invalid (``-1e6``); a valid
    candidate with no usable library column is neutral (``1.0``).
    """
    candidates = np.asarray(candidate_rankic, dtype=np.float32)
    library = np.asarray(library_rankic, dtype=np.float32)
    if candidates.ndim == 1:
        candidates = candidates[None, :]
    if library.ndim == 1:
        library = library[:, None]
    if candidates.ndim != 2 or library.ndim != 2:
        raise ValueError("candidate_rankic and library_rankic must be 1D or 2D")
    if candidates.shape[1] != library.shape[0]:
        raise ValueError(
            "candidate_rankic periods must match library_rankic rows: "
            f"{candidates.shape[1]} != {library.shape[0]}"
        )

    n_candidates = candidates.shape[0]
    novelty = np.full(n_candidates, -1e6, dtype=np.float32)
    candidate_valid = np.isfinite(candidates)
    candidate_counts = candidate_valid.sum(axis=1)
    candidate_ok = candidate_counts >= int(min_valid)
    if not np.any(candidate_ok) or library.shape[1] == 0:
        return novelty

    max_abs_corr = np.full(n_candidates, -np.inf, dtype=np.float64)
    for start in range(0, library.shape[1], max(1, int(library_chunk))):
        stop = min(start + max(1, int(library_chunk)), library.shape[1])
        base = library[:, start:stop].astype(np.float64, copy=False)
        base_valid = np.isfinite(base)
        pair_valid = candidate_valid[:, :, None] & base_valid[None, :, :]
        counts = pair_valid.sum(axis=1)
        x = np.where(pair_valid, candidates[:, :, None], 0.0).astype(np.float64, copy=False)
        y = np.where(pair_valid, base[None, :, :], 0.0)
        counts_f = counts.astype(np.float64, copy=False)
        sx = x.sum(axis=1)
        sy = y.sum(axis=1)
        sxx = (x * x).sum(axis=1)
        syy = (y * y).sum(axis=1)
        sxy = (x * y).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_x = sx / counts_f
            mean_y = sy / counts_f
            cov = sxy / counts_f - mean_x * mean_y
            var_x = sxx / counts_f - mean_x * mean_x
            var_y = syy / counts_f - mean_y * mean_y
            corr = cov / np.sqrt(var_x * var_y)
        usable = (counts >= int(min_valid)) & (var_x > 1e-20) & (var_y > 1e-20)
        corr = np.where(usable, np.abs(corr), -np.inf)
        max_abs_corr = np.maximum(max_abs_corr, np.max(corr, axis=1))

    comparable = candidate_ok & np.isfinite(max_abs_corr)
    novelty[candidate_ok & ~np.isfinite(max_abs_corr)] = 1.0
    novelty[comparable] = (1.0 - np.clip(max_abs_corr[comparable], 0.0, 1.0)).astype(np.float32)
    return novelty


def _precompute_ret_rank(ret_slice, tradable_slice=None):
    r_valid = np.isfinite(ret_slice)
    if tradable_slice is not None:
        r_valid &= tradable_slice
    r_fill = np.where(r_valid, ret_slice, 1e18)
    r_rank = np.argsort(np.argsort(r_fill, axis=1), axis=1).astype(np.float32)
    r_rank = np.where(r_valid, r_rank, 0.0)
    return r_rank, r_valid


def _signal_pct_rank_stats_single(factor_pp, tradable_slice=None):
    valid = np.isfinite(factor_pp)
    if tradable_slice is not None:
        valid &= tradable_slice
    signal_counts = valid.sum(axis=1).astype(np.float32)
    fill = np.where(valid, factor_pp, 1e18)
    ranks = np.argsort(np.argsort(fill, axis=1), axis=1).astype(np.float32) + 1.0
    pct_ranks = ranks / np.maximum(signal_counts[:, None], 1.0)
    pct_ranks = np.where(valid, pct_ranks, np.nan)
    return valid, signal_counts, pct_ranks


def build_lagged_returns(period_returns, lag=LAG_PERIODS):
    if lag > 0:
        ret_lagged = np.full_like(period_returns, np.nan)
        ret_lagged[:-lag] = period_returns[lag:]
        return ret_lagged
    return period_returns


def prepare_fitness_context(lagged_returns, is_mask=None, tradable_mask=None):
    P = lagged_returns.shape[0]
    if is_mask is None:
        is_mask = np.ones(P, dtype=bool)

    idx = np.where(is_mask)[0]
    if len(idx) < 60:
        return None

    ret_slice = lagged_returns[idx]
    tradable_slice = tradable_mask[idx] if tradable_mask is not None else None
    ret_rank_cache = _precompute_ret_rank(ret_slice, tradable_slice=tradable_slice)
    return idx, ret_slice, tradable_slice, ret_rank_cache


def _required_effective_bars(n_periods):
    return max(
        int(MIN_EFFECTIVE_BARS),
        int(np.ceil(max(n_periods - 1, 1) * MIN_EFFECTIVE_BAR_RATIO)),
    )


def _empty_aux_matrix(pop):
    aux = np.full((pop, len(_AUX_NAMES)), np.nan, dtype=np.float32)
    aux[:, 0] = 1.0
    aux[:, 5] = 0.0
    aux[:, 6] = 0.0
    return aux


def _aux_matrix_to_dict(aux_matrix):
    return {
        name: aux_matrix[:, i].astype(np.float32, copy=False)
        for i, name in enumerate(_AUX_NAMES)
    }


def _normalize_direction_np(direction):
    direction = np.asarray(direction, dtype=np.float32)
    return np.where(direction < 0.0, np.float32(-1.0), np.float32(1.0))


def _min_soft_scale(value, target):
    if not np.isfinite(value):
        return np.float32(_EPS)
    if target <= _EPS:
        return np.float32(1.0)
    return np.float32(np.clip(value / target, _EPS, 1.0))


def _max_soft_scale(value, target):
    if not np.isfinite(value):
        return np.float32(_EPS)
    if target <= _EPS:
        return np.float32(1.0)
    if value <= target:
        return np.float32(1.0)
    return np.float32(np.clip(target / value, _EPS, 1.0))


def _signed_soft_scale(value, scale):
    scale = float(np.clip(scale, _EPS, 1.0))
    return float(value * scale) if value >= 0.0 else float(value / scale)


def _safe_mean_valid(x):
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.float32(np.nan)
    return np.float32(np.mean(x[valid], dtype=np.float32))


def _resolve_ic_direction_single(ic_all):
    mean_ic = _safe_mean_valid(ic_all)
    if not np.isfinite(mean_ic):
        return np.float32(1.0)
    return np.float32(-1.0 if mean_ic < 0.0 else 1.0)


def _compute_ic_quality_single(ic_all, periods_per_year):
    valid = np.isfinite(ic_all)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return np.float32(np.nan), np.float32(np.nan), np.float32(np.nan), 0

    ic_valid = ic_all[valid].astype(np.float32, copy=False)
    ic_mean = np.float32(np.mean(ic_valid, dtype=np.float32))
    ic_std = np.float32(np.std(ic_valid, dtype=np.float32))
    if ic_std > _EPS:
        rankicir = np.float32(ic_mean / ic_std * np.sqrt(periods_per_year))
    else:
        rankicir = np.float32(0.0)
    positive_rate = np.float32(np.mean(ic_valid > 0.0, dtype=np.float32))
    return ic_mean, rankicir, positive_rate, n_valid


def _build_target_memberships_single(factor_pp, top_frac, tradable_slice=None):
    P, S = factor_pp.shape
    valid, signal_counts, pct_ranks = _signal_pct_rank_stats_single(
        factor_pp, tradable_slice=tradable_slice)
    if tradable_slice is not None:
        tradable_counts = tradable_slice.sum(axis=1).astype(np.float32)
    else:
        tradable_counts = np.full(P, S, dtype=np.float32)

    # Use the per-period available universe, not the global symbol count.
    # On Bybit, tradable names can be far below total listed symbols; using the
    # global count here can make every period fail the holding-size gate.
    available_counts = np.minimum(signal_counts, tradable_counts).astype(np.float32)
    min_k = np.maximum(np.ceil(available_counts * top_frac).astype(np.int32) // 2, 5)
    sufficient = signal_counts >= (min_k.astype(np.float32) * 2.0)
    target_long = valid & (pct_ranks >= (1.0 - top_frac)) & sufficient[:, None]
    target_short = valid & (pct_ranks <= top_frac) & sufficient[:, None]
    coverage = signal_counts / np.maximum(tradable_counts, 1.0)
    return target_long, target_short, coverage.astype(np.float32), min_k.astype(np.float32)


def _simulate_ls_holdings_single(factor_pp, ret_slice, top_frac, tradable_slice=None):
    target_long, target_short, coverage, min_k = _build_target_memberships_single(
        factor_pp, top_frac, tradable_slice=tradable_slice)
    P = factor_pp.shape[0]

    prev_long = np.zeros_like(target_long)
    prev_short = np.zeros_like(target_short)
    prev_long[1:] = target_long[:-1]
    prev_short[1:] = target_short[:-1]

    ret_valid = np.isfinite(ret_slice)
    held_long = prev_long & ret_valid
    held_short = prev_short & ret_valid

    prev_long_cnt = prev_long.sum(axis=1).astype(np.float32)
    prev_short_cnt = prev_short.sum(axis=1).astype(np.float32)
    long_cnt = held_long.sum(axis=1).astype(np.float32)
    short_cnt = held_short.sum(axis=1).astype(np.float32)

    long_sum = np.where(held_long, ret_slice, 0.0).sum(axis=1)
    short_sum = np.where(held_short, ret_slice, 0.0).sum(axis=1)
    long_rets = np.where(long_cnt > 0, long_sum / (long_cnt + _EPS), 0.0)
    short_rets = np.where(short_cnt > 0, -(short_sum / (short_cnt + _EPS)), 0.0)

    long_turn = np.zeros(P, dtype=np.float32)
    short_turn = np.zeros(P, dtype=np.float32)
    long_turn_valid = prev_long_cnt > 0
    short_turn_valid = prev_short_cnt > 0
    if long_turn_valid.any():
        long_turn[long_turn_valid] = (
            (prev_long[long_turn_valid] != target_long[long_turn_valid]).sum(axis=1) /
            (prev_long_cnt[long_turn_valid] + _EPS)
        )
    if short_turn_valid.any():
        short_turn[short_turn_valid] = (
            (prev_short[short_turn_valid] != target_short[short_turn_valid]).sum(axis=1) /
            (prev_short_cnt[short_turn_valid] + _EPS)
        )

    effective = (
        (prev_long_cnt >= min_k) &
        (prev_short_cnt >= min_k) &
        (long_cnt >= min_k) &
        (short_cnt >= min_k)
    )
    ls_ret = np.full(P, np.nan, dtype=np.float32)
    ls_turn = np.full(P, np.nan, dtype=np.float32)
    if effective.any():
        ls_ret[effective] = (long_rets[effective] + short_rets[effective]) * np.float32(0.5)
        ls_turn[effective] = (long_turn[effective] + short_turn[effective]) * np.float32(0.5)

    effective_count = int(effective.sum())
    if effective_count > 0:
        long_turn_mean = float(np.mean(long_turn[effective], dtype=np.float32))
    else:
        long_turn_mean = np.nan

    return (
        ls_ret,
        ls_turn,
        float(np.mean(coverage, dtype=np.float32)),
        effective_count,
        float(long_turn_mean),
    )


def _max_drawdown_1d(cum_returns):
    peak = np.maximum.accumulate(cum_returns)
    dd = 1.0 - cum_returns / (peak + _EPS)
    return float(np.nanmax(dd))


def _estimate_turnover(factor_pp, top_frac, tradable_slice=None):
    """Estimate average turnover of top/bottom quantile portfolios."""
    P, S = factor_pp.shape
    k = max(1, int(S * top_frac))
    if P < 2:
        return 1.0

    valid = np.isfinite(factor_pp)
    if tradable_slice is not None:
        valid &= tradable_slice
    fv = np.where(valid, factor_pp, -np.inf)
    eligible_rows = np.flatnonzero(valid.sum(axis=1) >= k * 2)
    if eligible_rows.size < 2:
        return 1.0

    fv = fv[eligible_rows]
    top_idx = np.argpartition(fv, -k, axis=1)[:, -k:]
    bot_idx = np.argpartition(-fv, -k, axis=1)[:, -k:]

    row_idx = np.arange(eligible_rows.size)[:, None]
    top_membership = np.zeros((eligible_rows.size, S), dtype=bool)
    bot_membership = np.zeros((eligible_rows.size, S), dtype=bool)
    top_membership[row_idx, top_idx] = True
    bot_membership[row_idx, bot_idx] = True

    top_overlap = (top_membership[1:] & top_membership[:-1]).sum(axis=1)
    bot_overlap = (bot_membership[1:] & bot_membership[:-1]).sum(axis=1)
    turnover = 1.0 - (top_overlap + bot_overlap) / (2.0 * k)
    return float(np.mean(turnover))


def _compute_one(factor_2d, ret_slice, top_frac, periods_per_year, rolling_window,
                 tradable_slice=None, ret_rank_cache=None,
                 ic_direction=None, return_aux=False, return_rankic_series=False,
                 trading_cost=TRADING_COST):
    obj = np.full(5, -1e6, dtype=np.float32)
    aux = _empty_aux_matrix(1)[0]

    factor_pp_raw = _postprocess_single(factor_2d)
    ic_all_raw = _rank_ic_single(
        factor_pp_raw, ret_slice, tradable_slice=tradable_slice, ret_rank_cache=ret_rank_cache)
    ic_direction_value = (
        _resolve_ic_direction_single(ic_all_raw)
        if ic_direction is None
        else _normalize_direction_np([ic_direction])[0]
    )
    factor_pp = factor_pp_raw * ic_direction_value
    ic_all = ic_all_raw * ic_direction_value

    ic_mean, rankicir, positive_rate, ic_valid = _compute_ic_quality_single(
        ic_all, periods_per_year)
    aux[0] = ic_direction_value
    aux[1] = ic_mean
    aux[2] = rankicir
    aux[3] = positive_rate
    aux[5] = float(ic_valid)
    if ic_valid < int(MIN_IC_VALID_BARS):
        return (obj, aux, ic_all) if return_aux and return_rankic_series else ((obj, aux) if return_aux else (obj, ic_all) if return_rankic_series else obj)

    ls_ret, ls_turnover, coverage_mean, effective_bars, long_turnover_mean = (
        _simulate_ls_holdings_single(
            factor_pp, ret_slice, top_frac, tradable_slice=tradable_slice)
    )
    aux[4] = np.float32(coverage_mean)
    aux[6] = float(effective_bars)
    aux[7] = np.float32(_safe_mean_valid(ls_turnover))
    aux[8] = np.float32(long_turnover_mean)

    required_bars = _required_effective_bars(len(ls_ret))
    if coverage_mean < MIN_SIGNAL_COVERAGE or effective_bars < required_bars:
        return (obj, aux, ic_all) if return_aux and return_rankic_series else ((obj, aux) if return_aux else (obj, ic_all) if return_rankic_series else obj)

    ls_ret_net = ls_ret - ls_turnover * np.float32(trading_cost / 2.0)
    ls_valid = int(np.sum(np.isfinite(ls_ret)))
    if ls_valid < required_bars:
        return (obj, aux, ic_all) if return_aux and return_rankic_series else ((obj, aux) if return_aux else (obj, ic_all) if return_rankic_series else obj)

    ls_mean = float(np.nanmean(ls_ret_net))
    ls_std = float(np.nanstd(ls_ret_net))
    ls_positive_rate = float(np.mean(ls_ret_net[np.isfinite(ls_ret_net)] > 0.0)) if ls_valid > 0 else np.nan
    ann_return = float(ls_mean * periods_per_year)

    ls_filled = np.nan_to_num(ls_ret_net, nan=0.0)
    cum = np.cumprod(1.0 + ls_filled)
    max_dd = _max_drawdown_1d(cum)

    turnover_pressure = float(np.nanmax([aux[7], aux[8]]))
    aux[9] = np.float32(ls_positive_rate)
    aux[10] = np.float32(ann_return)

    obj[0] = float(rankicir) if np.isfinite(rankicir) else -1e6
    obj[1] = float(
        ls_mean / max(ls_std, _EPS) * np.sqrt(periods_per_year)
        if ls_std > _EPS else 0.0
    )
    obj[2] = -max(0.0, turnover_pressure - TURNOVER_HINGE_TARGET) if np.isfinite(turnover_pressure) else -1e6
    obj[3] = 0.0
    obj[4] = float(1.0 - max_dd)

    if return_aux:
        return (obj, aux, ic_all) if return_rankic_series else (obj, aux)
    if return_rankic_series:
        return obj, ic_all
    return obj


def _compute_one_from_index(i, daily_factors, idx, ret_slice, top_frac,
                            periods_per_year, rolling_window,
                            tradable_slice, ret_rank_cache,
                            ic_directions, return_aux, return_rankic_series,
                            trading_cost):
    return _compute_one(
        daily_factors[i, idx, :], ret_slice, top_frac,
        periods_per_year, rolling_window,
        tradable_slice=tradable_slice,
        ret_rank_cache=ret_rank_cache,
        ic_direction=None if ic_directions is None else ic_directions[i],
        return_aux=return_aux,
        return_rankic_series=return_rankic_series,
        trading_cost=trading_cost)


def _gpu_row_nanmedian_3d(x):
    import cupy as cp

    valid = cp.isfinite(x)
    counts = valid.sum(axis=2).astype(cp.int32)
    filled = cp.where(valid, x, cp.float32(cp.inf))
    filled.sort(axis=2)

    out = cp.full((x.shape[0], x.shape[1], 1), cp.nan, dtype=cp.float32)

    odd_mask = (counts % 2) == 1
    if int(cp.count_nonzero(odd_mask).item()) > 0:
        b_idx, p_idx = cp.where(odd_mask)
        out[b_idx, p_idx, 0] = filled[b_idx, p_idx, counts[b_idx, p_idx] // 2]

    even_mask = (counts > 0) & ((counts % 2) == 0)
    if int(cp.count_nonzero(even_mask).item()) > 0:
        b_idx, p_idx = cp.where(even_mask)
        hi = counts[b_idx, p_idx] // 2
        lo = hi - 1
        out[b_idx, p_idx, 0] = (
            filled[b_idx, p_idx, lo] + filled[b_idx, p_idx, hi]) * cp.float32(0.5)
    return out


def _gpu_postprocess_batched(factors_gpu):
    import cupy as cp

    med = _gpu_row_nanmedian_3d(factors_gpu)
    mad = _gpu_row_nanmedian_3d(cp.abs(factors_gpu - med))
    ok = mad > _EPS
    lo = med - 5 * mad
    hi = med + 5 * mad
    clipped = cp.where(ok, cp.clip(factors_gpu, lo, hi), factors_gpu)
    mean = cp.nanmean(clipped, axis=2, keepdims=True)
    std = cp.nanstd(clipped, axis=2, keepdims=True)
    std = cp.where(std < _EPS, cp.float32(1.0), std)
    return (clipped - mean) / std


def _gpu_rank_ic_batched(factor_pp_gpu, ret_slice_gpu,
                         tradable_slice_gpu=None, ret_rank_cache_gpu=None):
    import cupy as cp

    valid = cp.isfinite(factor_pp_gpu) & cp.isfinite(ret_slice_gpu[None, :, :])
    if tradable_slice_gpu is not None:
        valid &= tradable_slice_gpu[None, :, :]
    n_valid = valid.sum(axis=2).astype(cp.float32)

    f_fill = cp.where(valid, factor_pp_gpu, cp.float32(1e18))
    f_rank = cp.argsort(cp.argsort(f_fill, axis=2), axis=2).astype(cp.float32)
    f_rank = cp.where(valid, f_rank, cp.float32(0.0))
    f_mean = cp.sum(f_rank, axis=2, keepdims=True) / (n_valid[:, :, None] + _EPS)
    f_diff = cp.where(valid, f_rank - f_mean, cp.float32(0.0))

    if ret_rank_cache_gpu is not None:
        r_rank_full, _ = ret_rank_cache_gpu
        r_rank = cp.where(valid, r_rank_full[None, :, :], cp.float32(0.0))
        r_mean = cp.sum(r_rank, axis=2, keepdims=True) / (n_valid[:, :, None] + _EPS)
        r_diff = cp.where(valid, r_rank - r_mean, cp.float32(0.0))
    else:
        r_fill = cp.where(valid, ret_slice_gpu[None, :, :], cp.float32(1e18))
        r_rank = cp.argsort(cp.argsort(r_fill, axis=2), axis=2).astype(cp.float32)
        r_rank = cp.where(valid, r_rank, cp.float32(0.0))
        r_mean = cp.sum(r_rank, axis=2, keepdims=True) / (n_valid[:, :, None] + _EPS)
        r_diff = cp.where(valid, r_rank - r_mean, cp.float32(0.0))

    cov = cp.sum(f_diff * r_diff, axis=2) / (n_valid + _EPS)
    sf = cp.sqrt(cp.sum(f_diff ** 2, axis=2) / (n_valid + _EPS) + _EPS)
    sr = cp.sqrt(cp.sum(r_diff ** 2, axis=2) / (n_valid + _EPS) + _EPS)
    ic = cov / (sf * sr + _EPS)
    return cp.where(n_valid >= 20, ic, cp.nan)


def _gpu_signal_pct_rank_stats_batched(factor_pp_gpu, tradable_slice_gpu=None):
    import cupy as cp

    valid = cp.isfinite(factor_pp_gpu)
    if tradable_slice_gpu is not None:
        valid &= tradable_slice_gpu[None, :, :]
    signal_counts = valid.sum(axis=2).astype(cp.float32)
    fill = cp.where(valid, factor_pp_gpu, cp.float32(1e18))
    ranks = cp.argsort(cp.argsort(fill, axis=2), axis=2).astype(cp.float32) + cp.float32(1.0)
    pct_ranks = ranks / cp.maximum(signal_counts[:, :, None], cp.float32(1.0))
    return valid, signal_counts, pct_ranks


def _gpu_simulate_ls_holdings_batched(factor_pp_gpu, ret_slice_gpu, top_frac,
                                      tradable_slice_gpu=None):
    import cupy as cp

    _, n_periods, n_symbols = factor_pp_gpu.shape
    valid, signal_counts, pct_ranks = _gpu_signal_pct_rank_stats_batched(
        factor_pp_gpu, tradable_slice_gpu=tradable_slice_gpu)
    if tradable_slice_gpu is not None:
        tradable_counts = tradable_slice_gpu.sum(axis=1).astype(cp.float32)
    else:
        tradable_counts = cp.full(n_periods, n_symbols, dtype=cp.float32)

    available_counts = cp.minimum(signal_counts, tradable_counts[None, :]).astype(cp.float32)
    min_k = cp.maximum(
        cp.ceil(available_counts * cp.float32(top_frac)).astype(cp.int32) // 2,
        cp.int32(5),
    )
    sufficient = signal_counts >= (min_k.astype(cp.float32) * cp.float32(2.0))
    target_long = valid & (pct_ranks >= cp.float32(1.0 - top_frac)) & sufficient[:, :, None]
    target_short = valid & (pct_ranks <= cp.float32(top_frac)) & sufficient[:, :, None]

    prev_long = cp.zeros_like(target_long)
    prev_short = cp.zeros_like(target_short)
    prev_long[:, 1:] = target_long[:, :-1]
    prev_short[:, 1:] = target_short[:, :-1]

    ret_valid = cp.isfinite(ret_slice_gpu)[None, :, :]
    held_long = prev_long & ret_valid
    held_short = prev_short & ret_valid

    prev_long_cnt = prev_long.sum(axis=2).astype(cp.float32)
    prev_short_cnt = prev_short.sum(axis=2).astype(cp.float32)
    long_cnt = held_long.sum(axis=2).astype(cp.float32)
    short_cnt = held_short.sum(axis=2).astype(cp.float32)

    long_sum = cp.where(held_long, ret_slice_gpu[None, :, :], cp.float32(0.0)).sum(axis=2)
    short_sum = cp.where(held_short, ret_slice_gpu[None, :, :], cp.float32(0.0)).sum(axis=2)
    long_rets = cp.where(long_cnt > 0, long_sum / (long_cnt + _EPS), cp.float32(0.0))
    short_rets = cp.where(short_cnt > 0, -(short_sum / (short_cnt + _EPS)), cp.float32(0.0))

    long_turn = cp.where(
        prev_long_cnt > 0,
        cp.sum(prev_long != target_long, axis=2).astype(cp.float32) / (prev_long_cnt + _EPS),
        cp.float32(0.0),
    )
    short_turn = cp.where(
        prev_short_cnt > 0,
        cp.sum(prev_short != target_short, axis=2).astype(cp.float32) / (prev_short_cnt + _EPS),
        cp.float32(0.0),
    )

    effective = (
        (prev_long_cnt >= min_k) &
        (prev_short_cnt >= min_k) &
        (long_cnt >= min_k) &
        (short_cnt >= min_k)
    )
    ls_ret = cp.where(
        effective,
        (long_rets + short_rets) * cp.float32(0.5),
        cp.float32(cp.nan),
    )
    ls_turn = cp.where(
        effective,
        (long_turn + short_turn) * cp.float32(0.5),
        cp.float32(cp.nan),
    )
    coverage_mean = cp.mean(
        signal_counts / cp.maximum(tradable_counts[None, :], cp.float32(1.0)),
        axis=1,
        dtype=cp.float32,
    )
    effective_bars = effective.sum(axis=1).astype(cp.float32)
    effective_den = cp.maximum(effective_bars, cp.float32(1.0))
    long_turn_mean = cp.sum(
        cp.where(effective, long_turn, cp.float32(0.0)),
        axis=1,
        dtype=cp.float32,
    ) / effective_den
    long_turn_mean = cp.where(effective_bars > 0, long_turn_mean, cp.float32(cp.nan))
    return ls_ret, ls_turn, coverage_mean, effective_bars, long_turn_mean


def _gpu_max_drawdown_batched(ls_ret_net_gpu):
    import cupy as cp

    ls_filled = cp.nan_to_num(ls_ret_net_gpu, nan=cp.float32(0.0))
    cum = cp.cumprod(1.0 + ls_filled, axis=1, dtype=cp.float32)
    peak = cp.empty_like(cum)
    global _GPU_ROW_PREFIX_MAX_KERNEL
    if _GPU_ROW_PREFIX_MAX_KERNEL is None:
        _GPU_ROW_PREFIX_MAX_KERNEL = cp.RawKernel(r'''
        extern "C" __global__
        void row_prefix_max(const float* x, float* out, const int n_rows, const int n_cols) {
            int row = blockDim.x * blockIdx.x + threadIdx.x;
            if (row >= n_rows) return;
            int base = row * n_cols;
            float running = x[base];
            out[base] = running;
            for (int col = 1; col < n_cols; ++col) {
                float v = x[base + col];
                running = running > v ? running : v;
                out[base + col] = running;
            }
        }
        ''', 'row_prefix_max')
    n_rows, n_cols = cum.shape
    threads = 128
    blocks = (int(n_rows) + threads - 1) // threads
    _GPU_ROW_PREFIX_MAX_KERNEL(
        (blocks,), (threads,),
        (cum, peak, np.int32(n_rows), np.int32(n_cols)),
    )
    return cp.nanmax(1.0 - cum / (peak + _EPS), axis=1).astype(cp.float32)


def _compute_five_objectives_prepared_gpu(daily_factors, fitness_context,
                                          periods_per_year=PERIODS_PER_YEAR,
                                          top_frac=TOP_QUANTILE,
                                          trading_cost=TRADING_COST,
                                          rolling_window=ROLLING_IC_WINDOW,
                                          perf_stats=None,
                                          ic_directions=None,
                                          return_aux=False,
                                          return_rankic_series=False):
    import cupy as cp

    total_t0 = time.perf_counter()
    pop = daily_factors.shape[0]
    if fitness_context is None:
        out = np.full((pop, 5), -1e6, dtype=np.float32)
        if return_aux:
            aux_out = _aux_matrix_to_dict(_empty_aux_matrix(pop))
            if return_rankic_series:
                return out, aux_out, np.full((pop, 0), np.nan, dtype=np.float32)
            return out, aux_out
        if return_rankic_series:
            return out, np.full((pop, 0), np.nan, dtype=np.float32)
        return out

    io_t0 = time.perf_counter()
    idx, ret_slice, tradable_slice, ret_rank_cache = fitness_context
    idx_np = to_numpy(idx)
    factors_gpu = to_xp(daily_factors[:, idx_np, :].astype(np.float32, copy=False))
    ret_slice_gpu = to_xp(ret_slice.astype(np.float32, copy=False))
    tradable_slice_gpu = (
        to_xp(tradable_slice.astype(bool, copy=False))
        if tradable_slice is not None else None
    )
    ret_rank_cache_gpu = None
    if ret_rank_cache is not None:
        ret_rank_cache_gpu = (
            to_xp(ret_rank_cache[0].astype(np.float32, copy=False)),
            to_xp(ret_rank_cache[1].astype(bool, copy=False)),
        )
    _perf_add(perf_stats, 'fit_gpu_io_s', time.perf_counter() - io_t0)

    post_t0 = time.perf_counter()
    factor_pp_gpu = _gpu_postprocess_batched(factors_gpu)
    _perf_add(perf_stats, 'fit_gpu_postprocess_s', time.perf_counter() - post_t0)

    ic_t0 = time.perf_counter()
    ic_all_gpu = _gpu_rank_ic_batched(
        factor_pp_gpu, ret_slice_gpu,
        tradable_slice_gpu=tradable_slice_gpu,
        ret_rank_cache_gpu=ret_rank_cache_gpu)
    _perf_add(perf_stats, 'fit_gpu_ic_s', time.perf_counter() - ic_t0)

    objectives_gpu = cp.full((pop, 5), cp.float32(-1e6), dtype=cp.float32)
    aux_gpu = cp.full((pop, len(_AUX_NAMES)), cp.float32(cp.nan), dtype=cp.float32)
    aux_gpu[:, 0] = cp.float32(1.0)
    aux_gpu[:, 5] = cp.float32(0.0)
    aux_gpu[:, 6] = cp.float32(0.0)

    if ic_directions is None:
        ic_raw_finite = cp.isfinite(ic_all_gpu)
        ic_raw_count = ic_raw_finite.sum(axis=1).astype(cp.float32)
        ic_raw_sum = cp.sum(
            cp.where(ic_raw_finite, ic_all_gpu, cp.float32(0.0)),
            axis=1,
            dtype=cp.float32,
        )
        ic_raw_mean = cp.where(
            ic_raw_count > 0,
            ic_raw_sum / cp.maximum(ic_raw_count, cp.float32(_EPS)),
            cp.float32(0.0),
        )
        ic_direction_gpu = cp.where(
            ic_raw_mean < 0.0,
            cp.float32(-1.0),
            cp.float32(1.0),
        )
    else:
        ic_direction_gpu = to_xp(_normalize_direction_np(ic_directions))
    factor_pp_gpu = factor_pp_gpu * ic_direction_gpu[:, None, None]
    ic_all_gpu = ic_all_gpu * ic_direction_gpu[:, None]

    ic_finite = cp.isfinite(ic_all_gpu)
    ic_valid = ic_finite.sum(axis=1).astype(cp.float32)
    ic_sum = cp.sum(cp.where(ic_finite, ic_all_gpu, cp.float32(0.0)), axis=1, dtype=cp.float32)
    ic_mean = cp.where(ic_valid > 0, ic_sum / cp.maximum(ic_valid, cp.float32(_EPS)), cp.float32(cp.nan))
    ic_std = cp.nanstd(ic_all_gpu, axis=1)
    rankicir = cp.where(
        (ic_valid > 0) & (ic_std > _EPS),
        ic_mean / cp.maximum(ic_std, cp.float32(_EPS)) * cp.sqrt(cp.float32(periods_per_year)),
        cp.float32(0.0),
    )
    positive_rate = cp.where(
        ic_valid > 0,
        cp.sum(ic_finite & (ic_all_gpu > 0.0), axis=1).astype(cp.float32) /
        cp.maximum(ic_valid, cp.float32(_EPS)),
        cp.float32(cp.nan),
    )
    aux_gpu[:, 0] = ic_direction_gpu
    aux_gpu[:, 1] = ic_mean
    aux_gpu[:, 2] = rankicir
    aux_gpu[:, 3] = positive_rate
    aux_gpu[:, 5] = ic_valid

    valid_ic = ic_valid >= cp.float32(MIN_IC_VALID_BARS)
    n_periods = ic_all_gpu.shape[1]

    ls_t0 = time.perf_counter()
    ls_ret_gpu, ls_turnover_gpu, coverage_mean, effective_bars, long_turnover_mean = (
        _gpu_simulate_ls_holdings_batched(
            factor_pp_gpu, ret_slice_gpu, top_frac,
            tradable_slice_gpu=tradable_slice_gpu)
    )
    required_bars = _required_effective_bars(n_periods)
    ls_valid = cp.sum(cp.isfinite(ls_ret_gpu), axis=1).astype(cp.float32)
    ls_turnover_mean = cp.where(
        ls_valid > 0,
        cp.nansum(ls_turnover_gpu, axis=1, dtype=cp.float32) /
        cp.maximum(ls_valid, cp.float32(_EPS)),
        cp.float32(cp.nan),
    )
    ls_ret_net = ls_ret_gpu - ls_turnover_gpu * cp.float32(trading_cost / 2.0)

    ls_mean = cp.nanmean(ls_ret_net, axis=1)
    ls_std = cp.nanstd(ls_ret_net, axis=1)
    ls_positive_rate = cp.where(
        ls_valid > 0,
        cp.sum(cp.isfinite(ls_ret_net) & (ls_ret_net > 0.0), axis=1).astype(cp.float32)
        / cp.maximum(ls_valid, cp.float32(_EPS)),
        cp.float32(cp.nan),
    )
    _perf_add(perf_stats, 'fit_gpu_ls_s', time.perf_counter() - ls_t0)

    aux_gpu[:, 4] = coverage_mean
    aux_gpu[:, 6] = effective_bars
    aux_gpu[:, 7] = ls_turnover_mean
    aux_gpu[:, 8] = long_turnover_mean
    aux_gpu[:, 9] = ls_positive_rate

    dd_t0 = time.perf_counter()
    max_dd = _gpu_max_drawdown_batched(ls_ret_net)
    _perf_add(perf_stats, 'fit_gpu_dd_s', time.perf_counter() - dd_t0)
    sharpe = cp.where(
        ls_std > _EPS,
        ls_mean / cp.maximum(ls_std, _EPS) * cp.sqrt(cp.float32(periods_per_year)),
        cp.float32(0.0),
    )
    ann_return = ls_mean * cp.float32(periods_per_year)
    turnover_pressure = cp.maximum(ls_turnover_mean, long_turnover_mean)
    aux_gpu[:, 10] = ann_return

    valid_mask = (
        valid_ic &
        (coverage_mean >= cp.float32(MIN_SIGNAL_COVERAGE)) &
        (effective_bars >= cp.float32(required_bars)) &
        (ls_valid >= cp.float32(required_bars))
    )
    objectives_gpu[valid_mask, 0] = rankicir[valid_mask]
    objectives_gpu[valid_mask, 1] = sharpe[valid_mask]
    turnover_excess = cp.maximum(cp.float32(0.0), turnover_pressure - cp.float32(TURNOVER_HINGE_TARGET))
    objectives_gpu[valid_mask, 2] = -turnover_excess[valid_mask]
    objectives_gpu[valid_mask, 3] = cp.float32(0.0)
    objectives_gpu[valid_mask, 4] = (1.0 - max_dd[valid_mask])
    _perf_add(perf_stats, 'fit_gpu_total_s', time.perf_counter() - total_t0)
    out = to_numpy(objectives_gpu)
    if return_aux:
        aux_out = _aux_matrix_to_dict(to_numpy(aux_gpu))
        if return_rankic_series:
            return out, aux_out, to_numpy(ic_all_gpu)
        return out, aux_out
    if return_rankic_series:
        return out, to_numpy(ic_all_gpu)
    return out


def compute_five_objectives_prepared(daily_factors, fitness_context,
                                     periods_per_year=PERIODS_PER_YEAR,
                                     top_frac=TOP_QUANTILE,
                                     trading_cost=TRADING_COST,
                                     rolling_window=ROLLING_IC_WINDOW,
                                     perf_stats=None,
                                     ic_directions=None,
                                     return_aux=False,
                                     return_rankic_series=False):
    pop = daily_factors.shape[0]
    if fitness_context is None:
        out = np.full((pop, 5), -1e6, dtype=np.float32)
        if return_aux:
            aux_out = _aux_matrix_to_dict(_empty_aux_matrix(pop))
            if return_rankic_series:
                return out, aux_out, np.full((pop, 0), np.nan, dtype=np.float32)
            return out, aux_out
        if return_rankic_series:
            return out, np.full((pop, 0), np.nan, dtype=np.float32)
        return out

    total_t0 = time.perf_counter()
    if get_backend() == 'gpu' and gpu_fitness_enabled():
        out = _compute_five_objectives_prepared_gpu(
            daily_factors, fitness_context,
            periods_per_year=periods_per_year,
            top_frac=top_frac,
            trading_cost=trading_cost,
            rolling_window=rolling_window,
            perf_stats=perf_stats,
            ic_directions=ic_directions,
            return_aux=return_aux,
            return_rankic_series=return_rankic_series)
        _perf_add(perf_stats, 'fit_total_s', time.perf_counter() - total_t0)
        return out

    idx, ret_slice, tradable_slice, ret_rank_cache = fitness_context

    if pop == 1:
        results = [_compute_one_from_index(
            0, daily_factors, idx, ret_slice, top_frac,
            periods_per_year, rolling_window,
            tradable_slice, ret_rank_cache,
            ic_directions, return_aux, return_rankic_series, trading_cost)]
    else:
        executor = _get_fitness_executor()
        results = list(executor.map(
            _compute_one_from_index,
            range(pop),
            repeat(daily_factors),
            repeat(idx),
            repeat(ret_slice),
            repeat(top_frac),
            repeat(periods_per_year),
            repeat(rolling_window),
            repeat(tradable_slice),
            repeat(ret_rank_cache),
            repeat(ic_directions),
            repeat(return_aux),
            repeat(return_rankic_series),
            repeat(trading_cost),
        ))
    if return_aux:
        out = np.array([r[0] for r in results], dtype=np.float32)
        aux = np.vstack([r[1] for r in results]).astype(np.float32, copy=False)
        aux_out = _aux_matrix_to_dict(aux)
        if return_rankic_series:
            rankic = np.vstack([r[2] for r in results]).astype(np.float32, copy=False)
            result = (out, aux_out, rankic)
        else:
            result = (out, aux_out)
    elif return_rankic_series:
        out = np.array([r[0] for r in results], dtype=np.float32)
        rankic = np.vstack([r[1] for r in results]).astype(np.float32, copy=False)
        result = (out, rankic)
    else:
        result = np.array(results, dtype=np.float32)
    _perf_add(perf_stats, 'fit_total_s', time.perf_counter() - total_t0)
    return result


def compute_five_objectives(daily_factors, period_returns,
                            lag=LAG_PERIODS, periods_per_year=PERIODS_PER_YEAR,
                            top_frac=TOP_QUANTILE, is_mask=None, tradable_mask=None,
                            trading_cost=TRADING_COST,
                            rolling_window=ROLLING_IC_WINDOW,
                            ic_directions=None,
                            return_aux=False, return_rankic_series=False):
    lagged_returns = build_lagged_returns(period_returns, lag=lag)
    fitness_context = prepare_fitness_context(
        lagged_returns, is_mask=is_mask, tradable_mask=tradable_mask)
    return compute_five_objectives_prepared(
        daily_factors, fitness_context,
        periods_per_year=periods_per_year,
        top_frac=top_frac,
        trading_cost=trading_cost,
        rolling_window=rolling_window,
        ic_directions=ic_directions,
        return_aux=return_aux,
        return_rankic_series=return_rankic_series)


def apply_shortcoming_penalty(objectives, percentile=SHORTCOMING_PERCENTILE):
    pop, n_obj = objectives.shape
    penalty = np.ones(pop, dtype=np.float32)
    for j in range(n_obj):
        col = objectives[:, j]
        valid = col > -1e5
        if valid.sum() < 10:
            continue
        vals = col[valid]
        order = np.argsort(np.argsort(vals)).astype(np.float32)
        rank_pct = order / max(1.0, float(len(vals) - 1))
        mult = (0.5 + 0.5 * rank_pct).astype(np.float32)
        penalty[valid] *= mult
    for j in range(n_obj):
        objectives[:, j] *= penalty
    return objectives


def sample_diversity_features(daily_factors, tradable_mask=None,
                              max_periods=DIVERSITY_MAX_PERIODS,
                              out_dtype=np.float16):
    """Sample factor phenotypes for diversity without storing full factor tensors."""
    pop, P, S = daily_factors.shape
    if P == 0:
        return np.empty((pop, 0), dtype=out_dtype)

    step = max(1, P // max_periods)
    sampled = daily_factors[:, ::step, :]
    if tradable_mask is not None:
        sampled_mask = tradable_mask[::step]
        if isinstance(sampled, np.ndarray):
            sampled = np.where(sampled_mask[None, :, :], sampled, np.nan)
        else:
            sampled = xp.where(
                to_xp(sampled_mask.astype(bool, copy=False))[None, :, :],
                sampled,
                xp.float32(xp.nan),
            )
    sampled = to_numpy(sampled.reshape(pop, -1)).astype(np.float32, copy=False)
    if np.dtype(out_dtype) == np.float16:
        finfo = np.finfo(np.float16)
        np.clip(sampled, finfo.min, finfo.max, out=sampled)
    return sampled.astype(out_dtype, copy=False)


def prepare_residual_basis_context(basis_panels, fitness_context,
                                   max_periods=128, ridge_rel=1e-4,
                                   min_valid=None):
    """Precompute per-period cross-sectional regression state for residual IC.

    basis_panels: (K, P_total, S) float32 array of existing-factor panels,
    already direction-adjusted and per-period z-scored (same convention as
    _postprocess_single output). fitness_context is the numpy IS context from
    prepare_fitness_context.

    Returns a dict with per-sampled-period centered basis (Bc), inverse normal
    equations (Ginv), validity masks and cached return ranks, or None when
    too few usable periods exist. All arrays are numpy float32.
    """
    if fitness_context is None or basis_panels is None:
        return None
    idx, ret_slice, tradable_slice, ret_rank_cache = fitness_context
    idx = np.asarray(idx, dtype=np.int64)
    panels = np.asarray(basis_panels, dtype=np.float32)
    if panels.ndim != 3 or panels.shape[0] == 0:
        return None
    K, _, S = panels.shape
    if min_valid is None:
        min_valid = max(20, 2 * K)
    n_is = idx.shape[0]
    if n_is < 8:
        return None
    step = max(1, n_is // max(1, int(max_periods)))
    rows = np.arange(0, n_is, step, dtype=np.int64)

    bc_list, ginv_list, valid_list, rrank_list, row_list = [], [], [], [], []
    for r in rows:
        p = int(idx[r])
        if p >= panels.shape[1]:
            continue
        B = panels[:, p, :].T  # (S, K)
        valid = np.isfinite(B).all(axis=1) & np.isfinite(ret_slice[r])
        if tradable_slice is not None:
            valid &= tradable_slice[r]
        n = int(valid.sum())
        if n < min_valid:
            continue
        Bv = np.where(valid[:, None], B, 0.0)
        mu = Bv.sum(axis=0) / float(n)
        Bc = np.where(valid[:, None], B - mu[None, :], 0.0).astype(np.float32)
        # Unnormalized normal equations: beta = Ginv @ (Bc' Fc), so Ginv must
        # match the unnormalized H = Fc @ Bc used at eval time.
        G = Bc.T @ Bc
        ridge = float(ridge_rel) * max(float(np.mean(np.diag(G))), 1e-8)
        G += np.eye(K, dtype=np.float32) * ridge
        try:
            Ginv = np.linalg.inv(G).astype(np.float32)
        except np.linalg.LinAlgError:
            continue
        if ret_rank_cache is not None:
            r_rank = np.asarray(ret_rank_cache[0][r], dtype=np.float32)
        else:
            r_fill = np.where(valid, ret_slice[r], 1e18)
            r_rank = np.argsort(np.argsort(r_fill)).astype(np.float32)
            r_rank = np.where(valid, r_rank, 0.0)
        bc_list.append(Bc)
        ginv_list.append(Ginv)
        valid_list.append(valid)
        rrank_list.append(r_rank)
        row_list.append(r)
    if len(row_list) < 8:
        return None
    rows_arr = np.asarray(row_list, dtype=np.int64)
    return {
        "Bc": np.stack(bc_list).astype(np.float32),          # (Ps, S, K)
        "Ginv": np.stack(ginv_list).astype(np.float32),      # (Ps, K, K)
        "valid": np.stack(valid_list),                       # (Ps, S) bool
        "ret_rank": np.stack(rrank_list).astype(np.float32), # (Ps, S)
        "rows": rows_arr,                                    # rows into ret_slice
        "period_pos": idx[rows_arr],                         # absolute period indices
        "n_basis": K,
    }


def compute_residual_rankic(daily_factors, basis_ctx, ic_directions=None,
                            periods_per_year=PERIODS_PER_YEAR,
                            min_periods=30, chunk=64):
    """RankIC of each candidate after removing the basis factors' span.

    For every sampled IS period the (direction-adjusted, per-period
    postprocessed) candidate is cross-sectionally regressed on the basis
    panels; the residual's RankIC vs forward returns measures the incremental
    signal not explained by the existing factor base. Signed: a negative
    value means the candidate adds nothing (or hurts) once the base is
    accounted for.

    daily_factors: (C, P_total, S) numpy/xp array of raw candidate values.
    Returns (resid_rankicir, resid_ic_mean), each (C,) float32, NaN when the
    candidate has fewer than min_periods usable periods.
    """
    C = daily_factors.shape[0]
    out_rankicir = np.full(C, np.nan, dtype=np.float32)
    out_ic_mean = np.full(C, np.nan, dtype=np.float32)
    if basis_ctx is None:
        return out_rankicir, out_ic_mean

    Bc_all = basis_ctx["Bc"]
    Ginv_all = basis_ctx["Ginv"]
    valid_all = basis_ctx["valid"]
    rrank_all = basis_ctx["ret_rank"]
    period_pos = basis_ctx["period_pos"]
    Ps = Bc_all.shape[0]

    factors_np = to_numpy(daily_factors).astype(np.float32, copy=False)
    dirs = (
        np.ones(C, dtype=np.float32)
        if ic_directions is None
        else _normalize_direction_np(ic_directions)
    )

    for c0 in range(0, C, chunk):
        c1 = min(c0 + chunk, C)
        F = factors_np[c0:c1][:, period_pos, :].astype(np.float32, copy=False)  # (Cc, Ps, S)
        Cc = F.shape[0]
        for i in range(Cc):
            F[i] = _postprocess_single(F[i])
        F *= dirs[c0:c1, None, None]

        ics = np.full((Cc, Ps), np.nan, dtype=np.float32)
        for t in range(Ps):
            v = valid_all[t]
            n = float(v.sum())
            Ft = F[:, t, :]
            fm = np.sum(np.where(v[None, :], Ft, 0.0), axis=1, keepdims=True) / (n + _EPS)
            Fc = np.where(v[None, :], Ft - fm, 0.0)
            H = Fc @ Bc_all[t]                       # (Cc, K)
            beta = H @ Ginv_all[t]                   # (Cc, K)
            resid = Fc - beta @ Bc_all[t].T          # (Cc, S)

            f_fill = np.where(v[None, :], resid, 1e18)
            f_rank = np.argsort(np.argsort(f_fill, axis=1), axis=1).astype(np.float32)
            f_rank = np.where(v[None, :], f_rank, 0.0)
            frm = np.sum(f_rank, axis=1, keepdims=True) / (n + _EPS)
            fd = np.where(v[None, :], f_rank - frm, 0.0)

            r_rank = np.where(v, rrank_all[t], 0.0)
            rm = np.sum(r_rank) / (n + _EPS)
            rd = np.where(v, r_rank - rm, 0.0)

            cov = np.sum(fd * rd[None, :], axis=1) / (n + _EPS)
            sf = np.sqrt(np.sum(fd ** 2, axis=1) / (n + _EPS) + _EPS)
            sr = np.sqrt(np.sum(rd ** 2) / (n + _EPS) + _EPS)
            ics[:, t] = cov / (sf * sr + _EPS)

        n_valid = np.isfinite(ics).sum(axis=1)
        ic_mean = np.nanmean(ics, axis=1)
        ic_std = np.nanstd(ics, axis=1)
        rankicir = np.where(
            (n_valid >= min_periods) & (ic_std > _EPS),
            ic_mean / np.maximum(ic_std, _EPS) * np.sqrt(np.float32(periods_per_year)),
            np.nan,
        )
        out_rankicir[c0:c1] = rankicir.astype(np.float32)
        out_ic_mean[c0:c1] = np.where(n_valid >= min_periods, ic_mean, np.nan).astype(np.float32)
    return out_rankicir, out_ic_mean


def standardize_phenotypes(phenotypes):
    """Standardise + unit-normalise phenotype vectors so dot product = Pearson r.

    Used by evolution-time novelty and diversity penalties.
    Returns (pop, n_feat) float32 array with unit-length rows.
    """
    pheno = phenotypes.astype(np.float32, copy=True)
    mu = _row_nanmean_2d(pheno)
    std = _row_nanstd_2d(pheno, mean=mu)
    std = np.where(std < 1e-8, 1.0, std)
    pheno = (pheno - mu) / std
    np.nan_to_num(pheno, copy=False, nan=0.0)
    norms = np.linalg.norm(pheno, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    pheno /= norms
    return pheno
