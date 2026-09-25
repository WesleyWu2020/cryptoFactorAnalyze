"""Evaluator v2: base factor evaluation + temporal/cross-sectional composition.

Key changes from v1:
- Composition layer applied after base factor computation
- Cross-sectional ops (rank, zscore, demean, scale) per period
- Temporal ops (delta, pct_change, zscore, rank, accel, decay) across periods
"""
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
import time

from .backend import xp, to_numpy, get_backend
from .config import (
    WINDOW_CHOICES, SLICE_CHOICES, MASK_RULES,
    INDICATOR_NAMES,
    MODE1_OPS, MODE2_OPS, MODE3_OPS, MODE4_OPS, B_SHIFT_CHOICES,
    INTRADAY_GROUP_CHOICES,
    TS_COMP_OPS, TS_COMP_WINDOWS, CS_COMP_OPS,
    MINUTES_PER_PERIOD, GPU_EVAL_WORKERS,
)
from .operators import MODE1_DISPATCH, MODE2_DISPATCH, MODE3_DISPATCH, MODE4_DISPATCH
from .composition_extensions import TS_EXTENSION_DISPATCH, CS_EXTENSION_DISPATCH, cs_neutralize, cs_group_rank

_EPS = 1e-8


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


_N_CPU_EVAL_WORKERS = _int_from_env(
    ("GP_V2_CPU_EVAL_WORKERS", "GP_CPU_WORKERS"),
    max(1, os.cpu_count() or 4),
)
_GPU_EVAL_WORKERS = _int_from_env(
    ("GP_V2_GPU_EVAL_WORKERS",),
    max(1, int(GPU_EVAL_WORKERS)),
)


def _has_active_composition(population):
    if population.size == 0 or population.shape[1] < 13:
        return False
    return bool(np.any(population[:, 10] != 0) or np.any(population[:, 12] != 0))


def _pad_population(population, target_len):
    if population.shape[1] >= target_len:
        return population
    pad = np.zeros((population.shape[0], target_len - population.shape[1]), dtype=population.dtype)
    return np.concatenate([population, pad], axis=1)


def _get_eval_workers(use_backend_output=False, task_count=0):
    if get_backend() != 'gpu':
        return min(_N_CPU_EVAL_WORKERS, max(1, int(task_count or 1)))
    if use_backend_output or task_count <= 1:
        return 1
    return min(int(_GPU_EVAL_WORKERS), int(task_count))


def _perf_add(perf_stats, key, value):
    if perf_stats is not None:
        perf_stats[key] = perf_stats.get(key, 0.0) + float(value)


def _diagnostic_topk():
    raw = os.environ.get("GP_V2_EVAL_DIAGNOSTICS_TOPK")
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _diagnostic_template(row):
    mode_raw = int(row[6]) if len(row) > 6 else 0
    if mode_raw == 0:
        operator = MODE1_OPS[int(row[7])]
    elif mode_raw == 1:
        operator = MODE2_OPS[int(row[8])]
    elif mode_raw == 2:
        operator = MODE3_OPS[int(row[13])]
    else:
        operator = MODE4_OPS[int(row[13])]
    return {
        "mode": mode_raw + 1,
        "operator": operator,
        "window": WINDOW_CHOICES[int(row[2])],
        "slice": SLICE_CHOICES[int(row[3])],
        "mask_field": INDICATOR_NAMES[int(row[4])],
        "mask_rule": MASK_RULES[int(row[5])],
        "ts_comp_op": TS_COMP_OPS[int(row[10])] if len(row) > 10 else "none",
        "cs_comp_op": CS_COMP_OPS[int(row[12])] if len(row) > 12 else "none",
    }


def _update_factor_diagnostics(results, population, perf_stats):
    if perf_stats is None or results.size == 0:
        return
    finite = np.isfinite(results)
    finite_counts = finite.sum(axis=(1, 2)).astype(np.int32)
    total_cells = max(1, int(results.shape[1] * results.shape[2]))
    finite_ratios = finite_counts.astype(np.float32) / float(total_cells)
    all_nan = finite_counts == 0

    perf_stats["eval_finite_cell_ratio"] = float(finite.sum() / max(1, results.size))
    perf_stats["eval_all_nan_individual_count"] = int(all_nan.sum())
    perf_stats["eval_all_nan_individual_share"] = float(np.mean(all_nan))
    perf_stats["eval_low_finite_individual_share"] = float(np.mean(finite_ratios < 0.01))

    topk = _diagnostic_topk()
    if topk <= 0 or not np.any(all_nan):
        return

    grouped = {}
    for idx in np.flatnonzero(all_nan):
        row = np.asarray(population[int(idx)])
        key = tuple(int(x) for x in row.tolist())
        entry = grouped.get(key)
        if entry is None:
            entry = _diagnostic_template(row)
            entry["count"] = 0
            entry["example_index"] = int(idx)
            grouped[key] = entry
        entry["count"] += 1

    top = sorted(grouped.values(), key=lambda item: (-item["count"], item["example_index"]))[:topk]
    perf_stats["eval_nan_top_templates"] = top


def _compute_slice_bounds(window, slice_val):
    M = MINUTES_PER_PERIOD
    if window >= M:
        return 0, M
    if slice_val is None:
        return M - window, M
    center = int(slice_val * (M - 1))
    half = window // 2
    start = max(0, center - half)
    end = min(M, start + window)
    if end == M:
        start = max(0, M - window)
    return start, end


def _build_mask(field_data, mask_rule):
    if mask_rule == 'none':
        return xp.ones(field_data.shape, dtype=xp.float32)
    parts = mask_rule.split('_')
    direction, quantile = parts[0], float(parts[1])
    valid = xp.isfinite(field_data)
    n_valid = xp.sum(valid, axis=1, keepdims=True).astype(xp.float32)
    fill_val = xp.float32(-1e18) if direction == 'high' else xp.float32(1e18)
    filled = xp.where(valid, field_data, fill_val)
    rank = xp.argsort(xp.argsort(filled, axis=1), axis=1).astype(xp.float32)
    del filled
    rank_norm = rank / (n_valid - 1 + _EPS)
    del rank
    if direction == 'high':
        mask = (rank_norm >= xp.float32(1.0 - quantile)).astype(xp.float32)
    else:
        mask = (rank_norm <= xp.float32(quantile)).astype(xp.float32)
    del rank_norm
    return mask * valid.astype(xp.float32)


# =========================================================================
# Cross-sectional composition: applied per period across coins
# Input/output: (P, S) numpy arrays
# =========================================================================
def _cs_rank(factor):
    """Percentile rank across coins per period."""
    valid = np.isfinite(factor)
    n = valid.sum(axis=1, keepdims=True).astype(np.float32)
    filled = np.where(valid, factor, 1e18)
    rank = np.argsort(np.argsort(filled, axis=1), axis=1).astype(np.float32)
    return np.where(valid, rank / (n - 1 + _EPS), np.nan)


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


def _cs_zscore(factor):
    """Z-score across coins per period."""
    mu = _row_nanmean_2d(factor)
    sigma = _row_nanstd_2d(factor, mean=mu)
    sigma = np.where(sigma < _EPS, 1.0, sigma)
    return (factor - mu) / sigma


def _cs_demean(factor):
    """Subtract cross-sectional mean per period."""
    return factor - _row_nanmean_2d(factor)


def _cs_scale(factor):
    """Divide by cross-sectional std per period."""
    sigma = _row_nanstd_2d(factor)
    sigma = np.where(sigma < _EPS, 1.0, sigma)
    return factor / sigma


CS_DISPATCH = {
    'none': None, 'cs_rank': _cs_rank, 'cs_zscore': _cs_zscore,
    'cs_demean': _cs_demean, 'cs_scale': _cs_scale,
}
CS_DISPATCH.update(CS_EXTENSION_DISPATCH)


# =========================================================================
# Temporal composition: applied across periods for each coin
# Input/output: (P, S) numpy arrays. May shorten P dimension.
# Returns (result, start_offset) where start_offset is how many periods
# are consumed from the front.
# =========================================================================
def _ts_delta(factor, w):
    """f(t) - f(t-w)"""
    result = np.full_like(factor, np.nan)
    result[w:] = factor[w:] - factor[:-w]
    return result, 0


def _ts_pct_change(factor, w):
    """f(t) / f(t-w) - 1"""
    result = np.full_like(factor, np.nan)
    denom = factor[:-w]
    safe = np.abs(denom) > _EPS
    result[w:] = np.where(safe, factor[w:] / denom - 1.0, np.nan)
    return result, 0


def _ts_zscore(factor, w):
    """Rolling z-score: (f(t) - rolling_mean(w)) / rolling_std(w)"""
    P, S = factor.shape
    result = np.full_like(factor, np.nan)
    if P < w:
        return result, 0
    # Cumsum trick for rolling mean/var
    cs = np.nancumsum(factor, axis=0)
    cs2 = np.nancumsum(factor**2, axis=0)
    cnt = np.nancumsum(np.isfinite(factor).astype(np.float32), axis=0)

    for t in range(w - 1, P):
        if t == w - 1:
            s = cs[t]
            s2 = cs2[t]
            n = cnt[t]
        else:
            s = cs[t] - cs[t - w]
            s2 = cs2[t] - cs2[t - w]
            n = cnt[t] - cnt[t - w]
        n = np.maximum(n, 1.0)
        mu = s / n
        var = s2 / n - mu**2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var) + _EPS
        result[t] = (factor[t] - mu) / std
    return result, 0


def _ts_rank(factor, w):
    """Rolling percentile rank within window."""
    P, S = factor.shape
    result = np.full_like(factor, np.nan)
    for t in range(w - 1, P):
        window_data = factor[t - w + 1:t + 1]  # (w, S)
        current = factor[t:t + 1]  # (1, S)
        valid = np.isfinite(window_data)
        n_valid = valid.sum(axis=0).astype(np.float32)
        # Count how many values in window are <= current
        leq = np.nansum(window_data <= current, axis=0).astype(np.float32)
        result[t] = np.where(n_valid >= 3, leq / (n_valid + _EPS), np.nan)
    return result, 0


def _ts_accel(factor, w):
    """Acceleration: delta of delta."""
    d1 = np.full_like(factor, np.nan)
    d1[w:] = factor[w:] - factor[:-w]
    result = np.full_like(factor, np.nan)
    result[2 * w:] = d1[2 * w:] - d1[w:-w]
    return result, 0


def _ts_decay(factor, w):
    """Exponential weighted mean with alpha = 2/(w+1)."""
    P, S = factor.shape
    alpha = 2.0 / (w + 1)
    result = np.full_like(factor, np.nan)
    ema = factor[0].copy()
    result[0] = ema
    for t in range(1, P):
        valid = np.isfinite(factor[t])
        ema_valid = np.isfinite(ema)
        ema = np.where(
            valid,
            np.where(ema_valid, alpha * factor[t] + (1 - alpha) * ema, factor[t]),
            ema,
        )
        result[t] = ema
    return result, 0


TS_DISPATCH = {
    'none': None, 'delta': _ts_delta, 'pct_change': _ts_pct_change,
    'ts_zscore': _ts_zscore, 'ts_rank': _ts_rank,
    'ts_accel': _ts_accel, 'ts_decay': _ts_decay,
}
TS_DISPATCH.update(TS_EXTENSION_DISPATCH)


# =========================================================================
# Main evaluation
# =========================================================================
def evaluate_population_base(population, indicator_data, output_backend='cpu',
                             perf_stats=None):
    """Evaluate base factors (before composition) on one chunk.

    Same as v1 evaluator but extracted as 'base' step.

    Args:
        population: np.ndarray (pop_size, 13) int
        indicator_data: xp.ndarray (P, N_IND, M, S)
        output_backend: 'cpu' for numpy output, 'xp' for current backend output
    Returns:
        Array (pop_size, P, S) float32 on requested backend
    """
    pop_size = population.shape[0]
    P, n_ind, M, S = indicator_data.shape
    use_backend_output = output_backend == 'xp'
    if use_backend_output:
        results = xp.full((pop_size, P, S), xp.float32(xp.nan), dtype=xp.float32)
    else:
        results = np.full((pop_size, P, S), np.nan, dtype=np.float32)

    group_t0 = time.perf_counter()
    full_groups = defaultdict(list)
    for i in range(pop_size):
        p = population[i]
        mode = int(p[6])
        if mode == 0:
            op_idx = int(p[7])
            long_mult_idx = 0
        elif mode == 1:
            op_idx = int(p[8])
            long_mult_idx = 0
        elif mode == 2:
            op_idx = int(p[13])
            long_mult_idx = int(p[11])
        else:
            op_idx = int(p[13])
            long_mult_idx = int(p[14])
        key = (int(p[2]), int(p[3]), int(p[4]), int(p[5]), mode, op_idx, long_mult_idx)
        full_groups[key].append(i)
    _perf_add(perf_stats, 'eval_group_s', time.perf_counter() - group_t0)

    mask_t0 = time.perf_counter()
    _mask_cache = {}
    group_masks = {}
    for (win_idx, sl_idx, mf_idx, mr_idx, mode, op_idx, long_mult_idx), members in full_groups.items():
        window = WINDOW_CHOICES[win_idx]
        slice_val = SLICE_CHOICES[sl_idx]
        start, end = _compute_slice_bounds(window, slice_val)

        mask_key = (mf_idx, mr_idx, start, end)
        if mask_key not in _mask_cache:
            mf_data = indicator_data[:, mf_idx, start:end, :]
            _mask_cache[mask_key] = _build_mask(mf_data, MASK_RULES[mr_idx])

        long_start = long_end = None
        mask_long = None
        if mode == 2:
            long_window = min(window * TS_COMP_WINDOWS[long_mult_idx], M)
            long_start, long_end = _compute_slice_bounds(long_window, slice_val)
            long_mask_key = (mf_idx, mr_idx, long_start, long_end)
            if long_mask_key not in _mask_cache:
                mf_data_l = indicator_data[:, mf_idx, long_start:long_end, :]
                _mask_cache[long_mask_key] = _build_mask(mf_data_l, MASK_RULES[mr_idx])
            mask_long = _mask_cache[long_mask_key]

        group_key = (win_idx, sl_idx, mf_idx, mr_idx, mode, op_idx, long_mult_idx)
        group_masks[group_key] = (_mask_cache[mask_key], start, end, mask_long, long_start, long_end)
    _perf_add(perf_stats, 'eval_mask_s', time.perf_counter() - mask_t0)

    task_build_t0 = time.perf_counter()
    tasks = []
    for (win_idx, sl_idx, mf_idx, mr_idx, mode, op_idx, long_mult_idx), members in full_groups.items():
        group_key = (win_idx, sl_idx, mf_idx, mr_idx, mode, op_idx, long_mult_idx)
        mask, start, end, mask_long, long_start, long_end = group_masks[group_key]

        if mode == 0:
            op_name = MODE1_OPS[op_idx]
            fn = MODE1_DISPATCH.get(op_name)
            if fn is None:
                continue
            a_to_members = defaultdict(list)
            for i in members:
                a_to_members[int(population[i, 0])].append(i)
            for ai, idxs in a_to_members.items():
                a_data = indicator_data[:, ai, start:end, :]
                tasks.append(('m1', fn, a_data, mask, idxs))
        elif mode == 1:
            op_name = MODE2_OPS[op_idx]
            fn = MODE2_DISPATCH.get(op_name)
            if fn is None:
                continue
            triple_to_members = defaultdict(list)
            for i in members:
                triple = (int(population[i, 0]), int(population[i, 1]), int(population[i, 9]))
                triple_to_members[triple].append(i)
            for (ai, bi, bs_idx), idxs in triple_to_members.items():
                a_data = indicator_data[:, ai, start:end, :]
                b_data = indicator_data[:, bi, start:end, :]
                b_shift = B_SHIFT_CHOICES[bs_idx]
                if b_shift != 0:
                    b_data = xp.roll(b_data, shift=b_shift, axis=1).copy()
                    if b_shift > 0:
                        b_data[:, :b_shift, :] = xp.float32(xp.nan)
                    else:
                        b_data[:, b_shift:, :] = xp.float32(xp.nan)
                tasks.append(('m2', fn, a_data, b_data, mask, idxs))
        elif mode == 2:
            op_name = MODE3_OPS[op_idx]
            fn = MODE3_DISPATCH.get(op_name)
            if fn is None or mask_long is None:
                continue
            a_to_members = defaultdict(list)
            for i in members:
                a_to_members[int(population[i, 0])].append(i)
            for ai, idxs in a_to_members.items():
                a_short = indicator_data[:, ai, start:end, :]
                a_long = indicator_data[:, ai, long_start:long_end, :]
                tasks.append(('m3', fn, a_short, a_long, mask, mask_long, idxs))
        else:
            op_name = MODE4_OPS[op_idx]
            fn = MODE4_DISPATCH.get(op_name)
            if fn is None:
                continue
            pair_to_members = defaultdict(list)
            for i in members:
                pair_to_members[(int(population[i, 0]), int(population[i, 1]))].append(i)
            group_count = INTRADAY_GROUP_CHOICES[long_mult_idx]
            for (ai, bi), idxs in pair_to_members.items():
                a_data = indicator_data[:, ai, start:end, :]
                b_data = indicator_data[:, bi, start:end, :]
                tasks.append(('m4', fn, a_data, b_data, mask, group_count, idxs))
    _perf_add(perf_stats, 'eval_task_build_s', time.perf_counter() - task_build_t0)

    if perf_stats is not None:
        perf_stats['eval_task_count'] = perf_stats.get('eval_task_count', 0) + len(tasks)
        perf_stats['eval_group_count'] = perf_stats.get('eval_group_count', 0) + len(full_groups)
        perf_stats['eval_mask_count'] = perf_stats.get('eval_mask_count', 0) + len(_mask_cache)

    def _run_task(task):
        mode_tag, fn = task[0], task[1]
        idxs = task[-1]
        t0 = time.perf_counter()
        d2h_s = 0.0
        try:
            if mode_tag == 'm1':
                factor = fn(task[2], task[3])
            elif mode_tag == 'm2':
                factor = fn(task[2], task[3], task[4])
            else:
                factor = fn(task[2], task[3], task[4], task[5])
            if use_backend_output:
                factor_arr = factor.astype(xp.float32, copy=False)
                factor_arr = xp.where(xp.isfinite(factor_arr), factor_arr, xp.float32(xp.nan))
            else:
                d2h_t0 = time.perf_counter()
                factor_arr = to_numpy(factor)
                d2h_s = time.perf_counter() - d2h_t0
                factor_arr[~np.isfinite(factor_arr)] = np.nan
            del factor
            return idxs, factor_arr, time.perf_counter() - t0, d2h_s
        except Exception:
            return None

    workers = _get_eval_workers(use_backend_output=use_backend_output, task_count=len(tasks))
    if workers == 1:
        iterator = map(_run_task, tasks)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        iterator = executor.map(_run_task, tasks)

    try:
        for result in iterator:
            if result is not None:
                idxs, factor_arr, task_elapsed, d2h_s = result
                results[idxs] = factor_arr
                _perf_add(perf_stats, 'eval_task_exec_s', task_elapsed)
                _perf_add(perf_stats, 'eval_d2h_s', d2h_s)
    finally:
        if workers != 1:
            executor.shutdown(wait=True)

    del tasks, _mask_cache, group_masks
    _update_factor_diagnostics(results, population, perf_stats)
    return results


def apply_composition(factor_values, population, indicator_data=None, indicator_field_indices=None):
    """Apply temporal and cross-sectional composition to base factor values.

    Args:
        factor_values: np.ndarray (pop_size, P, S) float32, base factor output
        population: np.ndarray (pop_size, 13) int, full gene encoding

    Returns:
        np.ndarray (pop_size, P, S) float32, composed factor values
    """
    pop_size = factor_values.shape[0]
    if pop_size == 0 or not _has_active_composition(population):
        return factor_values

    result = factor_values.copy()
    active_idx = np.flatnonzero((population[:, 10] != 0) | (population[:, 12] != 0))

    for i in active_idx:
        f = result[i]  # (P, S)

        # 1. Cross-sectional composition (applied first, per period)
        cs_op_idx = int(population[i, 12])
        cs_op_name = CS_COMP_OPS[cs_op_idx]
        cs_fn = CS_DISPATCH.get(cs_op_name)
        if cs_fn is not None:
            if cs_op_name in ("cs_neutralize_mcap", "cs_group_rank_mcap", "cs_neutralize_liquidity", "cs_group_rank_liquidity") and indicator_data is not None:
                field_name = "cm_mcap_pct_lag1" if "mcap" in cs_op_name else "dollar_volume_rank"
                global_field_idx = INDICATOR_NAMES.index(field_name)
                if indicator_field_indices is None:
                    field_idx = global_field_idx
                else:
                    fields = np.asarray(indicator_field_indices, dtype=np.intp)
                    pos = np.searchsorted(fields, global_field_idx)
                    if pos >= fields.size or fields[pos] != global_field_idx:
                        raise IndexError(f"composition dependency field {field_name!r} is not in cached indicator fields")
                    field_idx = int(pos)
                neutralizer = np.nanmean(to_numpy(indicator_data[:, field_idx, :, :]), axis=1)
                if "neutralize" in cs_op_name:
                    f = cs_neutralize(f, neutralizer)
                else:
                    f = cs_group_rank(f, neutralizer)
            elif cs_op_name not in ("cs_neutralize_mcap", "cs_group_rank_mcap", "cs_neutralize_liquidity", "cs_group_rank_liquidity"):
                f = cs_fn(f)

        # 2. Temporal composition (applied second, across periods)
        ts_op_idx = int(population[i, 10])
        ts_op_name = TS_COMP_OPS[ts_op_idx]
        ts_fn = TS_DISPATCH.get(ts_op_name)
        if ts_fn is not None:
            ts_window = TS_COMP_WINDOWS[int(population[i, 11])]
            f, _ = ts_fn(f, ts_window)

        f[~np.isfinite(f)] = np.nan
        result[i] = f

    return result


def evaluate_population(population, indicator_data, output_backend='cpu',
                        perf_stats=None, indicator_field_indices=None):
    """Full evaluation: base factors + composition.

    Args:
        population: np.ndarray (pop_size, 13) int
        indicator_data: xp.ndarray (P, N_IND, M, S)
        output_backend: 'cpu' or 'xp'
    Returns:
        Array (pop_size, P, S) float32
    """
    base_output_backend = output_backend if not _has_active_composition(population) else 'cpu'
    base_t0 = time.perf_counter()
    base = evaluate_population_base(
        population, indicator_data,
        output_backend=base_output_backend,
        perf_stats=perf_stats,
    )
    _perf_add(perf_stats, 'eval_base_s', time.perf_counter() - base_t0)
    comp_t0 = time.perf_counter()
    composed = apply_composition(base, population, indicator_data=indicator_data,
                                 indicator_field_indices=indicator_field_indices)
    _perf_add(perf_stats, 'eval_comp_s', time.perf_counter() - comp_t0)
    if output_backend == 'xp' and isinstance(composed, np.ndarray):
        h2d_t0 = time.perf_counter()
        out = xp.asarray(composed, dtype=xp.float32)
        _perf_add(perf_stats, 'eval_h2d_s', time.perf_counter() - h2d_t0)
        return out
    return composed
