"""Fitness evaluation for GP factor mining."""
import os
import numpy as np


class _Fitness:
    __slots__ = ('function', 'greater_is_better', 'sign')

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)


def _dense_minmax_rank_2d(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A)
    if A.ndim != 2: raise ValueError("A must be 2D")
    mask = np.isfinite(A)
    n_rows, n_cols = A.shape
    if n_cols == 0: return np.empty_like(A, dtype=np.float32)
    dtype = A.dtype if A.dtype in (np.float32, np.float64) else np.float32
    filled = np.where(mask, A, np.inf).astype(dtype, copy=False)
    sorter = np.argsort(filled, axis=1, kind="quicksort")
    sorted_vals = np.take_along_axis(filled, sorter, axis=1)
    sorted_valid = np.take_along_axis(mask, sorter, axis=1)
    first = sorted_valid[:, :1].astype(np.int32)
    change = (sorted_vals[:, 1:] != sorted_vals[:, :-1]) & sorted_valid[:, 1:]
    dense_sorted = np.cumsum(np.concatenate([first, change.astype(np.int32)], axis=1), axis=1).astype(np.int32)
    dense_sorted = np.where(sorted_valid, dense_sorted, 0)
    dense = np.empty((n_rows, n_cols), dtype=np.int32)
    np.put_along_axis(dense, sorter, dense_sorted, axis=1)
    max_rank = dense.max(axis=1).astype(np.int32)
    denom = (max_rank - 1).astype(dtype)
    out = np.full((n_rows, n_cols), np.nan, dtype=dtype)
    ok = (max_rank - 1) > 0
    if np.any(ok):
        out_ok = (dense[ok].astype(dtype) - 1.0) / denom[ok][:, None]
        out[ok] = np.where(dense[ok] > 0, out_ok, np.nan)
    return out


def topk_mean_return(factor, y, uni_mask=None, top_frac=0.2, min_top_cnt=50):
    valid = np.isfinite(factor) & np.isfinite(y)
    if uni_mask is not None: valid &= uni_mask
    fv = np.where(valid, factor, -np.inf)
    n = fv.shape[1]
    k = int(max(1, min(n, round(n * float(top_frac)))))
    idx = np.argpartition(fv, n - k, axis=1)[:, n - k:]
    rows = np.arange(fv.shape[0])[:, None]
    top_f, top_y = fv[rows, idx], y[rows, idx]
    top_ok = np.isfinite(top_y) & np.isfinite(top_f) & (top_f > -1e300)
    top_sum = np.nansum(np.where(top_ok, top_y, np.nan), axis=1)
    top_cnt = np.sum(top_ok, axis=1)
    return np.divide(top_sum, top_cnt, out=np.zeros_like(top_sum), where=top_cnt > min_top_cnt)


def score_from_factor(formula, factor, y, uni_mask=None, top_frac=0.2, min_top_cnt=50,
                      min_years=1, n_trading_days=243, lag=2, min_unique_cnt=50, min_unique_day_frac=0.5):
    if lag > 0:
        factor, y = factor[:-lag], y[lag:]
        if uni_mask is not None: uni_mask = uni_mask[:-lag]

    if min_unique_cnt and min_unique_cnt > 1:
        nd = factor.shape[0]
        step = max(1, nd // min(200, nd))
        ok_days = total = 0
        for d in range(0, nd, step):
            row = factor[d]
            v = np.isfinite(row)
            if v.sum() < 10: continue
            total += 1
            if len(np.unique(row[v])) >= min_unique_cnt: ok_days += 1
        if total > 0 and ok_days / total < min_unique_day_frac: return None

    daily_ret = topk_mean_return(factor, y, uni_mask=uni_mask, top_frac=top_frac, min_top_cnt=min_top_cnt)
    daily_ret = np.clip(np.nan_to_num(daily_ret, nan=0.0), -0.99, 10.0)
    n_days = daily_ret.shape[0]
    n_years = n_days // n_trading_days
    if n_years < min_years: return None

    mat = daily_ret[:n_years * n_trading_days].reshape(n_years, n_trading_days)
    yearly = np.expm1(np.log1p(mat).sum(axis=1)).tolist()
    if not yearly: return None

    total_log = np.log1p(daily_ret[:n_years * n_trading_days]).sum()
    cagr = float(np.expm1(total_log / n_years))
    curve = np.exp(np.cumsum(np.log1p(daily_ret[:n_years * n_trading_days])))
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.max(1.0 - curve / (peak + 1e-12)))
    n_recent = min(3, len(yearly))
    return {
        'formula': formula, 'yearly': yearly,
        'mean': float(np.mean(yearly)),
        'recent': float(np.mean(yearly[-n_recent:])),
        'early': float(np.mean(yearly[:-n_recent])) if len(yearly) > n_recent else float(np.mean(yearly)),
        'win_rate': sum(1 for r in yearly if r > 0) / len(yearly),
        'min': float(min(yearly)), 'max': float(max(yearly)),
        'std': float(np.std(yearly)) if len(yearly) > 1 else 0.0,
        'cagr': cagr, 'max_dd': max_dd,
    }


def compute_factor_score(r):
    if r is None: return -999
    cagr = float(r.get('cagr', r.get('mean', 0.0)))
    base = 200.0 * cagr
    max_dd = float(r.get('max_dd', 1.0))
    dd_pen = 0.0 if max_dd <= 0.2 else (max_dd - 0.2) * 120.0 if max_dd <= 0.6 else 80.0 + (max_dd - 0.6) * 300.0
    recent_bonus = 30.0 * float(r.get('recent', 0.0))
    min_y = float(r.get('min', 0.0))
    min_pen = (-0.3 - min_y) * 80.0 if min_y < -0.3 else 0.0
    return base + recent_bonus - dd_pen - min_pen


def _quality_guard(y_pred, y, quantile, min_valid_frac, min_valid_day_frac,
                   min_top_frac, min_top_day_frac, min_valid_cnt=None, min_top_cnt=None,
                   min_nonzero_frac=None, min_nonzero_day_frac=0.0, nonzero_epsilon=1e-6,
                   min_unique_bins=None, min_unique_day_frac=0.0):
    if y_pred.ndim != 2: return False, None
    nd, ns = y_pred.shape
    valid = np.isfinite(y_pred)
    vcnt = valid.sum(axis=1)
    min_vc = int(min_valid_cnt) if min_valid_cnt else max(1, int(min_valid_frac * ns))
    ok_cov = vcnt >= min_vc
    if ok_cov.sum() < 30 or ok_cov.mean() < min_valid_day_frac: return False, None

    if min_nonzero_frac and float(min_nonzero_frac) > 0:
        eps = float(nonzero_epsilon) if nonzero_epsilon else 1e-6
        nz_cnt = (valid & (np.abs(y_pred) > eps)).sum(axis=1)
        ok_nz = nz_cnt >= max(1, int(float(min_nonzero_frac) * ns))
        if ok_nz.mean() < float(min_nonzero_day_frac): return False, None

    pred_rank = _dense_minmax_rank_2d(np.where(valid, y_pred, np.nan))
    exp_top = max(1, int(quantile * ns))
    min_tc = int(min_top_cnt) if min_top_cnt else max(1, int(exp_top * min_top_frac))
    min_tc = min(min_tc, exp_top)
    top_mask = pred_rank > (1 - quantile)
    ok_top = (top_mask & valid).sum(axis=1) >= min_tc
    if np.mean(ok_top[ok_cov]) < min_top_day_frac: return False, None

    if min_unique_bins and int(min_unique_bins) > 1:
        k = int(min_unique_bins)
        p = float(min_unique_day_frac) if min_unique_day_frac else 0.0
        day_all = np.flatnonzero(ok_cov)
        if day_all.size == 0: return False, None
        ds = max(32, min(128, day_all.size))
        day_idx = day_all[::max(1, day_all.size // ds)][:ds]
        min_uniq = max(k * 8, 50)
        ok_cnt = sum(1 for d in day_idx if valid[d].sum() >= 10 and len(np.unique(y_pred[d][valid[d]])) >= min_uniq)
        if day_idx.size > 0 and ok_cnt / day_idx.size < p: return False, None

    return True, pred_rank


def _rolling_sum_strides(a, w):
    if w == 1: return a
    af = a.copy()
    np.nan_to_num(af, copy=False, nan=0.0)
    ret = np.cumsum(af, axis=0)
    ret[w:] = ret[w:] - ret[:-w]
    return ret


def make_fitness_long_return(quantile=0.05, n_trading_days=252, horizons=None, horizon_weights=None,
                             neutralize_fn=None, benchmark=None, lag_periods=0,
                             use_quality_guard=True, align_backtest=True, universe_mask=None,
                             min_valid_frac=0.3, min_valid_day_frac=0.7, min_top_frac=0.6,
                             min_top_day_frac=0.8, min_valid_cnt=None, min_top_cnt=None,
                             min_nonzero_frac=None, min_nonzero_day_frac=0.0, nonzero_epsilon=None,
                             min_unique_bins=None, min_unique_day_frac=0.0, **_kw):
    _cache = {}
    _max_cache = int(os.environ.get("GP_FITNESS_CACHE_MAX", "8") or 8)

    def _fitness(y, y_pred, w):
        mask = (w > 0)
        ys, yps = y[mask], y_pred[mask]
        bms = benchmark[mask] if (benchmark is not None and len(benchmark) == len(w)) else benchmark
        unis = universe_mask[mask] if (universe_mask is not None and len(universe_mask) == len(w)) else None

        if not np.isfinite(yps).all() and np.isnan(yps).mean() > 0.8: return -1e6
        if unis is not None: yps = np.where(unis, yps, np.nan)
        if neutralize_fn:
            try: yps = neutralize_fn(yps, time_mask=mask)
            except Exception: pass
        if yps.ndim != 2: return -1e6

        lp = int(lag_periods) if lag_periods else 0
        if lp > 0:
            tmp = np.full_like(yps, np.nan, dtype=np.float32)
            if lp < yps.shape[0]: tmp[lp:] = yps[:-lp]
            yps = tmp

        if use_quality_guard:
            ok, pred_rank = _quality_guard(
                yps, ys, quantile, min_valid_frac, min_valid_day_frac,
                min_top_frac, min_top_day_frac, min_valid_cnt=min_valid_cnt, min_top_cnt=min_top_cnt,
                min_nonzero_frac=min_nonzero_frac, min_nonzero_day_frac=min_nonzero_day_frac,
                nonzero_epsilon=nonzero_epsilon, min_unique_bins=min_unique_bins, min_unique_day_frac=min_unique_day_frac)
            if not ok: return -1e6
        else:
            ns = int(yps.shape[1])
            k = int(max(1, min(ns, round(float(quantile) * ns))))
            fv = np.where(np.isfinite(yps), yps, -np.inf).astype(np.float32, copy=False)
            idx_top = np.argpartition(fv, ns - k, axis=1)[:, ns - k:]
            rows = np.arange(fv.shape[0])[:, None]

        y_key = (id(y), tuple(horizons) if horizons else (1,), id(w))
        if y_key not in _cache:
            if _max_cache > 0 and len(_cache) >= _max_cache:
                try: _cache.pop(next(iter(_cache)))
                except Exception: _cache.clear()
            if horizons:
                cy, cb = [], []
                for h in horizons:
                    cy.append(_rolling_sum_strides(ys, h))
                    if bms is not None:
                        bm_t = bms if bms.ndim == 2 else bms.reshape(-1, 1)
                        cb.append(_rolling_sum_strides(bm_t, h))
                _cache[y_key] = (cy, cb)
            else:
                _cache[y_key] = ([ys], [bms] if bms is not None else [None])

        all_y, all_bm = _cache[y_key]
        ws = horizon_weights or [1.0] * len(all_y)
        total = 0.0

        for i, yh in enumerate(all_y):
            h = horizons[i] if horizons else 1
            bh = all_bm[i] if all_bm else None

            if use_quality_guard:
                tmask = pred_rank > (1 - quantile)
                tsel = tmask & np.isfinite(yh)
                tsum = np.nansum(np.where(tsel, yh, np.nan), axis=1)
                tcnt = np.sum(tsel, axis=1)
                tr = np.divide(tsum, tcnt, out=np.zeros_like(tsum), where=tcnt != 0)
            else:
                tf = fv[rows, idx_top]
                ty = yh[rows, idx_top]
                tok = np.isfinite(ty) & np.isfinite(tf) & (tf > -1e30)
                tsum = np.nansum(np.where(tok, ty, np.nan), axis=1)
                tcnt = np.sum(tok, axis=1)
                tr = np.divide(tsum, tcnt, out=np.zeros_like(tsum), where=tcnt != 0)

            excess = tr - (np.nanmean(bh, axis=1) if (bh is not None and bh.ndim == 2) else bh) if bh is not None else tr

            if align_backtest and horizons is None:
                daily = np.clip(np.nan_to_num(excess, nan=0.0), -0.999999999, 10.0)
                n_d = daily.shape[0]
                ny = n_d / float(n_trading_days) if n_trading_days > 0 else 0.0
                t = float(np.prod(1.0 + daily))
                score = (t ** (1.0 / ny) - 1.0) if ny > 0 else 0.0
            else:
                de = np.clip(np.nan_to_num(excess, nan=0.0) / float(h), -0.999999999, 10.0)
                score = np.expm1(np.mean(np.log1p(de)) * n_trading_days)

            total += ws[i] * score
        return total

    return _Fitness(function=_fitness, greater_is_better=True)
