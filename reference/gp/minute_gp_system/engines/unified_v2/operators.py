"""Backend-agnostic operators: work with both numpy (CPU) and cupy (GPU).

Base operators: same as v1 (mode1 single-var, mode2 dual-var).
All operators: (D, W, S) [, (D, W, S)] -> (D, S)
"""
from .backend import xp

_EPS = 1e-8


def _masked(x, mask):
    return xp.where(mask > 0.5, x, xp.float32(xp.nan))


def _safe_div(a, b):
    return xp.where(xp.abs(b) > _EPS, a / b, xp.float32(xp.nan))


def _first_valid(x):
    valid = xp.isfinite(x)
    any_valid = xp.any(valid, axis=1)
    idx = xp.argmax(valid, axis=1)
    gathered = xp.take_along_axis(x, idx[:, None, :], axis=1)[:, 0, :]
    return xp.where(any_valid, gathered, xp.float32(xp.nan))


def _last_valid(x):
    valid = xp.isfinite(x)
    any_valid = xp.any(valid, axis=1)
    idx = x.shape[1] - 1 - xp.argmax(valid[:, ::-1, :], axis=1)
    gathered = xp.take_along_axis(x, idx[:, None, :], axis=1)[:, 0, :]
    return xp.where(any_valid, gathered, xp.float32(xp.nan))


# ========== Mode 1: Single-variable ==========
def m1_mean(a, mask): return xp.nanmean(_masked(a, mask), axis=1)
def m1_std(a, mask): return xp.nanstd(_masked(a, mask), axis=1, ddof=1)


def m1_zscore(a, mask):
    """(A[last] - rolling_mean) / rolling_std within window — Bollinger-like.
    Scale-invariant, captures "how many sigmas from recent mean".
    """
    x = _masked(a, mask)
    mu = xp.nanmean(x, axis=1)
    sd = xp.nanstd(x, axis=1, ddof=1)
    last = _last_valid(x)
    return _safe_div(last - mu, sd + xp.float32(_EPS))


def m1_momentum(a, mask):
    """Window return: A[last] / A[first] - 1. Captures directional drift."""
    x = _masked(a, mask)
    first = _first_valid(x)
    last = _last_valid(x)
    return _safe_div(last - first, xp.abs(first) + xp.float32(_EPS))


def m1_range_position(a, mask):
    """(A[last] - min) / (max - min) — stochastic-K.
    Scale-invariant, 0 at window low, 1 at window high.
    """
    x = _masked(a, mask)
    lo = xp.nanmin(x, axis=1)
    hi = xp.nanmax(x, axis=1)
    last = _last_valid(x)
    return _safe_div(last - lo, hi - lo + xp.float32(_EPS))

def m1_skew(a, mask):
    x = _masked(a, mask)
    mu = xp.nanmean(x, axis=1, keepdims=True)
    d = x - mu
    n = xp.sum(mask > 0.5, axis=1).astype(xp.float32)
    m2 = xp.nanmean(d**2, axis=1)
    m3 = xp.nanmean(d**3, axis=1)
    return xp.where(n >= 3, m3 / (xp.sqrt(m2 + _EPS)**3 + _EPS), xp.float32(xp.nan))

def m1_kurt(a, mask):
    x = _masked(a, mask)
    mu = xp.nanmean(x, axis=1, keepdims=True)
    d = x - mu
    n = xp.sum(mask > 0.5, axis=1).astype(xp.float32)
    m2 = xp.nanmean(d**2, axis=1)
    m4 = xp.nanmean(d**4, axis=1)
    return xp.where(n >= 4, m4 / (m2**2 + _EPS) - 3.0, xp.float32(xp.nan))

def m1_median(a, mask): return xp.nanmedian(_masked(a, mask), axis=1)
def m1_sum(a, mask): return xp.nansum(_masked(a, mask), axis=1)
def m1_max(a, mask): return xp.nanmax(_masked(a, mask), axis=1)
def m1_min(a, mask): return xp.nanmin(_masked(a, mask), axis=1)
def m1_range(a, mask): return xp.nanmax(_masked(a, mask), axis=1) - xp.nanmin(_masked(a, mask), axis=1)
def m1_first(a, mask): return _first_valid(_masked(a, mask))
def m1_last(a, mask): return _last_valid(_masked(a, mask))

def m1_ret(a, mask):
    x = _masked(a, mask)
    first = _first_valid(x)
    last = _last_valid(x)
    return _safe_div(last - first, xp.abs(first) + _EPS)

def m1_autocorr(a, mask):
    x = _masked(a, mask)
    x0, x1 = x[:, :-1, :], x[:, 1:, :]
    v = xp.isfinite(x0) & xp.isfinite(x1)
    x0 = xp.where(v, x0, 0.0); x1 = xp.where(v, x1, 0.0)
    n = xp.sum(v, axis=1).astype(xp.float32)
    m0 = xp.sum(x0, axis=1) / (n + _EPS)
    m1v = xp.sum(x1, axis=1) / (n + _EPS)
    d0 = xp.where(v, x0 - m0[:, None, :], 0.0)
    d1 = xp.where(v, x1 - m1v[:, None, :], 0.0)
    cov = xp.sum(d0 * d1, axis=1) / (n + _EPS)
    s0 = xp.sqrt(xp.sum(d0**2, axis=1) / (n + _EPS) + _EPS)
    s1 = xp.sqrt(xp.sum(d1**2, axis=1) / (n + _EPS) + _EPS)
    return xp.where(n >= 5, cov / (s0 * s1 + _EPS), xp.float32(xp.nan))

def m1_entropy(a, mask):
    x = _masked(a, mask)
    D, W, S = x.shape
    lo = xp.nanmin(x, axis=1, keepdims=True)
    hi = xp.nanmax(x, axis=1, keepdims=True)
    bins = xp.clip(((x - lo) / (hi - lo + _EPS) * 9.99).astype(xp.int32), 0, 9)
    bins = xp.where(xp.isfinite(x), bins, -1)
    n = xp.sum(xp.isfinite(x), axis=1).astype(xp.float32)
    ent = xp.zeros((D, S), dtype=xp.float32)
    for b in range(10):
        cnt = xp.sum(bins == b, axis=1).astype(xp.float32)
        p = cnt / (n + _EPS)
        ent -= xp.where(p > _EPS, p * xp.log(p + _EPS), 0.0)
    return ent

def m1_zero_cross(a, mask):
    x = _masked(a, mask)
    signs = xp.sign(x)
    diff = xp.abs(signs[:, 1:, :] - signs[:, :-1, :])
    v = xp.isfinite(x[:, 1:, :]) & xp.isfinite(x[:, :-1, :])
    return _safe_div(xp.sum((diff > 0.5) & v, axis=1).astype(xp.float32),
                     xp.sum(v, axis=1).astype(xp.float32))


# ========== Mode 2: Dual-variable ==========
def _paired(a, b, mask):
    v = (mask > 0.5) & xp.isfinite(a) & xp.isfinite(b)
    af = xp.where(v, a, 0.0); bf = xp.where(v, b, 0.0)
    n = xp.sum(v, axis=1).astype(xp.float32)
    sa = xp.sum(af, axis=1); sb = xp.sum(bf, axis=1)
    sa2 = xp.sum(af**2, axis=1); sb2 = xp.sum(bf**2, axis=1)
    sab = xp.sum(af*bf, axis=1)
    ma = sa/(n+_EPS); mb = sb/(n+_EPS)
    va = sa2/(n+_EPS) - ma**2; vb = sb2/(n+_EPS) - mb**2
    cov = sab/(n+_EPS) - ma*mb
    return n, ma, mb, va, vb, cov, v

def m2_corr(a, b, mask):
    n,ma,mb,va,vb,cov,_ = _paired(a,b,mask)
    return xp.where(n>=5, xp.clip(cov/(xp.sqrt(va*vb)+_EPS), -1, 1), xp.float32(xp.nan))

def m2_slope(a, b, mask):
    n,_,_,_,vb,cov,_ = _paired(a,b,mask)
    return xp.where(n>=5, cov/(vb+_EPS), xp.float32(xp.nan))

def m2_intercept(a, b, mask):
    n,ma,mb,_,vb,cov,_ = _paired(a,b,mask)
    slope = cov/(vb+_EPS)
    return xp.where(n>=5, ma - slope*mb, xp.float32(xp.nan))

def m2_r2(a, b, mask):
    n,_,_,va,vb,cov,_ = _paired(a,b,mask)
    r = cov/(xp.sqrt(va*vb)+_EPS)
    return xp.where(n>=5, xp.clip(r**2, 0, 1), xp.float32(xp.nan))

def m2_euc_dist(a, b, mask):
    v = (mask>0.5) & xp.isfinite(a) & xp.isfinite(b)
    n = xp.sum(v, axis=1).astype(xp.float32)
    ma = xp.nanmean(xp.where(v,a,xp.nan), axis=1, keepdims=True)
    mb = xp.nanmean(xp.where(v,b,xp.nan), axis=1, keepdims=True)
    sa = xp.nanstd(xp.where(v,a,xp.nan), axis=1, keepdims=True)+_EPS
    sb = xp.nanstd(xp.where(v,b,xp.nan), axis=1, keepdims=True)+_EPS
    za = (a-ma)/sa; zb = (b-mb)/sb
    d = xp.where(v, (za-zb)**2, 0.0)
    return xp.where(n>=5, xp.sqrt(xp.sum(d,axis=1)/(n+_EPS)), xp.float32(xp.nan))

def m2_cov(a, b, mask):
    n,_,_,_,_,cov,_ = _paired(a,b,mask)
    return xp.where(n>=5, cov, xp.float32(xp.nan))

def m2_cos_sim(a, b, mask):
    v = (mask>0.5) & xp.isfinite(a) & xp.isfinite(b)
    af = xp.where(v,a,0.0); bf = xp.where(v,b,0.0)
    dot = xp.sum(af*bf, axis=1)
    na = xp.sqrt(xp.sum(af**2, axis=1)+_EPS)
    nb = xp.sqrt(xp.sum(bf**2, axis=1)+_EPS)
    return _safe_div(dot, na*nb)

def m2_wmean(a, b, mask):
    v = (mask>0.5) & xp.isfinite(a) & xp.isfinite(b)
    w = xp.where(v & (b>0), b, 0.0)
    av = xp.where(v, a, 0.0)
    return _safe_div(xp.sum(w*av, axis=1), xp.sum(w, axis=1))

def m2_rank_corr(a, b, mask):
    v = (mask>0.5) & xp.isfinite(a) & xp.isfinite(b)
    af = xp.where(v, a, 1e18); bf = xp.where(v, b, 1e18)
    ar = xp.argsort(xp.argsort(af, axis=1), axis=1).astype(xp.float32)
    br = xp.argsort(xp.argsort(bf, axis=1), axis=1).astype(xp.float32)
    return m2_corr(xp.where(v,ar,xp.nan), xp.where(v,br,xp.nan), mask)

def m2_beta(a, b, mask): return m2_slope(a, b, mask)

def m2_residual_std(a, b, mask):
    n,ma,mb,_,vb,cov,v = _paired(a,b,mask)
    slope = cov/(vb+_EPS)
    intercept = ma - slope*mb
    pred = slope[:,None,:]*b + intercept[:,None,:]
    resid = xp.where(v, a-pred, xp.nan)
    return xp.nanstd(resid, axis=1)

def m2_diff_mean(a, b, mask):
    x = _masked(a,mask); y = _masked(b,mask)
    return xp.nanmean(x, axis=1) - xp.nanmean(y, axis=1)

def m2_diff_std(a, b, mask):
    x = _masked(a,mask); y = _masked(b,mask)
    return xp.nanstd(x, axis=1) - xp.nanstd(y, axis=1)

def m2_ratio_mean(a, b, mask):
    x = _masked(a,mask); y = _masked(b,mask)
    return _safe_div(xp.nanmean(x, axis=1), xp.nanmean(y, axis=1))

def m2_ratio_std(a, b, mask):
    x = _masked(a,mask); y = _masked(b,mask)
    return _safe_div(xp.nanstd(x, axis=1), xp.nanstd(y, axis=1))


# ========== Dispatch tables ==========
MODE1_DISPATCH = {
    'mean': m1_mean, 'std': m1_std, 'skew': m1_skew, 'kurt': m1_kurt,
    'median': m1_median, 'sum': m1_sum, 'max': m1_max, 'min': m1_min,
    'zscore': m1_zscore, 'momentum': m1_momentum, 'range_position': m1_range_position,
    'range': m1_range, 'first': m1_first, 'last': m1_last, 'ret': m1_ret,
    'autocorr': m1_autocorr, 'entropy': m1_entropy, 'zero_cross': m1_zero_cross,
}

MODE2_DISPATCH = {
    'corr': m2_corr, 'slope': m2_slope, 'intercept': m2_intercept,
    'r2': m2_r2, 'euc_dist': m2_euc_dist, 'cov': m2_cov,
    'cos_sim': m2_cos_sim, 'wmean': m2_wmean, 'rank_corr': m2_rank_corr,
    'beta': m2_beta, 'residual_std': m2_residual_std,
    'diff_mean': m2_diff_mean, 'diff_std': m2_diff_std,
    'ratio_mean': m2_ratio_mean, 'ratio_std': m2_ratio_std,
}


def m3_mean_ratio(a_s, a_l, mask_s, mask_l):
    ms = xp.nanmean(_masked(a_s, mask_s), axis=1)
    ml = xp.nanmean(_masked(a_l, mask_l), axis=1)
    return _safe_div(ms, ml)


def m3_std_ratio(a_s, a_l, mask_s, mask_l):
    ss = xp.nanstd(_masked(a_s, mask_s), axis=1, ddof=1)
    sl = xp.nanstd(_masked(a_l, mask_l), axis=1, ddof=1)
    return _safe_div(ss, sl)


def m3_mean_diff(a_s, a_l, mask_s, mask_l):
    ms = xp.nanmean(_masked(a_s, mask_s), axis=1)
    ml = xp.nanmean(_masked(a_l, mask_l), axis=1)
    return ms - ml


def m3_zscore_anchor(a_s, a_l, mask_s, mask_l):
    last = _last_valid(_masked(a_s, mask_s))
    ml = xp.nanmean(_masked(a_l, mask_l), axis=1)
    sl = xp.nanstd(_masked(a_l, mask_l), axis=1, ddof=1)
    return _safe_div(last - ml, sl + xp.float32(_EPS))


def m3_std_normalized_diff(a_s, a_l, mask_s, mask_l):
    ms = xp.nanmean(_masked(a_s, mask_s), axis=1)
    ml = xp.nanmean(_masked(a_l, mask_l), axis=1)
    sl = xp.nanstd(_masked(a_l, mask_l), axis=1, ddof=1)
    return _safe_div(ms - ml, sl + xp.float32(_EPS))


MODE3_DISPATCH = {
    'mean_ratio': m3_mean_ratio,
    'std_ratio': m3_std_ratio,
    'mean_diff': m3_mean_diff,
    'zscore_anchor': m3_zscore_anchor,
    'std_normalized_diff': m3_std_normalized_diff,
}


# ========== Mode 4: Intraday grouped compression ==========
def _group_bounds(width, group_count):
    g = max(1, min(int(group_count), int(width)))
    return [(i * width // g, (i + 1) * width // g) for i in range(g)]


def _group_mean(a, mask, group_count):
    vals = []
    cnts = []
    for start, end in _group_bounds(a.shape[1], group_count):
        v = (mask[:, start:end, :] > 0.5) & xp.isfinite(a[:, start:end, :])
        n = xp.sum(v, axis=1).astype(xp.float32)
        s = xp.sum(xp.where(v, a[:, start:end, :], xp.float32(0.0)), axis=1)
        vals.append(xp.where(n > 0, s / (n + xp.float32(_EPS)), xp.float32(xp.nan)))
        cnts.append(n)
    return xp.stack(vals, axis=1), xp.stack(cnts, axis=1)


def _group_sum_positive(a, mask, group_count):
    vals = []
    for start, end in _group_bounds(a.shape[1], group_count):
        v = (mask[:, start:end, :] > 0.5) & xp.isfinite(a[:, start:end, :])
        s = xp.sum(xp.where(v, a[:, start:end, :], xp.float32(0.0)), axis=1)
        vals.append(xp.maximum(s, xp.float32(0.0)))
    return xp.stack(vals, axis=1)


def _b_sorted_group_stats(a, b, mask, group_count):
    """Sort by B within each window, then split valid samples into equal-count buckets."""
    g = max(1, min(int(group_count), int(a.shape[1])))
    valid = (mask > 0.5) & xp.isfinite(a) & xp.isfinite(b)
    positions = xp.arange(a.shape[1], dtype=xp.float32)[None, :, None]
    # Preserve time order when B ties while keeping B as the dominant sort key.
    key = xp.where(valid, b + positions * xp.float32(1e-6), xp.float32(xp.inf))
    order = xp.argsort(key, axis=1)
    a_sorted = xp.take_along_axis(a, order, axis=1)
    b_sorted = xp.take_along_axis(b, order, axis=1)
    valid_sorted = xp.take_along_axis(valid.astype(xp.float32), order, axis=1) > 0.5
    counts = xp.sum(valid_sorted, axis=1).astype(xp.int32)

    a_means = []
    b_means = []
    positive_sums = []
    bucket_counts = []
    rank = xp.arange(a.shape[1], dtype=xp.int32)[None, :, None]
    for gi in range(g):
        start = (counts * gi) // g
        end = (counts * (gi + 1)) // g
        sel = valid_sorted & (rank >= start[:, None, :]) & (rank < end[:, None, :])
        n = xp.sum(sel, axis=1).astype(xp.float32)
        sa = xp.sum(xp.where(sel, a_sorted, xp.float32(0.0)), axis=1)
        sb = xp.sum(xp.where(sel, b_sorted, xp.float32(0.0)), axis=1)
        pos = xp.sum(xp.where(sel, xp.maximum(a_sorted, xp.float32(0.0)), xp.float32(0.0)), axis=1)
        a_means.append(xp.where(n > 0, sa / (n + xp.float32(_EPS)), xp.float32(xp.nan)))
        b_means.append(xp.where(n > 0, sb / (n + xp.float32(_EPS)), xp.float32(xp.nan)))
        positive_sums.append(pos)
        bucket_counts.append(n)
    return (
        xp.stack(a_means, axis=1),
        xp.stack(b_means, axis=1),
        xp.stack(positive_sums, axis=1),
        xp.stack(bucket_counts, axis=1),
    )


def _slope_over_groups(gm):
    valid = xp.isfinite(gm)
    n = xp.sum(valid, axis=1).astype(xp.float32)
    x = xp.arange(gm.shape[1], dtype=xp.float32)[None, :, None]
    xv = xp.where(valid, x, xp.float32(0.0))
    yv = xp.where(valid, gm, xp.float32(0.0))
    mx = xp.sum(xv, axis=1) / (n + xp.float32(_EPS))
    my = xp.sum(yv, axis=1) / (n + xp.float32(_EPS))
    dx = xp.where(valid, x - mx[:, None, :], xp.float32(0.0))
    dy = xp.where(valid, gm - my[:, None, :], xp.float32(0.0))
    cov = xp.sum(dx * dy, axis=1)
    var = xp.sum(dx * dx, axis=1)
    return xp.where(n >= 2, cov / (var + xp.float32(_EPS)), xp.float32(xp.nan))


def m4_group_slope(a, b, mask, group_count):
    gm, _, _, _ = _b_sorted_group_stats(a, b, mask, group_count)
    return _slope_over_groups(gm)


def m4_group_dispersion(a, b, mask, group_count):
    gm, _, _, _ = _b_sorted_group_stats(a, b, mask, group_count)
    valid = xp.isfinite(gm)
    n = xp.sum(valid, axis=1).astype(xp.float32)
    return xp.where(n >= 2, xp.nanstd(gm, axis=1, ddof=1), xp.float32(xp.nan))


def m4_group_early_late_diff(a, b, mask, group_count):
    gm, _, _, _ = _b_sorted_group_stats(a, b, mask, group_count)
    g = gm.shape[1]
    mid = max(1, g // 2)
    early = xp.nanmean(gm[:, :mid, :], axis=1)
    late = xp.nanmean(gm[:, mid:, :], axis=1)
    n_early = xp.sum(xp.isfinite(gm[:, :mid, :]), axis=1)
    n_late = xp.sum(xp.isfinite(gm[:, mid:, :]), axis=1)
    return xp.where((n_early > 0) & (n_late > 0), late - early, xp.float32(xp.nan))


def m4_group_top_share(a, b, mask, group_count):
    _, _, gs, _ = _b_sorted_group_stats(a, b, mask, group_count)
    total = xp.sum(gs, axis=1)
    top = xp.max(gs, axis=1)
    return xp.where(total > xp.float32(_EPS), top / (total + xp.float32(_EPS)), xp.float32(xp.nan))


def _paired_group_stats(a, b, mask, group_count):
    ga, gb, _, _ = _b_sorted_group_stats(a, b, mask, group_count)
    valid = xp.isfinite(ga) & xp.isfinite(gb)
    n = xp.sum(valid, axis=1).astype(xp.float32)
    af = xp.where(valid, ga, xp.float32(0.0))
    bf = xp.where(valid, gb, xp.float32(0.0))
    ma = xp.sum(af, axis=1) / (n + xp.float32(_EPS))
    mb = xp.sum(bf, axis=1) / (n + xp.float32(_EPS))
    da = xp.where(valid, ga - ma[:, None, :], xp.float32(0.0))
    db = xp.where(valid, gb - mb[:, None, :], xp.float32(0.0))
    cov = xp.sum(da * db, axis=1) / (n + xp.float32(_EPS))
    va = xp.sum(da * da, axis=1) / (n + xp.float32(_EPS))
    vb = xp.sum(db * db, axis=1) / (n + xp.float32(_EPS))
    return n, va, vb, cov


def m4_group_corr(a, b, mask, group_count):
    n, va, vb, cov = _paired_group_stats(a, b, mask, group_count)
    corr = cov / (xp.sqrt(va * vb) + xp.float32(_EPS))
    return xp.where(n >= 3, xp.clip(corr, -1.0, 1.0), xp.float32(xp.nan))


def m4_group_beta(a, b, mask, group_count):
    n, _, vb, cov = _paired_group_stats(a, b, mask, group_count)
    beta = cov / (vb + xp.float32(_EPS))
    return xp.where(n >= 3, beta, xp.float32(xp.nan))


def m4_group_skew(a, b, mask, group_count):
    gm, _, _, _ = _b_sorted_group_stats(a, b, mask, group_count)
    valid = xp.isfinite(gm)
    n = xp.sum(valid, axis=1).astype(xp.float32)
    mu = xp.sum(xp.where(valid, gm, xp.float32(0.0)), axis=1, keepdims=True) / (n[:, None, :] + xp.float32(_EPS))
    d = gm - mu
    m2 = xp.sum(xp.where(valid, d ** 2, xp.float32(0.0)), axis=1) / (n + xp.float32(_EPS))
    m3 = xp.sum(xp.where(valid, d ** 3, xp.float32(0.0)), axis=1) / (n + xp.float32(_EPS))
    return xp.where(n >= 3, m3 / (m2 ** 1.5 + xp.float32(_EPS)), xp.float32(xp.nan))


def m4_group_path_length(a, b, mask, group_count):
    gm, _, _, _ = _b_sorted_group_stats(a, b, mask, group_count)
    # path_efficiency = |net displacement| / total path length
    # - 1.0 表示组均值序列完全单调（从首组直接走到末组）
    # - 接近 0 表示组均值序列反复震荡、路径迂回
    diff = xp.abs(gm[:, 1:, :] - gm[:, :-1, :])
    valid_step = xp.isfinite(gm[:, 1:, :]) & xp.isfinite(gm[:, :-1, :])
    path_length = xp.sum(xp.where(valid_step, diff, xp.float32(0.0)), axis=1)

    first = gm[:, 0, :]
    last = gm[:, -1, :]
    net_displacement = xp.abs(last - first)
    valid_net = xp.isfinite(first) & xp.isfinite(last)

    return xp.where(
        valid_net & (path_length > xp.float32(_EPS)),
        net_displacement / (path_length + xp.float32(_EPS)),
        xp.float32(xp.nan),
    )


def m4_group_jump_ratio(a, b, mask, group_count):
    gm, _, _, _ = _b_sorted_group_stats(a, b, mask, group_count)
    valid = xp.isfinite(gm)
    n = xp.sum(valid, axis=1).astype(xp.float32)
    mu = xp.sum(xp.where(valid, gm, xp.float32(0.0)), axis=1, keepdims=True) / (n[:, None, :] + xp.float32(_EPS))
    sd = xp.sqrt(xp.sum(xp.where(valid, (gm - mu) ** 2, xp.float32(0.0)), axis=1, keepdims=True) / (n[:, None, :] + xp.float32(_EPS)))
    jumps = xp.abs(gm - mu) > (sd + xp.float32(_EPS))
    jump_count = xp.sum(jumps & valid, axis=1).astype(xp.float32)
    return xp.where(n >= 3, jump_count / (n + xp.float32(_EPS)), xp.float32(xp.nan))


def m4_group_first_last_diff(a, b, mask, group_count):
    gm, _, _, _ = _b_sorted_group_stats(a, b, mask, group_count)
    first = gm[:, 0, :]
    last = gm[:, -1, :]
    n_first = xp.sum(xp.isfinite(gm[:, 0:1, :]), axis=1)
    n_last = xp.sum(xp.isfinite(gm[:, -1:, :]), axis=1)
    return xp.where((n_first > 0) & (n_last > 0), last - first, xp.float32(xp.nan))


# ========== Mode 4: Volume-clock variants ==========
# 使用 pair 的 b 通道作为成交量代理（权重），按累计成交量等分时间轴

def _volume_group_labels(b, mask, group_count):
    """按累计 |b|（成交量代理）把时间轴分成 group_count 组，返回组标签 (D,W,S)。
    b 通道是成交量代理，无需改 evaluator 签名。
    """
    g = max(1, int(group_count))
    v = (mask > 0.5) & xp.isfinite(b)
    w = xp.where(v, xp.abs(b), xp.float32(0.0))
    cum = xp.cumsum(w, axis=1)
    total = cum[:, -1:, :]
    frac = xp.where(total > _EPS, cum / (total + _EPS), xp.float32(0.0))
    labels = xp.clip((frac * g).astype(xp.int32), 0, g - 1)
    return xp.where(v, labels, xp.int32(-1)), g


def _volume_group_mean(a, b, mask, group_count):
    """按成交量时钟分组，计算 a 的组均值，返回 (D, g, S)。"""
    labels, g = _volume_group_labels(b, mask, group_count)
    va = (mask > 0.5) & xp.isfinite(a)
    vals = []
    for gi in range(g):
        sel = va & (labels == gi)
        n = xp.sum(sel, axis=1).astype(xp.float32)
        s = xp.sum(xp.where(sel, a, xp.float32(0.0)), axis=1)
        vals.append(xp.where(n > 0, s / (n + _EPS), xp.float32(xp.nan)))
    return xp.stack(vals, axis=1)


def m4_group_dispersion_vclock(a, b, mask, group_count):
    gm = _volume_group_mean(a, b, mask, group_count)
    valid = xp.isfinite(gm)
    n = xp.sum(valid, axis=1).astype(xp.float32)
    return xp.where(n >= 2, xp.nanstd(gm, axis=1, ddof=1), xp.float32(xp.nan))


def m4_group_slope_vclock(a, b, mask, group_count):
    gm = _volume_group_mean(a, b, mask, group_count)
    return _slope_over_groups(gm)


def m4_group_path_length_vclock(a, b, mask, group_count):
    gm = _volume_group_mean(a, b, mask, group_count)
    diff = xp.abs(gm[:, 1:, :] - gm[:, :-1, :])
    valid_step = xp.isfinite(gm[:, 1:, :]) & xp.isfinite(gm[:, :-1, :])
    path_length = xp.sum(xp.where(valid_step, diff, xp.float32(0.0)), axis=1)
    first = gm[:, 0, :]
    last = gm[:, -1, :]
    net = xp.abs(last - first)
    valid_net = xp.isfinite(first) & xp.isfinite(last)
    return xp.where(
        valid_net & (path_length > _EPS),
        net / (path_length + _EPS),
        xp.float32(xp.nan),
    )


def m4_group_early_late_diff_vclock(a, b, mask, group_count):
    gm = _volume_group_mean(a, b, mask, group_count)
    g = gm.shape[1]
    mid = max(1, g // 2)
    early = xp.nanmean(gm[:, :mid, :], axis=1)
    late = xp.nanmean(gm[:, mid:, :], axis=1)
    n_e = xp.sum(xp.isfinite(gm[:, :mid, :]), axis=1)
    n_l = xp.sum(xp.isfinite(gm[:, mid:, :]), axis=1)
    return xp.where((n_e > 0) & (n_l > 0), late - early, xp.float32(xp.nan))


MODE4_DISPATCH = {
    "group_slope": m4_group_slope,
    "group_dispersion": m4_group_dispersion,
    "group_early_late_diff": m4_group_early_late_diff,
    "group_top_share": m4_group_top_share,
    "group_corr": m4_group_corr,
    "group_beta": m4_group_beta,
    "group_skew": m4_group_skew,
    "group_path_length": m4_group_path_length,
    "group_jump_ratio": m4_group_jump_ratio,
    "group_first_last_diff": m4_group_first_last_diff,
    "group_dispersion_vclock": m4_group_dispersion_vclock,
    "group_slope_vclock": m4_group_slope_vclock,
    "group_path_length_vclock": m4_group_path_length_vclock,
    "group_early_late_diff_vclock": m4_group_early_late_diff_vclock,
}
