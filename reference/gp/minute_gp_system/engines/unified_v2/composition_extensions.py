"""Additional temporal and cross-sectional composition operators."""

import numpy as np

_EPS = 1e-8


def _row_mean(x):
    valid = np.isfinite(x)
    count = valid.sum(axis=1, keepdims=True)
    return np.divide(np.where(valid, x, 0.0).sum(axis=1, keepdims=True),
                     np.maximum(count, 1),
                     out=np.full((x.shape[0], 1), np.nan, dtype=x.dtype),
                     where=count > 0)


def _cs_robust_zscore(factor):
    median = np.nanmedian(factor, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(factor - median), axis=1, keepdims=True) * 1.4826
    return (factor - median) / np.where(mad < _EPS, 1.0, mad)


def _cs_winsor_zscore(factor):
    lo = np.nanpercentile(factor, 5.0, axis=1, keepdims=True)
    hi = np.nanpercentile(factor, 95.0, axis=1, keepdims=True)
    clipped = np.clip(factor, lo, hi)
    mean = _row_mean(clipped)
    valid = np.isfinite(clipped)
    count = valid.sum(axis=1, keepdims=True)
    std = np.sqrt(np.sum(np.where(valid, clipped - mean, 0.0) ** 2, axis=1, keepdims=True) /
                  np.maximum(count, 1))
    return (clipped - mean) / np.where(std < _EPS, 1.0, std)


def _cs_tail_rank(factor):
    valid = np.isfinite(factor)
    count = valid.sum(axis=1, keepdims=True).astype(np.float32)
    filled = np.where(valid, factor, np.inf)
    order = np.argsort(np.argsort(filled, axis=1), axis=1).astype(np.float32)
    rank = np.where(valid, order / (count - 1 + _EPS), np.nan)
    centered = rank - 0.5
    threshold = 0.35
    return np.sign(centered) * np.maximum(np.abs(centered) - threshold, 0.0) / (0.5 - threshold + _EPS)


def cs_neutralize(factor, neutralizer):
    """Per-period residual from y = alpha + beta * neutralizer."""
    valid = np.isfinite(factor) & np.isfinite(neutralizer)
    count = valid.sum(axis=1, keepdims=True)
    x = np.where(valid, neutralizer, 0.0)
    y = np.where(valid, factor, 0.0)
    xm = x.sum(axis=1, keepdims=True) / np.maximum(count, 1)
    ym = y.sum(axis=1, keepdims=True) / np.maximum(count, 1)
    xc = np.where(valid, neutralizer - xm, 0.0)
    yc = np.where(valid, factor - ym, 0.0)
    denom = np.sum(xc * xc, axis=1, keepdims=True)
    beta = np.divide(np.sum(xc * yc, axis=1, keepdims=True), denom,
                     out=np.full_like(denom, np.nan), where=denom > _EPS)
    residual = factor - (ym + beta * (neutralizer - xm))
    return np.where(valid & (count >= 3) & np.isfinite(residual), residual, np.nan)


def cs_group_rank(factor, neutralizer, n_groups=5):
    """Percentile-rank factor within per-period neutralizer quantile buckets."""
    result = np.full_like(factor, np.nan, dtype=np.float32)
    for t in range(factor.shape[0]):
        valid = np.isfinite(factor[t]) & np.isfinite(neutralizer[t])
        if valid.sum() < 2:
            continue
        edges = np.nanpercentile(neutralizer[t, valid], np.linspace(0.0, 100.0, n_groups + 1))
        groups = np.searchsorted(edges[1:-1], neutralizer[t], side="right")
        for group in range(n_groups):
            idx = valid & (groups == group)
            count = int(idx.sum())
            if count == 1:
                result[t, idx] = 0.5
            elif count > 1:
                order = np.argsort(np.argsort(factor[t, idx])).astype(np.float32)
                result[t, idx] = order / (count - 1 + _EPS)
    return result


def _ts_ewm_residual(factor, w):
    decay, _ = _ts_decay(factor, w)
    return factor - decay, 0


def _ts_fast_slow_diff(factor, w):
    fast, _ = _ts_decay(factor, max(1, int(w) // 3))
    slow, _ = _ts_decay(factor, w)
    return fast - slow, 0


def _ts_rolling_tstat(factor, w):
    result = np.full_like(factor, np.nan)
    for t in range(w - 1, factor.shape[0]):
        window = factor[t - w + 1:t + 1]
        valid = np.isfinite(window)
        count = valid.sum(axis=0)
        mean = np.nanmean(window, axis=0)
        std = np.nanstd(window, axis=0)
        result[t] = np.where(count >= 3, mean / (std / np.sqrt(np.maximum(count, 1)) + _EPS), np.nan)
    return result, 0


def _ts_robust_zscore(factor, w):
    result = np.full_like(factor, np.nan)
    for t in range(w - 1, factor.shape[0]):
        window = factor[t - w + 1:t + 1]
        median = np.nanmedian(window, axis=0)
        mad = np.nanmedian(np.abs(window - median), axis=0) * 1.4826
        result[t] = (factor[t] - median) / np.where(mad < _EPS, 1.0, mad)
    return result, 0


def _ts_persistence_ratio(factor, w):
    result = np.full_like(factor, np.nan)
    for t in range(w - 1, factor.shape[0]):
        window = factor[t - w + 1:t + 1]
        valid = np.isfinite(window)
        count = valid.sum(axis=0)
        result[t] = np.where(count > 0,
                             np.maximum((window > 0).sum(axis=0), (window < 0).sum(axis=0)) / count,
                             np.nan)
    return result, 0


def _ts_event_count(factor, w):
    result = np.full_like(factor, np.nan)
    for t in range(w - 1, factor.shape[0]):
        window = factor[t - w + 1:t + 1]
        valid = np.isfinite(window)
        count = valid.sum(axis=0)
        result[t] = np.where(count > 0, ((np.abs(window) > 1.0) & valid).sum(axis=0) / count, np.nan)
    return result, 0


def _ts_drawup(factor, w):
    result = np.full_like(factor, np.nan)
    for t in range(w - 1, factor.shape[0]):
        result[t] = factor[t] - np.nanmin(factor[t - w + 1:t + 1], axis=0)
    return result, 0


def _ts_drawdown(factor, w):
    result = np.full_like(factor, np.nan)
    for t in range(w - 1, factor.shape[0]):
        result[t] = np.nanmax(factor[t - w + 1:t + 1], axis=0) - factor[t]
    return result, 0


def _ts_half_life_decay(factor, w):
    alpha = 1.0 - 0.5 ** (1.0 / max(1.0, float(w)))
    result = np.full_like(factor, np.nan)
    ema = factor[0].copy()
    result[0] = ema
    for t in range(1, factor.shape[0]):
        valid = np.isfinite(factor[t])
        ema = np.where(valid, np.where(np.isfinite(ema), alpha * factor[t] + (1 - alpha) * ema, factor[t]), ema)
        result[t] = ema
    return result, 0


def _ts_decay(factor, w):
    alpha = 2.0 / (w + 1)
    result = np.full_like(factor, np.nan)
    ema = factor[0].copy()
    result[0] = ema
    for t in range(1, factor.shape[0]):
        valid = np.isfinite(factor[t])
        ema = np.where(valid, np.where(np.isfinite(ema), alpha * factor[t] + (1 - alpha) * ema, factor[t]), ema)
        result[t] = ema
    return result, 0


TS_EXTENSION_DISPATCH = {
    "ewm_residual": _ts_ewm_residual, "fast_slow_diff": _ts_fast_slow_diff,
    "rolling_tstat": _ts_rolling_tstat, "robust_zscore": _ts_robust_zscore,
    "persistence_ratio": _ts_persistence_ratio, "event_count": _ts_event_count,
    "drawup": _ts_drawup, "drawdown": _ts_drawdown,
    "half_life_decay": _ts_half_life_decay,
}

CS_EXTENSION_DISPATCH = {
    "cs_robust_zscore": _cs_robust_zscore, "cs_winsor_zscore": _cs_winsor_zscore,
    "cs_tail_rank": _cs_tail_rank,
    "cs_neutralize_mcap": cs_neutralize, "cs_neutralize_liquidity": cs_neutralize,
    "cs_group_rank_mcap": cs_group_rank, "cs_group_rank_liquidity": cs_group_rank,
}
