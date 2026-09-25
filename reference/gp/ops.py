"""GP operator library - Numba-accelerated rolling/cross-sectional operators."""
import numpy as np
from copy import deepcopy
from numba import njit, prange

# ---------------------------------------------------------------------------
# _Function node
# ---------------------------------------------------------------------------
class _Function:
    __slots__ = ('function', 'name', 'arity', 'isRandom', 'RandRange',
                 'baseConst', 'role', 'is_ts_op', 'complexity_weight')

    def __init__(self, function, name, arity, isRandom=(False, (1, 100)),
                 role='other', is_ts_op=False, complexity_weight=1.0):
        self.function = function
        self.name = name
        self.arity = arity
        self.isRandom = isRandom[0]
        self.RandRange = isRandom[1]
        self.baseConst = -1
        self.role = role
        self.is_ts_op = bool(is_ts_op)
        self.complexity_weight = float(complexity_weight)

    def __call__(self, *args):
        if self.isRandom and self.baseConst > 0:
            return self.function(*args, self.baseConst)
        return self.function(*args)


# ---------------------------------------------------------------------------
# Numba JIT rolling kernels
# ---------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _rolling_mean(A, w):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    w = int(w)
    if w <= 1:
        return A.copy()
    for j in prange(ns):
        buf = np.empty(w, dtype=np.float64)
        okb = np.zeros(w, dtype=np.uint8)
        s, c = 0.0, 0
        for i in range(nd):
            bi = i % w
            if okb[bi] == 1:
                s -= buf[bi]; c -= 1; okb[bi] = 0
            v = A[i, j]
            if np.isfinite(v):
                buf[bi] = v; okb[bi] = 1; s += v; c += 1
            if c > 0:
                out[i, j] = s / c
    return out

@njit(cache=True, parallel=True)
def _rolling_std(A, w):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    w = int(w)
    if w <= 1:
        return np.full((nd, ns), np.nan)
    for j in prange(ns):
        buf = np.empty(w, dtype=np.float64)
        okb = np.zeros(w, dtype=np.uint8)
        s, ss, c = 0.0, 0.0, 0
        for i in range(nd):
            bi = i % w
            if okb[bi] == 1:
                old = buf[bi]; s -= old; ss -= old * old; c -= 1; okb[bi] = 0
            v = A[i, j]
            if np.isfinite(v):
                buf[bi] = v; okb[bi] = 1; s += v; ss += v * v; c += 1
            if c >= 2:
                var_num = ss - (s * s) / c
                if var_num < 0.0:
                    var_num = 0.0
                out[i, j] = np.sqrt(var_num / (c - 1))
    return out

@njit(cache=True, parallel=True)
def _rolling_max(A, w):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    w = int(w)
    if w <= 1:
        return A.copy()
    for j in prange(ns):
        dq = np.empty(nd, dtype=np.int32)
        head, tail = 0, 0
        for i in range(nd):
            start = i - w + 1
            while head < tail and dq[head] < start:
                head += 1
            v = A[i, j]
            if np.isfinite(v):
                while head < tail:
                    if (not np.isfinite(A[dq[tail - 1], j])) or (A[dq[tail - 1], j] < v):
                        tail -= 1
                    else:
                        break
                dq[tail] = i; tail += 1
            if head < tail:
                out[i, j] = A[dq[head], j]
    return out

@njit(cache=True, parallel=True)
def _rolling_min(A, w):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    w = int(w)
    if w <= 1:
        return A.copy()
    for j in prange(ns):
        dq = np.empty(nd, dtype=np.int32)
        head, tail = 0, 0
        for i in range(nd):
            start = i - w + 1
            while head < tail and dq[head] < start:
                head += 1
            v = A[i, j]
            if np.isfinite(v):
                while head < tail:
                    if (not np.isfinite(A[dq[tail - 1], j])) or (A[dq[tail - 1], j] > v):
                        tail -= 1
                    else:
                        break
                dq[tail] = i; tail += 1
            if head < tail:
                out[i, j] = A[dq[head], j]
    return out

@njit(cache=True, parallel=True)
def _rolling_sum(A, w):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    w = int(w)
    if w <= 1:
        return A.copy()
    for j in prange(ns):
        buf = np.empty(w, dtype=np.float64)
        okb = np.zeros(w, dtype=np.uint8)
        s, c = 0.0, 0
        for i in range(nd):
            bi = i % w
            if okb[bi] == 1:
                s -= buf[bi]; c -= 1; okb[bi] = 0
            v = A[i, j]
            if np.isfinite(v):
                buf[bi] = v; okb[bi] = 1; s += v; c += 1
            if c > 0:
                out[i, j] = s
    return out

@njit(cache=True, parallel=True)
def _rolling_rank_fast(A, w):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    w = int(w)
    if w <= 1:
        for j in prange(ns):
            for i in range(nd):
                out[i, j] = 1.0 if np.isfinite(A[i, j]) else np.nan
        return out
    import math
    for j in prange(ns):
        buf = np.empty(w, dtype=np.float64)
        okb = np.zeros(w, dtype=np.uint8)
        s, ss, c = 0.0, 0.0, 0
        for i in range(nd):
            bi = i % w
            if okb[bi] == 1:
                old = buf[bi]; s -= old; ss -= old * old; c -= 1; okb[bi] = 0
            v = A[i, j]
            if np.isfinite(v):
                buf[bi] = v; okb[bi] = 1; s += v; ss += v * v; c += 1
            if c <= 0:
                continue
            curr = A[i, j]
            if not np.isfinite(curr):
                continue
            if c == 1:
                out[i, j] = 1.0; continue
            mean = s / c
            var = (ss - (s * s) / c) / c
            if var <= 1e-12:
                out[i, j] = 1.0; continue
            z = (curr - mean) / math.sqrt(var)
            phi = 0.5 * (1.0 + math.tanh(0.7978845608028654 * (z + 0.044715 * z * z * z)))
            if phi < 0.0: phi = 0.0
            elif phi > 1.0: phi = 1.0
            out[i, j] = phi * (c - 1.0) + 1.0
    return out

@njit(cache=True, parallel=True)
def _rolling_cov(A, B, w):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    w = int(w)
    if w <= 1:
        return np.full((nd, ns), np.nan)
    for j in prange(ns):
        abuf = np.empty(w, dtype=np.float64)
        bbuf = np.empty(w, dtype=np.float64)
        okb = np.zeros(w, dtype=np.uint8)
        sa, sb, sab, c = 0.0, 0.0, 0.0, 0
        for i in range(nd):
            bi = i % w
            if okb[bi] == 1:
                sa -= abuf[bi]; sb -= bbuf[bi]; sab -= abuf[bi] * bbuf[bi]; c -= 1; okb[bi] = 0
            a, b = A[i, j], B[i, j]
            if np.isfinite(a) and np.isfinite(b):
                abuf[bi] = a; bbuf[bi] = b; okb[bi] = 1
                sa += a; sb += b; sab += a * b; c += 1
            if c >= 2:
                out[i, j] = (sab - (sa * sb) / c) / (c - 1)
    return out

@njit(cache=True, parallel=True)
def _rolling_corr(A, B, w):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    w = int(w)
    if w <= 1:
        return np.full((nd, ns), np.nan)
    for j in prange(ns):
        abuf = np.empty(w, dtype=np.float64)
        bbuf = np.empty(w, dtype=np.float64)
        okb = np.zeros(w, dtype=np.uint8)
        sa, sb, sa2, sb2, sab, c = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for i in range(nd):
            bi = i % w
            if okb[bi] == 1:
                oa, ob = abuf[bi], bbuf[bi]
                sa -= oa; sb -= ob; sa2 -= oa*oa; sb2 -= ob*ob; sab -= oa*ob; c -= 1; okb[bi] = 0
            a, b = A[i, j], B[i, j]
            if np.isfinite(a) and np.isfinite(b):
                abuf[bi] = a; bbuf[bi] = b; okb[bi] = 1
                sa += a; sb += b; sa2 += a*a; sb2 += b*b; sab += a*b; c += 1
            if c >= 2:
                va = sa2 - (sa * sa) / c
                vb = sb2 - (sb * sb) / c
                if va > 1e-10 and vb > 1e-10:
                    out[i, j] = (sab - (sa * sb) / c) / np.sqrt(va * vb)
    return out

@njit(cache=True, parallel=True)
def _rolling_skew(A, w):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    for j in prange(ns):
        for i in range(nd):
            s, c, t = max(0, i - w + 1), 0, 0.0
            for k in range(s, i + 1):
                if np.isfinite(A[k, j]): t += A[k, j]; c += 1
            if c >= 3:
                m = t / c
                m2, m3 = 0.0, 0.0
                for k in range(s, i + 1):
                    v = A[k, j]
                    if np.isfinite(v):
                        d = v - m; m2 += d*d; m3 += d*d*d
                std = np.sqrt(m2 / (c - 1))
                if std > 1e-10: out[i, j] = (m3 / c) / (std ** 3)
    return out

@njit(cache=True, parallel=True)
def _rolling_kurt(A, w):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    for j in prange(ns):
        for i in range(nd):
            s, c, t = max(0, i - w + 1), 0, 0.0
            for k in range(s, i + 1):
                if np.isfinite(A[k, j]): t += A[k, j]; c += 1
            if c >= 4:
                m = t / c
                m2, m4 = 0.0, 0.0
                for k in range(s, i + 1):
                    v = A[k, j]
                    if np.isfinite(v):
                        d = v - m; m2 += d*d; m4 += d*d*d*d
                var = m2 / (c - 1)
                if var > 1e-10: out[i, j] = (m4 / c) / (var * var) - 3
    return out

@njit(cache=True, parallel=True)
def _ts_rank(A, w, min_periods):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    for j in prange(ns):
        for i in range(nd):
            curr = A[i, j]
            if np.isfinite(curr):
                s, cnt, lt = max(0, i - w + 1), 0, 0
                for k in range(s, i + 1):
                    v = A[k, j]
                    if np.isfinite(v):
                        cnt += 1
                        if v < curr: lt += 1
                if cnt >= min_periods: out[i, j] = lt / cnt
    return out

@njit(cache=True, parallel=True)
def _ts_ema(A, alpha):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    for j in prange(ns):
        prev = np.nan
        for i in range(nd):
            x = A[i, j]
            if np.isfinite(x):
                prev = alpha * x + (1.0 - alpha) * prev if np.isfinite(prev) else x
                out[i, j] = prev
            else:
                out[i, j] = prev if np.isfinite(prev) else np.nan
    return out

@njit(cache=True, parallel=True)
def _ts_ir(A, w):
    nd, ns = A.shape
    out = np.full((nd, ns), np.nan)
    for j in prange(ns):
        for i in range(nd):
            s, c, t = max(0, i - w + 1), 0, 0.0
            for k in range(s, i + 1):
                v = A[k, j]
                if np.isfinite(v): t += v; c += 1
            if c >= 2:
                m = t / c
                vs = 0.0
                for k in range(s, i + 1):
                    v = A[k, j]
                    if np.isfinite(v): vs += (v - m) ** 2
                std = np.sqrt(vs / (c - 1))
                if std > 1e-10: out[i, j] = m / std
    return out

@njit(cache=True, parallel=True)
def _safe_div_numba(A, B, eps, max_ratio):
    nd, ns = A.shape
    out = np.empty((nd, ns), dtype=np.float64)
    for i in prange(nd):
        for j in range(ns):
            a, b = A[i, j], B[i, j]
            if not np.isfinite(a) or not np.isfinite(b):
                out[i, j] = np.nan
            else:
                denom = b if abs(b) >= eps else (eps if b >= 0 else -eps)
                r = a / denom
                out[i, j] = max(-max_ratio, min(max_ratio, r))
    return out

@njit(cache=True, parallel=True)
def _step_numba(A, row_means):
    nd, ns = A.shape
    out = np.empty((nd, ns), dtype=np.float64)
    for i in prange(nd):
        m = row_means[i]
        for j in range(ns):
            v = A[i, j]
            out[i, j] = (1.0 if v > m else 0.0) if (np.isfinite(v) and np.isfinite(m)) else np.nan
    return out

@njit(cache=True, parallel=True)
def _if_else_numba(Cond, A, B, row_means):
    nd, ns = Cond.shape
    out = np.empty((nd, ns), dtype=np.float64)
    for i in prange(nd):
        m = row_means[i]
        for j in range(ns):
            c, a, b = Cond[i, j], A[i, j], B[i, j]
            out[i, j] = (a if c > m else b) if (np.isfinite(c) and np.isfinite(m)) else np.nan
    return out


# ---------------------------------------------------------------------------
# Wrapped operators
# ---------------------------------------------------------------------------
def _ensure(A):
    if np.isscalar(A): return float(A)
    return np.asarray(A, dtype=np.float64)

def _wrap(func, A, w, *args):
    A = _ensure(A)
    if np.ndim(A) < 2: return A
    return func(A, int(w), *args)

def delay(A, w=1):
    A = _ensure(A)
    if np.ndim(A) < 2: return A
    w = int(w)
    out = np.full_like(A, np.nan)
    if w < A.shape[0]: out[w:] = A[:-w]
    return out

def delta(A, w=1):
    A = _ensure(A)
    if np.ndim(A) < 2: return np.zeros_like(A)
    w = int(w)
    out = np.full_like(A, np.nan)
    if w < A.shape[0]: out[w:] = A[w:] - A[:-w]
    return out

def rank(A):
    A = _ensure(A)
    if np.ndim(A) < 2: return A
    mask = np.isfinite(A)
    filled = np.where(mask, A, -np.inf)
    sorter = np.argsort(filled, axis=1)
    ranks = np.empty_like(A, dtype=np.float32)
    np.put_along_axis(ranks, sorter, np.arange(A.shape[1], dtype=np.float32), axis=1)
    counts = np.sum(mask, axis=1, keepdims=True, dtype=np.float32)
    out = (ranks + 1.0) / np.maximum(counts, 1.0)
    return np.where(mask, out, np.nan)

def cs_zscore(A, eps=1e-9):
    A = _ensure(A)
    if np.ndim(A) < 2: return A
    m = np.nanmean(A, axis=1, keepdims=True)
    s = np.nanstd(A, axis=1, keepdims=True)
    return (A - m) / np.where(s < eps, eps, s)

def scale(A, s=1):
    A = _ensure(A)
    if np.ndim(A) < 2: return A
    rs = np.nansum(np.abs(A), axis=1, keepdims=True)
    return s * A / np.where(rs < 1e-10, 1.0, rs)

def ts_zscore(A, w=60, eps=1e-9):
    A = _ensure(A)
    if np.ndim(A) < 2: return 0.0
    m = _rolling_mean(A, int(w))
    s = _rolling_std(A, int(w))
    return (A - m) / np.where(s < eps, eps, s)

def ts_rank_op(A, w=60): return _wrap(_ts_rank, A, w, 5)
def ts_ema_op(A, span=20): return _wrap(_ts_ema, A, 2.0 / (int(span) + 1))

def ts_ir_op(A, w=20):
    result = _wrap(_ts_ir, A, w)
    return np.clip(result, -20.0, 20.0)

def ts_return(A, w=1, eps=1e-3):
    A = _ensure(A)
    d = delay(A, int(w))
    result = np.where(np.abs(d) > eps, (A - d) / (np.abs(d) + eps), 0.0)
    return np.clip(result, -10.0, 10.0)

def ts_range(A, w=5):
    A = _ensure(A)
    return _wrap(_rolling_max, A, w) - _wrap(_rolling_min, A, w)

def safe_div(A, B, eps=1e-3, max_ratio=1e4):
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    if A.ndim == 2 and B.ndim == 2:
        return _safe_div_numba(A, B, eps, max_ratio)
    denom = np.where(np.abs(B) < eps, eps * np.sign(B) + (B == 0) * eps, B)
    return np.clip(A / denom, -max_ratio, max_ratio)

def step_op(A):
    A = _ensure(A)
    if np.ndim(A) == 2:
        A = np.asarray(A, dtype=np.float64)
        return _step_numba(A, np.nanmean(A, axis=1))
    return np.where(A > 0, 1.0, 0.0)

def if_else_op(Cond, A, B):
    Cond = np.asarray(_ensure(Cond), dtype=np.float64)
    if np.ndim(Cond) == 2:
        A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
        return _if_else_numba(Cond, A, B, np.nanmean(Cond, axis=1))
    return np.where(Cond > 0, A, B)

def rolling_nanmean(A, w=5): return _wrap(_rolling_mean, A, w)
def rolling_nanstd(A, w=5): return _wrap(_rolling_std, A, w)
def rolling_max(A, w=5): return _wrap(_rolling_max, A, w)
def rolling_min(A, w=5): return _wrap(_rolling_min, A, w)
def rolling_sum(A, w=5): return _wrap(_rolling_sum, A, w)
def rolling_rank(A, w=5): return _wrap(_rolling_rank_fast, A, w)

def _protected_division(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------
FINANCIAL_CYCLES = [3, 5, 10, 15, 20, 30, 40, 60, 120, 240]

_base_ops = {
    'add': _Function(np.add, 'add', 2, role='arith'),
    'sub': _Function(np.subtract, 'sub', 2, role='arith'),
    'mul': _Function(np.multiply, 'mul', 2, role='arith', complexity_weight=1.5),
    'neg': _Function(np.negative, 'neg', 1, role='arith', complexity_weight=0.5),
    'max': _Function(np.maximum, 'max', 2, role='arith', complexity_weight=1.5),
    'min': _Function(np.minimum, 'min', 2, role='arith', complexity_weight=1.5),
}

_ext_ops = {
    'abs': _Function(np.abs, 'abs', 1, role='arith', complexity_weight=1.5),
    'tanh': _Function(np.tanh, 'tanh', 1, role='arith', complexity_weight=1.5),
    'safe_div': _Function(safe_div, 'safe_div', 2, role='arith', complexity_weight=2.5),
    'delay': _Function(delay, 'delay', 1, role='lag', complexity_weight=0.8),
    'delta': _Function(delta, 'delta', 1, role='lag', complexity_weight=1.0),
    'dynamic_ts_mean': _Function(rolling_nanmean, 'dynamic_ts_mean', 1, isRandom=(True, (5, 60)), role='ts_agg', is_ts_op=True, complexity_weight=3.0),
    'dynamic_ts_std': _Function(rolling_nanstd, 'dynamic_ts_std', 1, isRandom=(True, (5, 60)), role='ts_agg', is_ts_op=True, complexity_weight=3.5),
    'dynamic_ts_max': _Function(rolling_max, 'dynamic_ts_max', 1, isRandom=(True, (5, 60)), role='ts_agg', is_ts_op=True, complexity_weight=3.5),
    'dynamic_ts_min': _Function(rolling_min, 'dynamic_ts_min', 1, isRandom=(True, (5, 60)), role='ts_agg', is_ts_op=True, complexity_weight=3.0),
    'dynamic_ts_rank': _Function(rolling_rank, 'dynamic_ts_rank', 1, isRandom=(True, (5, 60)), role='ts_norm', is_ts_op=True, complexity_weight=3.0),
    'ts_sum': _Function(rolling_sum, 'ts_sum', 1, isRandom=(True, (5, 60)), role='ts_agg', is_ts_op=True, complexity_weight=3.0),
    'ts_ema': _Function(ts_ema_op, 'ts_ema', 1, isRandom=(True, (10, 120)), role='ts_agg', is_ts_op=True, complexity_weight=3.0),
    'ts_return': _Function(ts_return, 'ts_return', 1, isRandom=(True, (1, 10)), role='ts_agg', is_ts_op=True, complexity_weight=2.5),
    'ts_range': _Function(ts_range, 'ts_range', 1, isRandom=(True, (5, 60)), role='ts_agg', is_ts_op=True, complexity_weight=3.0),
    'ts_ir': _Function(ts_ir_op, 'ts_ir', 1, isRandom=(True, (10, 60)), role='ts_norm', is_ts_op=True, complexity_weight=3.5),
    'pos': _Function(lambda A: np.maximum(A, 0.0), 'pos', 1, role='arith', complexity_weight=1.5),
    'step': _Function(step_op, 'step', 1, role='logic', complexity_weight=1.5),
    'if_else': _Function(if_else_op, 'if_else', 3, role='logic', complexity_weight=4.0),
    'cs_rank': _Function(rank, 'cs_rank', 1, role='cs_norm', complexity_weight=2.0),
    'cs_zscore': _Function(cs_zscore, 'cs_zscore', 1, role='cs_norm', complexity_weight=2.0),
}

OP_MAP = {**_base_ops, **_ext_ops}

DEFAULT_FUNCTION_SET = [
    'add', 'sub', 'mul', 'neg', 'safe_div', 'abs', 'tanh',
    'delta', 'delay',
    'dynamic_ts_mean', 'dynamic_ts_std', 'dynamic_ts_max', 'dynamic_ts_min',
    'dynamic_ts_rank', 'ts_sum', 'ts_ema', 'ts_return', 'ts_range', 'ts_ir',
    'pos', 'step', 'if_else',
]
