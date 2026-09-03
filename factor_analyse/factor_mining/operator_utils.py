import numpy as np
import pandas as pd


def _validate_window(window: int) -> None:
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be a positive integer, got: {window}")


def _resolve_min_periods(window: int, min_periods: int | None) -> int:
    if min_periods is None:
        return window
    if not isinstance(min_periods, int) or min_periods <= 0:
        raise ValueError(f"min_periods must be a positive integer, got: {min_periods}")
    return min_periods


def _group_shift(x: pd.Series, periods: int, by: pd.Series | None) -> pd.Series:
    if by is None:
        return x.shift(periods)
    return x.groupby(by, sort=False).shift(periods)


def _rolling_op(
    x: pd.Series,
    by: pd.Series | None,
    window: int,
    op_name: str,
    min_periods: int | None = None,
) -> pd.Series:
    _validate_window(window)
    min_p = _resolve_min_periods(window, min_periods)
    if by is None:
        return getattr(x.rolling(window=window, min_periods=min_p), op_name)()
    rolled = getattr(x.groupby(by, sort=False).rolling(window=window, min_periods=min_p), op_name)()
    return rolled.reset_index(level=0, drop=True)


def _rolling_apply(
    x: pd.Series,
    by: pd.Series | None,
    window: int,
    func,
    min_periods: int | None = None,
    raw: bool = True,
) -> pd.Series:
    _validate_window(window)
    min_p = _resolve_min_periods(window, min_periods)
    if by is None:
        return x.rolling(window=window, min_periods=min_p).apply(func, raw=raw)
    rolled = x.groupby(by, sort=False).rolling(window=window, min_periods=min_p).apply(func, raw=raw)
    return rolled.reset_index(level=0, drop=True)


def _groupwise_scan(cond: pd.Series, by: pd.Series | None, fn) -> pd.Series:
    cond_bool = cond.fillna(False).astype(bool)
    if by is None:
        return pd.Series(fn(cond_bool.to_numpy()), index=cond.index, dtype=float)

    out = pd.Series(np.nan, index=cond.index, dtype=float)
    for _, idx in cond_bool.groupby(by, sort=False).groups.items():
        arr = cond_bool.loc[idx].to_numpy()
        out.loc[idx] = fn(arr)
    return out


def _linreg_params(arr: np.ndarray) -> tuple[float, float]:
    if len(arr) == 0 or np.isnan(arr).any():
        return np.nan, np.nan
    x = np.arange(len(arr), dtype=float)
    x_mean = x.mean()
    y_mean = arr.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 1e-12:
        return np.nan, np.nan
    slope = np.sum((x - x_mean) * (arr - y_mean)) / denom
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)


# ======================
# 1) 直接操作型函数 (X)
# ======================
def abs_(x: pd.Series) -> pd.Series:
    return x.abs()


def log(x: pd.Series) -> pd.Series:
    return np.log(x)


def logabs(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    return np.log(np.abs(x) + eps)


def exp(x: pd.Series) -> pd.Series:
    return np.exp(x)


def as_float(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype(bool).astype(float)
    return pd.Series(np.asarray(x).astype(bool).astype(float))


def rd(x: pd.Series, n: int = 2) -> pd.Series:
    return x.round(n)


def sign(x: pd.Series) -> pd.Series:
    return np.sign(x)


def sin(x: pd.Series) -> pd.Series:
    return np.sin(x)


def cos(x: pd.Series) -> pd.Series:
    return np.cos(x)


def tan(x: pd.Series) -> pd.Series:
    return np.tan(x)


def arcsin(x: pd.Series) -> pd.Series:
    return np.arcsin(np.clip(x, -1.0, 1.0))


def arccos(x: pd.Series) -> pd.Series:
    return np.arccos(np.clip(x, -1.0, 1.0))


def arctan(x: pd.Series) -> pd.Series:
    return np.arctan(x)


# ======================
# 2) 截面操作型函数 (X)
# ======================
def cs_rank(x: pd.Series, by_date: pd.Series | None = None, method: str = "average") -> pd.Series:
    if by_date is None:
        return x.rank(method=method, pct=True)
    return x.groupby(by_date, sort=False).rank(method=method, pct=True)


def cs_scale(x: pd.Series, by_date: pd.Series | None = None, eps: float = 1e-12) -> pd.Series:
    def _scale(g: pd.Series) -> pd.Series:
        gmin = g.min()
        gmax = g.max()
        spread = gmax - gmin
        if spread < eps:
            return pd.Series(0.0, index=g.index)
        return 2.0 * (g - gmin) / (spread + eps) - 1.0

    if by_date is None:
        return _scale(x)
    return x.groupby(by_date, sort=False).transform(_scale)


def cs_zscore(x: pd.Series, by_date: pd.Series | None = None, eps: float = 1e-12) -> pd.Series:
    if by_date is None:
        return (x - x.mean()) / (x.std() + eps)
    mean_ = x.groupby(by_date, sort=False).transform("mean")
    std_ = x.groupby(by_date, sort=False).transform("std")
    return (x - mean_) / (std_ + eps)


# ======================
# 3) 时序操作型函数 (X)
# ======================
def const(x: pd.Series, by: pd.Series | None = None) -> pd.Series:
    if by is None:
        if x.empty:
            return x.copy()
        return pd.Series(x.iloc[-1], index=x.index)
    return x.groupby(by, sort=False).transform("last")


def barslast(x, by: pd.Series | None = None) -> pd.Series:
    def _fn(arr: np.ndarray) -> np.ndarray:
        out = np.full(len(arr), np.nan, dtype=float)
        last_true = -1
        for i, v in enumerate(arr):
            if v:
                last_true = i
                out[i] = 0.0
            elif last_true >= 0:
                out[i] = float(i - last_true)
        return out

    return _groupwise_scan(pd.Series(x), by, _fn)


def barslastcount(x, by: pd.Series | None = None) -> pd.Series:
    def _fn(arr: np.ndarray) -> np.ndarray:
        out = np.zeros(len(arr), dtype=float)
        run = 0
        for i, v in enumerate(arr):
            if v:
                run += 1
                out[i] = float(run)
            else:
                run = 0
                out[i] = 0.0
        return out

    return _groupwise_scan(pd.Series(x), by, _fn)


# ===========================
# 4) 双参数直接操作型 (X, N)
# ===========================
def power(x: pd.Series, n: float) -> pd.Series:
    return x**n


def signedpower(x: pd.Series, n: float) -> pd.Series:
    return np.sign(x) * (np.abs(x) ** n)


# ======================
# 5) 时序操作型函数 (X, N)
# ======================
def ref(x: pd.Series, n: int = 1, by: pd.Series | None = None) -> pd.Series:
    return _group_shift(x, n, by)


def delay(x: pd.Series, n: int = 1, by: pd.Series | None = None) -> pd.Series:
    return ref(x, n, by)


def diff(x: pd.Series, n: int = 1, by: pd.Series | None = None) -> pd.Series:
    return x - _group_shift(x, n, by)


def delta(x: pd.Series, n: int = 1, by: pd.Series | None = None) -> pd.Series:
    return diff(x, n, by)


def ma(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_op(x, by, n, "mean", min_periods=min_periods)


def ts_mean(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return ma(x, n, by=by, min_periods=min_periods)


def ts_sum(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_op(x, by, n, "sum", min_periods=min_periods)


def product(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_apply(
        x,
        by,
        n,
        func=lambda arr: float(np.prod(arr)),
        min_periods=min_periods,
        raw=True,
    )


def roc(x: pd.Series, n: int = 1, by: pd.Series | None = None) -> pd.Series:
    base = _group_shift(x, n, by)
    return x / base - 1.0


def pct_change(x: pd.Series, n: int = 1, by: pd.Series | None = None) -> pd.Series:
    return roc(x, n=n, by=by)


def ts_std(
    x: pd.Series,
    n: int,
    by: pd.Series | None = None,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.Series:
    _validate_window(n)
    min_p = _resolve_min_periods(n, min_periods)
    if by is None:
        return x.rolling(window=n, min_periods=min_p).std(ddof=ddof)
    rolled = x.groupby(by, sort=False).rolling(window=n, min_periods=min_p).std(ddof=ddof)
    return rolled.reset_index(level=0, drop=True)


def stddev(
    x: pd.Series,
    n: int,
    by: pd.Series | None = None,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.Series:
    return ts_std(x, n, by=by, min_periods=min_periods, ddof=ddof)


def var(
    x: pd.Series,
    n: int,
    by: pd.Series | None = None,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.Series:
    _validate_window(n)
    min_p = _resolve_min_periods(n, min_periods)
    if by is None:
        return x.rolling(window=n, min_periods=min_p).var(ddof=ddof)
    rolled = x.groupby(by, sort=False).rolling(window=n, min_periods=min_p).var(ddof=ddof)
    return rolled.reset_index(level=0, drop=True)


def ts_max(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_op(x, by, n, "max", min_periods=min_periods)


def ts_min(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_op(x, by, n, "min", min_periods=min_periods)


def ts_middle(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return (ts_max(x, n, by=by, min_periods=min_periods) + ts_min(x, n, by=by, min_periods=min_periods)) / 2.0


def ts_mad(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_apply(
        x,
        by,
        n,
        func=lambda arr: float(np.mean(np.abs(arr - np.mean(arr)))),
        min_periods=min_periods,
        raw=True,
    )


def ts_rank(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_apply(
        x,
        by,
        n,
        func=lambda arr: float(pd.Series(arr).rank(pct=True, method="average").iloc[-1]),
        min_periods=min_periods,
        raw=False,
    )


def ts_argmax(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_apply(
        x,
        by,
        n,
        func=lambda arr: float(np.argmax(arr) + 1),
        min_periods=min_periods,
        raw=True,
    )


def ts_argmin(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_apply(
        x,
        by,
        n,
        func=lambda arr: float(np.argmin(arr) + 1),
        min_periods=min_periods,
        raw=True,
    )


def hhv(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return ts_max(x, n, by=by, min_periods=min_periods)


def llv(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return ts_min(x, n, by=by, min_periods=min_periods)


def hhvbars(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_apply(
        x,
        by,
        n,
        func=lambda arr: float(len(arr) - 1 - int(np.argmax(arr))),
        min_periods=min_periods,
        raw=True,
    )


def llvbars(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_apply(
        x,
        by,
        n,
        func=lambda arr: float(len(arr) - 1 - int(np.argmin(arr))),
        min_periods=min_periods,
        raw=True,
    )


def count(x, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    xb = pd.Series(x).fillna(False).astype(bool).astype(float)
    return ts_sum(xb, n, by=by, min_periods=min_periods)


def every(x, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    c = count(x, n, by=by, min_periods=min_periods)
    return c == float(n)


def exist(x, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    c = count(x, n, by=by, min_periods=min_periods)
    return c > 0.0


def barssincen(x, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    def _first_true_distance(arr: np.ndarray) -> float:
        idx = np.flatnonzero(arr.astype(bool))
        if len(idx) == 0:
            return np.nan
        return float(len(arr) - 1 - idx[0])

    xb = pd.Series(x).fillna(False).astype(bool)
    return _rolling_apply(xb.astype(float), by, n, func=lambda arr: _first_true_distance(arr), min_periods=min_periods)


def slope(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_apply(
        x,
        by,
        n,
        func=lambda arr: _linreg_params(arr)[0],
        min_periods=min_periods,
        raw=True,
    )


def angle(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return np.degrees(np.arctan(slope(x, n, by=by, min_periods=min_periods)))


def intercept(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_apply(
        x,
        by,
        n,
        func=lambda arr: _linreg_params(arr)[1],
        min_periods=min_periods,
        raw=True,
    )


def forcast(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    def _forecast_next(arr: np.ndarray) -> float:
        s, b = _linreg_params(arr)
        if np.isnan(s) or np.isnan(b):
            return np.nan
        return float(s * len(arr) + b)

    return _rolling_apply(x, by, n, func=_forecast_next, min_periods=min_periods, raw=True)


def decaylinear(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    def _dl(arr: np.ndarray) -> float:
        w = np.arange(1, len(arr) + 1, dtype=float)
        w = w / w.sum()
        return float(np.dot(arr, w))

    return _rolling_apply(x, by, n, func=_dl, min_periods=min_periods, raw=True)


def ts_zscore(
    x: pd.Series,
    n: int,
    by: pd.Series | None = None,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.Series:
    mean_ = ts_mean(x, n, by=by, min_periods=min_periods)
    std_ = ts_std(x, n, by=by, min_periods=min_periods)
    return (x - mean_) / (std_ + eps)


def ts_skew(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_op(x, by, n, "skew", min_periods=min_periods)


def ts_kurt(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_op(x, by, n, "kurt", min_periods=min_periods)


def ts_median(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return _rolling_op(x, by, n, "median", min_periods=min_periods)


def avedev(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return ts_mad(x, n, by=by, min_periods=min_periods)


def ema(
    x: pd.Series,
    n: int,
    by: pd.Series | None = None,
    adjust: bool = False,
    min_periods: int | None = None,
) -> pd.Series:
    _validate_window(n)
    min_p = _resolve_min_periods(n, min_periods)
    if by is None:
        return x.ewm(span=n, adjust=adjust, min_periods=min_p).mean()
    return x.groupby(by, sort=False).transform(lambda s: s.ewm(span=n, adjust=adjust, min_periods=min_p).mean())


def dma(x: pd.Series, a: float, by: pd.Series | None = None) -> pd.Series:
    if not isinstance(a, (float, int)) or not (0 < float(a) < 1):
        raise ValueError(f"a must be in (0, 1), got: {a}")

    alpha = float(a)

    def _dma_scan(arr: np.ndarray) -> np.ndarray:
        out = np.full(len(arr), np.nan, dtype=float)
        prev = np.nan
        for i, v in enumerate(arr):
            if np.isnan(v):
                out[i] = prev
                continue
            if np.isnan(prev):
                prev = v
            else:
                prev = alpha * v + (1.0 - alpha) * prev
            out[i] = prev
        return out

    if by is None:
        return pd.Series(_dma_scan(x.to_numpy(dtype=float)), index=x.index)

    out = pd.Series(np.nan, index=x.index, dtype=float)
    for _, idx in x.groupby(by, sort=False).groups.items():
        out.loc[idx] = _dma_scan(x.loc[idx].to_numpy(dtype=float))
    return out


def wma(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    def _wma(arr: np.ndarray) -> float:
        w = np.arange(1, len(arr) + 1, dtype=float)
        w = w / w.sum()
        return float(np.dot(arr, w))

    return _rolling_apply(x, by, n, func=_wma, min_periods=min_periods, raw=True)


def returns(x: pd.Series, n: int = 1, by: pd.Series | None = None) -> pd.Series:
    return roc(x, n=n, by=by)


def future_returns(x: pd.Series, n: int = 1, by: pd.Series | None = None) -> pd.Series:
    if by is None:
        return x.shift(-n) / x - 1.0
    lead = x.groupby(by, sort=False).shift(-n)
    return lead / x - 1.0


def sharpe(
    x: pd.Series,
    n: int,
    by: pd.Series | None = None,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.Series:
    mean_ = ts_mean(x, n, by=by, min_periods=min_periods)
    std_ = ts_std(x, n, by=by, min_periods=min_periods)
    return mean_ / (std_ + eps)


def sum_abs_price_change(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return ts_sum(np.abs(diff(x, 1, by=by)), n, by=by, min_periods=min_periods)


def mean_abs_price_change(x: pd.Series, n: int, by: pd.Series | None = None, min_periods: int | None = None) -> pd.Series:
    return ts_mean(np.abs(diff(x, 1, by=by)), n, by=by, min_periods=min_periods)


def corr(
    x: pd.Series,
    y: pd.Series,
    n: int,
    by: pd.Series | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    _validate_window(n)
    min_p = _resolve_min_periods(n, min_periods)
    if by is None:
        return x.rolling(window=n, min_periods=min_p).corr(y)
    rolled = x.groupby(by, sort=False).rolling(window=n, min_periods=min_p).corr(y)
    return rolled.reset_index(level=0, drop=True)


def correlation(
    x: pd.Series,
    y: pd.Series,
    n: int,
    by: pd.Series | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    return corr(x, y, n, by=by, min_periods=min_periods)


def cov(
    x: pd.Series,
    y: pd.Series,
    n: int,
    by: pd.Series | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    _validate_window(n)
    min_p = _resolve_min_periods(n, min_periods)
    if by is None:
        return x.rolling(window=n, min_periods=min_p).cov(y)
    rolled = x.groupby(by, sort=False).rolling(window=n, min_periods=min_p).cov(y)
    return rolled.reset_index(level=0, drop=True)


def covariance(
    x: pd.Series,
    y: pd.Series,
    n: int,
    by: pd.Series | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    return cov(x, y, n, by=by, min_periods=min_periods)


def if_else(cond, a, b) -> pd.Series:
    if isinstance(cond, pd.Series):
        index = cond.index
    elif isinstance(a, pd.Series):
        index = a.index
    elif isinstance(b, pd.Series):
        index = b.index
    else:
        return pd.Series(np.where(cond, a, b))
    return pd.Series(np.where(cond, a, b), index=index)


# Formula-style uppercase aliases
ABS = abs_
LOG = log
LOGABS = logabs
EXP = exp
AS_FLOAT = as_float
RD = rd
SIGN = sign
SIN = sin
COS = cos
TAN = tan
ARCSIN = arcsin
ARCCOS = arccos
ARCTAN = arctan

RANK = cs_rank
SCALE = cs_scale
ZSCORE = cs_zscore

CONST = const
BARSLAST = barslast
BARSLASTCOUNT = barslastcount

POWER = power
SIGNEDPOWER = signedpower

REF = ref
DELAY = delay
DIFF = diff
DELTA = delta
MA = ma
TS_MEAN = ts_mean
SUM = ts_sum
PRODUCT = product
ROC = roc
PCT_CHANGE = pct_change
STD = ts_std
STDDEV = stddev
VAR = var
TS_MAX = ts_max
TS_MIN = ts_min
TS_MIDDLE = ts_middle
TS_MAD = ts_mad
TS_RANK = ts_rank
TS_ARGMAX = ts_argmax
TS_ARGMIN = ts_argmin
HHV = hhv
LLV = llv
HHVBARS = hhvbars
LLVBARS = llvbars
COUNT = count
EVERY = every
EXIST = exist
BARSSINCEN = barssincen
SLOPE = slope
ANGLE = angle
INTERCEPT = intercept
FORCAST = forcast
DECAYLINEAR = decaylinear
TS_ZSCORE = ts_zscore
TS_SKEW = ts_skew
TS_KURT = ts_kurt
TS_MEDIAN = ts_median
AVEDEV = avedev
EMA = ema
DMA = dma
WMA = wma
RETURNS = returns
FUTURE_RETURNS = future_returns
SHARPE = sharpe
SUM_ABS_PRICE_CHANGE = sum_abs_price_change
MEAN_ABS_PRICE_CHANGE = mean_abs_price_change
CORR = corr
CORRELATION = correlation
COV = cov
COVARIANCE = covariance
IF = if_else


__all__ = [
    # 1
    "abs_",
    "log",
    "logabs",
    "exp",
    "as_float",
    "rd",
    "sign",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    # 2
    "cs_rank",
    "cs_scale",
    "cs_zscore",
    # 3
    "const",
    "barslast",
    "barslastcount",
    # 4
    "power",
    "signedpower",
    # 5
    "ref",
    "delay",
    "diff",
    "delta",
    "ma",
    "ts_mean",
    "ts_sum",
    "product",
    "roc",
    "pct_change",
    "ts_std",
    "stddev",
    "var",
    "ts_max",
    "ts_min",
    "ts_middle",
    "ts_mad",
    "ts_rank",
    "ts_argmax",
    "ts_argmin",
    "hhv",
    "llv",
    "hhvbars",
    "llvbars",
    "count",
    "every",
    "exist",
    "barssincen",
    "slope",
    "angle",
    "intercept",
    "forcast",
    "decaylinear",
    "ts_zscore",
    "ts_skew",
    "ts_kurt",
    "ts_median",
    "avedev",
    "ema",
    "dma",
    "wma",
    "returns",
    "future_returns",
    "sharpe",
    "sum_abs_price_change",
    "mean_abs_price_change",
    "corr",
    "correlation",
    "cov",
    "covariance",
    "if_else",
    # aliases
    "ABS",
    "LOG",
    "LOGABS",
    "EXP",
    "AS_FLOAT",
    "RD",
    "SIGN",
    "SIN",
    "COS",
    "TAN",
    "ARCSIN",
    "ARCCOS",
    "ARCTAN",
    "RANK",
    "SCALE",
    "ZSCORE",
    "CONST",
    "BARSLAST",
    "BARSLASTCOUNT",
    "POWER",
    "SIGNEDPOWER",
    "REF",
    "DELAY",
    "DIFF",
    "DELTA",
    "MA",
    "TS_MEAN",
    "SUM",
    "PRODUCT",
    "ROC",
    "PCT_CHANGE",
    "STD",
    "STDDEV",
    "VAR",
    "TS_MAX",
    "TS_MIN",
    "TS_MIDDLE",
    "TS_MAD",
    "TS_RANK",
    "TS_ARGMAX",
    "TS_ARGMIN",
    "HHV",
    "LLV",
    "HHVBARS",
    "LLVBARS",
    "COUNT",
    "EVERY",
    "EXIST",
    "BARSSINCEN",
    "SLOPE",
    "ANGLE",
    "INTERCEPT",
    "FORCAST",
    "DECAYLINEAR",
    "TS_ZSCORE",
    "TS_SKEW",
    "TS_KURT",
    "TS_MEDIAN",
    "AVEDEV",
    "EMA",
    "DMA",
    "WMA",
    "RETURNS",
    "FUTURE_RETURNS",
    "SHARPE",
    "SUM_ABS_PRICE_CHANGE",
    "MEAN_ABS_PRICE_CHANGE",
    "CORR",
    "CORRELATION",
    "COV",
    "COVARIANCE",
    "IF",
]
