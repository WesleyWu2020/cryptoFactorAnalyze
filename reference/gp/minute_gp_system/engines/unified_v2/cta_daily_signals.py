"""Hand-written daily CTA signal adapter and evaluator.

The signals are computed on closed 1D bars and evaluated against next-bar
returns through the same fitness primitives used by unified_v2 mining.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .config import (
    DEFAULT_H5_PATH,
    LAG_PERIODS,
    PERIODS_PER_YEAR,
    TOP_QUANTILE,
    TRADING_COST,
)
from .data_loader import CryptoDataLoader
from .fitness import (
    build_lagged_returns,
    compute_five_objectives_prepared,
    prepare_fitness_context,
)
from .publish_pipeline import (
    MAX_OOS_TURNOVER,
    MIN_OOS_SHARPE,
)
from .evolution import (
    FINAL_MAX_LS_TURNOVER,
    FINAL_MAX_YEARLY_TURNOVER,
    FINAL_MIN_COVERAGE,
    FINAL_MIN_LS_NETRET,
    FINAL_MIN_LS_NET_SHARPE,
    FINAL_MIN_LS_POSITIVE_RATE,
    FINAL_MIN_YEARLY_COVERAGE,
    FINAL_MIN_YEARLY_LS_NETRET,
    FINAL_MIN_YEARLY_LS_NET_SHARPE,
    FINAL_YEARLY_STABILITY_WINDOWS,
)


_EPS = 1e-8
CTA_SIGNAL_SPECS = (
    ("S1", "BB+KC+KDJ compression reversal"),
    ("S2", "Williams PH/PL breakout"),
    ("V1", "Strict BB+KC+KDJ threshold"),
    ("V2_base", "Dual RSI + Chandelier Exit"),
    ("V2_opt", "V2 + RSI extreme pullback amplifier"),
    ("V3_base", "Dual Keltner five-zone position"),
    ("V3_opt", "V3 + short trend modulation"),
    ("V4_base", "MACD + EMA200 + SAR consensus"),
    ("V5_base", "Gaussian Channel + MACD + CCI pyramid"),
    ("V6_base", "Dual CCI contrarian cross"),
    ("V7_base", "SuperTrend + QQE + EMA200"),
    ("V8_base", "Three soldiers + PMAX + OCC + EMA200"),
    ("V9_base", "Hull Suite + Mobo Bands scalp"),
    ("V9_opt", "V9 + EMA200 alignment"),
    ("V10_base", "HMA + EMA200 + KDJ swing"),
    ("V11_base", "HMA + std multi-mode adaptive"),
    ("V12_base", "Breakout + volatility + trend + ATR adaptive"),
    ("V13_base", "Dual BB + BBW filter trend"),
    ("V14_base", "Reverse MA state + N-day breakout"),
)


def _default_daily_cache_path(loader: CryptoDataLoader) -> Path:
    h5_path = Path(loader.h5_path).resolve()
    base = h5_path.parents[2] if len(h5_path.parents) > 2 else Path.cwd()
    dates = loader.get_active_dates()
    date_key = f"{dates[0]}_{dates[-1]}" if len(dates) else "empty"
    name = f"daily_ohlcv_{loader.minutes_per_period}m_{date_key}_{loader.n_tradable_coins}c.npz"
    return base / "factor_platform" / "strategy_factory_artifacts" / "minute_gp_system" / "cta_daily_signals" / name


def _daily_cache_meta(loader: CryptoDataLoader) -> dict:
    stat = os.stat(loader.h5_path)
    return {
        "h5_path": str(Path(loader.h5_path).resolve()),
        "h5_mtime_ns": int(stat.st_mtime_ns),
        "h5_size": int(stat.st_size),
        "minutes_per_period": int(loader.minutes_per_period),
        "n_active_periods": int(loader.n_active_periods),
        "n_tradable_coins": int(loader.n_tradable_coins),
    }


def _load_daily_cache(loader: CryptoDataLoader, cache_path: Path):
    if not cache_path.exists():
        return None
    try:
        expected = _daily_cache_meta(loader)
        with np.load(cache_path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
            if meta != expected:
                return None
            if not np.array_equal(z["active_period_indices"], loader.active_period_indices):
                return None
            if not np.array_equal(z["coin_indices"], loader.coin_indices):
                return None
            return {
                "open": z["open"].astype(np.float32, copy=False),
                "high": z["high"].astype(np.float32, copy=False),
                "low": z["low"].astype(np.float32, copy=False),
                "close": z["close"].astype(np.float32, copy=False),
                "volume": z["volume"].astype(np.float32, copy=False),
                "dates": z["dates"].astype(str, copy=False),
            }
    except Exception:
        return None


def _save_daily_cache(loader: CryptoDataLoader, cache_path: Path, daily: dict[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
    np.savez(
        tmp,
        meta=np.array(json.dumps(_daily_cache_meta(loader)), dtype=np.str_),
        active_period_indices=loader.active_period_indices,
        coin_indices=loader.coin_indices,
        dates=np.asarray(daily["dates"]).astype("U8"),
        open=daily["open"],
        high=daily["high"],
        low=daily["low"],
        close=daily["close"],
        volume=daily["volume"],
    )
    os.replace(tmp, cache_path)


def _safe_div(a, b):
    return np.asarray(a, dtype=np.float32) / (np.asarray(b, dtype=np.float32) + np.float32(_EPS))


def _clean(x, clip=1e6):
    out = np.asarray(x, dtype=np.float32)
    out = np.where(np.isfinite(out), out, np.nan)
    if clip is not None:
        out = np.clip(out, -clip, clip)
    return out.astype(np.float32, copy=False)


def _df(x):
    return pd.DataFrame(np.asarray(x, dtype=np.float32))


def _window_sum_2d(x, window):
    arr = np.asarray(x, dtype=np.float64)
    csum = np.vstack([
        np.zeros((1, arr.shape[1]), dtype=np.float64),
        np.cumsum(arr, axis=0, dtype=np.float64),
    ])
    end = np.arange(1, arr.shape[0] + 1)
    start = np.maximum(0, end - int(window))
    return csum[end] - csum[start]


def _rolling(x, window, method, min_periods=None):
    if min_periods is None:
        min_periods = max(2, min(window, window // 2))
    roll = _df(x).rolling(window, min_periods=min_periods)
    return getattr(roll, method)().to_numpy(dtype=np.float32)


def _roll_mean(x, window, min_periods=None):
    arr = np.asarray(x, dtype=np.float32)
    min_periods = min_periods or max(2, min(window, window // 2))
    valid = np.isfinite(arr)
    count = _window_sum_2d(valid.astype(np.float32), window)
    total = _window_sum_2d(np.where(valid, arr, 0.0), window)
    out = total / np.maximum(count, 1.0)
    out[count < int(min_periods)] = np.nan
    return out.astype(np.float32, copy=False)


def _roll_std(x, window, min_periods=None):
    arr = np.asarray(x, dtype=np.float32)
    min_periods = min_periods or max(2, min(window, window // 2))
    valid = np.isfinite(arr)
    count = _window_sum_2d(valid.astype(np.float32), window)
    filled = np.where(valid, arr, 0.0)
    total = _window_sum_2d(filled, window)
    total_sq = _window_sum_2d(filled * filled, window)
    mean = total / np.maximum(count, 1.0)
    var = total_sq / np.maximum(count, 1.0) - mean * mean
    out = np.sqrt(np.maximum(var, 0.0))
    out[count < int(min_periods)] = np.nan
    return out.astype(np.float32, copy=False)


def _roll_min(x, window, min_periods=None):
    return _rolling(x, window, "min", min_periods=min_periods)


def _roll_max(x, window, min_periods=None):
    return _rolling(x, window, "max", min_periods=min_periods)


def _ema(x, span):
    return _df(x).ewm(span=span, adjust=False, min_periods=1).mean().to_numpy(dtype=np.float32)


def _rma(x, period):
    return _df(x).ewm(alpha=1.0 / float(period), adjust=False, min_periods=1).mean().to_numpy(dtype=np.float32)


def _wma(x, window):
    arr = np.asarray(x, dtype=np.float32)
    window = int(window)
    min_periods = max(2, window // 2)
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0).astype(np.float64, copy=False)
    valid_f = valid.astype(np.float64, copy=False)
    idx = np.arange(1, arr.shape[0] + 1, dtype=np.float64)[:, None]
    end = np.arange(1, arr.shape[0] + 1)
    start = np.maximum(0, end - window).astype(np.float64)

    count = _window_sum_2d(valid_f, window)
    sum_x = _window_sum_2d(filled, window)
    sum_ix = _window_sum_2d(filled * idx, window)
    sum_i = _window_sum_2d(valid_f * idx, window)
    numerator = sum_ix - start[:, None] * sum_x
    denominator = sum_i - start[:, None] * count
    out = numerator / np.maximum(denominator, _EPS)
    out[(count < min_periods) | (denominator <= _EPS)] = np.nan
    return out.astype(np.float32, copy=False)


def _hma(x, window):
    half = max(2, int(window) // 2)
    root = max(2, int(np.sqrt(window)))
    return _wma(2.0 * _wma(x, half) - _wma(x, int(window)), root)


def _true_range(high, low, close):
    prev_close = np.vstack([close[:1], close[:-1]])
    return np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ]).astype(np.float32)


def _atr(high, low, close, period=14, method="rma"):
    tr = _true_range(high, low, close)
    return _rma(tr, period) if method == "rma" else _ema(tr, period)


def _rsi(close, period):
    diff = np.vstack([np.zeros_like(close[:1]), np.diff(close, axis=0)])
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = _safe_div(avg_gain, avg_loss)
    return _clean(100.0 - 100.0 / (1.0 + rs), clip=100.0)


def _kdj_k(high, low, close, period, smooth=3):
    ll = _roll_min(low, period)
    hh = _roll_max(high, period)
    rsv = np.clip(_safe_div(close - ll, hh - ll) * 100.0, 0.0, 100.0)
    return _df(rsv).ewm(alpha=1.0 / float(smooth), adjust=False, min_periods=1).mean().to_numpy(dtype=np.float32)


def _cci(high, low, close, period):
    tp = (high + low + close) / 3.0
    ma = _roll_mean(tp, period)
    mad = _roll_mean(np.abs(tp - ma), period, min_periods=max(2, period // 2))
    return _clean((tp - ma) / (0.015 * mad + _EPS), clip=1e4)


def _macd_hist(close, fast=12, slow=26, signal=9):
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    return _clean(macd - sig)


def _psar(high, low, step=0.02, max_af=0.2):
    p, s = high.shape
    sar = np.full((p, s), np.nan, dtype=np.float32)
    trend_up = np.ones(s, dtype=bool)
    af = np.full(s, step, dtype=np.float32)
    ep = high[0].astype(np.float32).copy()
    sar[0] = low[0]
    for t in range(1, p):
        prev_sar = sar[t - 1] + af * (ep - sar[t - 1])
        if t >= 2:
            prev_sar = np.where(trend_up, np.minimum(prev_sar, np.minimum(low[t - 1], low[t - 2])), prev_sar)
            prev_sar = np.where(~trend_up, np.maximum(prev_sar, np.maximum(high[t - 1], high[t - 2])), prev_sar)
        flip_down = trend_up & (low[t] < prev_sar)
        flip_up = (~trend_up) & (high[t] > prev_sar)
        sar_t = prev_sar.copy()
        sar_t[flip_down | flip_up] = ep[flip_down | flip_up]
        trend_up = np.where(flip_down, False, np.where(flip_up, True, trend_up))
        ep = np.where(flip_down, low[t], np.where(flip_up, high[t], ep))
        new_high = trend_up & (high[t] > ep)
        new_low = (~trend_up) & (low[t] < ep)
        ep = np.where(new_high, high[t], np.where(new_low, low[t], ep))
        af = np.where(new_high | new_low, np.minimum(af + step, max_af), af)
        af = np.where(flip_down | flip_up, step, af)
        sar[t] = sar_t
    return sar


def _chandelier_state(high, low, close, period=22, mult=3.0):
    atr = _atr(high, low, close, period)
    long_stop = _roll_max(high, period) - mult * atr
    short_stop = _roll_min(low, period) + mult * atr
    p, s = close.shape
    state = np.ones((p, s), dtype=np.float32)
    for t in range(1, p):
        prev = state[t - 1]
        state[t] = np.where(close[t] > short_stop[t - 1], 1.0, np.where(close[t] < long_stop[t - 1], -1.0, prev))
    return state


def _supertrend_state(high, low, close, period=100, mult=5.5):
    atr = _atr(high, low, close, period)
    hl2 = (high + low) * 0.5
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    p, s = close.shape
    state = np.ones((p, s), dtype=np.float32)
    final_upper = upper.copy()
    final_lower = lower.copy()
    for t in range(1, p):
        final_upper[t] = np.where(
            (upper[t] < final_upper[t - 1]) | (close[t - 1] > final_upper[t - 1]),
            upper[t],
            final_upper[t - 1],
        )
        final_lower[t] = np.where(
            (lower[t] > final_lower[t - 1]) | (close[t - 1] < final_lower[t - 1]),
            lower[t],
            final_lower[t - 1],
        )
        state[t] = np.where(close[t] > final_upper[t - 1], 1.0, np.where(close[t] < final_lower[t - 1], -1.0, state[t - 1]))
    return state


def _pmax_line_state(high, low, close, period=10, mult=3.0):
    ma = _ema(close, period)
    atr = _atr(high, low, close, period)
    upper = ma + mult * atr
    lower = ma - mult * atr
    p, s = close.shape
    state = np.ones((p, s), dtype=np.float32)
    line = lower.copy()
    for t in range(1, p):
        state[t] = np.where(close[t] > upper[t - 1], 1.0, np.where(close[t] < lower[t - 1], -1.0, state[t - 1]))
        line[t] = np.where(state[t] > 0, lower[t], upper[t])
    return line, state


def _decay_memory(trigger, decay=0.95):
    out = np.zeros_like(trigger, dtype=np.float32)
    for t in range(trigger.shape[0]):
        if t == 0:
            out[t] = trigger[t].astype(np.float32)
        else:
            out[t] = np.where(trigger[t], 1.0, out[t - 1] * decay)
    return out


def _rolling_count(event, window):
    return _window_sum_2d(event.astype(np.float32), window).astype(np.float32, copy=False)


def _nanmax_axis1(block):
    valid = np.isfinite(block)
    out = np.where(valid, block, -np.inf).max(axis=1)
    out[~valid.any(axis=1)] = np.nan
    return out.astype(np.float32, copy=False)


def _nanmin_axis1(block):
    valid = np.isfinite(block)
    out = np.where(valid, block, np.inf).min(axis=1)
    out[~valid.any(axis=1)] = np.nan
    return out.astype(np.float32, copy=False)


def _days_since_change(state):
    p, s = state.shape
    out = np.zeros((p, s), dtype=np.float32)
    for t in range(1, p):
        out[t] = np.where(state[t] != state[t - 1], 0.0, out[t - 1] + 1.0)
    return out


def _qqe_signal_progress(close):
    rsi = _rsi(close, 14) - 50.0
    signal = _ema(rsi, 5)
    hist = signal - _ema(signal, 14)
    sign = np.sign(hist)
    p, s = hist.shape
    flips = np.zeros((p, s), dtype=np.float32)
    count = np.zeros(s, dtype=np.float32)
    last = sign[0]
    for t in range(1, p):
        changed = (sign[t] != 0) & (last != 0) & (sign[t] != last)
        count = np.where(changed, np.minimum(count + 1.0, 3.0), count)
        flips[t] = count / 3.0
        last = np.where(sign[t] != 0, sign[t], last)
    return signal, hist, flips


def _load_daily_ohlcv(loader: CryptoDataLoader, cache_path: str | Path | None = None):
    if cache_path is not None:
        cache_path = Path(cache_path)
        cached = _load_daily_cache(loader, cache_path)
        if cached is not None:
            return cached

    p = loader.n_active_periods
    s = loader.n_tradable_coins
    out = {k: np.full((p, s), np.nan, dtype=np.float32) for k in ("open", "high", "low", "close", "volume")}
    mpp = loader.minutes_per_period
    coin_indices = loader.coin_indices
    with h5py.File(loader.h5_path, "r") as f:
        for start in range(0, p, loader.chunk_periods):
            end = min(start + loader.chunk_periods, p)
            period_indices = loader.active_period_indices[start:end]
            chunk_p = len(period_indices)
            contiguous = chunk_p > 1 and int(period_indices[-1]) - int(period_indices[0]) == chunk_p - 1
            if contiguous:
                ms = int(period_indices[0]) * mpp
                me = (int(period_indices[-1]) + 1) * mpp
                abs_starts = period_indices.astype(np.int64) * mpp
                abs_ends = np.minimum(abs_starts + mpp - 1, loader.n_usable_minutes - 1)
                out["open"][start:end] = f["open"][abs_starts, :][:, coin_indices]
                out["close"][start:end] = f["close"][abs_ends, :][:, coin_indices]

                high_block = f["high"][ms:me, :][:, coin_indices].reshape(chunk_p, mpp, s)
                low_block = f["low"][ms:me, :][:, coin_indices].reshape(chunk_p, mpp, s)
                volume_block = f["volume"][ms:me, :][:, coin_indices].reshape(chunk_p, mpp, s)
                out["high"][start:end] = _nanmax_axis1(high_block)
                out["low"][start:end] = _nanmin_axis1(low_block)
                out["volume"][start:end] = np.sum(volume_block, axis=1, dtype=np.float32)
            else:
                for i, pi in enumerate(period_indices, start=start):
                    ms = int(pi) * mpp
                    me = ms + mpp
                    out["open"][i] = f["open"][ms, :][coin_indices]
                    out["close"][i] = f["close"][min(me - 1, loader.n_usable_minutes - 1), :][coin_indices]
                    high_block = f["high"][ms:me, :][:, coin_indices]
                    low_block = f["low"][ms:me, :][:, coin_indices]
                    out["high"][i] = _nanmax_axis1(high_block[None, :, :])[0]
                    out["low"][i] = _nanmin_axis1(low_block[None, :, :])[0]
                    out["volume"][i] = np.sum(f["volume"][ms:me, :][:, coin_indices], axis=0, dtype=np.float32)
    out["dates"] = loader.get_active_dates()
    if cache_path is not None:
        _save_daily_cache(loader, cache_path, out)
    return out


def _returns_from_daily_close(close):
    ret = np.full_like(close, np.nan, dtype=np.float32)
    ret[1:] = close[1:] / (close[:-1] + np.float32(1e-10)) - 1.0
    return np.clip(ret, -0.5, 5.0).astype(np.float32, copy=False)


def _tradable_mask_from_daily_volume(loader: CryptoDataLoader, daily: dict[str, np.ndarray]):
    mpp = loader.minutes_per_period
    minute_ends = np.minimum(
        (loader.active_period_indices + 1).astype(np.int64) * mpp - 1,
        loader.n_usable_minutes - 1,
    )
    base_mask = loader.tradable_mask_flat[minute_ends][:, loader.coin_indices].astype(bool, copy=False)
    period_volume = np.nan_to_num(daily["volume"], nan=0.0, posinf=0.0, neginf=0.0)
    lookback_periods = max(1, int(round((7 * 24 * 60) / mpp)))
    rolling_volume = _window_sum_2d(period_volume, lookback_periods).astype(np.float32, copy=False)

    active_counts = base_mask.sum(axis=1)
    thresholds = np.full(base_mask.shape[0], np.inf, dtype=np.float32)
    valid_rows = active_counts > 10
    if np.any(valid_rows):
        masked_volume = np.where(base_mask[valid_rows], rolling_volume[valid_rows], np.nan)
        thresholds[valid_rows] = np.nanquantile(masked_volume, 0.1, axis=1).astype(np.float32, copy=False)

    liquidity_mask = base_mask & (rolling_volume >= thresholds[:, None])
    small_rows = (active_counts > 0) & (active_counts <= 10)
    liquidity_mask[small_rows] = base_mask[small_rows]
    return liquidity_mask


def build_cta_signals(daily: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    op = daily["open"]
    hi = daily["high"]
    lo = daily["low"]
    cl = daily["close"]
    dates = daily["dates"]

    tr = _true_range(hi, lo, cl)
    atr6 = _atr(hi, lo, cl, 6)
    atr9 = _atr(hi, lo, cl, 9)
    atr14 = _atr(hi, lo, cl, 14)
    atr20 = _atr(hi, lo, cl, 20)
    atr50_ema = _ema(tr, 50)
    ema20 = _ema(cl, 20)
    ema50 = _ema(cl, 50)
    ema200 = _ema(cl, 200)
    macd_hist = _macd_hist(cl)
    k40 = _kdj_k(hi, lo, cl, 40, 3)
    k25 = _kdj_k(hi, lo, cl, 25, 3)

    bb20_mid = _roll_mean(cl, 20)
    bb20_std = _roll_std(cl, 20)
    bb20_width = 4.0 * bb20_std
    bb20_lower = bb20_mid - 2.0 * bb20_std
    kc20_width = 3.0 * _ema(tr, 20)
    compression20 = _safe_div(bb20_width, kc20_width)
    bb20_pos = np.clip(_safe_div(cl - bb20_lower, bb20_width), 0.0, 1.0)

    tp = (hi + lo + cl) / 3.0
    ph = 2.0 * tp - lo
    pl = 2.0 * tp - hi
    prev_ph23 = np.vstack([np.full_like(cl[:1], np.nan), _roll_max(ph, 23)[:-1]])
    prev_pl23 = np.vstack([np.full_like(cl[:1], np.nan), _roll_min(pl, 23)[:-1]])

    rsi25 = _rsi(cl, 25)
    rsi100 = _rsi(cl, 100)
    ce_state = _chandelier_state(hi, lo, cl, 22, 3.0)
    v2_base = ce_state * (rsi25 - rsi100) * np.abs((rsi25 - 50.0) / 50.0)
    rsi_peak20 = _roll_max(rsi25, 20)
    rsi_low20 = _roll_min(rsi25, 20)
    pullback_long = np.maximum(0.0, rsi_peak20 - 70.0) / 30.0
    pullback_short = np.maximum(0.0, 30.0 - rsi_low20) / 30.0

    kc_mid50 = ema50
    inner_up = kc_mid50 + 2.75 * atr50_ema
    inner_dn = kc_mid50 - 2.75 * atr50_ema
    outer_up = kc_mid50 + 3.75 * atr50_ema
    outer_dn = kc_mid50 - 3.75 * atr50_ema
    center_zone = np.clip(_safe_div(cl - kc_mid50, 2.75 * atr50_ema), -1.0, 1.0)
    v3 = center_zone.copy()
    v3 = np.where((cl >= inner_up) & (cl <= outer_up), 1.0 + _safe_div(cl - inner_up, outer_up - inner_up), v3)
    v3 = np.where(cl > outer_up, 2.0 + np.clip(_safe_div(cl - outer_up, atr50_ema), 0.0, 1.0), v3)
    v3 = np.where((cl <= inner_dn) & (cl >= outer_dn), -2.0 + _safe_div(cl - outer_dn, inner_dn - outer_dn), v3)
    v3 = np.where(cl < outer_dn, -2.0 - np.clip(_safe_div(outer_dn - cl, atr50_ema), 0.0, 1.0), v3)

    psar = _psar(hi, lo)
    ema_dev = _safe_div(cl - ema200, ema200)
    macd_mom = _safe_div(macd_hist, cl) * 1000.0
    sar_dist = _safe_div(cl - psar, atr14)
    ema_sign = np.sign(ema_dev)
    agree = (
        (np.sign(macd_mom) == ema_sign).astype(np.float32)
        + (np.sign(sar_dist) == ema_sign).astype(np.float32)
        + (ema_sign != 0).astype(np.float32)
    ) / 3.0

    gc_mid = _ema(_ema(tp, 20), 20)
    gc_width = 2.0 * _ema(_ema(tr, 20), 20)
    gc_color = np.sign(np.vstack([np.zeros_like(gc_mid[:1]), np.diff(gc_mid, axis=0)]))
    gc_pos = _safe_div(cl - gc_mid, gc_width)
    macd_delta = np.vstack([np.zeros_like(macd_hist[:1]), np.diff(macd_hist, axis=0)])
    cci20 = _cci(hi, lo, cl, 20)
    cci_delta = np.vstack([np.zeros_like(cci20[:1]), np.diff(cci20, axis=0)])
    cci_regress = ((np.vstack([cci20[:1], cci20[:-1]]) > 200.0) & (cci20 <= 200.0)) | (
        (np.vstack([cci20[:1], cci20[:-1]]) < -200.0) & (cci20 >= -200.0)
    )
    pyramid_depth = np.minimum(_rolling_count(cci_regress, 20), 3.0)
    v5_base = (
        gc_color
        * np.sign(macd_delta)
        * np.sign(cci20)
        * (np.abs(gc_pos) + np.abs(_safe_div(macd_delta, cl) * 1000.0) + np.abs(cci_delta / 100.0)) / 3.0
        * (1.0 + pyramid_depth * 0.33)
    )

    cci18 = _cci(hi, lo, cl, 18)
    cci54 = _cci(hi, lo, cl, 54)
    recent_oversold = _decay_memory(cci18 < -200.0, 0.95)
    recent_overbought = _decay_memory(cci18 > 200.0, 0.95)
    cci_spread = cci18 - cci54
    atr_dev = _safe_div(cl - _roll_mean(cl, 20), atr6)

    st_state = _supertrend_state(hi, lo, cl, 100, 5.5)
    qqe_signal, qqe_hist, qqe_progress = _qqe_signal_progress(cl)
    ema_filter = np.where(cl >= ema200, 1.0, -1.0)
    v7_long = (ema_filter > 0) * (st_state > 0) * np.maximum(0.0, qqe_hist) * np.abs(qqe_signal)
    v7_short = (ema_filter < 0) * (st_state < 0) * np.maximum(0.0, -qqe_hist) * np.abs(qqe_signal)

    body = cl - op
    body_abs = np.abs(body)
    red3 = (
        (body > 0)
        & (np.vstack([body[:1], body[:-1]]) > 0)
        & (np.vstack([body[:2], body[:-2]]) > 0)
        & (cl > np.vstack([cl[:1], cl[:-1]]))
        & (np.vstack([cl[:1], cl[:-1]]) > np.vstack([cl[:2], cl[:-2]]))
    )
    black3 = (
        (body < 0)
        & (np.vstack([body[:1], body[:-1]]) < 0)
        & (np.vstack([body[:2], body[:-2]]) < 0)
        & (cl < np.vstack([cl[:1], cl[:-1]]))
        & (np.vstack([cl[:1], cl[:-1]]) < np.vstack([cl[:2], cl[:-2]]))
    )
    body3 = body_abs + np.vstack([body_abs[:1], body_abs[:-1]]) + np.vstack([body_abs[:2], body_abs[:-2]])
    soldier_strength = _safe_div(body3, atr14)
    pmax_line, pmax_state = _pmax_line_state(hi, lo, cl)
    pmax_dist = _safe_div(cl - pmax_line, atr14)
    occ = _roll_mean(np.sign(body), 3, min_periods=1)
    align_long = (cl > ema200) & (pmax_line > ema200)
    align_short = (cl < ema200) & (pmax_line < ema200)

    hma20 = _hma(cl, 20)
    hma_slope = _safe_div(hma20 - np.vstack([hma20[:1], hma20[:-1]]), atr14)
    mobo_mid = _roll_mean(cl, 20)
    mobo_std = _roll_std(cl, 20)
    mobo_up = mobo_mid + 0.8 * mobo_std
    mobo_dn = mobo_mid - 0.8 * mobo_std
    long_entry = np.maximum(0.0, _safe_div(cl - mobo_up, atr14)) * np.maximum(0.0, hma_slope)
    short_entry = np.maximum(0.0, _safe_div(mobo_dn - cl, atr14)) * np.maximum(0.0, -hma_slope)
    net_entry = long_entry - short_entry
    long_count = _rolling_count(long_entry > 0, 10)
    short_count = _rolling_count(short_entry > 0, 10)
    pyramid_intensity = np.clip((long_count - short_count) / 10.0, -1.0, 1.0)
    coherence = np.sign(net_entry) * np.sign(pyramid_intensity)
    v9_base = pyramid_intensity * np.abs(net_entry) * np.maximum(0.0, coherence)
    ema200_sign = np.sign(cl - ema200)
    v9_alignment = np.sign(v9_base) * ema200_sign

    body_high = np.maximum(op, cl)
    body_low = np.minimum(op, cl)
    cross_above = np.where(body_low > hma20, 1.0, np.where(body_high > hma20, 0.3, 0.0))
    cross_below = np.where(body_high < hma20, 1.0, np.where(body_low < hma20, 0.3, 0.0))
    k_long = np.maximum(0.0, k25 - 50.0) / 50.0
    k_short = np.maximum(0.0, 50.0 - k25) / 50.0

    hma_dev = _safe_div(cl - hma20, _roll_std(cl, 20))
    signal_prev = np.vstack([np.zeros_like(hma_dev[:1]), np.sign(hma_dev[:-1])])
    ret_now = np.vstack([np.zeros_like(cl[:1]), cl[1:] / (cl[:-1] + _EPS) - 1.0])
    hit = (signal_prev != 0) & (np.sign(ret_now) == signal_prev)
    hit_rate6 = _roll_mean(hit.astype(np.float32), 6, min_periods=1)
    prev_h6 = np.vstack([np.full_like(cl[:1], np.nan), _roll_max(hi, 6)[:-1]])
    prev_l6 = np.vstack([np.full_like(cl[:1], np.nan), _roll_min(lo, 6)[:-1]])
    conservative_ok = ((hma_dev > 0) & (cl > prev_h6)) | ((hma_dev < 0) & (cl < prev_l6))
    mode = np.where(hit_rate6 >= 0.5, 1.0, np.where(conservative_ok, 0.5, 0.0))
    weekdays = pd.to_datetime(dates).weekday.to_numpy()
    time_filter = np.where(weekdays == 6, 0.0, np.where(weekdays == 0, 0.5, 1.0)).astype(np.float32)[:, None]
    peak = _roll_max(np.abs(hma_dev), 20, min_periods=1)
    maturity = 1.0 - 0.5 * np.clip(_safe_div(np.abs(hma_dev), peak), 0.0, 1.0)

    atr_short = _atr(hi, lo, cl, 14)
    atr_long = _atr(hi, lo, cl, 50)
    vol_expand = np.maximum(0.0, _safe_div(atr_short - atr_long, atr_long))
    trend_score = (
        ((ema20 > ema50) & (ema50 > ema200)).astype(np.float32)
        - ((ema20 < ema50) & (ema50 < ema200)).astype(np.float32)
    )
    adaptive = np.clip(1.0 + _safe_div(atr_short - np.vstack([atr_short[:1], atr_short[:-1]]), atr_short) * 0.5, 0.5, 2.0)
    long_break = np.maximum(0.0, _safe_div(cl - prev_ph23, atr14))
    short_break = np.maximum(0.0, _safe_div(prev_pl23 - cl, atr14))

    bb25_mid = _roll_mean(cl, 25)
    bb25_std = _roll_std(cl, 25)
    yellow_up = bb25_mid + 2.5 * bb25_std
    yellow_dn = bb25_mid - 2.5 * bb25_std
    blue_up = bb25_mid + 3.75 * bb25_std
    blue_dn = bb25_mid - 3.75 * bb25_std
    bbw = _safe_div(yellow_up - yellow_dn, bb25_mid)
    bbw_gate = bbw > 0.01
    buffer = _safe_div(cl - bb25_mid, atr14)
    long_v13 = (cl > yellow_up) & bbw_gate
    short_v13 = (cl < yellow_dn) & bbw_gate
    profit_decay_long = np.clip(_safe_div(blue_up - cl, 3.0 * atr14), 0.0, 1.0)
    profit_decay_short = np.clip(_safe_div(cl - blue_dn, 3.0 * atr14), 0.0, 1.0)

    ma2 = _roll_mean(cl, 2, min_periods=1)
    ma30 = _roll_mean(cl, 30)
    ma_state = np.where(ma2 >= ma30, 1.0, -1.0)
    just_flip = np.vstack([np.zeros_like(ma_state[:1], dtype=bool), ma_state[1:] != ma_state[:-1]])
    days = _days_since_change(ma_state)
    maturity14 = np.clip(days / 5.0, 0.0, 1.0)
    prev_h16 = np.vstack([np.full_like(cl[:1], np.nan), _roll_max(hi, 16)[:-1]])
    prev_l16 = np.vstack([np.full_like(cl[:1], np.nan), _roll_min(lo, 16)[:-1]])
    v14_long = (ma_state < 0) & (cl > prev_h16) & (~just_flip)
    v14_short = (ma_state > 0) & (cl < prev_l16) & (~just_flip)

    signals = {
        "S1": (1.0 - compression20) * (bb20_pos - 0.5) * (k40 - 50.0),
        "S2": _safe_div(cl - prev_ph23, atr9),
        "V1": (1.0 - compression20) * (bb20_pos - 0.5) * np.where(k25 > 80.0, k25 - 80.0, np.where(k25 < 20.0, k25 - 20.0, 0.0)),
        "V2_base": v2_base,
        "V2_opt": v2_base * (1.0 + np.where(ce_state > 0, pullback_long, pullback_short)),
        "V3_base": v3,
        "V3_opt": v3 * _safe_div(ema20 - ema50, atr50_ema),
        "V4_base": ema_sign * (np.abs(ema_dev) + np.abs(macd_mom) + np.abs(sar_dist)) * agree,
        "V5_base": v5_base,
        "V6_base": (recent_oversold * np.maximum(0.0, cci_spread) - recent_overbought * np.maximum(0.0, -cci_spread)) * (1.0 + np.abs(atr_dev) * 0.3) / 100.0,
        "V7_base": (v7_long - v7_short) * (1.0 - np.clip(qqe_progress, 0.0, 1.0)) * 0.01,
        "V8_base": (
            red3.astype(np.float32) * soldier_strength * np.maximum(0.0, pmax_dist) * np.maximum(0.0, occ) * align_long.astype(np.float32)
            - black3.astype(np.float32) * soldier_strength * np.maximum(0.0, -pmax_dist) * np.maximum(0.0, -occ) * align_short.astype(np.float32)
        ) * (1.0 + np.abs(ema_dev)),
        "V9_base": v9_base,
        "V9_opt": v9_base * (1.0 + 0.5 * v9_alignment),
        "V10_base": np.maximum(0.0, hma_slope) * cross_above * (cl > ema200) * k_long - np.maximum(0.0, -hma_slope) * cross_below * (cl < ema200) * k_short,
        "V11_base": np.sign(hma_dev) * np.minimum(np.abs(hma_dev), 3.0) * mode * time_filter * maturity,
        "V12_base": (long_break - short_break) * (0.5 + vol_expand) * np.maximum(0.0, np.sign(long_break - short_break) * trend_score) * adaptive,
        "V13_base": long_v13.astype(np.float32) * np.maximum(0.0, buffer) * profit_decay_long - short_v13.astype(np.float32) * np.maximum(0.0, -buffer) * profit_decay_short,
        "V14_base": v14_long.astype(np.float32) * np.minimum(long_break, 2.0) * maturity14 - v14_short.astype(np.float32) * np.minimum(short_break, 2.0) * maturity14,
    }
    return {name: _clean(signals[name]) for name, _ in CTA_SIGNAL_SPECS}


def _objective_dict(obj, aux, idx):
    return {
        "ls_netret": float(obj[idx, 0]),
        "ls_net_sharpe": float(obj[idx, 1]),
        "neg_turnover": float(obj[idx, 2]),
        "ls_1-maxdd": float(obj[idx, 4]),
        "ic_mean": float(aux["ic_mean"][idx]),
        "rankicir": float(aux["rankicir"][idx]),
        "positive_rate": float(aux["positive_rate"][idx]),
        "coverage": float(aux["coverage"][idx]),
        "ls_turnover": float(aux["ls_turnover"][idx]),
        "long_turnover": float(aux["long_turnover"][idx]),
        "ls_positive_rate": float(aux["ls_positive_rate"][idx]),
        "effective_bars": int(round(float(aux["effective_bars"][idx]))),
    }


def _gate_reasons(train, yearly_summary, oos):
    reasons = []
    turnover_pressure = max(train["ls_turnover"], train["long_turnover"])
    if train["ls_netret"] <= FINAL_MIN_LS_NETRET:
        reasons.append("is_netret")
    if train["ls_net_sharpe"] < FINAL_MIN_LS_NET_SHARPE:
        reasons.append("is_sharpe")
    if not np.isfinite(turnover_pressure) or turnover_pressure > FINAL_MAX_LS_TURNOVER:
        reasons.append("turnover")
    if train["ls_positive_rate"] < FINAL_MIN_LS_POSITIVE_RATE:
        reasons.append("positive_rate")
    if train["coverage"] < FINAL_MIN_COVERAGE:
        reasons.append("coverage")
    if yearly_summary["min_ls_netret"] <= FINAL_MIN_YEARLY_LS_NETRET:
        reasons.append("yearly_netret")
    if yearly_summary["min_ls_net_sharpe"] <= FINAL_MIN_YEARLY_LS_NET_SHARPE:
        reasons.append("yearly_sharpe")
    if yearly_summary["max_turnover"] > FINAL_MAX_YEARLY_TURNOVER:
        reasons.append("yearly_turnover")
    if yearly_summary["min_coverage"] < FINAL_MIN_YEARLY_COVERAGE:
        reasons.append("yearly_coverage")
    if oos["ls_netret"] <= 0.0:
        reasons.append("oos_netret")
    if oos["ls_net_sharpe"] <= MIN_OOS_SHARPE:
        reasons.append("oos_sharpe")
    if oos["ls_turnover"] > MAX_OOS_TURNOVER:
        reasons.append("oos_turnover")
    return reasons


def evaluate_cta_signals(
    h5_path=DEFAULT_H5_PATH,
    start_date="20230101",
    end_date="20260401",
    train_end_date="20260101",
    oos_start="20260101",
    oos_end="20260401",
    top_quantile=TOP_QUANTILE,
    trading_cost=TRADING_COST,
    cpu_workers=None,
    daily_cache_path="auto",
):
    if cpu_workers:
        os.environ["GP_CPU_WORKERS"] = str(int(cpu_workers))
    loader = CryptoDataLoader(
        h5_path=h5_path,
        minutes_per_period=1440,
        chunk_periods=60,
        start_date=start_date,
        end_date=end_date,
        backend="cpu",
    )
    if daily_cache_path == "auto":
        daily_cache_path = _default_daily_cache_path(loader)
    daily = _load_daily_ohlcv(loader, cache_path=daily_cache_path)
    signals = build_cta_signals(daily)

    names = [name for name, _ in CTA_SIGNAL_SPECS]
    factors = np.stack([signals[name] for name in names]).astype(np.float32, copy=False)
    tradable = _tradable_mask_from_daily_volume(loader, daily)
    factors = np.where(tradable[None, :, :], factors, np.nan).astype(np.float32, copy=False)

    dates = np.asarray(loader.get_active_dates())
    returns = _returns_from_daily_close(daily["close"])
    lagged = build_lagged_returns(returns, lag=LAG_PERIODS)

    masks = {
        "train": (dates >= start_date) & (dates < train_end_date),
        "full": (dates >= start_date) & (dates < oos_end),
        "oos": (dates >= oos_start) & (dates < oos_end),
    }
    contexts = {
        key: prepare_fitness_context(lagged, is_mask=mask, tradable_mask=tradable)
        for key, mask in masks.items()
    }
    eval_kwargs = {
        "periods_per_year": PERIODS_PER_YEAR,
        "top_frac": float(top_quantile),
        "trading_cost": float(trading_cost),
        "return_aux": True,
    }
    train_obj, train_aux = compute_five_objectives_prepared(factors, contexts["train"], **eval_kwargs)
    directions = train_aux["ic_direction"]
    full_obj, full_aux = compute_five_objectives_prepared(
        factors, contexts["full"], ic_directions=directions, **eval_kwargs)
    oos_obj, oos_aux = compute_five_objectives_prepared(
        factors, contexts["oos"], ic_directions=directions, **eval_kwargs)

    yearly_payload = {}
    yearly_metrics = {}
    for label, y_start, y_end in FINAL_YEARLY_STABILITY_WINDOWS:
        mask = (dates >= y_start) & (dates < y_end)
        ctx = prepare_fitness_context(lagged, is_mask=mask, tradable_mask=tradable)
        obj, aux = compute_five_objectives_prepared(
            factors, ctx, ic_directions=directions, **eval_kwargs)
        yearly_payload[label] = (obj, aux, int(mask.sum()))
        yearly_metrics[label] = [
            _objective_dict(obj, aux, i)
            for i in range(len(names))
        ]

    rows = []
    for i, name in enumerate(names):
        y_windows = {
            label: yearly_metrics[label][i]
            for label in yearly_metrics
        }
        y_summary = {
            "min_ls_netret": float(np.nanmin([v["ls_netret"] for v in y_windows.values()])),
            "min_ls_net_sharpe": float(np.nanmin([v["ls_net_sharpe"] for v in y_windows.values()])),
            "max_turnover": float(np.nanmax([v["ls_turnover"] for v in y_windows.values()])),
            "min_coverage": float(np.nanmin([v["coverage"] for v in y_windows.values()])),
        }
        train_m = _objective_dict(train_obj, train_aux, i)
        full_m = _objective_dict(full_obj, full_aux, i)
        oos_m = _objective_dict(oos_obj, oos_aux, i)
        reasons = _gate_reasons(train_m, y_summary, oos_m)
        rows.append({
            "name": name,
            "description": dict(CTA_SIGNAL_SPECS)[name],
            "direction": int(-1 if float(directions[i]) < 0.0 else 1),
            "passed": not reasons,
            "gate_reasons": reasons,
            "train": train_m,
            "full": full_m,
            "oos": oos_m,
            "yearly_summary": y_summary,
            "yearly": y_windows,
            "non_nan_ratio": float(np.isfinite(factors[i]).mean()),
        })

    rows.sort(
        key=lambda r: (
            bool(r["passed"]),
            r["oos"]["ls_net_sharpe"],
            r["yearly_summary"]["min_ls_net_sharpe"],
            r["train"]["ls_net_sharpe"],
        ),
        reverse=True,
    )
    return {
        "h5_path": h5_path,
        "start_date": start_date,
        "train_end_date": train_end_date,
        "oos_start": oos_start,
        "oos_end": oos_end,
        "top_quantile": float(top_quantile),
        "trading_cost": float(trading_cost),
        "daily_cache_path": None if daily_cache_path is None else str(daily_cache_path),
        "n_signals": len(names),
        "missing_from_user_prompt": 4,
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate hand-written CTA daily signals with unified_v2 fitness.")
    parser.add_argument("--h5_path", default=DEFAULT_H5_PATH)
    parser.add_argument("--start_date", default="20230101")
    parser.add_argument("--end_date", default="20260401")
    parser.add_argument("--train_end_date", default="20260101")
    parser.add_argument("--oos_start", default="20260101")
    parser.add_argument("--oos_end", default="20260401")
    parser.add_argument("--top_quantile", type=float, default=TOP_QUANTILE)
    parser.add_argument("--trading_cost", type=float, default=TRADING_COST)
    parser.add_argument("--cpu_workers", type=int, default=8)
    parser.add_argument("--daily_cache_path", default="auto")
    parser.add_argument("--no_daily_cache", action="store_true")
    parser.add_argument("--json_output", default=None)
    args = parser.parse_args()

    payload = evaluate_cta_signals(
        h5_path=args.h5_path,
        start_date=args.start_date,
        end_date=args.end_date,
        train_end_date=args.train_end_date,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
        top_quantile=args.top_quantile,
        trading_cost=args.trading_cost,
        cpu_workers=args.cpu_workers,
        daily_cache_path=None if args.no_daily_cache else args.daily_cache_path,
    )
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    print(f"CTA daily signals evaluated: {payload['n_signals']}")
    print(f"Missing prompt strategies: {payload['missing_from_user_prompt']}")
    print(f"{'Name':<10} {'Pass':<5} {'IS Shp':>8} {'MinY Shp':>9} {'OOS Shp':>8} {'OOS Ret':>8} {'TO':>6} Reasons")
    for row in payload["results"]:
        print(
            f"{row['name']:<10} {str(row['passed']):<5} "
            f"{row['train']['ls_net_sharpe']:8.3f} "
            f"{row['yearly_summary']['min_ls_net_sharpe']:9.3f} "
            f"{row['oos']['ls_net_sharpe']:8.3f} "
            f"{row['oos']['ls_netret']:8.3f} "
            f"{row['oos']['ls_turnover']:6.3f} "
            f"{','.join(row['gate_reasons'])}"
        )


if __name__ == "__main__":
    main()
