"""Crypto HDF5 data loader with GPU indicator caching.

v2: Precompute all indicators once at init, cache in CPU RAM (float16).
Eliminates per-generation h5 IO — the main v1 bottleneck.
"""
import gc
import os
import time
import numpy as np
import h5py
from datetime import datetime
from pathlib import Path

from .backend import xp, set_backend, to_xp, free_gpu, get_backend
from .config import (
    BASE_FIELDS, DEFAULT_H5_PATH, MINUTES_PER_PERIOD,
    CHUNK_PERIODS, PERIODS_PER_DAY, CACHE_DTYPE, N_INDICATORS,
    CROSS_PERIOD_INDICATOR_NAMES, FUNDAMENTAL_INDICATOR_NAMES, INDICATOR_INDEX,
    INDICATOR_NAMES,
    ORDER_FLOW_RAW_FIELDS, XBINANCE_ORDER_FLOW_RAW_FIELDS,
)
from .indicators import derive_indicators, derive_indicators_subset

DEFAULT_TRADABLE_MASK_OVERLAY = (
    "/root/crypto-research/common/data/raw/h5/bybit_tradable_mask_mining.npy"
)

_EPS = 1e-12


def _build_period_masks(period_dates, start_date=None, end_date=None, train_end_date=None):
    """Return active period mask and in-sample mask over active periods."""
    dates = np.asarray(period_dates)
    period_mask = np.ones(len(dates), dtype=bool)
    if start_date:
        period_mask &= dates >= str(start_date)
    if end_date:
        period_mask &= dates <= str(end_date)
    active_dates = dates[period_mask]
    is_mask = np.ones(len(active_dates), dtype=bool)
    if train_end_date:
        is_mask &= active_dates <= str(train_end_date)
    return period_mask, is_mask


def _rolling_sums_2d(x, window):
    arr = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(arr)
    values = np.where(finite, arr, 0.0)
    zero = np.zeros((1, arr.shape[1]), dtype=np.float64)
    csum = np.vstack([zero, np.cumsum(values, axis=0, dtype=np.float64)])
    ccnt = np.vstack([zero, np.cumsum(finite.astype(np.float64), axis=0)])
    right = np.arange(arr.shape[0]) + 1
    left = np.maximum(0, right - int(window))
    return csum[right] - csum[left], ccnt[right] - ccnt[left]


def _rolling_mean_std_2d(x, window, min_periods=None):
    arr = np.asarray(x, dtype=np.float64)
    min_periods = int(min_periods or max(3, min(window, window // 3)))
    total, count = _rolling_sums_2d(arr, window)
    total2, _ = _rolling_sums_2d(arr * arr, window)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count >= min_periods)
    var = np.divide(total2, count, out=np.full_like(total2, np.nan), where=count >= min_periods) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, std, count


def _rolling_zscore_2d(x, window, min_periods=None):
    mean, std, count = _rolling_mean_std_2d(x, window, min_periods=min_periods)
    out = (np.asarray(x, dtype=np.float64) - mean) / (std + _EPS)
    return np.where(count >= int(min_periods or max(3, min(window, window // 3))), out, np.nan)


def _shift_rows(x, lag):
    out = np.full_like(x, np.nan, dtype=np.float64)
    if lag <= 0:
        return np.asarray(x, dtype=np.float64)
    out[lag:] = np.asarray(x, dtype=np.float64)[:-lag]
    return out


def _ema_causal_2d(x, span):
    arr = np.asarray(x, dtype=np.float64)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    alpha = 2.0 / (float(span) + 1.0)
    prev = np.full(arr.shape[1], np.nan, dtype=np.float64)
    for i in range(arr.shape[0]):
        cur = arr[i]
        cur_ok = np.isfinite(cur)
        prev_ok = np.isfinite(prev)
        nxt = np.where(prev_ok & cur_ok, alpha * cur + (1.0 - alpha) * prev, np.where(cur_ok, cur, prev))
        out[i] = nxt
        prev = nxt
    return out


def _rolling_beta_corr_2d(x, y, window, min_periods=None):
    x = np.asarray(x, dtype=np.float64)
    y2 = np.broadcast_to(np.asarray(y, dtype=np.float64)[:, None], x.shape)
    valid = np.isfinite(x) & np.isfinite(y2)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y2, 0.0)
    xy = xv * yv
    x2 = xv * xv
    y2v = yv * yv
    min_periods = int(min_periods or max(5, min(window, window // 3)))

    def roll(values):
        zero = np.zeros((1, values.shape[1]), dtype=np.float64)
        csum = np.vstack([zero, np.cumsum(values, axis=0, dtype=np.float64)])
        right = np.arange(values.shape[0]) + 1
        left = np.maximum(0, right - int(window))
        return csum[right] - csum[left]

    count = roll(valid.astype(np.float64))
    sx, sy, sxy, sx2, sy2 = roll(xv), roll(yv), roll(xy), roll(x2), roll(y2v)
    mx = np.divide(sx, count, out=np.full_like(sx, np.nan), where=count >= min_periods)
    my = np.divide(sy, count, out=np.full_like(sy, np.nan), where=count >= min_periods)
    cov = np.divide(sxy, count, out=np.full_like(sxy, np.nan), where=count >= min_periods) - mx * my
    vx = np.divide(sx2, count, out=np.full_like(sx2, np.nan), where=count >= min_periods) - mx * mx
    vy = np.divide(sy2, count, out=np.full_like(sy2, np.nan), where=count >= min_periods) - my * my
    beta = cov / (vy + _EPS)
    corr = cov / (np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0)) + _EPS)
    ok = count >= min_periods
    return np.where(ok, beta, np.nan), np.where(ok, np.clip(corr, -1.0, 1.0), np.nan)


def _cross_section_rank_2d(x, mask=None):
    arr = np.asarray(x, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    if mask is None:
        valid_base = np.ones(arr.shape, dtype=bool)
    else:
        valid_base = np.asarray(mask, dtype=bool)
    for i in range(arr.shape[0]):
        idx = np.flatnonzero(valid_base[i] & np.isfinite(arr[i]))
        if idx.size < 2:
            continue
        order = np.argsort(arr[i, idx], kind="mergesort")
        ranks = np.empty(idx.size, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, idx.size, dtype=np.float32)
        out[i, idx] = ranks
    return out


def _cross_sectional_style_residualize(raw, style_panel, valid_mask):
    raw_arr = np.asarray(raw, dtype=np.float64)
    styles = np.asarray(style_panel, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if raw_arr.ndim != 2:
        raise ValueError("raw must have shape (time, assets)")
    if styles.ndim != 3:
        raise ValueError("style_panel must have shape (time, assets, n_style)")
    if mask.shape != raw_arr.shape or styles.shape[:2] != raw_arr.shape:
        raise ValueError("raw, style_panel, and valid_mask shapes are inconsistent")

    n_style = styles.shape[2]
    min_rows = max(10, n_style + 3)
    out = np.full(raw_arr.shape, np.nan, dtype=np.float32)
    finite_styles = np.all(np.isfinite(styles), axis=2)
    finite = mask & np.isfinite(raw_arr) & finite_styles
    ridge = np.float64(1e-6)
    for i in range(raw_arr.shape[0]):
        idx = np.flatnonzero(finite[i])
        if idx.size < min_rows:
            continue
        y = raw_arr[i, idx]
        x_style = styles[i, idx, :]
        x = np.empty((idx.size, n_style + 1), dtype=np.float64)
        x[:, 0] = 1.0
        x[:, 1:] = x_style
        xtx = x.T @ x
        penalty = np.eye(n_style + 1, dtype=np.float64) * ridge
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(xtx + penalty, x.T @ y)
        out[i, idx] = (y - x @ beta).astype(np.float32, copy=False)
    return out


class CryptoDataLoader:
    """Load Binance perp 1m data, precompute indicators, and serve cached chunks."""

    def __init__(self, h5_path=DEFAULT_H5_PATH, minutes_per_period=MINUTES_PER_PERIOD,
                 chunk_periods=CHUNK_PERIODS,
                 start_date=None, end_date=None, train_end_date=None,
                 min_tradable=50, backend='cpu', gpu_id=0,
                 cache_dtype=CACHE_DTYPE, tradable_mask_overlay_path=None,
                 min_presence_frac=0.01,
                 indicator_field_indices=None,
                 fundamental_panel_path=None):
        self.h5_path = h5_path
        self.minutes_per_period = minutes_per_period
        self.chunk_periods = chunk_periods
        self.backend = backend
        self.gpu_id = gpu_id
        self.cache_dtype = np.dtype(cache_dtype)
        self.fundamental_panel_path = None
        self._fundamental_panel = None
        self.indicator_field_indices = None
        if indicator_field_indices is not None:
            fields = np.asarray(indicator_field_indices, dtype=np.intp)
            if fields.ndim != 1 or fields.size == 0:
                raise ValueError("indicator_field_indices must be a non-empty 1d array")
            self.indicator_field_indices = np.unique(fields)
        self.min_presence_frac = float(min_presence_frac)
        self.start_date = start_date
        self.end_date = end_date
        self.train_end_date = train_end_date
        self.tradable_mask_overlay_path = None

        with h5py.File(h5_path, 'r') as f:
            self.timestamps = f['timestamp'][:]
            self.symbols = np.array([s.decode() if isinstance(s, bytes) else str(s)
                                     for s in f['symbols'][:]])
            self.tradable_mask_flat = f['tradable_mask'][:]
        overlay_path = tradable_mask_overlay_path
        overlay_explicit = overlay_path is not None
        if overlay_path is None:
            overlay_path = os.environ.get('GP_TRADABLE_MASK_OVERLAY', DEFAULT_TRADABLE_MASK_OVERLAY)
            overlay_explicit = 'GP_TRADABLE_MASK_OVERLAY' in os.environ
        if overlay_path and str(overlay_path).lower() not in {'0', 'false', 'none', 'off'}:
            overlay = Path(overlay_path)
            if overlay.exists():
                mask_overlay = np.load(overlay, mmap_mode='r')
                if mask_overlay.shape != self.tradable_mask_flat.shape:
                    raise ValueError(
                        f"tradable mask overlay shape mismatch: {mask_overlay.shape} "
                        f"!= {self.tradable_mask_flat.shape} ({overlay})"
                    )
                self.tradable_mask_flat &= mask_overlay[:].astype(bool, copy=False)
                self.tradable_mask_overlay_path = str(overlay)
            elif overlay_explicit:
                raise FileNotFoundError(f"tradable mask overlay not found: {overlay}")

        panel_path = fundamental_panel_path
        panel_explicit = panel_path is not None
        if panel_path is None:
            panel_path = os.environ.get('GP_FUNDAMENTAL_PANEL')
        if panel_path and str(panel_path).lower() not in {'0', 'false', 'none', 'off'}:
            panel = Path(panel_path)
            if panel.exists():
                from factor_platform.services.coinmetrics_market_cap import load_market_cap_panel

                self._fundamental_panel = load_market_cap_panel(panel)
                self.fundamental_panel_path = str(panel)
                os.environ['GP_FUNDAMENTAL_PANEL'] = str(panel)
            elif panel_explicit:
                raise FileNotFoundError(f"fundamental panel not found: {panel}")

        self.n_total_minutes = len(self.timestamps)
        self.n_coins = len(self.symbols)

        ts0 = self.timestamps[0]
        scale = 1000 if ts0 > 1e12 else 1
        self.dates_flat = np.array([
            datetime.utcfromtimestamp(t / scale).strftime('%Y%m%d')
            for t in self.timestamps[::minutes_per_period]
        ])

        self.n_total_periods = self.n_total_minutes // minutes_per_period
        self.n_usable_minutes = self.n_total_periods * minutes_per_period

        period_dates = self.dates_flat[:self.n_total_periods]
        self.period_mask, self.is_mask = _build_period_masks(
            period_dates,
            start_date=start_date,
            end_date=end_date,
            train_end_date=train_end_date,
        )

        self.active_period_indices = np.where(self.period_mask)[0]
        self.n_active_periods = len(self.active_period_indices)
        self.n_is = int(self.is_mask.sum())
        self.full_mask = np.ones(self.n_active_periods, dtype=bool)

        selection_periods = self.active_period_indices
        sample_periods = selection_periods[::max(1, len(selection_periods) // 100)]
        if len(sample_periods) == 0 and len(selection_periods) > 0:
            sample_periods = selection_periods[:1]

        tradable_counts = np.zeros(self.n_coins, dtype=np.int32)
        for pi in sample_periods:
            minute_end = min((int(pi) + 1) * minutes_per_period - 1, self.n_usable_minutes - 1)
            tradable_counts += self.tradable_mask_flat[minute_end].astype(np.int32)
        # The 2022+ train span would over-penalize newer listings at 30%.
        # 1% keeps the mining universe near its causal upper bound without
        # using future/OOS listings.
        min_presence = max(1, int(np.ceil(len(sample_periods) * self.min_presence_frac)))
        self.coin_mask = tradable_counts >= min_presence
        self.coin_indices = np.where(self.coin_mask)[0]
        self.n_tradable_coins = len(self.coin_indices)

        if self.n_tradable_coins < min_tradable:
            minute_ends = np.minimum(
                (sample_periods.astype(np.int64) + 1) * minutes_per_period - 1,
                self.n_usable_minutes - 1,
            )
            with h5py.File(h5_path, 'r') as f:
                turnover_ds = f['turnover']
                row_min, row_max = int(minute_ends.min()), int(minute_ends.max())
                turnover_block = turnover_ds[row_min:row_max + 1, :]
                local_idx = (minute_ends - row_min).astype(np.intp)
                turnover_sample = turnover_block[local_idx].astype(np.float32, copy=False)

            tradable_sample = self.tradable_mask_flat[minute_ends].astype(bool, copy=False)
            turnover_sample = np.where(tradable_sample, turnover_sample, np.nan)

            candidate_indices = np.where(tradable_counts > 0)[0]
            target_coins = min(self.n_coins, max(min_tradable, 120))
            if len(candidate_indices) > target_coins:
                candidate_scores = np.nanmean(turnover_sample[:, candidate_indices], axis=0)
                candidate_scores = np.nan_to_num(candidate_scores, nan=-np.inf)
                top_order = np.argsort(candidate_scores)[::-1][:target_coins]
                self.coin_indices = np.sort(candidate_indices[top_order])
            else:
                self.coin_indices = candidate_indices
            self.n_tradable_coins = len(self.coin_indices)

        self._period_returns = None
        self._period_tradable_mask = None
        self._period_volume = None
        self._period_sum_fields = {}
        self._period_end_fields = {}
        self._cross_period_features = None
        self._indicator_cache = None  # will hold list of np.float16 arrays

    def get_period_returns(self):
        if self._period_returns is not None:
            return self._period_returns

        mpp = self.minutes_per_period
        minute_ends = np.minimum(
            (self.active_period_indices + 1).astype(np.int64) * mpp - 1,
            self.n_usable_minutes - 1,
        )

        with h5py.File(self.h5_path, 'r') as f:
            close_ds = f['close']
            row_min, row_max = int(minute_ends.min()), int(minute_ends.max())
            close_block = close_ds[row_min:row_max + 1, :][:, self.coin_indices]
            local_idx = (minute_ends - row_min).astype(np.intp)
            period_close = close_block[local_idx].astype(np.float32)

        ret = np.full_like(period_close, np.nan)
        ret[1:] = period_close[1:] / (period_close[:-1] + 1e-10) - 1.0
        ret = np.clip(ret, -0.5, 5.0)
        self._period_returns = ret
        return ret

    def get_period_volume(self):
        if self._period_volume is not None:
            return self._period_volume

        mpp = self.minutes_per_period
        volume = np.zeros((self.n_active_periods, self.n_tradable_coins), dtype=np.float32)
        with h5py.File(self.h5_path, 'r') as f:
            volume_ds = f['volume']
            for start in range(0, self.n_active_periods, self.chunk_periods):
                end = min(start + self.chunk_periods, self.n_active_periods)
                period_indices = self.active_period_indices[start:end]
                chunk_p = len(period_indices)
                is_contiguous = (
                    chunk_p > 1
                    and int(period_indices[-1]) - int(period_indices[0]) == chunk_p - 1
                )
                if is_contiguous:
                    m_start = int(period_indices[0]) * mpp
                    m_end = (int(period_indices[-1]) + 1) * mpp
                    block = volume_ds[m_start:m_end, :][:, self.coin_indices]
                    reshaped = block.reshape(chunk_p, mpp, -1)
                    valid_count = np.isfinite(reshaped).sum(axis=1)
                    chunk = np.nansum(reshaped, axis=1, dtype=np.float32)
                    chunk[valid_count == 0] = np.nan
                else:
                    chunk = np.empty((chunk_p, self.n_tradable_coins), dtype=np.float32)
                    for i, pi in enumerate(period_indices):
                        ms = int(pi) * mpp
                        me = ms + mpp
                        block = volume_ds[ms:me, :][:, self.coin_indices]
                        valid_count = np.isfinite(block).sum(axis=0)
                        values = np.nansum(block, axis=0, dtype=np.float32)
                        values[valid_count == 0] = np.nan
                        chunk[i] = values
                volume[start:end] = chunk

        self._period_volume = volume
        return volume

    def _get_period_sum_field(self, field):
        if field in self._period_sum_fields:
            return self._period_sum_fields[field]

        out = np.full((self.n_active_periods, self.n_tradable_coins), np.nan, dtype=np.float32)
        with h5py.File(self.h5_path, 'r') as f:
            if field not in f:
                self._period_sum_fields[field] = out
                return out
            ds = f[field]
            mpp = self.minutes_per_period
            for start in range(0, self.n_active_periods, self.chunk_periods):
                end = min(start + self.chunk_periods, self.n_active_periods)
                period_indices = self.active_period_indices[start:end]
                chunk_p = len(period_indices)
                is_contiguous = (
                    chunk_p > 1
                    and int(period_indices[-1]) - int(period_indices[0]) == chunk_p - 1
                )
                if is_contiguous:
                    m_start = int(period_indices[0]) * mpp
                    m_end = (int(period_indices[-1]) + 1) * mpp
                    block = ds[m_start:m_end, :][:, self.coin_indices]
                    reshaped = block.reshape(chunk_p, mpp, -1)
                    valid_count = np.isfinite(reshaped).sum(axis=1)
                    chunk = np.nansum(reshaped, axis=1, dtype=np.float32)
                    chunk[valid_count == 0] = np.nan
                    out[start:end] = chunk
                else:
                    for i, pi in enumerate(period_indices, start):
                        ms = int(pi) * mpp
                        me = ms + mpp
                        block = ds[ms:me, :][:, self.coin_indices]
                        valid_count = np.isfinite(block).sum(axis=0)
                        values = np.nansum(block, axis=0, dtype=np.float32)
                        values[valid_count == 0] = np.nan
                        out[i] = values
        self._period_sum_fields[field] = out
        return out

    def get_period_turnover(self):
        return self._get_period_sum_field("turnover")

    def _get_period_end_field(self, field):
        if field in self._period_end_fields:
            return self._period_end_fields[field]

        out = np.full((self.n_active_periods, self.n_tradable_coins), np.nan, dtype=np.float32)
        with h5py.File(self.h5_path, 'r') as f:
            if field not in f:
                self._period_end_fields[field] = out
                return out
            mpp = self.minutes_per_period
            minute_ends = np.minimum(
                (self.active_period_indices + 1).astype(np.int64) * mpp - 1,
                self.n_usable_minutes - 1,
            )
            ds = f[field]
            for start in range(0, self.n_active_periods, self.chunk_periods):
                end = min(start + self.chunk_periods, self.n_active_periods)
                rows = minute_ends[start:end]
                row_min, row_max = int(rows.min()), int(rows.max())
                block = ds[row_min:row_max + 1, :][:, self.coin_indices]
                out[start:end] = block[(rows - row_min).astype(np.intp)].astype(np.float32, copy=False)
        self._period_end_fields[field] = out
        return out

    def get_active_dates(self):
        return self.dates_flat[self.active_period_indices]

    def get_period_tradable_mask(self):
        if self._period_tradable_mask is not None:
            return self._period_tradable_mask

        mpp = self.minutes_per_period
        minute_ends = np.minimum(
            (self.active_period_indices + 1).astype(np.int64) * mpp - 1,
            self.n_usable_minutes - 1,
        )
        base_mask = self.tradable_mask_flat[minute_ends][:, self.coin_indices].astype(bool, copy=False)
        period_volume = np.nan_to_num(self.get_period_volume(), nan=0.0, posinf=0.0, neginf=0.0)
        lookback_periods = max(1, int(round((7 * 24 * 60) / mpp)))

        csum = np.cumsum(period_volume, axis=0, dtype=np.float64)
        rolling_volume = np.empty_like(period_volume, dtype=np.float32)
        for idx in range(self.n_active_periods):
            left = idx - lookback_periods
            if left >= 0:
                window_sum = csum[idx] - csum[left]
            else:
                window_sum = csum[idx]
            rolling_volume[idx] = window_sum.astype(np.float32, copy=False)

        liquidity_mask = np.zeros_like(base_mask, dtype=bool)
        for idx in range(self.n_active_periods):
            active = base_mask[idx]
            if not np.any(active):
                continue
            active_volume = rolling_volume[idx, active]
            if active_volume.size <= 10:
                liquidity_mask[idx, active] = True
                continue
            threshold = np.nanquantile(active_volume, 0.1)
            liquidity_mask[idx] = active & (rolling_volume[idx] >= threshold)

        self._period_tradable_mask = liquidity_mask
        return self._period_tradable_mask

    def n_chunks(self):
        return (self.n_active_periods + self.chunk_periods - 1) // self.chunk_periods

    def _optional_raw_fields_to_load(self):
        from .config import EXTRA_RAW_FIELDS

        optional_fields = (*EXTRA_RAW_FIELDS, *ORDER_FLOW_RAW_FIELDS, *XBINANCE_ORDER_FLOW_RAW_FIELDS)
        if self.indicator_field_indices is None:
            return optional_fields

        names = {
            INDICATOR_NAMES[int(idx)]
            for idx in self.indicator_field_indices
            if 0 <= int(idx) < len(INDICATOR_NAMES)
        }
        required = {field for field in EXTRA_RAW_FIELDS if field in names}
        required.update(field for field in ORDER_FLOW_RAW_FIELDS if field in names)
        required.update(field for field in XBINANCE_ORDER_FLOW_RAW_FIELDS if field in names)

        if names.intersection({
            "oi_change_intraday",
            "oi_price_alignment",
            "new_long_pressure",
            "new_short_pressure",
            "short_cover_pressure",
            "long_liquidation_pressure",
        }):
            required.add("open_interest")
        if "premium_change_intraday" in names:
            required.add("premium_close")
        if names.intersection({"mark_index_spread", "mark_index_spread_change"}):
            required.update(("mark_close", "index_close"))
        if "funding_abs" in names:
            required.add("funding")
        if names.intersection({"taker_buy_ratio", "taker_pressure"}):
            required.add("taker_buy_volume")
        if "taker_quote_ratio" in names:
            required.update(("taker_buy_quote_volume", "turnover"))
        if "avg_trade_size" in names:
            required.update(("trade_count", "turnover"))
        if names.intersection({"xbinance_taker_quote_ratio", "xbinance_taker_pressure"}):
            required.update(("xbinance_quote_volume", "xbinance_taker_buy_quote_volume"))
        if "xbinance_avg_trade_size" in names:
            required.update(("xbinance_quote_volume", "xbinance_trade_count"))
        if self._needs_fundamental_fields():
            required.add("turnover")
        return tuple(field for field in optional_fields if field in required)

    def _load_chunk_raw(self, chunk_idx):
        """Load raw data for one chunk from h5."""
        start = chunk_idx * self.chunk_periods
        end = min(start + self.chunk_periods, self.n_active_periods)
        period_indices = self.active_period_indices[start:end]
        chunk_p = len(period_indices)
        mpp = self.minutes_per_period

        is_contiguous = (chunk_p > 1 and
                         int(period_indices[-1]) - int(period_indices[0]) == chunk_p - 1)

        raw = {}
        with h5py.File(self.h5_path, 'r') as f:
            # OHLCV is always required; optional raw fields are dependency-pruned
            # after field slate selection to keep H2D transfer and pinned memory bounded.
            optional_fields = self._optional_raw_fields_to_load()
            fields_to_load = list(BASE_FIELDS) + [field for field in optional_fields if field in f]
            for field in fields_to_load:
                ds = f[field]
                if is_contiguous:
                    m_start = int(period_indices[0]) * mpp
                    m_end = (int(period_indices[-1]) + 1) * mpp
                    big_slice = ds[m_start:m_end, :][:, self.coin_indices]
                    raw[field] = big_slice.reshape(chunk_p, mpp, -1).astype(
                        np.float32, copy=False)
                else:
                    arr = np.empty((chunk_p, mpp, self.n_tradable_coins),
                                   dtype=np.float32)
                    for i, pi in enumerate(period_indices):
                        ms = int(pi) * mpp
                        me = ms + mpp
                        arr[i] = ds[ms:me, :][:, self.coin_indices]
                    raw[field] = arr
        return raw, chunk_p

    def precompute_indicators(self):
        """Precompute all indicators and cache in CPU RAM.

        This is the key v2 optimization: compute indicators once at startup,
        then serve from cache every generation. Saves ~90% of per-generation time.
        """
        import time
        t0 = time.time()
        n_ch = self.n_chunks()
        self._indicator_cache = []
        self._chunk_sizes = []

        set_backend(self.backend, self.gpu_id)

        for ci in range(n_ch):
            raw, chunk_p = self._load_chunk_raw(ci)
            raw_xp = {k: to_xp(v) for k, v in raw.items()}
            field_positions = None
            if self.indicator_field_indices is not None:
                field_positions = {
                    int(field): pos for pos, field in enumerate(self.indicator_field_indices)
                }
                indicators = derive_indicators_subset(raw_xp, self.indicator_field_indices)
            else:
                indicators = derive_indicators(raw_xp)
            from .backend import to_numpy
            ind_np = to_numpy(indicators).astype(np.float32, copy=False)
            self._inject_fundamental_indicators(ind_np, raw, ci, chunk_p, field_positions)
            self._inject_cross_period_indicators(ind_np, ci, chunk_p, field_positions)
            if self.cache_dtype == np.float16:
                finfo = np.finfo(np.float16)
                np.clip(ind_np, finfo.min, finfo.max, out=ind_np)
            ind_np = ind_np.astype(self.cache_dtype, copy=False)
            self._indicator_cache.append(ind_np)
            self._chunk_sizes.append(chunk_p)
            del raw, raw_xp, indicators
            free_gpu()
            gc.collect()
            print(f"  [CACHE] chunk {ci+1}/{n_ch} ({chunk_p}p) "
                  f"{time.time()-t0:.1f}s", flush=True)

        elapsed = time.time() - t0
        total_p = sum(self._chunk_sizes)
        mem_mb = sum(c.nbytes for c in self._indicator_cache) / 1e6
        field_msg = ""
        if self.indicator_field_indices is not None:
            field_msg = f", {len(self.indicator_field_indices)} projected fields"
        print(f"[CACHE] {n_ch} chunks, {total_p} periods cached in {elapsed:.1f}s "
              f"({mem_mb:.0f} MB CPU RAM{field_msg})", flush=True)
        return self._chunk_sizes

    def _needs_fundamental_fields(self):
        if self._fundamental_panel is None:
            return False
        if self.indicator_field_indices is None:
            return True
        requested = set(int(x) for x in self.indicator_field_indices)
        return any(INDICATOR_INDEX[name] in requested for name in FUNDAMENTAL_INDICATOR_NAMES)

    @staticmethod
    def _indicator_position(name, field_positions=None):
        idx = INDICATOR_INDEX.get(name)
        if idx is None:
            return None
        if field_positions is None:
            return idx
        return field_positions.get(idx)

    def _inject_fundamental_indicators(self, ind_np, raw, chunk_idx, chunk_p, field_positions=None):
        if not self._needs_fundamental_fields():
            return
        from factor_platform.services.coinmetrics_market_cap import align_market_cap_features

        start = chunk_idx * self.chunk_periods
        end = start + chunk_p
        period_dates = self.dates_flat[self.active_period_indices[start:end]]
        symbols = self.symbols[self.coin_indices]
        period_turnover = None
        if "turnover" in raw:
            period_turnover = np.nansum(raw["turnover"], axis=1, dtype=np.float64).astype(np.float32, copy=False)
        features = align_market_cap_features(
            self._fundamental_panel,
            period_dates=period_dates,
            symbols=symbols,
            period_turnover=period_turnover,
            lag_days=1,
        )
        for name, values in features.items():
            idx = self._indicator_position(name, field_positions)
            if idx is None:
                continue
            ind_np[:, idx, :, :] = values[:, None, :]

    def _needs_cross_period_fields(self):
        if self.indicator_field_indices is None:
            return True
        requested = set(int(x) for x in self.indicator_field_indices)
        return any(INDICATOR_INDEX[name] in requested for name in CROSS_PERIOD_INDICATOR_NAMES)

    def _inject_cross_period_indicators(self, ind_np, chunk_idx, chunk_p, field_positions=None):
        if not self._needs_cross_period_fields():
            return
        features = self._get_cross_period_features()
        start = chunk_idx * self.chunk_periods
        end = start + chunk_p
        for name, values in features.items():
            idx = self._indicator_position(name, field_positions)
            if idx is not None:
                ind_np[:, idx, :, :] = values[start:end, None, :]

    def _get_cross_period_features(self):
        if self._cross_period_features is not None:
            return self._cross_period_features

        shape = (self.n_active_periods, self.n_tradable_coins)
        features = {
            name: np.full(shape, np.nan, dtype=np.float32)
            for name in CROSS_PERIOD_INDICATOR_NAMES
        }
        ret = self.get_period_returns().astype(np.float64, copy=False)
        volume = self.get_period_volume().astype(np.float64, copy=False)
        turnover = self.get_period_turnover().astype(np.float64, copy=False)

        funding = self._get_period_end_field("funding").astype(np.float64, copy=False)
        funding_mean20, funding_std20, _ = _rolling_mean_std_2d(funding, 20, min_periods=8)
        features["funding_z20"] = ((funding - funding_mean20) / (funding_std20 + _EPS)).astype(np.float32)
        funding_mean3, _, _ = _rolling_mean_std_2d(funding, 3, min_periods=2)
        funding_mean30, funding_std30, _ = _rolling_mean_std_2d(funding, 30, min_periods=10)
        features["funding_ma_diff_3_30"] = (funding_mean3 - funding_mean30).astype(np.float32)
        features["funding_vol_30d"] = funding_std30.astype(np.float32)
        funding_valid = np.isfinite(funding)
        pos_count, total_count = _rolling_sums_2d(np.where(funding_valid, funding > 0, np.nan), 20)
        neg_count, _ = _rolling_sums_2d(np.where(funding_valid, funding < 0, np.nan), 20)
        current_pos = funding > 0
        current_neg = funding < 0
        persistence = np.where(
            current_pos,
            pos_count / (total_count + _EPS),
            np.where(current_neg, -neg_count / (total_count + _EPS), np.nan),
        )
        features["funding_persistence_20d"] = persistence.astype(np.float32)

        oi = self._get_period_end_field("open_interest").astype(np.float64, copy=False)
        oi_log = np.log(np.where(oi > 0.0, oi, np.nan))
        features["oi_z20"] = _rolling_zscore_2d(oi_log, 20, min_periods=8).astype(np.float32)
        oi_lag5 = _shift_rows(oi, 5)
        oi_change_5d = np.where((oi > 0.0) & (oi_lag5 > 0.0), oi / (oi_lag5 + _EPS) - 1.0, np.nan)
        oi_change_5d = np.clip(oi_change_5d, -0.95, 10.0)
        features["oi_change_5d"] = oi_change_5d.astype(np.float32)
        oi_intraday = self._get_period_end_field("open_interest") / (
            self._load_period_start_field("open_interest") + np.float32(_EPS)
        ) - np.float32(1.0)
        features["oi_change_decay_20d"] = _ema_causal_2d(np.clip(oi_intraday, -0.95, 10.0), 20).astype(np.float32)
        features["funding_oi_joint_5d"] = (np.sign(funding) * oi_change_5d).astype(np.float32)

        btc_idx = self._btc_local_index()
        if btc_idx is not None:
            btc_ret = ret[:, btc_idx]
            beta60, _ = _rolling_beta_corr_2d(ret, btc_ret, 60, min_periods=20)
            _, corr20 = _rolling_beta_corr_2d(ret, btc_ret, 20, min_periods=8)
            residual = ret - beta60 * btc_ret[:, None]
            resid_sum20, resid_count20 = _rolling_sums_2d(residual, 20)
            _, idio20, _ = _rolling_mean_std_2d(residual, 20, min_periods=8)
            ret_sum5, ret_count5 = _rolling_sums_2d(ret, 5)
            btc_sum5, btc_count5 = _rolling_sums_2d(btc_ret[:, None], 5)
            features["beta_btc_60d"] = beta60.astype(np.float32)
            features["corr_btc_20d"] = corr20.astype(np.float32)
            features["resid_return_btc_20d"] = np.where(resid_count20 >= 8, resid_sum20, np.nan).astype(np.float32)
            features["idio_vol_btc_20d"] = idio20.astype(np.float32)
            features["relative_strength_btc_5d"] = np.where(
                (ret_count5 >= 3) & (btc_count5 >= 3),
                ret_sum5 - btc_sum5,
                np.nan,
            ).astype(np.float32)

        amihud = np.abs(ret) / (np.where(turnover > 0.0, turnover, np.nan) + _EPS) * 1e6
        features["amihud_illiq"] = amihud.astype(np.float32)
        turnover_lag1 = _shift_rows(turnover, 1)
        turnover_lookback30, _, _ = _rolling_mean_std_2d(turnover_lag1, 30, min_periods=1)
        log_dollar_volume = np.log1p(np.where(turnover_lookback30 > 0.0, turnover_lookback30, np.nan))
        features["dollar_volume_rank"] = _cross_section_rank_2d(
            log_dollar_volume,
            mask=self.get_period_tradable_mask(),
        )
        log_volume = np.log1p(np.where(volume >= 0.0, volume, np.nan))
        features["volume_zscore_60d"] = _rolling_zscore_2d(log_volume, 60, min_periods=20).astype(np.float32)
        _, rv5, _ = _rolling_mean_std_2d(ret, 5, min_periods=3)
        _, rv20, _ = _rolling_mean_std_2d(ret, 20, min_periods=8)
        _, vol_of_vol20, _ = _rolling_mean_std_2d(rv5, 20, min_periods=8)
        features["rv_5d"] = rv5.astype(np.float32)
        features["rv_20d"] = rv20.astype(np.float32)
        features["rv_ratio_5_20"] = (rv5 / (rv20 + _EPS)).astype(np.float32)
        features["vol_of_vol_20d"] = vol_of_vol20.astype(np.float32)

        tradable_mask = self.get_period_tradable_mask()
        style_controls = (
            "dollar_volume_rank",
            "amihud_illiq",
            "beta_btc_60d",
            "corr_btc_20d",
            "rv_20d",
            "rv_ratio_5_20",
        )

        def residualize_feature(source_name, raw_values):
            residual_name = f"{source_name}_resid_style"
            if residual_name not in features:
                return
            panels = [
                features[control]
                for control in style_controls
                if (
                    control != source_name
                    and control in features
                    and np.isfinite(features[control]).any()
                )
            ]
            if not panels:
                return
            style_panel = np.stack(panels, axis=2)
            features[residual_name] = _cross_sectional_style_residualize(
                raw_values,
                style_panel,
                tradable_mask,
            )

        residualize_feature("turnover", turnover)
        for source_name in ("rv_20d", "rv_ratio_5_20", "funding_z20", "oi_z20"):
            if source_name in features:
                residualize_feature(source_name, features[source_name])

        self._cross_period_features = features
        return features

    def _btc_local_index(self):
        matches = np.flatnonzero(self.symbols == "BTCUSDT")
        if matches.size == 0:
            return None
        local = np.flatnonzero(self.coin_indices == int(matches[0]))
        return int(local[0]) if local.size else None

    def _load_period_start_field(self, field):
        key = f"{field}:start"
        if key in self._period_end_fields:
            return self._period_end_fields[key]
        out = np.full((self.n_active_periods, self.n_tradable_coins), np.nan, dtype=np.float32)
        with h5py.File(self.h5_path, 'r') as f:
            if field not in f:
                self._period_end_fields[key] = out
                return out
            mpp = self.minutes_per_period
            rows = self.active_period_indices.astype(np.int64) * mpp
            ds = f[field]
            for start in range(0, self.n_active_periods, self.chunk_periods):
                end = min(start + self.chunk_periods, self.n_active_periods)
                r = rows[start:end]
                row_min, row_max = int(r.min()), int(r.max())
                block = ds[row_min:row_max + 1, :][:, self.coin_indices]
                out[start:end] = block[(r - row_min).astype(np.intp)].astype(np.float32, copy=False)
        self._period_end_fields[key] = out
        return out

    def get_cached_chunk(self, chunk_idx, perf_stats=None):
        """Return cached indicators on current backend.

        Returns:
            xp.ndarray (chunk_P, N_IND, M, S) float32
        """
        chunk = self._indicator_cache[chunk_idx]
        t0 = time.perf_counter() if perf_stats is not None else None
        if get_backend() == 'gpu':
            # Keep the PCIe transfer at cache precision, then widen on GPU.
            chunk_xp = to_xp(chunk)
            if chunk.dtype != np.float32:
                cast_t0 = time.perf_counter() if perf_stats is not None else None
                chunk_xp = chunk_xp.astype(xp.float32, copy=False)
                if perf_stats is not None:
                    perf_stats['chunk_cast_s'] = perf_stats.get('chunk_cast_s', 0.0) + (
                        time.perf_counter() - cast_t0)
        else:
            chunk_xp = to_xp(chunk.astype(np.float32, copy=False))
        if perf_stats is not None:
            perf_stats['chunk_upload_s'] = perf_stats.get('chunk_upload_s', 0.0) + (time.perf_counter() - t0)
            perf_stats['chunk_uploads'] = perf_stats.get('chunk_uploads', 0) + 1
            perf_stats['chunk_upload_bytes'] = perf_stats.get('chunk_upload_bytes', 0) + int(chunk.nbytes)
        return chunk_xp

    def get_cached_chunk_fields(self, chunk_idx, field_indices, perf_stats=None):
        """Return a cached indicator chunk projected to the requested fields.

        Evolution evaluates each batch with only a small subset of A/B/mask
        fields. Projecting on CPU before H2D transfer preserves formula
        semantics after local index remapping while avoiding full-library
        uploads when the indicator set grows.
        """
        fields = np.asarray(field_indices, dtype=np.intp)
        if fields.ndim != 1 or fields.size == 0:
            return self.get_cached_chunk(chunk_idx, perf_stats=perf_stats)
        fields = np.unique(fields)

        chunk = self._indicator_cache[chunk_idx]
        if self.indicator_field_indices is not None:
            pos = np.searchsorted(self.indicator_field_indices, fields)
            ok = np.zeros(fields.shape, dtype=bool)
            in_bounds = pos < len(self.indicator_field_indices)
            ok[in_bounds] = self.indicator_field_indices[pos[in_bounds]] == fields[in_bounds]
            if not np.all(ok):
                missing = fields[~ok].tolist()
                raise IndexError(f"requested fields not in projected indicator cache: {missing}")
            fields = pos.astype(np.intp, copy=False)
        if fields.size >= chunk.shape[1]:
            return self.get_cached_chunk(chunk_idx, perf_stats=perf_stats)

        t0 = time.perf_counter() if perf_stats is not None else None
        projected = np.take(chunk, fields, axis=1)
        if get_backend() == 'gpu':
            projected_xp = to_xp(projected)
            if projected.dtype != np.float32:
                cast_t0 = time.perf_counter() if perf_stats is not None else None
                projected_xp = projected_xp.astype(xp.float32, copy=False)
                if perf_stats is not None:
                    perf_stats['chunk_cast_s'] = perf_stats.get('chunk_cast_s', 0.0) + (
                        time.perf_counter() - cast_t0)
        else:
            projected_xp = to_xp(projected.astype(np.float32, copy=False))

        if perf_stats is not None:
            perf_stats['chunk_upload_s'] = perf_stats.get('chunk_upload_s', 0.0) + (time.perf_counter() - t0)
            perf_stats['chunk_uploads'] = perf_stats.get('chunk_uploads', 0) + 1
            perf_stats['chunk_upload_bytes'] = perf_stats.get('chunk_upload_bytes', 0) + int(projected.nbytes)
            perf_stats['chunk_field_uploads'] = perf_stats.get('chunk_field_uploads', 0) + int(fields.size)
            perf_stats['chunk_field_full_uploads'] = perf_stats.get('chunk_field_full_uploads', 0) + int(chunk.shape[1])
        return projected_xp

    def get_chunk_size(self, chunk_idx):
        return self._chunk_sizes[chunk_idx]

    def print_info(self):
        print(f"[DATA] {self.h5_path}", flush=True)
        if self.tradable_mask_overlay_path:
            print(f"[DATA] TradableMaskOverlay: {self.tradable_mask_overlay_path}", flush=True)
        if self.fundamental_panel_path:
            print(f"[DATA] FundamentalPanel: {self.fundamental_panel_path}", flush=True)
        if self.indicator_field_indices is not None:
            print(
                f"[DATA] IndicatorFieldCache: {len(self.indicator_field_indices)}/{N_INDICATORS} fields",
                flush=True,
            )
        print(f"[DATA] Total: {self.n_total_minutes} minutes, {self.n_total_periods} periods "
              f"({self.minutes_per_period}min/period)", flush=True)
        if self.n_is < self.n_active_periods:
            print(
                f"[DATA] Active: {self.n_active_periods} periods; "
                f"IS train: {self.n_is} periods through {self.train_end_date}",
                flush=True,
            )
        else:
            print(f"[DATA] Active: {self.n_active_periods} periods (full active sample search)", flush=True)
        print(f"[DATA] Coins: {self.n_tradable_coins}/{self.n_coins} tradable", flush=True)
        print(f"[DATA] Chunks: {self.n_chunks()} x {self.chunk_periods} periods", flush=True)
