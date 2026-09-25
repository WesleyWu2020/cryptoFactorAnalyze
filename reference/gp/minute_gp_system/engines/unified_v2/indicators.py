"""Derive GP indicators from OHLCV, crypto-native fields, and intraday shape.
Supports numpy(CPU) and cupy(GPU)."""
from .backend import xp
from .config import (
    CROSS_PERIOD_INDICATOR_NAMES,
    EXTRA_RAW_FIELDS,
    FUNDAMENTAL_INDICATOR_NAMES,
    INDICATOR_INDEX,
    INDICATOR_NAMES,
    ORDER_FLOW_RAW_FIELDS,
    XBINANCE_ORDER_FLOW_RAW_FIELDS,
)


N_IND = len(INDICATOR_NAMES)
PATH_EFFICIENCY_SLOT = INDICATOR_INDEX["path_efficiency"]


def _assign_slot(out, name, value):
    idx = INDICATOR_INDEX.get(name)
    if idx is not None:
        out[:, idx] = value


def _assign_day_slot(out, name, value, shape):
    idx = INDICATOR_INDEX.get(name)
    if idx is not None:
        out[:, idx] = xp.broadcast_to(value, shape)


def derive_indicators_subset(raw: dict, field_indices):
    """Compute only requested indicator slots, preserving global field order."""
    fields = [int(idx) for idx in field_indices]
    P, M, S = raw['open'].shape
    out = xp.empty((P, len(fields), M, S), dtype=xp.float32)
    out[...] = xp.float32(xp.nan)
    pos = {field: i for i, field in enumerate(fields)}
    eps = xp.float32(1e-8)

    def has(name):
        idx = INDICATOR_INDEX.get(name)
        return idx in pos if idx is not None else False

    def assign(name, value):
        idx = INDICATOR_INDEX.get(name)
        if idx in pos:
            out[:, pos[idx]] = value

    def assign_day(name, value):
        idx = INDICATOR_INDEX.get(name)
        if idx in pos:
            out[:, pos[idx]] = xp.broadcast_to(value, (P, M, S))

    op = raw['open']; hi = raw['high']; lo = raw['low']
    cl = raw['close']; vol = raw['volume']

    for name, value in (
        ('open', op), ('high', hi), ('low', lo), ('close', cl), ('volume', vol),
    ):
        assign(name, value)

    ret = xp.empty_like(cl)
    ret[:, 0, :] = 0.0
    ret[:, 1:, :] = cl[:, 1:, :] / (cl[:, :-1, :] + eps) - 1.0
    ret = xp.clip(ret, -0.5, 5.0)
    assign('returns', ret)
    assign('return_abs', xp.abs(ret))
    assign('log_volume', xp.log(vol + 1.0))
    assign('spread', hi - lo)
    assign('mid_price', (hi + lo) * 0.5)

    typical = (hi + lo + cl) / 3.0
    assign('typical_price', typical)
    assign('vwap_proxy', typical)
    assign('body', xp.abs(cl - op))
    assign('upper_shadow', hi - xp.maximum(op, cl))
    assign('lower_shadow', xp.minimum(op, cl) - lo)
    assign('price_range_pct', (hi - lo) / (op + eps))
    assign('close_position', (cl - lo) / (hi - lo + eps))
    assign('money_flow', typical * vol)

    if any(has(name) for name in ('volume_intensity', 'up_volume_pressure', 'down_volume_pressure')):
        vol_valid = xp.isfinite(vol)
        vol_count = xp.sum(vol_valid, axis=1, keepdims=True)
        vol_mean = xp.where(
            vol_count > 0,
            xp.sum(xp.where(vol_valid, vol, xp.float32(0.0)), axis=1, keepdims=True) / xp.maximum(vol_count, 1),
            xp.float32(xp.nan),
        )
        volume_intensity = vol / (vol_mean + eps)
        assign('volume_intensity', volume_intensity)
        assign('up_volume_pressure', xp.maximum(ret, 0.0) * volume_intensity)
        assign('down_volume_pressure', xp.maximum(-ret, 0.0) * volume_intensity)

    need_cum = has('cum_return') or has('market_cum_return')
    cum_return = xp.cumprod(1.0 + xp.nan_to_num(ret, nan=0.0), axis=1) if need_cum else None
    if cum_return is not None:
        assign('cum_return', cum_return)

    for i, field in enumerate(EXTRA_RAW_FIELDS):
        name = INDICATOR_NAMES[20 + i]
        if has(name) and field in raw:
            assign(name, raw[field])

    day_ret = cl[:, -1:, :] / (op[:, :1, :] + eps) - 1.0
    if has('path_efficiency'):
        path_len = xp.nansum(xp.abs(ret), axis=1, keepdims=True)
        assign_day('path_efficiency', xp.abs(day_ret) / (path_len + eps))

    need_extrema = any(has(name) for name in ('high_time_frac', 'low_time_frac', 'high_before_low', 'intraday_reversal'))
    if need_extrema:
        hi_valid = xp.isfinite(hi)
        lo_valid = xp.isfinite(lo)
        hi_any = xp.any(hi_valid, axis=1, keepdims=True)
        lo_any = xp.any(lo_valid, axis=1, keepdims=True)
        hi_fill = xp.where(hi_valid, hi, xp.float32(-xp.inf))
        lo_fill = xp.where(lo_valid, lo, xp.float32(xp.inf))
        high_idx = xp.argmax(hi_fill, axis=1).astype(xp.float32)[:, None, :]
        low_idx = xp.argmin(lo_fill, axis=1).astype(xp.float32)[:, None, :]
        denom_m = xp.float32(max(M - 1, 1))
        assign_day('high_time_frac', xp.where(hi_any, high_idx / denom_m, xp.float32(xp.nan)))
        assign_day('low_time_frac', xp.where(lo_any, low_idx / denom_m, xp.float32(xp.nan)))
        assign_day(
            'high_before_low',
            xp.where(hi_any & lo_any, xp.where(high_idx <= low_idx, 1.0, 0.0), xp.float32(xp.nan)),
        )
        if has('intraday_reversal'):
            day_high = xp.where(hi_any, xp.max(hi_fill, axis=1, keepdims=True), xp.float32(xp.nan))
            day_low = xp.where(lo_any, xp.min(lo_fill, axis=1, keepdims=True), xp.float32(xp.nan))
            day_range = day_high - day_low
            last_close = cl[:, -1:, :]
            assign_day('intraday_reversal', ((day_high - last_close) - (last_close - day_low)) / (day_range + eps))

    if any(has(name) for name in ('am_return', 'pm_return', 'am_pm_return_diff')):
        mid = max(1, min(M - 1, M // 2))
        first_open = op[:, :1, :]
        mid_close = cl[:, mid:mid + 1, :]
        last_close = cl[:, -1:, :]
        am_return = mid_close / (first_open + eps) - 1.0
        pm_return = last_close / (mid_close + eps) - 1.0
        assign_day('am_return', am_return)
        assign_day('pm_return', pm_return)
        assign_day('am_pm_return_diff', am_return - pm_return)

    if any(has(name) for name in ('late_volume_share', 'late_range_share', 'volume_burst_share')):
        late_start = min(M - 1, max(0, int(M * 0.75)))
        vol_sum = xp.nansum(vol, axis=1, keepdims=True)
        vol_any = xp.any(xp.isfinite(vol), axis=1, keepdims=True)
        if has('late_volume_share'):
            late_vol = xp.nansum(vol[:, late_start:, :], axis=1, keepdims=True)
            assign_day('late_volume_share', xp.where(vol_any & (vol_sum > eps), late_vol / (vol_sum + eps), xp.float32(xp.nan)))
        if has('late_range_share'):
            path_sum = xp.nansum(xp.abs(ret), axis=1, keepdims=True)
            late_path = xp.nansum(xp.abs(ret[:, late_start:, :]), axis=1, keepdims=True)
            assign_day('late_range_share', xp.where(path_sum > eps, late_path / (path_sum + eps), xp.float32(xp.nan)))
        if has('volume_burst_share'):
            vol_max = xp.max(xp.where(xp.isfinite(vol), vol, xp.float32(0.0)), axis=1, keepdims=True)
            assign_day('volume_burst_share', xp.where(vol_any & (vol_sum > eps), vol_max / (vol_sum + eps), xp.float32(xp.nan)))

    if has('market_cum_return') or has('market_vol_intensity'):
        valid_close = xp.isfinite(cl)
        market_count = xp.sum(valid_close, axis=2, keepdims=True)
        market_denom = xp.maximum(market_count, 1)
        if has('market_cum_return') and cum_return is not None:
            market_cum_base = xp.where(valid_close, cum_return - 1.0, xp.float32(xp.nan))
            assign_day(
                'market_cum_return',
                xp.where(
                    market_count > 0,
                    xp.sum(xp.where(valid_close, market_cum_base, xp.float32(0.0)), axis=2, keepdims=True) / market_denom,
                    xp.float32(xp.nan),
                ),
            )
        if has('market_vol_intensity'):
            market_vol_base = xp.where(valid_close, xp.abs(ret), xp.float32(xp.nan))
            assign_day(
                'market_vol_intensity',
                xp.where(
                    market_count > 0,
                    xp.sum(xp.where(valid_close, market_vol_base, xp.float32(0.0)), axis=2, keepdims=True) / market_denom,
                    xp.float32(xp.nan),
                ),
            )

    oi_directional_names = (
        'new_long_pressure',
        'new_short_pressure',
        'short_cover_pressure',
        'long_liquidation_pressure',
    )
    if "open_interest" in raw and (
        has('oi_change_intraday')
        or has('oi_price_alignment')
        or any(has(name) for name in oi_directional_names)
    ):
        oi = raw["open_interest"]
        oi_start = oi[:, :1, :]
        oi_end = oi[:, -1:, :]
        oi_change = oi_end / (oi_start + eps) - 1.0
        valid_oi = xp.isfinite(oi_start) & xp.isfinite(oi_end) & (oi_start > eps) & (oi_end >= 0.0)
        directional_oi_change = xp.where(
            valid_oi,
            xp.clip(oi_change, -0.95, 10.0),
            xp.float32(xp.nan),
        )
        assign_day('oi_change_intraday', oi_change)
        assign_day('oi_price_alignment', oi_change * day_ret)
        up_ret = xp.maximum(xp.clip(day_ret, -0.5, 5.0), 0.0)
        down_ret = xp.maximum(-xp.clip(day_ret, -0.5, 5.0), 0.0)
        oi_up = xp.maximum(directional_oi_change, 0.0)
        oi_down = xp.maximum(-directional_oi_change, 0.0)
        assign_day('new_long_pressure', up_ret * oi_up)
        assign_day('new_short_pressure', down_ret * oi_up)
        assign_day('short_cover_pressure', up_ret * oi_down)
        assign_day('long_liquidation_pressure', down_ret * oi_down)

    if "premium_close" in raw and has('premium_change_intraday'):
        premium = raw["premium_close"]
        assign_day('premium_change_intraday', premium[:, -1:, :] - premium[:, :1, :])

    if "mark_close" in raw and "index_close" in raw and (has('mark_index_spread') or has('mark_index_spread_change')):
        index_close = raw["index_close"]
        mark_spread = (raw["mark_close"] - index_close) / (xp.abs(index_close) + eps)
        assign('mark_index_spread', mark_spread)
        assign_day('mark_index_spread_change', mark_spread[:, -1:, :] - mark_spread[:, :1, :])

    if "funding" in raw and has('funding_abs'):
        assign('funding_abs', xp.abs(raw["funding"]))

    for field in ORDER_FLOW_RAW_FIELDS:
        if field in raw:
            assign(field, raw[field])

    for field in XBINANCE_ORDER_FLOW_RAW_FIELDS:
        if field in raw:
            assign(field, raw[field])

    if any(has(name) for name in ('taker_buy_ratio', 'taker_quote_ratio', 'taker_pressure', 'avg_trade_size')):
        taker_vol = raw.get("taker_buy_volume")
        taker_quote = raw.get("taker_buy_quote_volume")
        trade_count = raw.get("trade_count")
        if taker_vol is not None:
            taker_ratio = taker_vol / (vol + eps)
            assign('taker_buy_ratio', xp.clip(taker_ratio, 0.0, 1.0))
            assign('taker_pressure', xp.clip(2.0 * taker_ratio - 1.0, -1.0, 1.0))
        if taker_quote is not None and "turnover" in raw:
            assign('taker_quote_ratio', xp.clip(taker_quote / (raw["turnover"] + eps), 0.0, 1.0))
        if trade_count is not None and "turnover" in raw:
            assign('avg_trade_size', raw["turnover"] / (trade_count + eps))

    if any(has(name) for name in ('xbinance_taker_quote_ratio', 'xbinance_taker_pressure', 'xbinance_avg_trade_size')):
        xb_quote = raw.get("xbinance_quote_volume")
        xb_taker_quote = raw.get("xbinance_taker_buy_quote_volume")
        xb_trade_count = raw.get("xbinance_trade_count")
        if xb_quote is not None and xb_taker_quote is not None:
            valid_xb = (xb_quote > eps) & xp.isfinite(xb_quote) & xp.isfinite(xb_taker_quote)
            xb_ratio_raw = xp.clip(xb_taker_quote / (xb_quote + eps), 0.0, 1.0)
            xb_ratio = xp.where(valid_xb, xb_ratio_raw, xp.float32(0.5))
            assign('xbinance_taker_quote_ratio', xb_ratio)
            assign('xbinance_taker_pressure', xp.clip(2.0 * xb_ratio - 1.0, -1.0, 1.0))
        if xb_quote is not None and xb_trade_count is not None:
            assign('xbinance_avg_trade_size', xb_quote / (xb_trade_count + eps))

    return xp.where(xp.isfinite(out), out, xp.float32(xp.nan))


def derive_indicators(raw: dict):
    """Compute N_IND indicators.
    Slots 0..19: derived from OHLCV.
    Slots 20..25: passthrough from EXTRA_RAW_FIELDS
        (funding, open_interest, premium_close, mark_close, index_close, turnover).
    Slots 26..: appended closed-period path/regime/positioning extensions.

    Args:
        raw: dict mapping field name -> xp.ndarray of shape (P, M, S), float32.
             Must contain OHLCV. EXTRA_RAW_FIELDS optional per field — missing
             ones become all-NaN slot.
    Returns:
        xp.ndarray of shape (P, N_IND, M, S), float32.
    """
    P, M, S = raw['open'].shape
    out = xp.empty((P, N_IND, M, S), dtype=xp.float32)
    eps = xp.float32(1e-8)

    op = raw['open']; hi = raw['high']; lo = raw['low']
    cl = raw['close']; vol = raw['volume']

    out[:, 0] = op
    out[:, 1] = hi
    out[:, 2] = lo
    out[:, 3] = cl
    out[:, 4] = vol

    ret = xp.empty_like(cl)
    ret[:, 0, :] = 0.0
    ret[:, 1:, :] = cl[:, 1:, :] / (cl[:, :-1, :] + eps) - 1.0
    ret = xp.clip(ret, -0.5, 5.0)
    out[:, 5] = ret

    out[:, 6] = xp.log(vol + 1.0)
    out[:, 7] = hi - lo
    out[:, 8] = (hi + lo) * 0.5
    typical = (hi + lo + cl) / 3.0
    out[:, 9] = typical
    out[:, 10] = xp.abs(cl - op)
    out[:, 11] = hi - xp.maximum(op, cl)
    out[:, 12] = xp.minimum(op, cl) - lo
    out[:, 13] = (hi - lo) / (op + eps)
    out[:, 14] = xp.abs(ret)
    out[:, 15] = typical
    vol_valid = xp.isfinite(vol)
    vol_count = xp.sum(vol_valid, axis=1, keepdims=True)
    vol_mean = xp.where(
        vol_count > 0,
        xp.sum(xp.where(vol_valid, vol, xp.float32(0.0)), axis=1, keepdims=True) / xp.maximum(vol_count, 1),
        xp.float32(xp.nan),
    )
    out[:, 16] = vol / (vol_mean + eps)
    _assign_slot(out, "up_volume_pressure", xp.maximum(ret, 0.0) * out[:, 16])
    _assign_slot(out, "down_volume_pressure", xp.maximum(-ret, 0.0) * out[:, 16])
    out[:, 17] = xp.cumprod(1.0 + xp.nan_to_num(ret, nan=0.0), axis=1)
    out[:, 18] = (cl - lo) / (hi - lo + eps)
    out[:, 19] = typical * vol

    # Crypto-native passthroughs (slots 20..25). Missing field → NaN slot.
    for i, field in enumerate(EXTRA_RAW_FIELDS):
        slot = 20 + i
        if field in raw:
            out[:, slot] = raw[field]
        else:
            out[:, slot] = xp.float32(xp.nan)

    out_shape = (P, M, S)
    day_ret = cl[:, -1:, :] / (op[:, :1, :] + eps) - 1.0
    path_len = xp.nansum(xp.abs(ret), axis=1, keepdims=True)
    path_efficiency = xp.abs(day_ret) / (path_len + eps)
    out[:, PATH_EFFICIENCY_SLOT] = xp.broadcast_to(path_efficiency, out_shape)

    hi_valid = xp.isfinite(hi)
    lo_valid = xp.isfinite(lo)
    hi_any = xp.any(hi_valid, axis=1, keepdims=True)
    lo_any = xp.any(lo_valid, axis=1, keepdims=True)
    hi_fill = xp.where(hi_valid, hi, xp.float32(-xp.inf))
    lo_fill = xp.where(lo_valid, lo, xp.float32(xp.inf))
    high_idx = xp.argmax(hi_fill, axis=1).astype(xp.float32)[:, None, :]
    low_idx = xp.argmin(lo_fill, axis=1).astype(xp.float32)[:, None, :]
    denom_m = xp.float32(max(M - 1, 1))
    high_time = xp.where(hi_any, high_idx / denom_m, xp.float32(xp.nan))
    low_time = xp.where(lo_any, low_idx / denom_m, xp.float32(xp.nan))
    _assign_day_slot(out, "high_time_frac", high_time, out_shape)
    _assign_day_slot(out, "low_time_frac", low_time, out_shape)
    _assign_day_slot(
        out,
        "high_before_low",
        xp.where(hi_any & lo_any, xp.where(high_idx <= low_idx, 1.0, 0.0), xp.float32(xp.nan)),
        out_shape,
    )

    mid = max(1, min(M - 1, M // 2))
    first_open = op[:, :1, :]
    mid_close = cl[:, mid:mid + 1, :]
    last_close = cl[:, -1:, :]
    am_return = mid_close / (first_open + eps) - 1.0
    pm_return = last_close / (mid_close + eps) - 1.0
    _assign_day_slot(out, "am_return", am_return, out_shape)
    _assign_day_slot(out, "pm_return", pm_return, out_shape)
    _assign_day_slot(out, "am_pm_return_diff", am_return - pm_return, out_shape)

    late_start = min(M - 1, max(0, int(M * 0.75)))
    vol_sum = xp.nansum(vol, axis=1, keepdims=True)
    late_vol = xp.nansum(vol[:, late_start:, :], axis=1, keepdims=True)
    vol_any = xp.any(xp.isfinite(vol), axis=1, keepdims=True)
    late_volume_share = xp.where(vol_any & (vol_sum > eps), late_vol / (vol_sum + eps), xp.float32(xp.nan))
    path_sum = xp.nansum(xp.abs(ret), axis=1, keepdims=True)
    late_path = xp.nansum(xp.abs(ret[:, late_start:, :]), axis=1, keepdims=True)
    late_range_share = xp.where(path_sum > eps, late_path / (path_sum + eps), xp.float32(xp.nan))
    vol_max = xp.max(xp.where(xp.isfinite(vol), vol, xp.float32(0.0)), axis=1, keepdims=True)
    volume_burst_share = xp.where(vol_any & (vol_sum > eps), vol_max / (vol_sum + eps), xp.float32(xp.nan))
    _assign_day_slot(out, "late_volume_share", late_volume_share, out_shape)
    _assign_day_slot(out, "late_range_share", late_range_share, out_shape)
    _assign_day_slot(out, "volume_burst_share", volume_burst_share, out_shape)

    day_high = xp.max(hi_fill, axis=1, keepdims=True)
    day_low = xp.min(lo_fill, axis=1, keepdims=True)
    day_high = xp.where(hi_any, day_high, xp.float32(xp.nan))
    day_low = xp.where(lo_any, day_low, xp.float32(xp.nan))
    day_range = day_high - day_low
    intraday_reversal = ((day_high - last_close) - (last_close - day_low)) / (day_range + eps)
    _assign_day_slot(out, "intraday_reversal", intraday_reversal, out_shape)

    valid_close = xp.isfinite(cl)
    market_cum_base = xp.where(valid_close, out[:, INDICATOR_INDEX["cum_return"]] - 1.0, xp.float32(xp.nan))
    market_vol_base = xp.where(valid_close, xp.abs(ret), xp.float32(xp.nan))
    market_count = xp.sum(valid_close, axis=2, keepdims=True)
    market_denom = xp.maximum(market_count, 1)
    market_cum_return = xp.where(
        market_count > 0,
        xp.sum(xp.where(valid_close, market_cum_base, xp.float32(0.0)), axis=2, keepdims=True) / market_denom,
        xp.float32(xp.nan),
    )
    market_vol_intensity = xp.where(
        market_count > 0,
        xp.sum(xp.where(valid_close, market_vol_base, xp.float32(0.0)), axis=2, keepdims=True) / market_denom,
        xp.float32(xp.nan),
    )
    _assign_day_slot(out, "market_cum_return", market_cum_return, out_shape)
    _assign_day_slot(out, "market_vol_intensity", market_vol_intensity, out_shape)

    if "open_interest" in raw:
        oi = raw["open_interest"]
        oi_start = oi[:, :1, :]
        oi_end = oi[:, -1:, :]
        oi_change = oi_end / (oi_start + eps) - 1.0
        valid_oi = xp.isfinite(oi_start) & xp.isfinite(oi_end) & (oi_start > eps) & (oi_end >= 0.0)
        directional_oi_change = xp.where(
            valid_oi,
            xp.clip(oi_change, -0.95, 10.0),
            xp.float32(xp.nan),
        )
        _assign_day_slot(out, "oi_change_intraday", oi_change, out_shape)
        _assign_day_slot(out, "oi_price_alignment", oi_change * day_ret, out_shape)
        up_ret = xp.maximum(xp.clip(day_ret, -0.5, 5.0), 0.0)
        down_ret = xp.maximum(-xp.clip(day_ret, -0.5, 5.0), 0.0)
        oi_up = xp.maximum(directional_oi_change, 0.0)
        oi_down = xp.maximum(-directional_oi_change, 0.0)
        _assign_day_slot(out, "new_long_pressure", up_ret * oi_up, out_shape)
        _assign_day_slot(out, "new_short_pressure", down_ret * oi_up, out_shape)
        _assign_day_slot(out, "short_cover_pressure", up_ret * oi_down, out_shape)
        _assign_day_slot(out, "long_liquidation_pressure", down_ret * oi_down, out_shape)
    else:
        _assign_slot(out, "oi_change_intraday", xp.float32(xp.nan))
        _assign_slot(out, "oi_price_alignment", xp.float32(xp.nan))
        for name in (
            "new_long_pressure",
            "new_short_pressure",
            "short_cover_pressure",
            "long_liquidation_pressure",
        ):
            _assign_slot(out, name, xp.float32(xp.nan))

    if "premium_close" in raw:
        premium = raw["premium_close"]
        _assign_day_slot(out, "premium_change_intraday", premium[:, -1:, :] - premium[:, :1, :], out_shape)
    else:
        _assign_slot(out, "premium_change_intraday", xp.float32(xp.nan))

    if "mark_close" in raw and "index_close" in raw:
        index_close = raw["index_close"]
        mark_spread = (raw["mark_close"] - index_close) / (xp.abs(index_close) + eps)
        _assign_slot(out, "mark_index_spread", mark_spread)
        _assign_day_slot(out, "mark_index_spread_change", mark_spread[:, -1:, :] - mark_spread[:, :1, :], out_shape)
    else:
        _assign_slot(out, "mark_index_spread", xp.float32(xp.nan))
        _assign_slot(out, "mark_index_spread_change", xp.float32(xp.nan))

    if "funding" in raw:
        _assign_slot(out, "funding_abs", xp.abs(raw["funding"]))
    else:
        _assign_slot(out, "funding_abs", xp.float32(xp.nan))

    for field in ORDER_FLOW_RAW_FIELDS:
        if field in raw:
            _assign_slot(out, field, raw[field])
        else:
            _assign_slot(out, field, xp.float32(xp.nan))

    for field in XBINANCE_ORDER_FLOW_RAW_FIELDS:
        if field in raw:
            _assign_slot(out, field, raw[field])
        else:
            _assign_slot(out, field, xp.float32(xp.nan))

    if "taker_buy_volume" in raw:
        taker_ratio = raw["taker_buy_volume"] / (vol + eps)
        _assign_slot(out, "taker_buy_ratio", xp.clip(taker_ratio, 0.0, 1.0))
        _assign_slot(out, "taker_pressure", xp.clip(2.0 * taker_ratio - 1.0, -1.0, 1.0))
    else:
        _assign_slot(out, "taker_buy_ratio", xp.float32(xp.nan))
        _assign_slot(out, "taker_pressure", xp.float32(xp.nan))

    if "taker_buy_quote_volume" in raw and "turnover" in raw:
        _assign_slot(out, "taker_quote_ratio", xp.clip(raw["taker_buy_quote_volume"] / (raw["turnover"] + eps), 0.0, 1.0))
    else:
        _assign_slot(out, "taker_quote_ratio", xp.float32(xp.nan))

    if "trade_count" in raw and "turnover" in raw:
        _assign_slot(out, "avg_trade_size", raw["turnover"] / (raw["trade_count"] + eps))
    else:
        _assign_slot(out, "avg_trade_size", xp.float32(xp.nan))

    if "xbinance_quote_volume" in raw and "xbinance_taker_buy_quote_volume" in raw:
        xb_quote = raw["xbinance_quote_volume"]
        xb_taker_quote = raw["xbinance_taker_buy_quote_volume"]
        valid_xb = (xb_quote > eps) & xp.isfinite(xb_quote) & xp.isfinite(xb_taker_quote)
        xb_ratio_raw = xp.clip(xb_taker_quote / (xb_quote + eps), 0.0, 1.0)
        xb_ratio = xp.where(valid_xb, xb_ratio_raw, xp.float32(0.5))
        _assign_slot(out, "xbinance_taker_quote_ratio", xb_ratio)
        _assign_slot(out, "xbinance_taker_pressure", xp.clip(2.0 * xb_ratio - 1.0, -1.0, 1.0))
    else:
        _assign_slot(out, "xbinance_taker_quote_ratio", xp.float32(0.5))
        _assign_slot(out, "xbinance_taker_pressure", xp.float32(0.0))

    if "xbinance_quote_volume" in raw and "xbinance_trade_count" in raw:
        _assign_slot(out, "xbinance_avg_trade_size", raw["xbinance_quote_volume"] / (raw["xbinance_trade_count"] + eps))
    else:
        _assign_slot(out, "xbinance_avg_trade_size", xp.float32(xp.nan))

    for name in (*FUNDAMENTAL_INDICATOR_NAMES, *CROSS_PERIOD_INDICATOR_NAMES):
        _assign_slot(out, name, xp.float32(xp.nan))

    return xp.where(xp.isfinite(out), out, xp.float32(xp.nan))
