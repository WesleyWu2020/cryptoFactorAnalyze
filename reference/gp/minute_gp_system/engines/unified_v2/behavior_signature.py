from __future__ import annotations

import numpy as np

from .config import INDICATOR_INDEX, TOP_QUANTILE
from .backend import to_numpy
from .fitness import _build_target_memberships_single, _postprocess_single, standardize_phenotypes


BEHAVIOR_SAMPLE_PERIODS = 96
BEHAVIOR_PATH_PERIODS = 48
REGIME_FIELDS = (
    "funding",
    "open_interest",
    "premium_close",
    "price_range_pct",
    "turnover",
    "return_abs",
)
_EPS = 1e-10


def subsample_index(idx: np.ndarray, max_points: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int32)
    if idx.size <= max_points:
        return idx
    take = np.linspace(0, idx.size - 1, num=max_points, dtype=np.int32)
    return idx[np.unique(take)].astype(np.int32, copy=False)


def _unit_norm(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= _EPS:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32, copy=False)


def _standardize_1d(vec: np.ndarray) -> np.ndarray:
    out = np.zeros(vec.shape[0], dtype=np.float32)
    valid = np.isfinite(vec)
    if int(valid.sum()) < 2:
        return out
    values = vec[valid].astype(np.float32, copy=False)
    mean = float(values.mean())
    std = float(values.std())
    out[valid] = values - mean if not np.isfinite(std) or std <= _EPS else (values - mean) / std
    return out


def _corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 8:
        return 0.0
    xv = x[valid].astype(np.float32, copy=False)
    yv = y[valid].astype(np.float32, copy=False)
    xv = xv - float(xv.mean())
    yv = yv - float(yv.mean())
    denom = float(np.sqrt(np.sum(xv * xv) * np.sum(yv * yv)))
    if not np.isfinite(denom) or denom <= _EPS:
        return 0.0
    return float(np.sum(xv * yv) / denom)


def _dominant_regime(exposures: np.ndarray) -> tuple[str, float]:
    if exposures.size == 0 or not np.any(np.isfinite(exposures)):
        return "neutral", 0.0
    idx = int(np.nanargmax(np.abs(exposures)))
    strength = float(abs(exposures[idx]))
    if strength < 0.15:
        return "neutral", strength
    return REGIME_FIELDS[idx], strength


def _behavior_bucket(value: float, *, low: float, high: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < low:
        return "low"
    if value < high:
        return "mid"
    return "high"


def build_regime_matrix(loader, tradable_mask: np.ndarray) -> np.ndarray:
    regime = np.full((loader.n_active_periods, len(REGIME_FIELDS)), np.nan, dtype=np.float32)
    indicator_cache = getattr(loader, "_indicator_cache", None)
    projected_fields = getattr(loader, "indicator_field_indices", None)
    if projected_fields is not None:
        projected_fields = np.asarray(projected_fields, dtype=np.intp)
    offset = 0
    for ci in range(loader.n_chunks()):
        if indicator_cache is not None:
            chunk = np.asarray(indicator_cache[ci], dtype=np.float32)
        else:
            chunk = to_numpy(loader.get_cached_chunk(ci)).astype(np.float32, copy=False)
        cp = loader.get_chunk_size(ci)
        tradable_chunk = tradable_mask[offset:offset + cp]
        for col, name in enumerate(REGIME_FIELDS):
            field_idx = INDICATOR_INDEX[name]
            if projected_fields is not None:
                pos = int(np.searchsorted(projected_fields, field_idx))
                if pos >= len(projected_fields) or int(projected_fields[pos]) != field_idx:
                    continue
                field_idx = pos
            values = chunk[:, field_idx, -1, :]
            valid = np.isfinite(values) & tradable_chunk
            sums = np.where(valid, values, 0.0).sum(axis=1, dtype=np.float32)
            counts = valid.sum(axis=1).astype(np.float32)
            regime[offset:offset + cp, col] = np.where(counts > 0.0, sums / counts, np.nan)
        offset += cp
    for col in range(regime.shape[1]):
        series = regime[:, col]
        valid = np.isfinite(series)
        if int(valid.sum()) < 8:
            regime[:, col] = 0.0
            continue
        mean = float(series[valid].mean())
        std = float(series[valid].std())
        if not np.isfinite(std) or std <= _EPS:
            regime[:, col] = 0.0
            continue
        regime[:, col] = np.where(valid, (series - mean) / std, 0.0)
    return regime


def build_behavior_context(
    loader,
    period_returns: np.ndarray,
    tradable_mask: np.ndarray,
    *,
    max_periods: int = BEHAVIOR_SAMPLE_PERIODS,
) -> dict:
    sample_idx = subsample_index(np.arange(loader.n_active_periods, dtype=np.int32), max_periods)
    return {
        "idx": sample_idx,
        "ret_sample": period_returns[sample_idx].astype(np.float32, copy=False),
        "tradable_sample": tradable_mask[sample_idx],
        "regime_sample": build_regime_matrix(loader, tradable_mask)[sample_idx],
    }


def build_behavior_signature_single(
    factor_2d: np.ndarray,
    ret_sample: np.ndarray,
    tradable_sample: np.ndarray,
    regime_sample: np.ndarray,
    top_frac: float,
    ic_direction: float | None,
) -> tuple[np.ndarray, dict]:
    factor_pp = _postprocess_single(np.asarray(factor_2d, dtype=np.float32))
    direction = np.float32(ic_direction if ic_direction is not None and np.isfinite(ic_direction) else 1.0)
    factor_pp = factor_pp * direction

    target_long, target_short, _, min_k = _build_target_memberships_single(
        factor_pp,
        top_frac,
        tradable_slice=tradable_sample,
    )
    prev_long = np.zeros_like(target_long)
    prev_short = np.zeros_like(target_short)
    prev_long[1:] = target_long[:-1]
    prev_short[1:] = target_short[:-1]

    ret_valid = np.isfinite(ret_sample)
    held_long = prev_long & ret_valid
    held_short = prev_short & ret_valid

    prev_long_cnt = prev_long.sum(axis=1).astype(np.float32)
    prev_short_cnt = prev_short.sum(axis=1).astype(np.float32)
    long_cnt = held_long.sum(axis=1).astype(np.float32)
    short_cnt = held_short.sum(axis=1).astype(np.float32)
    effective = (
        (prev_long_cnt >= min_k)
        & (prev_short_cnt >= min_k)
        & (long_cnt >= min_k)
        & (short_cnt >= min_k)
    )

    long_sum = np.where(held_long, ret_sample, 0.0).sum(axis=1, dtype=np.float32)
    short_sum = np.where(held_short, ret_sample, 0.0).sum(axis=1, dtype=np.float32)
    long_rets = np.where(long_cnt > 0, long_sum / (long_cnt + _EPS), 0.0)
    short_rets = np.where(short_cnt > 0, -(short_sum / (short_cnt + _EPS)), 0.0)
    ls_ret = np.full(ret_sample.shape[0], np.nan, dtype=np.float32)
    if effective.any():
        ls_ret[effective] = (long_rets[effective] + short_rets[effective]) * np.float32(0.5)

    if int(effective.sum()) > 0:
        long_freq = prev_long[effective].mean(axis=0, dtype=np.float32)
        short_freq = prev_short[effective].mean(axis=0, dtype=np.float32)
    else:
        long_freq = np.zeros(ret_sample.shape[1], dtype=np.float32)
        short_freq = np.zeros(ret_sample.shape[1], dtype=np.float32)

    valid_path = np.flatnonzero(np.isfinite(ls_ret))
    if valid_path.size > BEHAVIOR_PATH_PERIODS:
        take = np.linspace(0, valid_path.size - 1, num=BEHAVIOR_PATH_PERIODS, dtype=np.int32)
        valid_path = valid_path[take]
    path_sig = (
        _unit_norm(_standardize_1d(ls_ret[valid_path]))
        if valid_path.size
        else np.zeros(BEHAVIOR_PATH_PERIODS, dtype=np.float32)
    )
    if path_sig.size < BEHAVIOR_PATH_PERIODS:
        padded = np.zeros(BEHAVIOR_PATH_PERIODS, dtype=np.float32)
        padded[:path_sig.size] = path_sig
        path_sig = padded

    exposures = np.array(
        [_corr_1d(ls_ret, regime_sample[:, col]) for col in range(regime_sample.shape[1])],
        dtype=np.float32,
    )
    dominant_regime, dominant_strength = _dominant_regime(exposures)
    long_peak = float(np.nanmax(long_freq)) if long_freq.size else 0.0
    short_peak = float(np.nanmax(short_freq)) if short_freq.size else 0.0
    concentration = float(max(long_peak, short_peak))
    path_positive_rate = float(np.mean(ls_ret[np.isfinite(ls_ret)] > 0.0)) if valid_path.size else np.nan
    behavior_key = "|".join(
        (
            f"regime={dominant_regime}",
            f"conc={_behavior_bucket(concentration, low=0.05, high=0.12)}",
            f"path={_behavior_bucket(path_positive_rate, low=0.45, high=0.60)}",
        )
    )

    signature = np.concatenate(
        (
            _unit_norm(long_freq) * np.float32(0.45),
            _unit_norm(short_freq) * np.float32(0.45),
            _unit_norm(path_sig) * np.float32(0.70),
            _unit_norm(np.nan_to_num(exposures, nan=0.0).astype(np.float32)) * np.float32(0.55),
        )
    ).astype(np.float32, copy=False)
    signature = _unit_norm(signature)

    return signature, {
        "behavior_key": behavior_key,
        "dominant_regime": dominant_regime,
        "dominant_regime_loading": dominant_strength,
        "effective_behavior_bars": int(effective.sum()),
        "holding_concentration": concentration,
        "path_positive_rate": path_positive_rate,
        "regime_exposures": {
            name: float(exposures[idx]) for idx, name in enumerate(REGIME_FIELDS)
        },
    }


def build_behavior_signatures(
    factors: np.ndarray,
    behavior_context: dict,
    *,
    top_frac: float = TOP_QUANTILE,
    ic_directions: np.ndarray | None = None,
    standardize: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    idx = behavior_context["idx"]
    ret_sample = behavior_context["ret_sample"]
    tradable_sample = behavior_context["tradable_sample"]
    regime_sample = behavior_context["regime_sample"]
    signatures = []
    feedback = []
    for i in range(factors.shape[0]):
        direction = None if ic_directions is None else float(ic_directions[i])
        sig, meta = build_behavior_signature_single(
            factors[i, idx, :],
            ret_sample,
            tradable_sample,
            regime_sample,
            top_frac,
            direction,
        )
        signatures.append(sig)
        feedback.append(meta)
    if not signatures:
        return np.zeros((0, 0), dtype=np.float32), feedback
    stacked = np.vstack(signatures)
    return (standardize_phenotypes(stacked) if standardize else stacked), feedback
