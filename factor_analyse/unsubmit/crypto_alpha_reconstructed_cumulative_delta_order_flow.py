"""crypto_alpha_reconstructed_cumulative_delta_order_flow 因子（factor_common 日频契约版）。

原始定义（已是日频语义，分钟版仅做工程分块）：
    每日 delta = volume * signed_log_path(OHLC) / ||log_path||^2
    cumulative_delta = cumsum(daily_delta)
    delta_mva50 = rolling_mean(cumulative_delta, 50d, min_periods=20)
    change = delta_mva50 - delta_mva50.shift(20d)
    factor = 截面 qualification mask：change >= 当日 0.7 分位 -> 1.0，否则 0.0

日频改写判断：
    - 输入即为日频 OHLCV，分钟版的分块/分块 numpy 管线直接整帧执行。
    - 删除 FACTOR_* 环境变量覆盖，魔法常量移入 SETTING["params"]。
    - change 用 diff(change_lag) 实现，只使用历史数据，无未来函数。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_reconstructed_cumulative_delta_order_flow",
    "author": "chatgpt",
    "level": "daily",
    "category": "order_flow",
    "description": "reconstructed cumulative delta order flow: CS mask of 20d change in 50d MVA of cumulative signed volume",
}

SETTING = {
    "data_needed": ["open", "high", "low", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 75,
    "preprocessing": "mad_rank",
    "params": {"window": 50, "min_periods": 20, "change_lag": 20, "qual_quantile": 0.7},
    "factor_direction": 1,  # 默认：高值=delta 动量上行（qualified）优先做多
}

_EPS = 1e-12


def _as_float_array(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=np.float64, copy=True)


def _clean_price_array(values: np.ndarray) -> np.ndarray:
    values[~np.isfinite(values) | (values <= 0.0)] = np.nan
    return values


def _clean_volume_array(values: np.ndarray) -> np.ndarray:
    values[~np.isfinite(values) | (values < 0.0)] = np.nan
    return values


def _signed_log_distance_sq(log_start: np.ndarray, log_end: np.ndarray) -> np.ndarray:
    signed_distance = log_end - log_start
    return signed_distance * np.abs(signed_distance)


def _log_distance_sq(log_start: np.ndarray, log_end: np.ndarray) -> np.ndarray:
    distance = log_end - log_start
    return distance * distance


def _rolling_mean_np(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    csum = np.cumsum(filled, axis=0, dtype=np.float64)
    ccount = np.cumsum(valid.astype(np.int32), axis=0, dtype=np.int32)

    sums = csum.copy()
    counts = ccount.copy()
    if values.shape[0] > window:
        sums[window:] = csum[window:] - csum[:-window]
        counts[window:] = ccount[window:] - ccount[:-window]

    out = sums / np.where(counts > 0, counts, np.nan)
    out[counts < min_periods] = np.nan
    return out


def _calc_daily_delta_np(
    open_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    close_values: np.ndarray,
    volume_values: np.ndarray,
) -> np.ndarray:
    open_values = _clean_price_array(open_values)
    high_values = _clean_price_array(high_values)
    low_values = _clean_price_array(low_values)
    close_values = _clean_price_array(close_values)
    volume_values = _clean_volume_array(volume_values)

    valid_price = (
        np.isfinite(open_values)
        & np.isfinite(high_values)
        & np.isfinite(low_values)
        & np.isfinite(close_values)
        & (high_values >= low_values)
        & (high_values >= open_values)
        & (high_values >= close_values)
        & (low_values <= open_values)
        & (low_values <= close_values)
    )
    valid = valid_price & np.isfinite(volume_values)

    log_open = np.log(open_values)
    log_high = np.log(high_values)
    log_low = np.log(low_values)
    log_close = np.log(close_values)

    midpoint = np.sqrt(high_values * low_values)
    low_first = (open_values < midpoint) | ((open_values == midpoint) & (close_values >= open_values))

    low_first_num = (
        _signed_log_distance_sq(log_open, log_low)
        + _signed_log_distance_sq(log_low, log_high)
        + _signed_log_distance_sq(log_high, log_close)
    )
    low_first_den = (
        _log_distance_sq(log_open, log_low)
        + _log_distance_sq(log_low, log_high)
        + _log_distance_sq(log_high, log_close)
    )

    high_first_num = (
        _signed_log_distance_sq(log_open, log_high)
        + _signed_log_distance_sq(log_high, log_low)
        + _signed_log_distance_sq(log_low, log_close)
    )
    high_first_den = (
        _log_distance_sq(log_open, log_high)
        + _log_distance_sq(log_high, log_low)
        + _log_distance_sq(log_low, log_close)
    )

    signed_num = np.where(low_first, low_first_num, high_first_num)
    denom = np.where(low_first, low_first_den, high_first_den)
    delta = np.full(volume_values.shape, np.nan, dtype=np.float64)

    tradable = valid & (denom > _EPS)
    delta[tradable] = volume_values[tradable] * signed_num[tradable] / denom[tradable]
    delta[valid & ((denom <= _EPS) | (volume_values == 0.0))] = 0.0
    return delta


def _calc_output_np(data_ctx: dict[str, pd.DataFrame], output: str) -> np.ndarray:
    open_values = _as_float_array(data_ctx["open"])
    high_values = _as_float_array(data_ctx["high"])
    low_values = _as_float_array(data_ctx["low"])
    close_values = _as_float_array(data_ctx["close"])
    volume_values = _as_float_array(data_ctx["volume"])

    window = SETTING["params"]["window"]
    min_periods = SETTING["params"]["min_periods"]
    change_lag = SETTING["params"]["change_lag"]

    daily_delta = _calc_daily_delta_np(
        open_values, high_values, low_values, close_values, volume_values
    )
    daily_delta_valid = np.isfinite(daily_delta)
    cumulative_delta = np.cumsum(np.where(daily_delta_valid, daily_delta, 0.0), axis=0)
    cumulative_delta[~daily_delta_valid] = np.nan
    delta_mva50 = _rolling_mean_np(cumulative_delta, window, min_periods)

    if output == "mva50":
        result = delta_mva50
    elif output == "change":
        result = delta_mva50.copy()
        result[:change_lag] = np.nan
        result[change_lag:] = delta_mva50[change_lag:] - delta_mva50[:-change_lag]
    else:
        volume_clean = _clean_volume_array(volume_values)
        rolling_volume = _rolling_mean_np(volume_clean, window, min_periods)
        result = delta_mva50 / np.where(rolling_volume > _EPS, rolling_volume, np.nan)

    result[~np.isfinite(result)] = np.nan
    return result.astype(np.float32, copy=False)


def calc_reconstructed_delta_order_flow_factor(
    data_ctx: dict[str, pd.DataFrame],
    output: str = "norm",
) -> pd.DataFrame:
    """Return one reconstructed cumulative delta variant.

    output can be "norm", "mva50", or "change".
    """
    if output not in {"norm", "mva50", "change"}:
        raise ValueError("output must be one of: norm, mva50, change")

    close_like = data_ctx["close"]
    return pd.DataFrame(
        _calc_output_np(data_ctx, output),
        index=close_like.index,
        columns=close_like.columns,
        copy=False,
    )


def calc_reconstructed_delta_order_flow_factors(
    data_ctx: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Return all variants for diagnostics; avoid this in large production runs."""
    return {
        "delta_mva50": calc_reconstructed_delta_order_flow_factor(data_ctx, output="mva50"),
        "delta_mva50_norm": calc_reconstructed_delta_order_flow_factor(data_ctx, output="norm"),
        "delta_mva50_change": calc_reconstructed_delta_order_flow_factor(data_ctx, output="change"),
    }


def _to_qualification_mask(raw: pd.DataFrame) -> pd.DataFrame:
    qual_quantile = SETTING["params"]["qual_quantile"]
    if not 0.0 <= qual_quantile <= 1.0:
        raise ValueError("SETTING.params.qual_quantile must be between 0 and 1")

    threshold = raw.quantile(qual_quantile, axis=1)
    qualified = raw.ge(threshold, axis=0)
    return qualified.where(raw.notna()).astype(np.float32)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the daily qualification-mask factor matrix (date x instrument)."""
    raw = calc_reconstructed_delta_order_flow_factor(data_ctx, output="change")
    return _to_qualification_mask(raw)
