"""crypto_alpha_rv_resid_btc_cum_liquidity_ensemble 因子（factor_common 日频契约版）。

原始定义（已是日频语义，分钟版仅做 resample）：
    factor = +0.5 * rank40(mean10(rv_ratio_5_20 风格残差))
             - 1.0 * WMA20(rank10(resid_return_btc_20d))
             + 0.5 * MAD10(cum_return)
             + 0.167 * |rank10(dollar_volume_rank)|
    其中 rv_ratio_5_20 对 [dollar_volume_rank, amihud_illiq, beta60_btc,
    corr20_btc, rv_20d] 做截面回归取残差。

日频改写判断：
    - 分钟聚合（close=last, dollar_volume=sum, tradable_mask=last）在日频输入下
      为恒等：dollar_volume -> quote_volume；tradable_mask -> data_ctx["__eligible__"]
      （context_eligible=True，截面 rank/残差化需要池内掩码）。
    - dollar_volume.shift(1) 是因子定义内部的"只用已知流动性"约束，予以保留。
    - 该因子无执行延迟 shift，直接返回日频矩阵。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_rv_resid_btc_cum_liquidity_ensemble",
    "author": "wesleywu",
    "level": "daily",
    "category": "ensemble",
    "description": "+0.5*rank40(mean10(rv_ratio_resid)) - WMA20(rank10(resid_btc_20d)) + 0.5*MAD10(cumret) + 0.167*|rank10(dollar_rank)|",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 90,
    "preprocessing": "mad_rank",
    "params": {
        "rv_short_days": 5,
        "rv_long_days": 20,
        "beta_window_days": 60,
        "beta_min_days": 20,
        "corr_window_days": 20,
        "corr_min_days": 8,
        "resid_sum_days": 20,
        "resid_sum_min_days": 8,
        "dollar_volume_lookback_days": 30,
        "mean_window_days": 10,
        "rank_window_days": 40,
        "btc_rank_days": 10,
        "btc_wma_days": 20,
        "mad_window_days": 10,
        "dollar_rank_days": 10,
    },
    "factor_direction": 1,  # 默认：ensemble 混合符号，高值优先
    "context_eligible": True,
}

_EPS = 1e-12
_BTC_SYMBOL_CANDIDATES = ("BTCUSDT", "BTC-USD", "BTCUSD")


def _rolling_mean(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).mean()


def _rolling_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    def _rank_last(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        last = values[-1]
        left = np.count_nonzero(last < values)
        right = np.count_nonzero(last <= values)
        return float((right + left + (right > left)) / (2.0 * len(values)))

    return frame.rolling(window, min_periods=window).apply(_rank_last, raw=True)


def _rolling_wma(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    weights = np.arange(window, dtype=np.float64)
    weights /= weights.sum()

    def _wma(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        return float(np.dot(values, weights))

    return frame.rolling(window, min_periods=window).apply(_wma, raw=True)


def _rolling_mad(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    def _mad(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        center = values.mean()
        return float(np.abs(values - center).mean())

    return frame.rolling(window, min_periods=window).apply(_mad, raw=True)


def _cross_section_rank(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=np.float64, copy=False)
    valid_mask = mask.reindex(index=frame.index, columns=frame.columns).fillna(False).to_numpy(dtype=bool, copy=False)
    out = np.full(values.shape, np.nan, dtype=np.float32)

    for row in range(values.shape[0]):
        idx = np.flatnonzero(valid_mask[row] & np.isfinite(values[row]))
        if idx.size == 0:
            continue
        order = np.argsort(values[row, idx], kind="mergesort")
        ranks = np.empty(idx.size, dtype=np.float32)
        ranks[order] = np.arange(idx.size, dtype=np.float32) / np.float32(idx.size)
        out[row, idx] = ranks

    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _style_residualize(
    values: pd.DataFrame,
    controls: list[pd.DataFrame],
    mask: pd.DataFrame,
) -> pd.DataFrame:
    if not controls:
        return pd.DataFrame(np.nan, index=values.index, columns=values.columns, dtype=np.float32)

    y_arr = values.to_numpy(dtype=np.float64, copy=False)
    control_arrs = [
        control.reindex_like(values).to_numpy(dtype=np.float64, copy=False)
        for control in controls
    ]
    mask_arr = mask.reindex_like(values).fillna(False).to_numpy(dtype=bool, copy=False)
    out_arr = np.full(y_arr.shape, np.nan, dtype=np.float32)

    for row in range(len(values.index)):
        valid = mask_arr[row] & np.isfinite(y_arr[row])
        for control_arr in control_arrs:
            valid &= np.isfinite(control_arr[row])
        idx = np.flatnonzero(valid)
        if idx.size < len(control_arrs) + 2:
            continue
        x = np.column_stack([
            np.ones(idx.size, dtype=np.float64),
            *[control_arr[row, idx] for control_arr in control_arrs],
        ])
        beta, *_ = np.linalg.lstsq(x, y_arr[row, idx], rcond=None)
        out_arr[row, idx] = (y_arr[row, idx] - x @ beta).astype(np.float32)

    return pd.DataFrame(out_arr, index=values.index, columns=values.columns)


def _find_btc_column(columns: pd.Index) -> Optional[str]:
    for candidate in _BTC_SYMBOL_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def _calc_daily_cum_return(daily_ret: pd.DataFrame) -> pd.DataFrame:
    cum_return = np.cumprod(
        1.0 + np.nan_to_num(daily_ret.to_numpy(dtype=np.float64, copy=False), nan=0.0), axis=0
    )
    return pd.DataFrame(cum_return, index=daily_ret.index, columns=daily_ret.columns, dtype=np.float32)


def _daily_base_features(data_ctx: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    params = SETTING["params"]

    daily_close = data_ctx["close"].astype("float64")
    daily_dollar_volume = data_ctx["quote_volume"].astype("float64").clip(lower=0)
    daily_tradable = data_ctx["__eligible__"].fillna(False).astype(bool)
    daily_ret = daily_close.pct_change(fill_method=None)

    log_dollar_volume = np.log1p(
        daily_dollar_volume.shift(1)
        .rolling(params["dollar_volume_lookback_days"], min_periods=1)
        .mean()
        .where(lambda x: x > 0.0)
    )
    dollar_volume_rank = _cross_section_rank(log_dollar_volume, daily_tradable)

    btc_col = _find_btc_column(daily_close.columns)
    if btc_col is None:
        beta60 = pd.DataFrame(np.nan, index=daily_close.index, columns=daily_close.columns)
        corr20 = pd.DataFrame(np.nan, index=daily_close.index, columns=daily_close.columns)
        resid_return_btc_20d = pd.DataFrame(np.nan, index=daily_close.index, columns=daily_close.columns)
    else:
        btc_ret = daily_ret[btc_col]
        btc_var60 = btc_ret.rolling(params["beta_window_days"], min_periods=params["beta_min_days"]).var()
        beta60 = daily_ret.rolling(params["beta_window_days"], min_periods=params["beta_min_days"]).cov(btc_ret).div(btc_var60 + _EPS, axis=0)
        corr20 = daily_ret.rolling(params["corr_window_days"], min_periods=params["corr_min_days"]).corr(btc_ret)
        residual = daily_ret.sub(beta60.mul(btc_ret, axis=0), axis=0)
        resid_return_btc_20d = residual.rolling(params["resid_sum_days"], min_periods=params["resid_sum_min_days"]).sum()

    rv_5d = daily_ret.rolling(params["rv_short_days"], min_periods=3).std()
    rv_20d = daily_ret.rolling(params["rv_long_days"], min_periods=8).std()
    rv_ratio_5_20 = rv_5d / (rv_20d + _EPS)
    amihud_illiq = daily_ret.abs() / (daily_dollar_volume.where(daily_dollar_volume > 0.0) + _EPS) * 1e6

    rv_ratio_5_20_resid_style = _style_residualize(
        rv_ratio_5_20.replace([np.inf, -np.inf], np.nan),
        [
            dollar_volume_rank,
            amihud_illiq.replace([np.inf, -np.inf], np.nan),
            beta60.replace([np.inf, -np.inf], np.nan),
            corr20.replace([np.inf, -np.inf], np.nan),
            rv_20d.replace([np.inf, -np.inf], np.nan),
        ],
        daily_tradable,
    )

    return {
        "daily_ret": daily_ret,
        "dollar_volume_rank": dollar_volume_rank.replace([np.inf, -np.inf], np.nan),
        "resid_return_btc_20d": resid_return_btc_20d.replace([np.inf, -np.inf], np.nan),
        "rv_ratio_5_20_resid_style": rv_ratio_5_20_resid_style.replace([np.inf, -np.inf], np.nan),
    }


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily ensemble matrix (date x instrument)."""
    params = SETTING["params"]

    base_features = _daily_base_features(data_ctx)
    rv_resid = base_features["rv_ratio_5_20_resid_style"]
    resid_btc = base_features["resid_return_btc_20d"]
    dollar_rank = base_features["dollar_volume_rank"]
    cum_return = _calc_daily_cum_return(base_features["daily_ret"])

    component_1 = _rolling_rank(_rolling_mean(rv_resid, params["mean_window_days"]), params["rank_window_days"])
    component_2 = _rolling_wma(_rolling_rank(resid_btc, params["btc_rank_days"]), params["btc_wma_days"])
    component_3 = _rolling_mad(cum_return, params["mad_window_days"])
    component_4 = _rolling_rank(dollar_rank, params["dollar_rank_days"]).abs()

    factor = 0.5 * component_1 - 1.0 * component_2 + 0.5 * component_3 + 0.167 * component_4
    return factor.replace([np.inf, -np.inf], np.nan).astype(np.float32)
