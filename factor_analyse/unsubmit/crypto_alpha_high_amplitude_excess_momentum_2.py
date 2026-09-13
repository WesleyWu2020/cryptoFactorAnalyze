"""crypto_alpha_high_amplitude_excess_momentum_2 因子（factor_common 日频契约版）。

原始定义（分钟版语义）：
    daily_return    = log(close / prev_close)（日频）
    daily_amplitude = (high - low) / prev_close（日频）
    valid           = 收益/振幅有限 且 quote_volume > 0 且 |收益| < 0.9
    market_return   = 当日截面 quote_volume 加权平均收益（仅有效币）
    daily_excess    = daily_return - market_return
    alpha_ret_high  = 最近 60 个有效日中振幅最高 top 30%（18 天）的超额收益之和
    reversal        = -log(close / close.shift(10d))
    factor          = - cross_sectional_residual(alpha_ret_high ~ reversal)
    （逐日截面 OLS 去均值残差，要求当日有效样本 >= 3）

日频改写说明：
    - data_ctx 已是日频矩阵，日聚合恒等（close=last/high=max/low=min、
      dollar_volume 日求和 == 日频 quote_volume），直接使用 quote_volume。
    - 截面市场收益与逐日截面回归只使用当日横截面数据，因果无未来函数。
    - 删除分钟版末尾的 .shift(1)（执行延迟由框架次日开盘成交处理）与分钟 index
      重广播；保留构造末端的整体取负（作者对原始动量信号翻号后的可交易信号）。
    - 窗口单位由分钟 bar 改写为日历日，语义不变。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_high_amplitude_excess_momentum_2",
    "author": "wesleywu",
    "level": "daily",
    "category": "momentum",
    "description": "高振幅日超额收益动量对 10d 反转做截面中性化后取负（翻号后的残差信号）",
}

SETTING = {
    "data_needed": ["high", "low", "close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 75,
    "preprocessing": "mad_rank",
    "params": {
        "window_days": 60,
        "high_amp_ratio": 0.3,
        "reversal_days": 10,
        "return_clip": 0.9,
        "min_daily_quote_volume": 0.0,
    },
    "factor_direction": 1,
}


def _selective_top_amp_sum(
    excess: pd.DataFrame,
    amp: pd.DataFrame,
    window: int,
    select: int,
) -> pd.DataFrame:
    """Sum of excess returns over the top-``select`` amplitude days in each
    trailing ``window``-day window; NaN unless all ``window`` days are valid."""

    exc_np = excess.to_numpy(dtype=np.float64)
    amp_np = amp.to_numpy(dtype=np.float64)
    n_rows, n_cols = exc_np.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    if n_rows < window:
        return pd.DataFrame(out, index=excess.index, columns=excess.columns)

    exc_w = np.lib.stride_tricks.sliding_window_view(exc_np, window, axis=0)
    amp_w = np.lib.stride_tricks.sliding_window_view(amp_np, window, axis=0)

    valid = np.isfinite(exc_w).all(axis=2) & np.isfinite(amp_w).all(axis=2)

    top_idx = np.argpartition(amp_w, window - select, axis=2)[:, :, window - select:]
    selected = np.take_along_axis(exc_w, top_idx, axis=2)
    selected_sum = selected.sum(axis=2)

    out[window - 1:] = np.where(valid, selected_sum, np.nan)
    return pd.DataFrame(out, index=excess.index, columns=excess.columns)


def _neutralize_by_reversal(
    alpha_ret_high: pd.DataFrame,
    reversal: pd.DataFrame,
) -> pd.DataFrame:
    """Per-day cross-sectional OLS residual of y on x (demeaned), requiring
    at least 3 valid instruments per day. Same-day cross-section only."""

    x = reversal.astype(np.float64)
    y = alpha_ret_high.astype(np.float64)

    valid = x.notna() & y.notna()
    count = valid.sum(axis=1)

    x_valid = x.where(valid)
    y_valid = y.where(valid)

    x_mean = x_valid.sum(axis=1) / count.where(count > 0)
    y_mean = y_valid.sum(axis=1) / count.where(count > 0)

    x_demean = x.sub(x_mean, axis=0).where(valid)
    y_demean = y.sub(y_mean, axis=0).where(valid)

    beta = (x_demean * y_demean).sum(axis=1) / (x_demean * x_demean).sum(axis=1).replace(0, np.nan)
    residual = y.sub(y_mean, axis=0) - x.sub(x_mean, axis=0).mul(beta, axis=0)

    residual = residual.where(count >= 3)
    return residual.replace([np.inf, -np.inf], np.nan)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily high-amplitude excess momentum matrix
    (date x instrument)."""

    window_days = SETTING["params"]["window_days"]
    high_amp_ratio = SETTING["params"]["high_amp_ratio"]
    reversal_days = SETTING["params"]["reversal_days"]
    return_clip = SETTING["params"]["return_clip"]
    min_daily_quote_volume = SETTING["params"]["min_daily_quote_volume"]
    select_days = max(1, int(window_days * high_amp_ratio))

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    quote_volume = data_ctx["quote_volume"].clip(lower=0).astype("float64")

    prev_close = close.shift(1).replace(0.0, np.nan)
    daily_return = np.log(close.where(close > 0) / prev_close)
    daily_amplitude = (high - low) / prev_close

    valid = (
        daily_return.replace([np.inf, -np.inf], np.nan).notna()
        & daily_amplitude.replace([np.inf, -np.inf], np.nan).notna()
        & quote_volume.replace([np.inf, -np.inf], np.nan).gt(min_daily_quote_volume)
    )
    if return_clip > 0:
        valid &= daily_return.abs().lt(return_clip)

    # 当日截面 quote_volume 加权市场收益（只用同日横截面，无未来函数）
    valid_return = daily_return.where(valid)
    valid_weight = quote_volume.where(valid)
    weighted_sum = (valid_return * valid_weight).sum(axis=1, skipna=True)
    weight_sum = valid_weight.sum(axis=1, skipna=True)
    market_return = (weighted_sum / weight_sum.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    daily_excess = daily_return.sub(market_return, axis=0).where(valid)
    daily_amplitude = daily_amplitude.where(valid)

    alpha_ret_high = _selective_top_amp_sum(
        daily_excess, daily_amplitude, window_days, select_days
    )

    prev_close_rev = close.shift(reversal_days).replace(0.0, np.nan)
    reversal = (-np.log(close.where(close > 0) / prev_close_rev)).where(valid)
    reversal = reversal.replace([np.inf, -np.inf], np.nan)

    residual = _neutralize_by_reversal(alpha_ret_high, reversal)

    # 保留分钟版构造末端的整体取负（翻号后的可交易信号）
    return (-residual).replace([np.inf, -np.inf], np.nan)
