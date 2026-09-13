"""crypto_alpha_rv_20d_resid_style_wma_rank 因子（factor_common 日频契约版）。

原始定义（分钟版改造）：
    daily_ret = close.pct_change()
    rv_5d / rv_20d = rolling std(daily_ret, 5d/20d)
    dollar_volume_rank = rank_cs(log1p(mean(quote_volume.shift(1), 30d)))
    resid = 截面回归 rv_20d ~ dollar_volume_rank + rv_ratio_5_20 的残差
    factor = rolling_last_rank(WMA(resid, 10d), 10d)

日频改写判断：
    - 分钟聚合（close=last, dollar_volume=sum, tradable_mask=last）在日频输入下
      为恒等：dollar_volume -> quote_volume；tradable_mask -> data_ctx["__eligible__"]
      （context_eligible=True，残差化/截面 rank 需要池内掩码）。
    - 删除 factor_daily.shift(1)（纯执行延迟，框架按次日开盘执行）与分钟 index 广播。
    - dollar_volume.shift(1) 是因子定义内部的"只用已知流动性"约束，予以保留。
    - 分块循环删除，整帧计算。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_rv_20d_resid_style_wma_rank",
    "author": "wesleywu",
    "level": "daily",
    "category": "volatility",
    "description": "rolling_last_rank(WMA(resid(rv_20d ~ liquidity+rv_ratio), 10d), 10d): 风格中性化 20d 已实现波动",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 75,
    "preprocessing": "mad_rank",
    "params": {
        "rv_short_days": 5,
        "rv_long_days": 20,
        "min_rv_short_days": 3,
        "min_rv_long_days": 8,
        "dollar_volume_rank_days": 30,
        "wma_days": 10,
        "rank_days": 10,
    },
    "factor_direction": 1,  # 默认：高值=中性化残差波动排名高
    "context_eligible": True,
}

_EPS = 1e-12


def _cross_section_rank(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.where(mask).rank(axis=1, pct=True)
    return ranked.replace([np.inf, -np.inf], np.nan)


def _style_residualize(
    values: pd.DataFrame,
    controls: list[pd.DataFrame],
    mask: pd.DataFrame,
) -> pd.DataFrame:
    if not controls:
        return pd.DataFrame(np.nan, index=values.index, columns=values.columns, dtype=np.float64)

    y_arr = values.to_numpy(dtype=np.float64, copy=False)
    control_arrs = [
        control.reindex_like(values).to_numpy(dtype=np.float64, copy=False)
        for control in controls
    ]
    mask_arr = mask.reindex_like(values).fillna(False).to_numpy(dtype=bool, copy=False)
    out_arr = np.full(y_arr.shape, np.nan, dtype=np.float64)

    for row in range(len(values.index)):
        valid = mask_arr[row] & np.isfinite(y_arr[row])
        for control_arr in control_arrs:
            valid &= np.isfinite(control_arr[row])
        idx = np.flatnonzero(valid)
        if idx.size < len(control_arrs) + 2:
            continue
        x = np.column_stack(
            [
                np.ones(idx.size, dtype=np.float64),
                *[control_arr[row, idx] for control_arr in control_arrs],
            ]
        )
        beta, *_ = np.linalg.lstsq(x, y_arr[row, idx], rcond=None)
        out_arr[row, idx] = y_arr[row, idx] - x @ beta

    return pd.DataFrame(out_arr, index=values.index, columns=values.columns)


def _weighted_moving_average(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    weights = np.arange(1, window + 1, dtype=np.float64)
    weights /= weights.sum()
    return frame.rolling(window, min_periods=window).apply(
        lambda values: np.dot(values, weights),
        raw=True,
    )


def _rolling_last_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    def last_rank(values: np.ndarray) -> float:
        last = values[-1]
        if not np.isfinite(last):
            return np.nan
        valid = values[np.isfinite(values)]
        if len(valid) < window:
            return np.nan
        return (np.sum(valid < last) + 0.5 * np.sum(valid == last)) / len(valid)

    return frame.rolling(window, min_periods=window).apply(last_rank, raw=True)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily style-residualized RV rank matrix (date x instrument)."""

    params = SETTING["params"]

    daily_close = data_ctx["close"].astype("float64")
    daily_dollar_volume = data_ctx["quote_volume"].astype("float64").clip(lower=0)
    daily_tradable = data_ctx["__eligible__"].fillna(False).astype(bool)

    daily_ret = daily_close.pct_change(fill_method=None)
    rv_5d = daily_ret.rolling(
        params["rv_short_days"], min_periods=params["min_rv_short_days"]
    ).std()
    rv_20d = daily_ret.rolling(
        params["rv_long_days"], min_periods=params["min_rv_long_days"]
    ).std()
    rv_ratio_5_20 = rv_5d / (rv_20d + _EPS)

    dollar_volume_lag1 = daily_dollar_volume.shift(1)
    dollar_volume_mean30 = dollar_volume_lag1.rolling(
        params["dollar_volume_rank_days"],
        min_periods=1,
    ).mean()
    dollar_volume_rank = _cross_section_rank(
        np.log1p(dollar_volume_mean30.where(dollar_volume_mean30 > 0.0)),
        daily_tradable,
    )

    rv_20d_resid_style = _style_residualize(
        rv_20d.replace([np.inf, -np.inf], np.nan),
        [
            dollar_volume_rank,
            rv_ratio_5_20.replace([np.inf, -np.inf], np.nan),
        ],
        daily_tradable,
    )

    wma = _weighted_moving_average(rv_20d_resid_style, params["wma_days"])
    factor = _rolling_last_rank(wma, params["rank_days"])
    return factor.replace([np.inf, -np.inf], np.nan)
