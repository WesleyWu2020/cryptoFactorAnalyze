"""crypto_alpha_bias_neutralized_zscore 因子（factor_common 日频契约版）。

原始定义（bias neutralized exp-weighted MA z-score）：
    weighted_ma = sum_i(w_i * MA(close, w_i_days))，w_i 正比 exp(-0.3*i)，
                  MA 窗口 (20, 40, 60, 90, 120) 日，短窗权重更高
    raw_z = -(close - weighted_ma) / rolling_std(close, 120d)
    对每个交易日横截面：raw_z ~ log1p(quote_volume) 回归取残差
    factor = -residual   （旧版末尾的负号修正，保留）

反转类：价格相对多窗口加权均线的偏离经波动率标准化、再对成交额（规模/流动性）
做截面中性化后的反转信号。负号已内嵌，factor_direction=1（因子值越高越偏多）。
转换判断说明：
- dollar_volume -> quote_volume（分钟成交额日求和 == 日 quote_volume；log1p 截面
  中性化语义不变）。
- 旧分钟版 rolling(120*1440) 的 std 窗口在日频下为 120 日；旧版 min_periods 仅
  1440 分钟（=1 日）过于宽松，日频版改为 min_periods=120（整窗），与 MA 窗口
  本就要求 120 日全有效保持一致（MA 门控已使 120 日前无输出，此改动不改变实际
  首个有效点）。
- 删除 ROW_CHUNK_SIZE 行分块（分钟级内存控制），日频整帧直接截面回归。
- 该因子无执行延迟 shift，保持原语义。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_bias_neutralized_zscore",
    "author": "wesleywu",
    "level": "daily",
    "category": "reversal",
    "description": "-(residual of -(close - exp-weighted multi-MA)/std120 ~ log1p(quote_volume))",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 125,
    "preprocessing": "mad_rank",
    "params": {
        "window_days": 120,
        "ma_windows_days": [20, 40, 60, 90, 120],
        "ma_weight_decay": 0.3,
    },
    "factor_direction": 1,
}


def _exp_weighted_multi_ma(close: pd.DataFrame) -> pd.DataFrame:
    """多窗口 MA 指数加权均线；仅在所有 MA 窗口都有效时输出。"""
    ma_windows_days = SETTING["params"]["ma_windows_days"]
    weight_decay = SETTING["params"]["ma_weight_decay"]

    weights = np.array(
        [np.exp(-weight_decay * i) for i in range(len(ma_windows_days))],
        dtype=np.float64,
    )
    weights /= weights.sum()

    mas = [
        close.rolling(window, min_periods=window).mean()
        for window in ma_windows_days
    ]
    values = np.stack(
        [ma.to_numpy(dtype=np.float64, copy=False) for ma in mas],
        axis=0,
    )
    valid = np.isfinite(values)
    weights_3d = weights[:, None, None]
    valid_weight_sum = np.where(valid, weights_3d, 0.0).sum(axis=0)
    weighted_sum = np.where(valid, values * weights_3d, 0.0).sum(axis=0)
    weighted_mean = np.divide(
        weighted_sum,
        valid_weight_sum,
        out=np.full(valid_weight_sum.shape, np.nan, dtype=np.float64),
        where=valid_weight_sum > 0,
    )
    weighted_mean = np.where(valid.all(axis=0), weighted_mean, np.nan)
    return pd.DataFrame(weighted_mean, index=close.index, columns=close.columns)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily bias-neutralized-zscore matrix (date x instrument)."""

    window_days = SETTING["params"]["window_days"]

    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")

    mean = _exp_weighted_multi_ma(close)
    std = close.rolling(window_days, min_periods=window_days).std().replace(0, np.nan)

    raw_z = -(close - mean) / std

    # 横截面中性化：raw_z ~ log1p(quote_volume)，逐日截面回归取残差
    x = np.log1p(quote_volume.clip(lower=0))
    mean_x = x.mean(axis=1)
    mean_y = raw_z.mean(axis=1)
    dev_x = x.sub(mean_x, axis=0)
    dev_y = raw_z.sub(mean_y, axis=0)
    beta = (dev_x * dev_y).mean(axis=1) / (dev_x**2).mean(axis=1).replace(0, np.nan)
    residual = dev_y.sub(dev_x.mul(beta, axis=0))

    # 保留旧版末尾的负号修正（方向反转）
    factor = -residual
    return factor.replace([np.inf, -np.inf], np.nan)
