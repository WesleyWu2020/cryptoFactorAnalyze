"""crypto_alpha_volume_amplitude_divergence_corr 因子（factor_common 日频契约版）。

原始定义（分钟版）：
    amplitude  = (daily_high - daily_low) / daily_close
    log_volume = log1p(daily_volume)
    factor     = -rolling_corr(log_volume, amplitude, 10d, min_periods=10)

分钟->日频改写说明：
    - 原实现对分钟数据按 tradable_mask 掩码后 resample("1D") 聚合出日频
      high(max)/low(min)/close(last)/volume(sum)。日频契约下这些字段直接为
      日频矩阵，聚合为恒等。
    - tradable_mask 在分钟版中用于剔除非可交易 bar 对日聚合的污染；日频契约下
      框架按 universe（historical_top50）成员关系自动掩码因子输出，因子内部不再
      需要该掩码，故从 data_needed 中移除且不设 context_eligible。
    - 删除原实现的 .shift(1)（执行延迟）与 broadcast 回分钟索引步骤；
      框架按次日开盘执行，factor(t) 允许使用 t 及以前数据。
    - 分钟版用 cumsum 手写滚动 Pearson 相关并要求分母 > 0；日频版直接用
      DataFrame.rolling().corr()，零方差自动产出 NaN，语义一致。
    - 删除逐列循环与 gc 管理，整表向量化计算。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_volume_amplitude_divergence_corr",
    "author": "wesleywu",
    "level": "daily",
    "category": "volume_price",
    "description": "negative 10d rolling corr between log1p(volume) and daily amplitude "
                   "((high-low)/close): volume-amplitude divergence",
}

SETTING = {
    "data_needed": ["high", "low", "close", "volume"],
    "universe": "historical_top50",
    # 10 (corr window) + buffer
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {"window_days": 10, "min_periods": 10},
    # 无 ITER_NOTE 方向记录；factor=-corr，高值=量价振幅背离，默认 1
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily divergence-corr matrix (date x instrument)."""

    window_days = SETTING["params"]["window_days"]
    min_periods = SETTING["params"]["min_periods"]

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    amplitude = ((high - low) / close).where(close > 0)
    log_volume = np.log1p(volume.clip(lower=0))

    corr = log_volume.rolling(window_days, min_periods=min_periods).corr(amplitude)
    factor = -corr.clip(-1.0, 1.0)
    return factor.replace([np.inf, -np.inf], np.nan)
