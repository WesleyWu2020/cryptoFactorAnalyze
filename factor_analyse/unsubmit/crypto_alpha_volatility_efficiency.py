"""crypto_alpha_volatility_efficiency 因子（factor_common 日频契约版）。

原始定义（分钟版）：
    roc = close / close.shift(20d) - 1
    range_pct = rolling_mean((high - low) / prev_close, 20d)
    factor = roc / range_pct

分钟->日频改写说明（语义判断）：
    - 原实现在 20*1440 根分钟 bar 上对每分钟 (high-low)/prev_close 求均值作为分母。
      日频下无法还原分钟级 range 分布，按公式意图改写为：分母 = 20 个"日"的
      (daily_high - daily_low) / prev_daily_close 的均值。分子 ROC 按公式意图改写为
      20 个交易日的涨跌幅（close / close.shift(20) - 1）。窗口语义从"20 天的分钟 bar"
      变为"20 天"，属忠实日频等价。
    - 分钟版无执行延迟 shift（实时分钟因子），日频版同样不加 shift；
      框架按次日开盘执行。
    - 删除 CHUNK_SIZE 分块循环，整表向量化计算。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_volatility_efficiency",
    "author": "wesleywu",
    "level": "daily",
    "category": "volatility",
    "description": "20d ROC divided by 20d mean daily relative range "
                   "((high-low)/prev_close): trend per unit of volatility",
}

SETTING = {
    "data_needed": ["close", "high", "low"],
    "universe": "historical_top50",
    # 20 (roc base) + 20 (range mean) + 1 (prev_close) + buffer
    "warmup_bars": 45,
    "preprocessing": "mad_rank",
    "params": {"window_days": 20},
    # 无 ITER_NOTE 方向记录；波动效率=单位波动的趋势幅度，默认高值做多
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily volatility-efficiency matrix (date x instrument)."""

    window_days = SETTING["params"]["window_days"]

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")

    close = close.where(close > 0)
    prev_close = close.shift(1)
    base_close = close.shift(window_days)

    roc = close / base_close - 1.0
    range_pct = ((high - low) / prev_close).rolling(
        window_days,
        min_periods=window_days,
    ).mean()

    factor = roc / (range_pct + _EPS)
    return factor.replace([np.inf, -np.inf], np.nan)
