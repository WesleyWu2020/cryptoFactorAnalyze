"""Alpha101 Alpha#24 因子（factor_common 契约版）。

原始定义:
    Alpha#24 =
        if (delta((sum(close, 100) / 100), 100) / delay(close, 100) <= 0.05):
            -1 * (close - ts_min(close, 100))
        else:
            -1 * delta(close, 3)

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准，与经典公式有偏离）:
    原脚本将经典公式的 100 日窗口全部替换为 30 日:
    ma            = ts_mean(close, 30)
    trend_rate    = delta(ma, 30) / delay(close, 30)
    min_price     = ts_min(close, 30)
    factor        = (trend_rate <= 0.05) ? -1 * (close - min_price) : -1 * delta(close, 3)

参数取原脚本 __main__ 实际调用值: 窗口固定为 30 / delta_window=3 / trend_threshold=0.05
（函数内固定值，写入 params）。rebalance_period 只用于旧版 future_ret，不属于因子值计算，丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha24_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #24: (delta(ts_mean(close,30),30)/delay(close,30) <= 0.05) ? -(close - ts_min(close,30)) : -delta(close,3)",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 65,
    "preprocessing": "mad_rank",
    "params": {
        "ma_window": 30,
        "trend_window": 30,
        "min_window": 30,
        "delta_window": 3,
        "trend_threshold": 0.05,
    },
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#24 matrix (date x instrument)."""

    ma_window = SETTING["params"]["ma_window"]
    trend_window = SETTING["params"]["trend_window"]
    min_window = SETTING["params"]["min_window"]
    delta_window = SETTING["params"]["delta_window"]
    trend_threshold = SETTING["params"]["trend_threshold"]

    close = data_ctx["close"].astype("float64")

    # 长期趋势变化率
    ma = close.rolling(window=ma_window, min_periods=ma_window).mean()
    ma_delta = ma - ma.shift(trend_window)
    close_delay = close.shift(trend_window)
    long_trend_rate = ma_delta / close_delay

    # 自适应分支：趋势平缓用相对区间低点的位置，趋势明显用短期价格变化
    min_price = close.rolling(window=min_window, min_periods=min_window).min()
    delta_close = close - close.shift(delta_window)

    factor = (-1.0 * (close - min_price)).where(
        long_trend_rate <= trend_threshold, -1.0 * delta_close
    )

    # 数据未就绪时输出 NaN（与旧脚本行级 isna 检查等价）
    valid = long_trend_rate.notna() & min_price.notna() & delta_close.notna()
    return factor.where(valid)
