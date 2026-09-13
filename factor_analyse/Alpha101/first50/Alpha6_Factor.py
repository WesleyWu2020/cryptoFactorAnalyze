"""Alpha101 Alpha#6 因子（factor_common 契约版，币圈7×24h优化版）。

原始定义:
    Alpha#6 = -1 * correlation(open, volume, 10)

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准，与经典公式不同）:
    factor = -1 * correlation(high - close, taker_buy_quote_volume, correlation_window)
即最高价与收盘价之差（上影线压力代理）与主动买入金额的时序相关性，取反捕捉
量价背离的反转机会。窗口内任一序列近似常数（std <= 1e-10）时相关性定义为 0
（与原脚本守卫一致），数据不足窗口期为 NaN。

参数取原脚本 ``__main__`` 实际调用值: correlation_window=20
（rebalance_period 仅用于旧版 future_ret，已丢弃；可用性池过滤由框架
universe="historical_top50" 承担）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha6_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #6 (crypto): -1 * correlation(high - close, taker_buy_quote_volume, 20)",
}

SETTING = {
    "data_needed": ["high", "close", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 25,
    "preprocessing": "mad_rank",
    "params": {"correlation_window": 20},
    "factor_direction": 1,
}

_CONST_EPS = 1e-10


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#6 matrix (date x instrument)."""

    window = SETTING["params"]["correlation_window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.correlation_window must be an integer >= 2")

    high = data_ctx["high"].astype("float64")
    close = data_ctx["close"].astype("float64")
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # 上影线压力代理: high - close（原脚本以代码实现为准，替代经典公式 rank(open)）
    high_close_diff = high - close

    rolling = high_close_diff.rolling(window=window, min_periods=window)
    correlation = rolling.corr(taker_buy_quote)

    # 常数窗口守卫: 任一序列窗口内 std <= 1e-10 时相关性定义为 0（与原脚本一致）；
    # 数据不足的窗口期保持 NaN
    x_std = high_close_diff.rolling(window=window, min_periods=window).std()
    y_std = taker_buy_quote.rolling(window=window, min_periods=window).std()
    both_vary = (x_std > _CONST_EPS) & (y_std > _CONST_EPS)
    valid_window = x_std.notna() & y_std.notna()
    correlation = correlation.where(both_vary, 0.0).where(valid_window)

    return -1.0 * correlation
