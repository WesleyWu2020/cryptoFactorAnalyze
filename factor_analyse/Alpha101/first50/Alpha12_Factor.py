"""Alpha101 Alpha#12 因子（factor_common 契约版）。

原始定义:
    Alpha#12 = sign(delta(volume, 1)) * (-1 * delta(close, 1))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    原脚本将 volume 替换为主动买入金额 taker_buy_quote（即 taker_buy_quote_volume）:
    factor = sign(delta(taker_buy_quote_volume, 30)) * (-1 * delta(close, 30))
混合策略: 主动买入增加时做价格反转，主动买入减少时做价格动量。
参数取原脚本 __main__ 实际调用值: delta_window=30
（rebalance_period 仅用于旧版 future_ret，已丢弃）。
与经典公式的差异: volume -> taker_buy_quote_volume，窗口由 1 改为 30。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha12_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #12: sign(delta(taker_buy_quote_volume,N)) * (-delta(close,N))",
}

SETTING = {
    "data_needed": ["close", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 35,
    "preprocessing": "mad_rank",
    "params": {"delta_window": 30},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#12 matrix (date x instrument)."""

    delta_window = SETTING["params"]["delta_window"]
    if isinstance(delta_window, bool) or not isinstance(delta_window, int) or delta_window < 1:
        raise ValueError("SETTING.params.delta_window must be an integer >= 1")

    close = data_ctx["close"].astype("float64")
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # 步骤1: 主动买入金额变化符号 sign(delta(taker_buy_quote, N))
    taker_buy_sign = np.sign(taker_buy_quote - taker_buy_quote.shift(delta_window))

    # 步骤2: 价格变化 delta(close, N)
    price_delta = close - close.shift(delta_window)

    # 步骤3: 复合信号（反转/动量混合）
    return taker_buy_sign * (-1.0 * price_delta)
