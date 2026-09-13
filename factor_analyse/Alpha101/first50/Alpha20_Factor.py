"""Alpha101 Alpha#20 因子（factor_common 契约版）。

原始定义:
    Alpha#20 = (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1))))
                * rank((open - delay(low, 1))))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准，与经典公式有偏离）:
    - 代码中 delay 窗口为 7 而非经典公式的 1
    - 代码去掉了 open - delay(close, 1) 一项，只保留 high/low 两项
    factor = (-1 * rank(open - delay(high, 7))) * rank(open - delay(low, 7))

参数取原脚本 __main__ 实际调用值: 无窗口参数，delay 固定为 7（写入 params.delay_window）。
原脚本的 normalization_method 仅生成未被因子使用的 normalized_* 列，丢弃；
rebalance_period 只用于旧版 future_ret，不属于因子值计算，丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
公式内部的横截面 rank 保留；最终去极值与归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha20_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #20: (-1 * rank(open - delay(high, 7))) * rank(open - delay(low, 7))",
}

SETTING = {
    "data_needed": ["open", "high", "low"],
    "universe": "historical_top50",
    "warmup_bars": 15,
    "preprocessing": "mad_rank",
    "params": {"delay_window": 7},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#20 matrix (date x instrument)."""

    delay_window = SETTING["params"]["delay_window"]

    open_ = data_ctx["open"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")

    # 开盘价相对 delay_window 日前最高/最低价的位置
    open_high_diff = open_ - high.shift(delay_window)
    open_low_diff = open_ - low.shift(delay_window)

    # 公式内部横截面 rank 保留；复合信号（原脚本去掉了 open-close 项）
    rank_open_high = open_high_diff.rank(axis=1, pct=True)
    rank_open_low = open_low_diff.rank(axis=1, pct=True)
    return (-1.0 * rank_open_high) * rank_open_low
