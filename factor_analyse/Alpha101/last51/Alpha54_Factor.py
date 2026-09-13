"""Alpha101 Alpha#54 因子（factor_common 契约版）。

原始定义:
    ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    1. 分子: -1 * (low - close) * (open^power_exponent)
    2. 分母: (low - high) * (close^power_exponent)（+ 1e-8 防除零）
    3. 因子值: 分子 / 分母
    注意：原脚本中 20 日价格归一化代码已被注释掉，实际使用原始价格，本实现保持一致。

参数取原脚本 __main__ 实际调用值（power_exponent=3；函数签名默认 5 但 __main__ 用 3）。
计算只使用当日数据，无未来函数。FactorManager 只调用下面的标准模块接口；
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha54_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #54: (-(low-close)*open^p) / ((low-high)*close^p), p=3",
}

SETTING = {
    "data_needed": ["open", "close", "high", "low"],
    "universe": "historical_top50",
    "warmup_bars": 5,
    "preprocessing": "mad_rank",
    "params": {"power_exponent": 3},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#54 matrix (date x instrument)."""

    power_exponent = SETTING["params"]["power_exponent"]
    if (
        isinstance(power_exponent, bool)
        or not isinstance(power_exponent, int)
        or power_exponent < 1
    ):
        raise ValueError("SETTING.params.power_exponent must be an integer >= 1")

    open_ = data_ctx["open"].astype("float64")
    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")

    numerator = -1.0 * (low - close) * (open_ ** power_exponent)
    denominator = (low - high) * (close ** power_exponent)
    return numerator / (denominator + _EPSILON)
