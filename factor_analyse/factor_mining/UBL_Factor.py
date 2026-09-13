"""UBL 因子（威廉下影线相对强度）。

旧版 CSV 管线脚本的新框架改写。旧版文档描述的完整公式为
``UBL = zscore(上影线标准差) + zscore(威廉下影线均值)``，但旧代码中上影线项
在最终合成时被注释掉，实际生效的因子仅为::

    lower_shadow      = close - low                     # 威廉下影线
    lower_shadow_mean = lower_shadow / (rolling_mean(lower_shadow, std_window) + 1e-8)
    UBL               = 按日横截面 zscore(lower_shadow_mean)

新版严格复刻旧版实际生效的计算：rolling 未指定 min_periods（等价于
min_periods=window），分母加 1e-8 防零，横截面 zscore 在 std 为 0 或 NaN 时
退化为中心化。输出为原始因子值，由框架 mad_rank 统一做去极值与按日 rank。

计算只使用当日及历史数据，无未来函数。
"""

from __future__ import annotations

import pandas as pd

TYPE = "regular"

META = {
    "factor_name": "UBL_Factor",
    "author": "local",
    "level": "daily",
    "category": "price_action",
    "description": "William lower-shadow relative strength (close-low vs its N-day mean)",
}

SETTING = {
    "data_needed": ["close", "low"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"std_window": 30},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回 UBL 原始因子矩阵（date × instrument）。

    rolling 沿时间轴（axis 0），横截面 zscore 沿 axis 1，均只依赖当日及
    历史数据。
    """

    std_window = SETTING["params"]["std_window"]
    if isinstance(std_window, bool) or not isinstance(std_window, int) or std_window < 2:
        raise ValueError("SETTING.params.std_window must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    low = data_ctx["low"].astype("float64")

    # 威廉下影线及其相对强度（与旧版 compute_one 逐点等价）
    lower_shadow = close - low
    lower_shadow_ma = lower_shadow.rolling(window=std_window, min_periods=std_window).mean()
    lower_shadow_mean = lower_shadow / (lower_shadow_ma + _EPSILON)

    # 按日横截面 zscore；std 为 0 或 NaN 时退化为中心化（同旧版 cs_zscore）
    mean = lower_shadow_mean.mean(axis=1)
    std = lower_shadow_mean.std(axis=1)
    centered = lower_shadow_mean.sub(mean, axis=0)
    valid_std = std.where((std != 0) & std.notna())
    zscore = centered.div(valid_std, axis=0)
    return zscore.where(zscore.notna(), centered)
