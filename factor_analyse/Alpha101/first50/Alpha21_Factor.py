"""Alpha101 Alpha#21 因子（factor_common 契约版）。

原始定义:
    Alpha#21 =
        if (8日均价 + 8日std < 2日均价): -1
        elif (2日均价 < 8日均价 - 8日std): 1
        elif (volume / adv20 >= 1): 1
        else: -1

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    signal = -1 / +1 的离散信号（价格通道突破 + 成交量确认）
    price_position = (close - channel_lower) / (channel_upper - channel_lower)
    trend_strength = (ma_short - ma_long) / ma_long
    factor = clip(signal * (0.5 + (|price_position - 0.5| + |trend_strength|) * 0.5), -1, 1)

参数取原脚本 __main__ 实际调用值: ma_window=8, std_window=8, short_ma_window=2,
adv_window=20, vol_threshold=1.0。
偏离说明: 原脚本 normalize_discrete_signal 使用 np.random.choice 对过度偏向的
信号做随机降级——该步骤非确定性且属于横截面分布调整，由框架 preprocessing
"mad_rank" 替代，本实现不保留；rebalance_period 只用于旧版 future_ret，丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha21_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #21: 价格通道(8日MA±8日std)突破/成交量确认的离散信号，乘以价格位置与趋势强度",
}

SETTING = {
    "data_needed": ["close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 25,
    "preprocessing": "mad_rank",
    "params": {
        "ma_window": 8,
        "std_window": 8,
        "short_ma_window": 2,
        "adv_window": 20,
        "vol_threshold": 1.0,
    },
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#21 matrix (date x instrument)."""

    ma_window = SETTING["params"]["ma_window"]
    std_window = SETTING["params"]["std_window"]
    short_ma_window = SETTING["params"]["short_ma_window"]
    adv_window = SETTING["params"]["adv_window"]
    vol_threshold = SETTING["params"]["vol_threshold"]

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 移动平均线、波动性与成交量活跃度
    ma_long = close.rolling(window=ma_window, min_periods=ma_window).mean()
    ma_short = close.rolling(window=short_ma_window, min_periods=short_ma_window).mean()
    std_long = close.rolling(window=std_window, min_periods=std_window).std()
    adv = volume.rolling(window=adv_window, min_periods=adv_window).mean()
    vol_ratio = volume / (adv + _EPSILON)

    # 价格通道
    channel_upper = ma_long + std_long
    channel_lower = ma_long - std_long

    # Alpha#21 核心离散信号（逐分支与旧脚本行级逻辑等价）
    cond_break_upper = channel_upper < ma_short          # 短期均线突破上轨 -> -1
    cond_break_lower = ma_short < channel_lower          # 短期均线跌破下轨 -> +1
    cond_vol_active = vol_ratio >= vol_threshold         # 成交量活跃 -> +1
    bullish = cond_break_lower | (cond_vol_active & ~cond_break_upper)
    signal = bullish.astype(float) * 2.0 - 1.0

    # 数据未就绪时输出 NaN（与旧脚本行级 isna 检查等价）
    valid = (
        ma_long.notna()
        & std_long.notna()
        & ma_short.notna()
        & adv.notna()
        & vol_ratio.notna()
    )
    signal = signal.where(valid)

    # 离散信号转连续因子值：价格位置 + 趋势强度调整
    price_position = (close - channel_lower) / (channel_upper - channel_lower + _EPSILON)
    trend_strength = (ma_short - ma_long) / (ma_long + _EPSILON)
    signal_strength = (price_position - 0.5).abs() + trend_strength.abs()

    factor = signal * (0.5 + signal_strength * 0.5)
    return factor.clip(-1.0, 1.0)
