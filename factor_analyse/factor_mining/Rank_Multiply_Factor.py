"""Rank 组合因子：量稳（负向转正向）+ 波动效率，等权 0.5/0.5 组合截面 rank。

公式（window 来自 SETTING["params"]）：

    average_trade_size   = quote_volume / (trade_count + eps)
    volume_stability     = rolling_mean(average_trade_size) / (rolling_std(average_trade_size) + eps)
    roc                  = close / close.shift(window) - 1
    range_pct            = rolling_mean((high - low) / (close.shift(1) + eps))
    volatility_efficiency = roc / (range_pct + eps)
    factor               = w_vs * (1 - rank_cs(volume_stability)) + w_ve * rank_cs(volatility_efficiency)

其中 rank_cs 为按日截面 rank（pct=True）。

与旧版的偏差说明：旧版用全样本 future_ret 计算历史 IC 并据此动态分配两个子因子的
权重，这在新框架下属于未来函数（全样本统计量），且 ``calc_factor`` 拿不到标签。
因此改为文档化的固定等权回退（0.5 / 0.5），组合结构与截面 rank 部分保持不变。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Rank_Multiply_Factor",
    "author": "local",
    "level": "daily",
    "category": "composite",
    "description": "等权组合量稳（取负）与波动效率的截面 rank 因子",
}

SETTING = {
    "data_needed": ["quote_volume", "trade_count", "close", "high", "low"],
    "universe": "historical_top50",
    # roc 需 close.shift(window)，range_pct 需 close.shift(1) 后再 rolling(window)，
    # 最长回看 window + 1，取保守值保证 start 处因子值已有效。
    "warmup_bars": 21,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "volstab_weight": 0.5, "vol_eff_weight": 0.5},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回等权 rank 组合原始因子矩阵（行=日期，列=标的）。

    rolling/shift 沿时间轴（axis 0），截面 rank 沿列轴（axis 1, pct=True），
    每个值只依赖当日及历史数据。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.window must be an integer >= 2")
    volstab_weight = SETTING["params"]["volstab_weight"]
    vol_eff_weight = SETTING["params"]["vol_eff_weight"]

    quote_volume = data_ctx["quote_volume"].astype("float64")
    trade_count = data_ctx["trade_count"].astype("float64")
    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")

    # === Volume Stability 子因子（负向：值越小越好） ===
    turnover = quote_volume / (trade_count + _EPSILON)
    turnover_mean = turnover.rolling(window=window, min_periods=window).mean()
    turnover_std = turnover.rolling(window=window, min_periods=window).std()
    volume_stability = turnover_mean / (turnover_std + _EPSILON)

    # === Volatility Efficiency 子因子（正向：值越大越好） ===
    roc = close / close.shift(window) - 1
    range_pct = ((high - low) / (close.shift(1) + _EPSILON)).rolling(
        window=window, min_periods=window
    ).mean()
    volatility_efficiency = roc / (range_pct + _EPSILON)

    # === 截面 rank 组合（等权 0.5/0.5 回退，替代旧版全样本 IC 权重） ===
    volstab_rank = volume_stability.rank(axis=1, pct=True)
    vol_eff_rank = volatility_efficiency.rank(axis=1, pct=True)

    return volstab_weight * (1 - volstab_rank) + vol_eff_weight * vol_eff_rank
