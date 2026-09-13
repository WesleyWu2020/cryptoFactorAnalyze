"""Alpha101 Alpha#50 因子（factor_common 契约版）。

原始定义:
    Alpha#50 = (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))

本实现按上述经典公式计算:
    price_base = mean(close, 20)（min_periods=1，滚动归一化基准）
    vwap = quote_volume / volume（仓库矩阵版约定，volume>0 掩码）
    norm_vwap = vwap / price_base
    corr = correlation(rank_cs(volume), rank_cs(norm_vwap), corr_window)（时间序列相关）
    corr_rank = rank_cs(corr)（按日横截面 pct rank）
    factor = -1 * ts_max(corr_rank, ts_max_window)
    其中 rank_cs 为按日横截面 pct rank（公式内部的 rank，予以保留）。

与旧脚本的重要偏离说明: 旧脚本 compute_one 中 volume_vwap_corr 初始化为
NaN 后从未真正计算（时间序列 correlation 一步缺失），导致 corr_rank /
ts_max / 因子值恒为 NaN、输出为空——属于实现 bug 而非有意语义。此处按
docstring 中的经典公式补全该链路。另旧脚本 vwap 用 (high+low+close)/3
近似，按仓库约定改为 quote_volume / volume。参数取原脚本 __main__ 实际
调用值 corr_window=10, ts_max_window=10；旧版 rebalance_period 仅用于
future_ret，已丢弃。factor_direction=1：公式已内置 -1，因子值越高对应
量-价相关性排名越低（未过热），与“高值看多”一致。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha50_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #50: -ts_max(rank(correlation(rank(volume), rank(vwap), N)), N)",
}

SETTING = {
    "data_needed": ["close", "volume", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"corr_window": 10, "ts_max_window": 10, "norm_window": 20},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#50 matrix (date x instrument)."""

    corr_window = SETTING["params"]["corr_window"]
    ts_max_window = SETTING["params"]["ts_max_window"]
    norm_window = SETTING["params"]["norm_window"]
    for name, value in (
        ("corr_window", corr_window),
        ("ts_max_window", ts_max_window),
        ("norm_window", norm_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")

    # 价格归一化（适应币圈价格量级差异），与原脚本一致 min_periods=1
    price_base = close.rolling(window=norm_window, min_periods=1).mean()

    # vwap 按仓库约定 = quote_volume / volume，仅在有成交的日子可观测
    observed = volume > 0
    vwap = (quote_volume / (volume + _EPSILON)).where(observed)
    norm_vwap = vwap / (price_base + _EPSILON)

    # 公式内部的横截面 rank（保留）
    volume_rank = volume.rank(axis=1, pct=True)
    vwap_rank = norm_vwap.rank(axis=1, pct=True)

    # 量-价排名的时间序列相关性（每个 instrument 各自沿时间轴）
    volume_vwap_corr = volume_rank.rolling(
        window=corr_window, min_periods=corr_window
    ).corr(vwap_rank)

    # 相关性的横截面排名，再取时间序列最大值，反向
    corr_rank = volume_vwap_corr.rank(axis=1, pct=True)
    ts_max_corr_rank = corr_rank.rolling(
        window=ts_max_window, min_periods=ts_max_window
    ).max()
    return -1.0 * ts_max_corr_rank
