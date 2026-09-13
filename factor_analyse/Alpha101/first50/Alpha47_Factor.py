"""Alpha101 Alpha#47 因子（factor_common 契约版）。

原始定义:
    Alpha#47 = ((((rank((1 / close)) * volume) / adv20)
                 * ((high * rank((high - close))) / (sum(high, 5) / 5)))
                - rank((vwap - delay(vwap, 5))))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    price_base = mean(close, 20)（min_periods=1，滚动归一化基准）
    norm_close = close / price_base, norm_high = high / price_base
    adv20 = mean(volume, 20)，vol_ratio = volume / adv20（注意是量，不是成交额）
    high_close_diff = norm_high - norm_close（冲高回落）
    high_avg_5 = sum(norm_high, 5) / 5
    vwap_change = norm_vwap - delay(norm_vwap, 5)
    factor = (rank_cs(1/norm_close) * vol_ratio)
             * (norm_high * rank_cs(high_close_diff) / high_avg_5)
             - rank_cs(vwap_change)
    其中 rank_cs 为按日横截面 pct rank（公式内部的 rank，予以保留）。

与旧脚本的一处偏离: 旧脚本在无 vwap 字段时用 (high+low+close)/3 近似；
按仓库矩阵版算子约定，vwap 统一为 quote_volume / volume（分母加 1e-8 并用
volume>0 掩码），再除以 price_base 得到 norm_vwap。参数取原脚本 __main__
实际调用值 adv_window=20, high_sum_window=5, vwap_delay=5；旧版
rebalance_period 仅用于 future_ret，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha47_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #47: low-price volume spike * intraday fade - vwap trend rank",
}

SETTING = {
    "data_needed": ["close", "high", "volume", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {
        "adv_window": 20,
        "high_sum_window": 5,
        "vwap_delay": 5,
        "norm_window": 20,
    },
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#47 matrix (date x instrument)."""

    adv_window = SETTING["params"]["adv_window"]
    high_sum_window = SETTING["params"]["high_sum_window"]
    vwap_delay = SETTING["params"]["vwap_delay"]
    norm_window = SETTING["params"]["norm_window"]
    for name, value in (
        ("adv_window", adv_window),
        ("high_sum_window", high_sum_window),
        ("vwap_delay", vwap_delay),
        ("norm_window", norm_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 1")

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    volume = data_ctx["volume"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")

    # 价格归一化（适应币圈价格量级差异），与原脚本一致 min_periods=1
    price_base = close.rolling(window=norm_window, min_periods=1).mean()
    norm_close = close / (price_base + _EPSILON)
    norm_high = high / (price_base + _EPSILON)

    # vwap 按仓库约定 = quote_volume / volume，仅在有成交的日子可观测
    observed = volume > 0
    vwap = (quote_volume / (volume + _EPSILON)).where(observed)
    norm_vwap = vwap / (price_base + _EPSILON)

    # 低价股识别与成交量异常（adv 基于 volume，与原脚本一致）
    inv_close = 1.0 / (norm_close + _EPSILON)
    adv = volume.rolling(window=adv_window, min_periods=adv_window).mean()
    vol_ratio = volume / (adv + _EPSILON)

    # 日内价格强度: 冲高回落，除以 high_sum_window 日最高价均值标准化
    high_close_diff = norm_high - norm_close
    high_avg = (
        norm_high.rolling(window=high_sum_window, min_periods=high_sum_window).sum()
        / high_sum_window
    )

    # VWAP 中期变化
    vwap_change = norm_vwap - norm_vwap.shift(vwap_delay)

    # 公式内部的横截面 rank（保留）
    rank_inv_close = inv_close.rank(axis=1, pct=True)
    rank_high_close_diff = high_close_diff.rank(axis=1, pct=True)
    rank_vwap_change = vwap_change.rank(axis=1, pct=True)

    return (rank_inv_close * vol_ratio) * (
        (norm_high * rank_high_close_diff) / (high_avg + _EPSILON)
    ) - rank_vwap_change
