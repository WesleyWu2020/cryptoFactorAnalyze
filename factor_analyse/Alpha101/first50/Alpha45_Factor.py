"""Alpha101 Alpha#45 因子（factor_common 契约版）。

原始定义:
    Alpha#45 = (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2))
               * rank(correlation(sum(close, 5), sum(close, 20), 2))))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑:
    price_base = mean(close, 20)（min_periods=1）
    norm_close = close / price_base
    mid_term_avg = mean(delay(norm_close, delay_days), mid_window)
    vol_norm = volume / mean(volume, 20)（min_periods=1）
    price_vol_corr = correlation(norm_close, vol_norm, short_corr_window)
    short_sum = sum(norm_close, short_sum_window); mid_sum = sum(norm_close, mid_window)
    period_corr = correlation(short_sum, mid_sum, short_corr_window)
    factor = -rank(mid_term_avg) * price_vol_corr * rank(period_corr)
其中 rank 为横截面 pct rank（axis=1）。
偏离说明: 旧脚本对 period_corr 先 dropna 再在压缩后的序列上滑窗（窗口可跨越日历缺口），
本实现直接在原始日历轴上做 rolling corr，更贴近标准语义且无前视。
参数取原脚本 __main__ 实际调用值: delay=5, mid=20, short_corr=2, short_sum=5。
旧脚本 rebalance_period 仅用于旧版 future_ret，已丢弃。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha45_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #45: -rank(mean(delay(close,N),N)) * corr(close, volume, N) * rank(corr(sum(close,N), sum(close,N), N))",
}

SETTING = {
    "data_needed": ["close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {
        "delay_days": 5,
        "mid_window": 20,
        "short_corr_window": 2,
        "short_sum_window": 5,
    },
    "factor_direction": 1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#45 matrix (date x instrument)."""

    delay_days = SETTING["params"]["delay_days"]
    mid_window = SETTING["params"]["mid_window"]
    short_corr_window = SETTING["params"]["short_corr_window"]
    short_sum_window = SETTING["params"]["short_sum_window"]

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 价格归一化（适应币圈价格差异），与原脚本一致 min_periods=1
    price_base = close.rolling(window=20, min_periods=1).mean()
    norm_close = close / (price_base + _EPSILON)

    # 中期价格趋势基准: 从 delay_days 天前开始的 mid_window 天收盘价均值
    delay_close = norm_close.shift(delay_days)
    mid_term_avg = delay_close.rolling(window=mid_window, min_periods=mid_window).mean()

    # 短期量价相关性: 过去 short_corr_window 天收盘价与归一化交易量的相关性
    vol_norm = volume / (volume.rolling(window=20, min_periods=1).mean() + _EPSILON)
    price_vol_corr = norm_close.rolling(window=short_corr_window, min_periods=short_corr_window).corr(
        vol_norm
    )

    # 多周期价格相关性: 短期与中期收盘价总和的相关性
    short_sum = norm_close.rolling(window=short_sum_window, min_periods=short_sum_window).sum()
    mid_sum = norm_close.rolling(window=mid_window, min_periods=mid_window).sum()
    period_corr = short_sum.rolling(window=short_corr_window, min_periods=short_corr_window).corr(
        mid_sum
    )

    # 横截面排名合成，取负
    rank_mid_term_avg = mid_term_avg.rank(axis=1, pct=True)
    rank_period_corr = period_corr.rank(axis=1, pct=True)
    return -1.0 * rank_mid_term_avg * price_vol_corr * rank_period_corr
