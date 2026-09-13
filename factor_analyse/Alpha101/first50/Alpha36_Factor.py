"""Alpha101 Alpha#36 因子（factor_common 契约版）。

原始定义:
    Alpha#36 = ((((2.21 * rank(correlation((close - open), delay(volume, 1), 15)))
        + (0.7 * rank((open - close))))
        + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5))))
        + rank(abs(correlation(vwap, adv20, 6))))
        + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open))))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（与经典公式差异较大，
以代码实现为准；参数取旧脚本 __main__ 策略1 实际调用值:
corr_window=10, ts_rank_window=5, delay_lag=5, vwap_adv_corr=5, long_ma_window=50）:

    1. 自适应价格基准: 按 20 日波动率(=std/mean)在 20/60/120 日均线间选择
       (>0.1 高波动用短期 20 日基准, <0.05 低波动用长期 120 日基准, 否则 60 日;
       NaN 时用 60 日基准)，对 open/close/vwap 做归一化。
    2. 自适应成交量基准: 同一波动率规则在 10/30/60 日均量间选择，归一化 volume。
    3. vwap 按旧脚本显式定义为 (high + low + close) / 3（不用 quote_volume/volume）。
    4. 收益率用对数形式: log_returns = log(norm_close / delay(norm_close,1)),
       log_intraday_ret = log(norm_close / norm_open)。
    5. 五个组件（各自做横截面 pct rank 后按权重 2.21/0.7/0.73/1.0/0.6 相加）:
       - corr(log_intraday_ret, delay(normalized_volume, 1), corr_window)
       - -log_intraday_ret（日内反转）
       - Ts_Rank(delay(-log_returns, delay_lag), ts_rank_window)
       - abs(corr(normalized_vwap, adv20(normalized_volume), vwap_adv_corr))
       - (mean(norm_close, long_ma_window) - norm_open) * (norm_close - norm_open)
    价格/成交量基准均线均用 min_periods=1（与旧脚本一致），组件窗口用完整窗口。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha36_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #36 (7x24 优化版): 自适应归一化多组件加权 rank 复合因子",
}

SETTING = {
    "data_needed": ["open", "high", "low", "close", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 60,
    "preprocessing": "mad_rank",
    "params": {
        "corr_window": 10,
        "ts_rank_window": 5,
        "delay_lag": 5,
        "vwap_adv_corr": 5,
        "long_ma_window": 50,
    },
    "factor_direction": 1,
}

_EPSILON = 1e-8
# 自适应基准内部参数（旧脚本硬编码）
_VOL_WINDOW = 20
_VOL_HIGH = 0.1
_VOL_LOW = 0.05


def _ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series percentile rank of the current value within the window."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: pd.Series(arr).rank(pct=True).iloc[-1], raw=False
    )


def _adaptive_base(
    base_short: pd.DataFrame,
    base_medium: pd.DataFrame,
    base_long: pd.DataFrame,
    volatility: pd.DataFrame,
) -> pd.DataFrame:
    """Select base by volatility: high vol -> short, low vol -> long, else medium."""
    base = base_medium.mask(volatility > _VOL_HIGH, base_short)
    return base.mask(volatility < _VOL_LOW, base_long)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#36 matrix (date x instrument)."""

    corr_window = SETTING["params"]["corr_window"]
    ts_rank_window = SETTING["params"]["ts_rank_window"]
    delay_lag = SETTING["params"]["delay_lag"]
    vwap_adv_corr = SETTING["params"]["vwap_adv_corr"]
    long_ma_window = SETTING["params"]["long_ma_window"]

    open_ = data_ctx["open"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    # 1. 自适应价格基准（20/60/120 日均线，按 20 日相对波动率选择）
    price_base_short = close.rolling(window=20, min_periods=1).mean()
    price_base_medium = close.rolling(window=60, min_periods=1).mean()
    price_base_long = close.rolling(window=120, min_periods=1).mean()
    price_volatility = close.rolling(window=_VOL_WINDOW).std() / close.rolling(window=_VOL_WINDOW).mean()
    price_base = _adaptive_base(price_base_short, price_base_medium, price_base_long, price_volatility)

    normalized_open = open_ / (price_base + _EPSILON)
    normalized_close = close / (price_base + _EPSILON)
    # 旧脚本显式定义 vwap = (high + low + close) / 3
    vwap = (high + low + close) / 3.0
    normalized_vwap = vwap / (price_base + _EPSILON)

    # 2. 自适应成交量基准（10/30/60 日均量，同一波动率规则）
    volume_ma_short = volume.rolling(window=10, min_periods=1).mean()
    volume_ma_medium = volume.rolling(window=30, min_periods=1).mean()
    volume_ma_long = volume.rolling(window=60, min_periods=1).mean()
    volume_base = _adaptive_base(volume_ma_short, volume_ma_medium, volume_ma_long, price_volatility)
    normalized_volume = volume / (volume_base + _EPSILON)

    # 3. 对数收益率
    log_returns = np.log(normalized_close / normalized_close.shift(1))
    log_intraday_ret = np.log(normalized_close / normalized_open)

    # 4. 五个组件
    # 组件1: corr(日内对数收益, delay(归一化成交量, 1), corr_window)，权重 2.21
    volume_delay = normalized_volume.shift(1)
    corr_intraday_volume = log_intraday_ret.rolling(
        window=corr_window, min_periods=corr_window
    ).corr(volume_delay)

    # 组件2: 日内反转 -log_intraday_ret，权重 0.7
    intraday_reversal = -log_intraday_ret

    # 组件3: Ts_Rank(delay(-log_returns, delay_lag), ts_rank_window)，权重 0.73
    neg_returns_delay = (-log_returns).shift(delay_lag)
    ts_rank_neg_returns = _ts_rank(neg_returns_delay, ts_rank_window)

    # 组件4: abs(corr(normalized_vwap, adv20(normalized_volume), vwap_adv_corr))，权重 1.0
    adv20 = normalized_volume.rolling(window=20, min_periods=20).mean()
    abs_corr_vwap_adv20 = normalized_vwap.rolling(
        window=vwap_adv_corr, min_periods=vwap_adv_corr
    ).corr(adv20).abs()

    # 组件5: (mean(norm_close, long_ma) - norm_open) * (norm_close - norm_open)，权重 0.6
    long_ma = normalized_close.rolling(window=long_ma_window, min_periods=long_ma_window).mean()
    trend_intraday = (long_ma - normalized_open) * (normalized_close - normalized_open)

    # 5. 公式内部横截面 rank 后按权重组合
    return (
        2.21 * corr_intraday_volume.rank(axis=1, pct=True)
        + 0.7 * intraday_reversal.rank(axis=1, pct=True)
        + 0.73 * ts_rank_neg_returns.rank(axis=1, pct=True)
        + 1.0 * abs_corr_vwap_adv20.rank(axis=1, pct=True)
        + 0.6 * trend_intraday.rank(axis=1, pct=True)
    )
