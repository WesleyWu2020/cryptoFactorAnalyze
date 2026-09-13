"""crypto_alpha_ppo_10_0_20260608_ensemble 因子（factor_common 日频契约版）。

原始定义（AlphaGen PPO ensemble，pool: crypto_10_0_20260608174731_ppo）：
    8 个日频 alphagen 表达式的加权集成：
      f1: -0.00323 * Mean(Sign($high), 10d)
      f2: +0.00094 * Rank(Sub($vwap, 0.01), 20d)     （常数平移在 Rank 内无关）
      f3: +0.00619 * EMA(Sign($open), 40d)           （最大权重）
      f4: +0.00047 * Rank(Add(Sum($open, 5d), -10), 5d)
      f5: +0.00092 * Med(Rank(CSZscore($open_interest), 5d), 5d)
      f6: -0.00063 * Var(Sign($funding), 10d)
      f7: -0.00167 * Rank(Sum($low, 5d), 40d)
      f8: -0.00038 * Rank(Add($mark, -0.01), 40d)

分钟->日频改写判断（记录在案）：
  - 原实现对 1m 数据 groupby(day) 聚合后做日频运算；日频契约下聚合恒等，直接用日频字段。
  - vwap：原实现为日内最后一根分钟 VWAP；日频改写为 日VWAP = quote_volume / volume。
  - mark：原实现缺失时回退 close；日频契约无 mark 字段，直接用 close。
  - funding：原实现取每日 last；日频契约为日均 funding rate（mean），判断为等价意图。
  - open_interest：日频契约不支持该字段，f5（权重 0.00092，占总绝对权重约 6%）无法计算，
    予以省略并在文档中声明；集成权重未归一化，省略该项只相当于整体平移/缩放。
  - 原实现的流式 H5 分块加载（streaming/chunking）纯为内存控制，日频数据量小，全部删除。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_ppo_10_0_20260608_ensemble",
    "author": "wesleywu",
    "level": "daily",
    "category": "momentum",
    "description": "加权集成 8 个日频 PPO/alphagen 表达式（open/high/low/vwap/funding/mark）",
}

SETTING = {
    "data_needed": ["open", "high", "low", "close", "volume", "quote_volume", "funding"],
    "universe": "historical_top50",
    # 最长链式窗口：f7 = Sum(low, 5d) 内嵌 Rank(.., 40d) -> 44 天，加缓冲取 55
    "warmup_bars": 55,
    "preprocessing": "mad_rank",
    "params": {
        "mean_sign_high_window": 10,
        "rank_vwap_window": 20,
        "ema_sign_open_window": 40,
        "sum_open_window": 5,
        "rank_sum_open_window": 5,
        "var_sign_funding_window": 10,
        "sum_low_window": 5,
        "rank_sum_low_window": 40,
        "rank_mark_window": 40,
        "w1": -0.0032305300149235933,
        "w2": 0.0009442490772445488,
        "w3": 0.006190342932286082,
        "w4": 0.00046536619582840356,
        "w6": -0.0006257742215931986,
        "w7": -0.001671611565855515,
        "w8": -0.00037885012546794996,
    },
    "factor_direction": 1,  # 原实现 ensemble_direction = 1.0
}

_EPS = 1e-8


def _rolling_rank_ppo(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """AlphaGen Rank(x, window)：当前值在窗口内的降序百分位（最低值 -> 接近 1）。"""

    def _rank_last(values: np.ndarray) -> float:
        last = values[-1]
        if not np.isfinite(last) or not np.isfinite(values).all():
            return np.nan
        left = np.count_nonzero(last < values)
        right = np.count_nonzero(last <= values)
        return float((right + left + (right > left)) / (2.0 * len(values)))

    return frame.rolling(window, min_periods=window).apply(_rank_last, raw=True)


def _rolling_ema_ppo(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """AlphaGen EMA(x, window)：固定窗口加权平均（非递归 ewm）。"""
    alpha = 1.0 - 2.0 / (1.0 + float(window))
    power = np.arange(window, 0, -1, dtype=np.float64)
    weights = alpha ** power
    weights = weights / weights.sum()

    def _ema(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        return float(np.dot(values, weights))

    return frame.rolling(window, min_periods=window).apply(_ema, raw=True)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily PPO-ensemble matrix (date x instrument)."""

    p = SETTING["params"]

    open_ = data_ctx["open"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")
    funding = data_ctx["funding"].astype("float64").fillna(0.0)

    # 日 VWAP；mark 字段不可用，回退 close（与原实现一致）
    vwap = (quote_volume / (volume + _EPS)).where(volume > 0)
    mark = close

    ensemble = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    # f1: Mean(Sign($high), 10d)
    ensemble += p["w1"] * np.sign(high).rolling(
        p["mean_sign_high_window"], min_periods=p["mean_sign_high_window"]
    ).mean()

    # f2: Rank(Sub($vwap, 0.01), 20d) — 常数平移在 Rank 内无关
    ensemble += p["w2"] * _rolling_rank_ppo(vwap, p["rank_vwap_window"])

    # f3: EMA(Sign($open), 40d)
    ensemble += p["w3"] * _rolling_ema_ppo(np.sign(open_), p["ema_sign_open_window"])

    # f4: Rank(Add(Sum($open, 5d), -10), 5d) — 常数平移无关
    sum_open = open_.rolling(p["sum_open_window"], min_periods=p["sum_open_window"]).sum()
    ensemble += p["w4"] * _rolling_rank_ppo(sum_open, p["rank_sum_open_window"])

    # f5: Med(Rank(CSZscore($open_interest), 5d), 5d) — 日频契约无 open_interest，省略
    #     （原权重 w5=+0.00092，占总绝对权重约 6%；见模块 docstring）。

    # f6: Var(Sign($funding), 10d)
    ensemble += p["w6"] * np.sign(funding).rolling(
        p["var_sign_funding_window"], min_periods=p["var_sign_funding_window"]
    ).var(ddof=1)

    # f7: Rank(Sum($low, 5d), 40d)
    sum_low = low.rolling(p["sum_low_window"], min_periods=p["sum_low_window"]).sum()
    ensemble += p["w7"] * _rolling_rank_ppo(sum_low, p["rank_sum_low_window"])

    # f8: Rank(Add($mark, -0.01), 40d) — 常数平移无关
    ensemble += p["w8"] * _rolling_rank_ppo(mark, p["rank_mark_window"])

    return ensemble.replace([np.inf, -np.inf], np.nan)
