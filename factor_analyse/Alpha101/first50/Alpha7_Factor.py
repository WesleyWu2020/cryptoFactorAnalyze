"""Alpha101 Alpha#7 因子（factor_common 契约版，币圈7×24h优化版）。

原始定义:
    Alpha#7 = (adv20 < volume) ? (-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7)) : -1

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（以代码为准）:
    adv         = mean(taker_buy_quote_volume, adv_window)   # 用主动买入金额替代经典 volume
    price_delta = delta(close, delta_window)
    factor      = (adv < taker_buy_quote_volume)
                  ? -ts_rank(abs(price_delta), ts_rank_window) * sign(price_delta)
                  : -1
主动买入放量时按价格变动幅度与方向给反转信号，未放量时默认 -1（看跌）。

参数取原脚本 ``__main__`` 实际调用值: adv_window=14, delta_window=7,
ts_rank_window=49（rebalance_period 仅用于旧版 future_ret，已丢弃；可用性池过滤
由框架 universe="historical_top50" 承担）。
与原脚本的一处差异：adv 尚未形成（前 adv_window-1 天）时本实现输出 NaN 而非
默认 -1，避免 warmup 期混入伪信号。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha7_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": (
        "Alpha101 #7 (crypto): (mean(taker_buy_quote,14) < taker_buy_quote) ? "
        "-ts_rank(abs(delta(close,7)),49)*sign(delta(close,7)) : -1"
    ),
}

SETTING = {
    "data_needed": ["close", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 61,
    "preprocessing": "mad_rank",
    "params": {"adv_window": 14, "delta_window": 7, "ts_rank_window": 49},
    "factor_direction": 1,
}


def _ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series pct rank of the latest value within the window, causal."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: arr.rank(pct=True).iloc[-1], raw=False
    )


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#7 matrix (date x instrument)."""

    adv_window = SETTING["params"]["adv_window"]
    delta_window = SETTING["params"]["delta_window"]
    ts_rank_window = SETTING["params"]["ts_rank_window"]
    for name, value in (
        ("adv_window", adv_window),
        ("delta_window", delta_window),
        ("ts_rank_window", ts_rank_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 1")

    close = data_ctx["close"].astype("float64")
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    # 步骤1: 平均主动买入金额（原脚本用 taker_buy_quote 替代经典 adv20(volume)）
    adv = taker_buy_quote.rolling(window=adv_window, min_periods=adv_window).mean()

    # 步骤2: 价格变动及其幅度、符号
    price_delta = close - close.shift(delta_window)
    price_delta_abs = price_delta.abs()
    price_delta_sign = np.sign(price_delta)

    # 步骤3: 价格变动幅度的时序排名
    price_delta_abs_ts_rank = _ts_rank(price_delta_abs, ts_rank_window)

    # 步骤4: 条件逻辑 —— 主动买入放量给反转信号，否则默认 -1；
    # adv 未形成（warmup 期）输出 NaN
    buy_spike_value = -1.0 * price_delta_abs_ts_rank * price_delta_sign
    factor = buy_spike_value.where(adv < taker_buy_quote, -1.0)
    return factor.where(adv.notna())
