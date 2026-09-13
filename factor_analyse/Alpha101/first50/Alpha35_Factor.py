"""Alpha101 Alpha#35 因子（factor_common 契约版）。

原始定义:
    Alpha#35 = ((Ts_Rank(taker_buy_quote, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（参数取旧脚本 __main__
实际调用值 vol_window=30, price_window=10, ret_window=10，而非签名默认值 32/16/32）:
    price_strength_rel = ((close + high) - low) / close      # 旧脚本做了无量纲化
    factor = Ts_Rank(taker_buy_quote_volume, vol_window)
             * (1 - Ts_Rank(price_strength_rel, price_window))
             * (1 - Ts_Rank(returns, ret_window))
其中 returns = close.pct_change()；Ts_Rank 为时序百分位排名（窗口内当前值的
rank pct），全部只使用当日及历史数据，无未来函数。

FactorManager 只调用下面的标准模块接口；横截面去极值与秩归一化由框架
preprocessing="mad_rank" 完成，这里输出原始因子值。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha35_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #35: tsrank(taker_buy_quote,30) * (1-tsrank(price_strength,10)) * (1-tsrank(returns,10))",
}

SETTING = {
    "data_needed": ["close", "high", "low", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 40,
    "preprocessing": "mad_rank",
    "params": {"vol_window": 30, "price_window": 10, "ret_window": 10},
    "factor_direction": 1,
}

_EPSILON = 1e-8


def _ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series percentile rank of the current value within the window."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: pd.Series(arr).rank(pct=True).iloc[-1], raw=False
    )


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#35 matrix (date x instrument)."""

    vol_window = SETTING["params"]["vol_window"]
    price_window = SETTING["params"]["price_window"]
    ret_window = SETTING["params"]["ret_window"]

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    taker_buy_quote = data_ctx["taker_buy_quote_volume"].astype("float64")

    returns = close.pct_change()

    # 价格强度（旧脚本无量纲化版本）: ((close + high) - low) / close
    price_strength_rel = ((close + high) - low) / (close + _EPSILON)

    tsrank_taker_buy = _ts_rank(taker_buy_quote, vol_window)
    tsrank_price = _ts_rank(price_strength_rel, price_window)
    tsrank_ret = _ts_rank(returns, ret_window)

    return tsrank_taker_buy * (1.0 - tsrank_price) * (1.0 - tsrank_ret)
