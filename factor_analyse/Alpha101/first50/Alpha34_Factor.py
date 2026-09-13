"""Alpha101 Alpha#34 因子（factor_common 契约版）。

原始定义:
    Alpha#34 = rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))

本实现保留原脚本“币圈7×24h优化版”的实际计算逻辑（参数取旧脚本 __main__
实际调用值 std_short=5, std_long=10, delta_lag=5，而非函数签名默认值 2/5/1）:
    vol_ratio  = stddev(returns, std_short) / stddev(returns, std_long)
    delta_close = delta(close, delta_lag)
    factor = cs_rank((1 - cs_rank(vol_ratio)) + (1 - cs_rank(delta_close)))
公式内部的横截面 rank 予以保留（axis=1 pct rank）；最终去极值与秩归一化
由框架 preprocessing="mad_rank" 完成。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha34_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #34: rank((1-rank(std(ret,5)/std(ret,10))) + (1-rank(delta(close,5))))",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {"std_short": 5, "std_long": 10, "delta_lag": 5},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#34 matrix (date x instrument)."""

    std_short = SETTING["params"]["std_short"]
    std_long = SETTING["params"]["std_long"]
    delta_lag = SETTING["params"]["delta_lag"]

    close = data_ctx["close"].astype("float64")
    returns = close.pct_change()

    # 波动率比值: stddev(returns, std_short) / stddev(returns, std_long)
    std_s = returns.rolling(window=std_short, min_periods=std_short).std()
    std_l = returns.rolling(window=std_long, min_periods=std_long).std()
    vol_ratio = std_s / std_l

    # 价格变化: delta(close, delta_lag)
    delta_close = close - close.shift(delta_lag)

    # 公式内部横截面 rank：两项均倾向小值 → 1 - rank，求和后再做一次横截面 rank
    inv_vol = 1.0 - vol_ratio.rank(axis=1, pct=True)
    inv_delta = 1.0 - delta_close.rank(axis=1, pct=True)
    return (inv_vol + inv_delta).rank(axis=1, pct=True)
