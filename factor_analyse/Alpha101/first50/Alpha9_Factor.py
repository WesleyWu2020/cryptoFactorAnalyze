"""Alpha101 Alpha#9 因子（factor_common 契约版）。

原始定义:
    Alpha#9 = (0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
              ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : -1 * delta(close, 1))

即：强上涨/强下跌趋势中跟随当日收益（动量），震荡市中取反（反转）。
参数取旧脚本 ``__main__`` 实际调用值 trend_window=10。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha9_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #9: trend-conditional momentum/reversal on daily returns",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 15,
    "preprocessing": "mad_rank",
    "params": {"trend_window": 10},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#9 matrix (date x instrument)."""

    trend_window = SETTING["params"]["trend_window"]
    if isinstance(trend_window, bool) or not isinstance(trend_window, int) or trend_window < 1:
        raise ValueError("SETTING.params.trend_window must be an integer >= 1")

    close = data_ctx["close"].astype("float64")
    daily_return = close.pct_change()

    ts_min_ret = daily_return.rolling(window=trend_window, min_periods=trend_window).min()
    ts_max_ret = daily_return.rolling(window=trend_window, min_periods=trend_window).max()

    strong_uptrend = ts_min_ret > 0
    strong_downtrend = ts_max_ret < 0

    # 强趋势跟随（+return），震荡反转（-return）；窗口未形成时为 NaN
    factor = daily_return.where(strong_uptrend | strong_downtrend, -daily_return)
    return factor.where(ts_min_ret.notna() & ts_max_ret.notna())
