"""收益率偏度反转因子（Return Skew Reversal）。

公式（与旧版 compute_one 逐点等价）：

    ret = close.pct_change(1)                     # 日收益率
    ret_skew = ret.rolling(window=20, min_periods=20).skew()
    factor = -1.0 * ret_skew                      # 偏度反转：偏度越低越好

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口；
输出的原始因子值由框架统一做截面 MAD 去极值与按日 rank（preprocessing="mad_rank"）。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Return_Skew_Reversal_Factor",
    "author": "local",
    "level": "daily",
    "category": "momentum",
    "description": "N-day return skewness reversal: factor = -TS_SKEW(PCT_CHANGE(close, 1), window)",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 21,
    "preprocessing": "mad_rank",
    "params": {"window": 20},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始收益率偏度反转矩阵（date × instrument）。

    rolling/shift 均沿时间轴（axis 0）：日收益需 1 期 shift，
    skew 需 window 个有效收益（min_periods=window，与旧版默认一致），
    因此第 1 个有效因子值出现在第 window+1 行。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.window must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    ret = close.pct_change(1)
    ret_skew = ret.rolling(window=window, min_periods=window).skew()
    return -1.0 * ret_skew
