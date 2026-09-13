"""滚动相关-主动买额因子（Rolling_Corr_BuyQuote_Factor）。

公式（逐点等价旧版 CSV 管线的 compute_one/calc_factor 核心计算）：

    ret       = log(close).diff()                    # 日对数收益
    vol_mean  = rolling_mean(volume, window)         # 窗口内成交量均值
    tbq_mean  = rolling_mean(taker_buy_quote_volume, window)
    corr_rbq  = rolling_corr(ret, tbq_mean, window)  # ret 与买额均值的滚动相关
    factor    = corr_rbq * log1p(vol_mean)

经济含义：若主动买单成交额与价格同步上升且出现在高流动性阶段，因子为正，
代表买方力量强。只使用当日及历史数据，无未来函数；rolling 沿时间轴，
未指定 min_periods 时默认等于 window（与旧版行为一致）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Rolling_Corr_BuyQuote_Factor",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "window-day rolling corr of log-return vs smoothed taker buy quote volume, scaled by log mean volume",
}

SETTING = {
    "data_needed": ["close", "volume", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    # tbq_mean 自身需 window 根 K 线才有值，corr 又需 window 个有效对，
    # 实际首个有效因子值出现在第 2*window-1 根（旧版行为相同）。
    "warmup_bars": 39,
    "preprocessing": "mad_rank",
    "params": {"window": 20},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始因子矩阵（date × instrument）。

    rolling 沿时间轴（axis 0），每个因子值只依赖当前及之前 window 根
    K 线；旧版输出阶段的按日 rank 归一化已由框架 preprocessing 接管。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.window must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")
    tbq = data_ctx["taker_buy_quote_volume"].astype("float64")

    ret = np.log(close).diff()
    vol_mean = volume.rolling(window).mean()
    tbq_mean = tbq.rolling(window).mean()
    corr_rbq = ret.rolling(window).corr(tbq_mean)
    return corr_rbq * np.log1p(vol_mean)
