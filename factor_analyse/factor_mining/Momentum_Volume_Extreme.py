"""动量-成交量极值分组因子 (Momentum Volume Extreme)。

公式（window=20，半窗 half=window//2=10）：

    momentum_t = close_t / close_{t-window} - 1
    对每个标的、每个交易日 t，取过去 window 天 (t-window+1 .. t)，
    按当日 volume 从大到小排序；
    factor_t = 成交量最大的 half 天的 momentum 之和
             / 成交量最小的 half 天的 momentum 之和

其中对 momentum 求和时跳过 NaN（与旧版 pandas ``.sum()`` 默认 skipna=True
语义一致）；当低成交量半窗动量和为 0 或比值非有限值时输出 NaN。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Momentum_Volume_Extreme",
    "author": "local",
    "level": "daily",
    "category": "momentum",
    "description": "Ratio of momentum summed on top-volume days vs bottom-volume days within a trailing window",
}

SETTING = {
    "data_needed": ["close", "volume"],
    "universe": "historical_top50",
    # momentum 需要 shift(window)，rolling 窗口再回看 window 天，
    # 首个完全有效的因子值在第 2*window-1 行，取保守值 2*window。
    "warmup_bars": 40,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "rebalance_period": 10},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始动量-成交量极值因子矩阵（date × instrument）。

    沿时间轴逐日处理：对每个交易日取过去 ``window`` 天，按成交量降序排序后，
    分别对成交量最高/最低的各 ``window // 2`` 天的动量求和并相除。
    每个值只依赖当前及历史 ``2 * window - 1`` 根 K 线。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.window must be an integer >= 2")
    half = window // 2

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    momentum = (close / close.shift(window) - 1.0).to_numpy()
    vol = volume.to_numpy()

    n_days, n_symbols = momentum.shape
    out = np.full((n_days, n_symbols), np.nan, dtype="float64")

    for i in range(window - 1, n_days):
        m_win = momentum[i - window + 1 : i + 1]  # (window, n_symbols)
        v_win = vol[i - window + 1 : i + 1]
        # 每列（每个标的）按成交量降序排序；-NaN 仍为 NaN，argsort 将其排在最后，
        # 与旧版 pandas sort_values 默认 NaN 置尾一致。
        order = np.argsort(-v_win, axis=0)
        m_sorted = np.take_along_axis(m_win, order, axis=0)
        # nansum 与旧版 pandas .sum() 的 skipna=True 语义一致（全 NaN 得 0.0）。
        top = np.nansum(m_sorted[:half], axis=0)
        bot = np.nansum(m_sorted[-half:], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = top / bot
        ratio[bot == 0] = np.nan
        ratio[~np.isfinite(ratio)] = np.nan
        out[i] = ratio

    return pd.DataFrame(out, index=close.index, columns=close.columns)
