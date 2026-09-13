"""成交量切分-主动买入Base 极值因子。

公式（复刻旧版 CSV 管线 ``compute_one`` 的核心计算）：

    tb_base_mom = rolling_sum(taker_buy_base_volume, 10, min_periods=10)
    对每个交易日 t，取最近 window(=20) 天的窗口，
    在窗口内按 volume 降序排序：
        factor = sum(tb_base_mom[成交量 Top10 日]) / sum(tb_base_mom[成交量 Bottom10 日])

即"放量日的主动买入动能之和 / 缩量日的主动买入动能之和"。值越大说明
高成交量日的主动买入相对缩量日越强。分母为 0 时因子记为 NaN（旧版为跳过）。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TYPE = "regular"

META = {
    "factor_name": "Volume_TakerBuyBase_Extreme",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "Top/Bottom volume-day split of taker-buy base momentum",
}

SETTING = {
    "data_needed": ["taker_buy_base_volume", "volume"],
    "universe": "historical_top50",
    "warmup_bars": 29,
    "preprocessing": "mad_rank",
    "params": {"window": 20},
    "factor_direction": 1,
}

# 旧脚本中硬编码的 taker_buy_base 动量窗口
_TB_MOM_WIN = 10
# 窗口内按 volume 排序后取头尾各 10 天
_TOP_K = 10


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回原始因子矩阵（date × instrument）。

    rolling/窗口操作均沿时间轴（axis 0），每个值只依赖当日及历史观测。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2 * _TOP_K:
        raise ValueError(f"SETTING.params.window must be an integer >= {2 * _TOP_K}")

    taker_buy = data_ctx["taker_buy_base_volume"].astype("float64")
    volume = data_ctx["volume"].astype("float64")

    tb_mom = taker_buy.rolling(window=_TB_MOM_WIN, min_periods=_TB_MOM_WIN).sum()

    tb_vals = tb_mom.to_numpy()
    vol_vals = volume.to_numpy()
    n_dates, n_syms = tb_vals.shape
    out = np.full((n_dates, n_syms), np.nan, dtype="float64")

    if n_dates >= window:
        # (n_dates - window + 1, n_syms, window) 滑动窗口（窗口轴在最后）
        vol_win = np.lib.stride_tricks.sliding_window_view(vol_vals, window, axis=0)
        tb_win = np.lib.stride_tricks.sliding_window_view(tb_vals, window, axis=0)
        # 窗口内按 volume 降序排序（NaN 排最后，与旧版 sort_values 一致）
        order = np.argsort(-vol_win, axis=2)
        tb_sorted = np.take_along_axis(tb_win, order, axis=2)
        # nansum 复刻旧版 pandas .sum() 的 skipna 语义
        top_sum = np.nansum(tb_sorted[:, :, :_TOP_K], axis=2)
        bot_sum = np.nansum(tb_sorted[:, :, -_TOP_K:], axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = top_sum / bot_sum
        # 旧版：分母为 0 / NaN / 非有限值时跳过该点
        factor[(bot_sum == 0) | ~np.isfinite(factor)] = np.nan
        out[window - 1:, :] = factor

    return pd.DataFrame(out, index=taker_buy.index, columns=taker_buy.columns)
