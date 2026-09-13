"""理想振幅因子 (Ideal Amplitude Factor)。

公式（逐日滚动窗口，只使用当日及历史数据，无未来函数）：

    amplitude = high / (low + eps) - 1
    对每个 window 日窗口，按收盘价从高到低排序：
      vhigh = 收盘价最高的 high_pct 比例交易日的振幅均值
      vlow  = 收盘价最低的 low_pct 比例交易日的振幅均值
    factor = vhigh - vlow

高价区振幅相对低价区振幅越大，说明上涨伴随剧烈波动，通常预示未来
收益偏弱，故 factor_direction 取 -1（反向因子）。

新框架入口是 ``calc_factor(data_ctx)``：输入 date(升序) × instrument 的
矩阵字典，返回同形状原始因子矩阵，截面去极值与 rank 归一化由框架的
``mad_rank`` 预处理完成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Ideal_Amplitude_Factor",
    "author": "local",
    "level": "daily",
    "category": "volatility",
    "description": "理想振幅：高价区振幅均值与低价区振幅均值之差（反向）",
}

SETTING = {
    "data_needed": ["high", "low", "close"],
    "universe": "historical_top50",
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {"window": 20, "high_pct": 0.2, "low_pct": 0.2},
    "factor_direction": -1,
}

_EPSILON = 1e-8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回理想振幅原始因子矩阵（date × instrument）。

    与旧版 ``compute_one`` 逐点等价：每个完整 window 日窗口内按收盘价
    降序排序，取头部 ``max(1, int(window*high_pct))`` 天与尾部
    ``max(1, int(window*low_pct))`` 天的振幅均值之差；窗口不满 window 天
    时输出 NaN（等价旧版 rolling 未指定 min_periods 的默认行为）。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.window must be an integer >= 2")
    high_pct = float(SETTING["params"]["high_pct"])
    low_pct = float(SETTING["params"]["low_pct"])
    if not (0 < high_pct <= 1) or not (0 < low_pct <= 1):
        raise ValueError("high_pct/low_pct must be in (0, 1]")

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")

    amplitude = high / (low + _EPSILON) - 1

    n_high = max(1, int(window * high_pct))
    n_low = max(1, int(window * low_pct))

    close_values = close.to_numpy()
    amplitude_values = amplitude.to_numpy()
    n_rows, n_cols = close_values.shape
    if n_rows < window:
        return pd.DataFrame(
            np.nan, index=close.index, columns=close.columns, dtype="float64"
        )

    # 滑动窗口视图: (n_rows-window+1, n_cols, window)
    close_windows = np.lib.stride_tricks.sliding_window_view(
        close_values, window, axis=0
    )
    amplitude_windows = np.lib.stride_tricks.sliding_window_view(
        amplitude_values, window, axis=0
    )

    # 窗口内按收盘价降序排序振幅；argsort 为稳定升序，负号实现降序，
    # 与旧版 sort_values('close', ascending=False) 的 head/tail 取法一致。
    order = np.argsort(-close_windows, axis=2, kind="stable")
    sorted_amplitude = np.take_along_axis(amplitude_windows, order, axis=2)

    vhigh = np.nanmean(sorted_amplitude[..., :n_high], axis=2)
    vlow = np.nanmean(sorted_amplitude[..., n_low * -1 :], axis=2)
    factor_values = vhigh - vlow

    result = np.full((n_rows, n_cols), np.nan, dtype="float64")
    result[window - 1 :] = factor_values
    return pd.DataFrame(result, index=close.index, columns=close.columns)
