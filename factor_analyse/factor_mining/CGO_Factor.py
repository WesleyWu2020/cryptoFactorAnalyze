"""资本利得突出量因子 (CGO - Capital Gain Outstanding)。

公式（与旧版 compute_one 逐点等价）：

    vwap = where(volume > 0, quote_volume / volume, close)
    log_vol = log(volume + 1)
    position = (log_vol - rolling_min(log_vol, window)) / (rolling_max - rolling_min)  # 分母为 0 时替换为 1
    turnover = clip(0.02 + position * 0.8 * (1 - 0.02), 0, 0.99)
    RP_t = RP_{t-1} * (1 - turnover_{t-1}) + vwap_{t-1} * turnover_{t-1}   # RP_0 = vwap_0
    CGO = (close - RP) / (RP + 1e-8)

RP 的逐 symbol 递归改写为按时间轴的行循环向量化：每行 t 用整行矩阵更新 RP，
各列相互独立，因此与原逐 symbol 递归逐点一致。rolling 的 min_periods=1
与旧版保持一致。计算只使用当日及历史数据，无未来函数。

注意：旧文件中的 calculate_turnover_volume_ratio 函数未被主流程使用，
本实现以 compute_one 内的 log_vol 滚动极值映射为准。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "CGO_Factor",
    "author": "local",
    "level": "daily",
    "category": "momentum",
    "description": "Capital Gain Outstanding: (close - recursive reference cost) / reference cost",
}

SETTING = {
    "data_needed": ["close", "volume", "quote_volume"],
    "universe": "historical_top50",
    # RP 递归带有全历史记忆（衰减因子 1 - turnover 可接近 1），取 3 倍窗口作为保守 warmup
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"window": 10},
    "factor_direction": 1,
}

_EPSILON = 1e-8
_BASE_TURNOVER = 0.02
_MAX_TURNOVER_CAP = 0.8


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily CGO matrix (date x instrument)."""

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be an integer >= 1")

    close = data_ctx["close"].astype("float64")
    volume = data_ctx["volume"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")

    # --- A. 成交均价 (VWAP) ---
    vwap = (quote_volume / volume).where(volume > 0, close)

    # --- B. 换手率估计（log_vol 滚动极值映射，严格复现旧版 compute_one） ---
    log_vol = np.log(volume + 1)
    roll_min = log_vol.rolling(window=window, min_periods=1).min()
    roll_max = log_vol.rolling(window=window, min_periods=1).max()
    denominator = (roll_max - roll_min).replace(0, 1)
    position = (log_vol - roll_min) / denominator
    estimated_turnover = position * _MAX_TURNOVER_CAP
    turnover = (_BASE_TURNOVER + estimated_turnover * (1 - _BASE_TURNOVER)).clip(0, 0.99)

    # --- C. 参考价格 RP：按时间轴行循环向量化（各列独立，等价于逐 symbol 递归） ---
    vwap_values = vwap.to_numpy(dtype="float64")
    turnover_values = turnover.to_numpy(dtype="float64")
    rp_values = np.empty_like(vwap_values)
    rp_values[0] = vwap_values[0]
    for t in range(1, len(vwap_values)):
        v_prev = turnover_values[t - 1]
        p_prev = vwap_values[t - 1]
        rp_values[t] = rp_values[t - 1] * (1 - v_prev) + p_prev * v_prev
    rp = pd.DataFrame(rp_values, index=close.index, columns=close.columns)

    # --- D. CGO 因子 ---
    return (close - rp) / (rp + _EPSILON)
