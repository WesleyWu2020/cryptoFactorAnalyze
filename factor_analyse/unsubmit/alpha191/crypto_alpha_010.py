"""crypto_alpha_010 因子（factor_common 日频契约版）。

GTJA Alpha10 日频原始公式（见 alpha191_factor_prompts.md）：
    RANK(MAX(((RET < 0) ? STD(RET,20) : CLOSE)^2, 5))

即：RET<0 时取 20 日收益标准差、否则取收盘价，平方后取 5 日时序最大，再截面 rank。

转换判断说明：
- 旧分钟版先 resample("D") 取日收盘再算同一公式，日频输入下聚合为恒等，直接删除。
- STD 沿用旧版 ddof=0（总体标准差）。
- 删除旧版 factor_daily.shift(1)（仅为分钟级执行延迟）；t 日因子只用 <= t 数据。
- 删除分块列循环，直接整帧运算。
计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_010",
    "author": "wesleywu",
    "level": "daily",
    "category": "price_volume",
    "description": "RANK(MAX(((RET<0) ? STD(RET,20) : CLOSE)^2, 5))",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"ret_std_window": 20, "ts_max_window": 5, "std_ddof": 0},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha10 matrix (date x instrument)."""

    ret_std_window = SETTING["params"]["ret_std_window"]
    ts_max_window = SETTING["params"]["ts_max_window"]
    std_ddof = SETTING["params"]["std_ddof"]

    close = data_ctx["close"].astype("float64")
    prev_close = close.shift(1)
    ret = close / prev_close.where(prev_close.abs() > _EPS) - 1.0
    ret_std = ret.rolling(ret_std_window, min_periods=ret_std_window).std(ddof=std_ddof)

    # Formula branch: (RET < 0) ? STD(RET, 20) : CLOSE
    base = close.where(~(ret < 0), ret_std)
    ts_max = base.pow(2).rolling(ts_max_window, min_periods=ts_max_window).max()
    factor = ts_max.rank(axis=1, method="average", pct=True)
    return factor.replace([np.inf, -np.inf], np.nan)
