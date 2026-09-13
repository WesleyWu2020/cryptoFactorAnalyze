"""crypto_alpha_vroc_dollar_volume 因子（factor_common 日频契约版）。

原始定义：
    factor = (daily_dollar_volume - ref) / ref * 100，ref = 20d 前的日成交额
    即日成交额的 20d 变化率（Volume Rate of Change, VROC）。

分钟 -> 日频改写说明：
    - dollar_volume -> quote_volume（分钟 dollar_volume 全日求和 == 日频 quote_volume），
      resample("1440min").sum() 在日频输入下为恒等，直接使用日频字段。
    - 删除 factor_daily.shift(1) 与 reindex 回分钟索引：原 shift 仅实现执行延迟，
      框架已按次日开盘执行，t 日因子可用 <= t 数据。
    - 公式即 quote_volume.clip(lower=0).pct_change(20) * 100，与原实现逐日等价。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_vroc_dollar_volume",
    "author": "wesleywu",
    "level": "daily",
    "category": "volume_price",
    "description": "日成交额 20d 变化率（VROC）：(quote_volume - quote_volume_20d_ago) / ref * 100",
}

SETTING = {
    "data_needed": ["quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"lookback_days": 20},
    "factor_direction": 1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily dollar-volume VROC matrix (date x instrument)."""

    lookback_days = SETTING["params"]["lookback_days"]

    quote_volume = data_ctx["quote_volume"].astype("float64").clip(lower=0.0)
    ref = quote_volume.shift(lookback_days)
    factor = (quote_volume - ref) / ref.replace(0.0, np.nan) * 100.0
    return factor.replace([np.inf, -np.inf], np.nan)
