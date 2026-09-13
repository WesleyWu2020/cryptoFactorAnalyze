"""散户活跃度背离因子 (Retail Activity Divergence)。

公式:
1. trades_ma = rolling_mean(trade_count, lookback_days)
2. quote_volume_ma = rolling_mean(quote_volume, lookback_days)
3. 每日截面分别对两者做百分位排名 (axis=1, pct=True)
4. factor = rank(trades_ma) - rank(quote_volume_ma)

含义: 因子越高, 表示交易笔数排名显著高于成交额排名, 通常反映散户噪音
交易更活跃。direction=-1, 即因子值越小越好。

只使用当日及历史数据, 无未来函数。框架对输出统一做 MAD 去极值与按日
rank 归一化, 故 calc_factor 返回原始排名差即可。
"""

from __future__ import annotations

import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Retail_Activity_Divergence_Factor",
    "author": "local",
    "level": "daily",
    "category": "order_flow",
    "description": "Divergence between trade-count rank and quote-volume rank",
}

SETTING = {
    "data_needed": ["trade_count", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {"lookback_days": 20},
    "factor_direction": -1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily retail-activity divergence matrix.

    rolling 沿时间轴 (axis 0), 截面 rank 沿标的轴 (axis 1, pct=True),
    与旧版 groupby("date").rank(pct=True) 语义一致。
    """

    lookback_days = SETTING["params"]["lookback_days"]
    if (
        isinstance(lookback_days, bool)
        or not isinstance(lookback_days, int)
        or lookback_days < 2
    ):
        raise ValueError("SETTING.params.lookback_days must be an integer >= 2")

    trade_count = data_ctx["trade_count"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")

    # 旧版 rolling 未指定 min_periods, 默认等于 window, 保持相同行为
    trades_ma = trade_count.rolling(window=lookback_days).mean()
    quote_volume_ma = quote_volume.rolling(window=lookback_days).mean()

    trades_rank = trades_ma.rank(axis=1, method="average", pct=True)
    quote_volume_rank = quote_volume_ma.rank(axis=1, method="average", pct=True)

    return trades_rank - quote_volume_rank
