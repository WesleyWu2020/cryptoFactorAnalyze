"""crypto_alpha_ticket_shift_premium_20_120 因子（factor_common 日频契约版）。

原始定义（分钟版语义）：
    avg_ticket = sum(taker_buy_quote_volume, 1d) / sum(trade_count, 1d)
    shift_ratio = mean(avg_ticket, 20d) / mean(avg_ticket, 120d)
    factor = rank_cs(shift_ratio) - 0.5

方向说明（factor_direction = -1）：
原始先验为"聪明钱脚印"（多均额上行），但 ITER_NOTE 实测 RankIC=-0.0163/ICIR=-2.76
五年稳定为负——主动买均额上行=过热（whale FOMO/派发）而非知情建仓，高因子值组
（均额最升）稳定跑输。故高因子值不优选，direction 取 -1（等价于空均额上行、
多均额下行，与 _rev 版本的镜像结论一致）。

日频改写判断：
- 分钟级 groupby(day).sum() 聚合直接由日频 taker_buy_quote_volume/trade_count 字段替代。
- legacy data_needed 中的 close 仅用于取分钟 index/columns，日频契约下不再需要，移除。
- 删除末尾 shift(1)（原仅用于执行延迟，框架按次日开盘执行，不人为延迟）。
- 删除广播回分钟 index 与 CHUNK_SIZE 分块逻辑，直接返回日频矩阵。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_ticket_shift_premium_20_120",
    "author": "wesleywu",
    "level": "daily",
    "category": "orderflow",
    "description": "主动买单笔均额 20d/120d 结构迁移比截面 rank（实测均额上行=过热，应做空）",
}

SETTING = {
    "data_needed": ["taker_buy_quote_volume", "trade_count"],
    "universe": "historical_top50",
    "warmup_bars": 150,
    "preprocessing": "mad_rank",
    "params": {"shift_fast_days": 20, "shift_slow_days": 120},
    "factor_direction": -1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily ticket-shift premium matrix."""

    shift_fast_days = SETTING["params"]["shift_fast_days"]
    shift_slow_days = SETTING["params"]["shift_slow_days"]

    taker_daily = data_ctx["taker_buy_quote_volume"].astype("float64").clip(lower=0)
    tc_daily = data_ctx["trade_count"].astype("float64").clip(lower=0)

    # event trade_size_shift: 主动买单笔均额 -> 20d/120d 结构迁移比（去水平化）
    avg_ticket = taker_daily / (tc_daily + _EPS)
    ticket_fast = avg_ticket.rolling(shift_fast_days, min_periods=shift_fast_days).mean()
    ticket_slow = avg_ticket.rolling(shift_slow_days, min_periods=shift_slow_days).mean()
    shift_ratio = ticket_fast / (ticket_slow + _EPS)

    # 截面 rank 打分；实测方向为负（见模块 docstring），由 factor_direction=-1 表达
    factor = shift_ratio.rank(axis=1, pct=True) - 0.5
    return factor.replace([np.inf, -np.inf], np.nan)
