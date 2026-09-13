"""crypto_alpha_ticket_shift_premium_20_120_rev 因子（factor_common 日频契约版）。

原始定义（分钟版语义，符号按 C1c 实测修正的翻转版）：
    avg_ticket = sum(taker_buy_quote_volume, 1d) / sum(trade_count, 1d)
    shift_ratio = mean(avg_ticket, 20d) / mean(avg_ticket, 120d)
    factor = -(rank_cs(shift_ratio) - 0.5)

方向 premium（taker 热度空头溢价）：空均额上行（whale FOMO/派发=过热）、
多均额下行（散户磨合=冷清）。翻转版实测 RankIC=+0.0163/ICIR=+2.76，高因子值
（均额下行=冷清）为优选多腿，factor_direction=1。

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
    "factor_name": "crypto_alpha_ticket_shift_premium_20_120_rev",
    "author": "wesleywu",
    "level": "daily",
    "category": "orderflow",
    "description": "主动买单笔均额 20d/120d 迁移比取负 rank：空均额上行（过热）多均额下行（冷清）",
}

SETTING = {
    "data_needed": ["taker_buy_quote_volume", "trade_count"],
    "universe": "historical_top50",
    "warmup_bars": 150,
    "preprocessing": "mad_rank",
    "params": {"shift_fast_days": 20, "shift_slow_days": 120},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily reversed ticket-shift premium matrix."""

    shift_fast_days = SETTING["params"]["shift_fast_days"]
    shift_slow_days = SETTING["params"]["shift_slow_days"]

    taker_daily = data_ctx["taker_buy_quote_volume"].astype("float64").clip(lower=0)
    tc_daily = data_ctx["trade_count"].astype("float64").clip(lower=0)

    # event trade_size_shift: 主动买单笔均额 -> 20d/120d 结构迁移比（去水平化）
    avg_ticket = taker_daily / (tc_daily + _EPS)
    ticket_fast = avg_ticket.rolling(shift_fast_days, min_periods=shift_fast_days).mean()
    ticket_slow = avg_ticket.rolling(shift_slow_days, min_periods=shift_slow_days).mean()
    shift_ratio = ticket_fast / (ticket_slow + _EPS)

    # direction premium（符号按 C1c 实测修正）: 空均额上行（过热）多均额下行（冷清）
    factor = -(shift_ratio.rank(axis=1, pct=True) - 0.5)
    return factor.replace([np.inf, -np.inf], np.nan)
