"""crypto_alpha_taker_attention_premium_20_120_retneut 因子（factor_common 日频契约版）。

原始定义（分钟版语义）：
    att = mean(taker_buy_quote_volume, 20d) / mean(taker_buy_quote_volume, 120d)
    ret20 = close.pct_change(20d)
    逐日截面 OLS（rank 域）：resid = rank(att) - beta * rank(ret20)
    factor = -(rank_cs(resid) - 0.5)

日频改写判断：
- 分钟级 groupby(day).sum()/last() 聚合直接由日频 taker_buy_quote_volume/close 字段替代。
- 删除末尾 shift(1)（原仅用于执行延迟，框架按次日开盘执行，不人为延迟）。
- 删除广播回分钟 index 与 CHUNK_SIZE 分块逻辑，直接返回日频矩阵。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_taker_attention_premium_20_120_retneut",
    "author": "wesleywu",
    "level": "daily",
    "category": "sentiment",
    "description": "att=mean(taker,20d)/mean(taker,120d) 对 ret20 截面 rank 残差化，空纯买盘拥挤多冷清",
}

SETTING = {
    "data_needed": ["taker_buy_quote_volume", "close"],
    "universe": "historical_top50",
    "warmup_bars": 150,
    "preprocessing": "mad_rank",
    "params": {"att_fast_days": 20, "att_slow_days": 120, "ret_neut_days": 20},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily return-neutralized attention-premium matrix."""

    att_fast_days = SETTING["params"]["att_fast_days"]
    att_slow_days = SETTING["params"]["att_slow_days"]
    ret_neut_days = SETTING["params"]["ret_neut_days"]

    taker_daily = data_ctx["taker_buy_quote_volume"].astype("float64").clip(lower=0)
    close_daily = data_ctx["close"].astype("float64")

    # event volume_surge（慢速版）: 关注度 regime = 20d/120d 双窗均值比
    att_fast = taker_daily.rolling(att_fast_days, min_periods=att_fast_days).mean()
    att_slow = taker_daily.rolling(att_slow_days, min_periods=att_slow_days).mean()
    att = att_fast / (att_slow + _EPS)

    # 收益中性化：rank 域逐日截面 OLS，att 对 ret20 取残差
    # 剩下的"买盘涌入但价格没跟上"=纯派发吸收/过度定价，切断与 mom GP 共享成分
    ret20 = close_daily.pct_change(ret_neut_days, fill_method=None)
    a = att.rank(axis=1, pct=True)
    r = ret20.rank(axis=1, pct=True)
    a_c = a.sub(a.mean(axis=1), axis=0)
    r_c = r.sub(r.mean(axis=1), axis=0)
    beta = (a_c * r_c).sum(axis=1) / ((r_c * r_c).sum(axis=1) + _EPS)
    resid = a - r.mul(beta, axis=0)

    # direction premium: 空拥挤（高 resid）多冷清（低 resid），固定符号
    factor = -(resid.rank(axis=1, pct=True) - 0.5)
    return factor.replace([np.inf, -np.inf], np.nan)
