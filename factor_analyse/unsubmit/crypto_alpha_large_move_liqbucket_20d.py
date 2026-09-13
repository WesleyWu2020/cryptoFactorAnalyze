"""crypto_alpha_large_move_liqbucket_20d 因子（factor_common 日频契约版）。

原始定义（分钟版语义）：
    ret1     = 日频 close.pct_change()
    z        = clip(ret1 / rolling_std(ret1, 20d), ±3)      大动作（带方向）
    amihud   = rolling_mean(|ret1| / dollar_volume, 20d)     非流动性水平
    liq_mult = 0.5 + 截面 pct_rank(amihud)                   逐币乘子 [0.5, 1.5]
    factor   = ewm(z * liq_mult, span=5)                     persistence 平滑

方向：continuation —— 薄市知情建仓方向在日频延续（做多高 score）。

日频改写说明：
    - data_ctx 已是日频矩阵，日聚合恒等（close=last、dollar_volume 日求和 ==
      日频 quote_volume），直接使用 quote_volume。
    - 删除分钟版末尾的 .shift(1)（执行延迟由框架次日开盘成交处理）与分钟 index
      重广播。
    - 窗口单位由分钟 bar 改写为日历日，语义不变。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_large_move_liqbucket_20d",
    "author": "wesleywu",
    "level": "daily",
    "category": "mom",
    "description": "波动率标准化大动作 z 值乘以 amihud 非流动性桶乘子，ewm5 平滑（薄市大动作延续）",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 35,
    "preprocessing": "mad_rank",
    "params": {"vol_days": 20, "amihud_days": 20, "smooth_span": 5, "z_clip": 3.0},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily large-move liquidity-bucket matrix
    (date x instrument)."""

    vol_days = SETTING["params"]["vol_days"]
    amihud_days = SETTING["params"]["amihud_days"]
    smooth_span = SETTING["params"]["smooth_span"]
    z_clip = SETTING["params"]["z_clip"]

    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].clip(lower=0).astype("float64")

    ret1 = close.pct_change(fill_method=None)

    # event large_move: 自身波动率标准化的单日冲击，clip 防尾部爆炸
    vol20 = ret1.rolling(vol_days, min_periods=vol_days).std()
    z = (ret1 / (vol20 + _EPS)).clip(-z_clip, z_clip)

    # context liquidity_bucket: amihud 非流动性截面分位 -> 逐币乘子 [0.5, 1.5]
    # 薄市（高 amihud）大动作加分，厚市（低 amihud）大动作减分；逐币时变，非择时门
    amihud = (ret1.abs() / (quote_volume + _EPS)).rolling(
        amihud_days, min_periods=amihud_days
    ).mean()
    amihud_rank = amihud.rank(axis=1, pct=True)
    liq_mult = 0.5 + amihud_rank

    factor = z * liq_mult

    # quality persistence: ewm 平滑，要求信号数日延续才保持权重（兼压换手）
    factor = factor.ewm(span=smooth_span, adjust=False).mean()

    return factor.replace([np.inf, -np.inf], np.nan)
