"""crypto_alpha_large_move_liqbucket_rev_20d 因子（factor_common 日频契约版）。

原始定义（分钟版语义，F2 continuation 版的符号翻转 + rank 域平滑）：
    ret1     = 日频 close.pct_change()
    z        = clip(ret1 / rolling_std(ret1, 20d), ±3)      大动作（带方向）
    amihud   = rolling_mean(|ret1| / dollar_volume, 20d)     非流动性水平
    liq_mult = 0.5 + 截面 pct_rank(amihud)                   逐币乘子 [0.5, 1.5]
    score    = -z * liq_mult                                 reversal：大动作后回弹
    factor   = ewm(截面 pct_rank(score) - 0.5, span=10)      rank 域平滑（有界、压换手）

方向：reversal —— 薄市大动作多为流动性冲击/过度反应，冲击过后回弹更狠
（做多 score 高 = 大跌后回弹）。

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
    "factor_name": "crypto_alpha_large_move_liqbucket_rev_20d",
    "author": "wesleywu",
    "level": "daily",
    "category": "mom",
    "description": "大动作反转：-z*amihud 乘子后截面 rank 中心化再 ewm10 rank 域平滑（薄市冲击回弹）",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 40,
    "preprocessing": "mad_rank",
    "params": {"vol_days": 20, "amihud_days": 20, "smooth_span": 10, "z_clip": 3.0},
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily large-move liquidity-bucket reversal matrix
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
    # reversal 方向：薄市大动作=流动性冲击/过度反应，回弹更狠
    amihud = (ret1.abs() / (quote_volume + _EPS)).rolling(
        amihud_days, min_periods=amihud_days
    ).mean()
    amihud_rank = amihud.rank(axis=1, pct=True)
    liq_mult = 0.5 + amihud_rank

    score = -z * liq_mult  # direction reversal：大动作后回弹

    # quality persistence: 截面 rank 中心化后 ewm rank 域平滑（有界、压换手）
    factor = score.rank(axis=1, pct=True) - 0.5
    factor = factor.ewm(span=smooth_span, adjust=False).mean()

    return factor.replace([np.inf, -np.inf], np.nan)
