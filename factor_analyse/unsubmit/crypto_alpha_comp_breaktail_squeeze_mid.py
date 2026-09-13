"""crypto_alpha_comp_breaktail_squeeze_mid 因子（factor_common 日频契约版）。

原始定义（分钟版）：compose 因子。
- 成分 A breakout 质量分：20d 区间位置([-1,1]) × 20d Kaufman 效率比 × 5d/20d
  成交额确认，截面 rank 中心化后立方（尾部锐化）；
- 成分 B vol_squeeze 中段 score：vol20=std(日收益,20d) 在 120d 区间内的位置取负，
  截面 rank 中心化后 clip 到 [-0.4, 0.4]（去极端十分位）；
- factor = (comp_a + comp_b)/2，shift(1) 广播分钟级。

日频改写说明（judgment calls）：
- 日聚合恒等：close=日内最后价、turnover 日求和 = quote_volume，直接使用日频字段。
- 所有窗口（20d 区间/效率比、5d/20d 成交额、20d 波动、120d 波动区间）均为日频
  滚动窗口，与原版语义一致。
- 删除 factor_daily.shift(1)（仅为执行延迟；框架按次日开盘执行）。
- 删除分钟级 reindex 广播与分块列循环，直接返回日频矩阵。

方向说明：direction=continuation（A 尾部延续 + B 中段低波排序），合成后高因子值
为多头腿，故 factor_direction=1（注意原 ITER_NOTE 实测五年全负 REJECTED，
此处仅做契约转换，不改变语义）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "breakout+vol_squeeze", "statement": "组合事件：放量干净突破 + 波动压缩位置"},
    "context":   None,
    "qualities": [
        {"id": "path_cleanliness", "statement": "Kaufman 效率比，趋势平滑无拉锯（成分 A）"},
        {"id": "volume_confirm", "statement": "5d/20d 成交额放大确认（成分 A）"},
        {"id": "persistence", "statement": "20d 慢窗口（成分 B 与整体）"},
    ],
    "direction": {"id": "continuation", "statement": "尾部延续（A）+ 中段低波排序（B）"},
    "output":    {"id": "rank", "statement": "成分各自截面 rank 塑形后等权合成"},
    "mechanism": "A（breakout 质量分）的延续性只在极端十分位成立 -> 中心化 rank 立方锐化只放大尾部；"
                 "B（vol_squeeze）的单调性只在中段成立 -> 中心化 rank 截断 ±0.4 剔除有毒极端桶。"
                 "两成分 prod_corr 低（0.27/0.33 vs 库），形态修剪后 edge 叠加而成本不叠加",
    "fields":    ["close", "turnover"],
    "expected_horizon": "1d",
    "invalidation": "合成 ls_netret_ann < 0.15（形态修剪未能保留有效区段，组合假设不成立）；"
                    "或 max|rho| >= 0.85 vs 库内因子",
    "search_mode": "exploit",
    "semantic_key": "breakout+vol_squeeze|-|path_cleanliness+volume_confirm+persistence|continuation|rank",
}

ITER_NOTE: dict = {
    "op_type": "compose_factor",
    "factor_family": "mom/vol",
    "hypothesis": "F0007（breakout 质量）毛 edge ~13%/yr 但换手成本相抵，结构=尾部延续+中段反转；"
                  "F0005（vol_squeeze）IC 强但结构=中段单调+尾部有毒。两者缺陷互补、有效区段不重叠"
                  "（A 的尾部 + B 的中段）。形态修剪（A 立方锐化尾部、B 截断去尾）后等权合成，"
                  "毛 edge 应近似叠加而换手不叠加，净收益有望跨过 0.20 门槛。",
    "change": "新建 compose 因子：成分 A = 20d 区间位置 x Kaufman 效率比 x 放量确认，"
              "日频截面 rank 中心化后立方（尾部锐化）；成分 B = -vol 位置（同 F0005 构造），"
              "日频截面 rank 中心化后 clip 到 [-0.4, 0.4]（去极端十分位）；等权平均，"
              "shift(1) 广播到分钟。",
    "expected": "|RankIC| 0.02~0.035（两成分 IC 叠加但有重叠损失）, |RankICIR| > 3, "
                "ls_netret_ann 0.15~0.30（关键验证点：毛 edge 叠加是否成立）, "
                "turnover 0.35~0.55, coverage > 0.7, maxdd < 0.25; prod_corr max|rho| < 0.6。",
    "parent_iter": "F0007 breakout_quality_20d + F0005 vol_squeeze_liqfil（均 REJECTED，形态互补）",
    "reasoning": "七轮元结论：极端十分位延续、中段反转。A 取尾部、B 取中段，正好各自取其有效区段。",
    "result": "REJECTED (STATISTICAL_REJECT, 组合假设证伪): |RankIC|=0.0124, |ICIR|=2.59 PASS, "
              "turnover=0.32, prod_corr=0.23 PASS; 但 ls_netret_ann=-9.3%, ls_netir=-0.92, "
              "五个年度全负。失败归因（两层）：(1) 实现层权重失衡——立方后 comp_a 峰值 0.125 "
              "vs comp_b 峰值 0.4，B 主导 ~3:1，合成退化为'B 加少量 A 倾斜'，IC 结构印证了这点；"
              "(2) 更根本的算术层：A 的 ls 毛 +13%/yr、B 的 ls 毛 ~-8.5%/yr，任何权重下组合的 "
              "ls 毛都不超过满配 A 的 13%，扣成本后不可能过 0.20 净门槛——edge 叠加假设在"
              "'只有一个成分有组合级正毛收益'时不成立。compose 分支关闭，不追权重重调。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_comp_breaktail_squeeze_mid",
    "author": "wesleywu",
    "level": "daily",
    "category": "mom",
    "description": "compose：breakout 质量分 rank 立方(尾部锐化) + vol_squeeze 位置 rank 截断(取中段)，等权合成",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    # 最长链式窗口：sq_range_days + sq_vol_days = 140 天，加缓冲。
    "warmup_bars": 146,
    "preprocessing": "mad_rank",
    "params": {
        "range_days": 20,
        "eff_days": 20,
        "vol_fast_days": 5,
        "vol_slow_days": 20,
        "sq_vol_days": 20,
        "sq_range_days": 120,
        "tail_clip": 0.4,
    },
    "factor_direction": 1,
}

EPS = 1e-12


def _centered_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True) - 0.5


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily breaktail+squeeze compose matrix (date x instrument)."""

    params = SETTING["params"]
    range_days = params["range_days"]
    eff_days = params["eff_days"]
    vol_fast_days = params["vol_fast_days"]
    vol_slow_days = params["vol_slow_days"]
    sq_vol_days = params["sq_vol_days"]
    sq_range_days = params["sq_range_days"]
    tail_clip = params["tail_clip"]

    close_daily = data_ctx["close"].astype("float64")
    to_daily = data_ctx["quote_volume"].astype("float64").clip(lower=0)

    # ---- 成分 A: breakout 质量分，尾部锐化 ----
    lo20 = close_daily.rolling(range_days, min_periods=range_days).min()
    hi20 = close_daily.rolling(range_days, min_periods=range_days).max()
    pos = 2.0 * (close_daily - lo20) / ((hi20 - lo20) + EPS) - 1.0

    ret1 = close_daily.pct_change(fill_method=None)
    net_move = (close_daily - close_daily.shift(eff_days)).abs()
    path_len = ret1.abs().rolling(eff_days, min_periods=eff_days).sum() * close_daily.shift(eff_days)
    efficiency = (net_move / (path_len + EPS)).clip(0, 1)

    to_fast = to_daily.rolling(vol_fast_days, min_periods=vol_fast_days).mean()
    to_slow = to_daily.rolling(vol_slow_days, min_periods=vol_slow_days).mean()
    vol_conf = (to_fast / (to_slow + EPS)).clip(0, 2) / 2.0

    a_raw = pos * efficiency * vol_conf
    a_rank = _centered_rank(a_raw)
    comp_a = a_rank.pow(3)          # 立方：保号、压中段、放大极端十分位

    # ---- 成分 B: vol_squeeze 中段 score，截尾去毒 ----
    vol20 = ret1.rolling(sq_vol_days, min_periods=sq_vol_days).std()
    sq_lo = vol20.rolling(sq_range_days, min_periods=sq_range_days).min()
    sq_hi = vol20.rolling(sq_range_days, min_periods=sq_range_days).max()
    sq_pos = (vol20 - sq_lo) / ((sq_hi - sq_lo) + EPS)

    b_rank = _centered_rank(-sq_pos)
    comp_b = b_rank.clip(-tail_clip, tail_clip)     # 剔除两个极端十分位的毒性

    # ---- 等权合成 ----
    factor = (comp_a + comp_b) / 2.0
    return factor.replace([np.inf, -np.inf], np.nan)
