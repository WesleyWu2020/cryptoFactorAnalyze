"""crypto_alpha_comp_attention_burst_premium 因子（factor_common 日频契约版）。

原始定义（分钟版）：compose 因子。
- 成分 C1 volume_surge：taker 买盘美元额 20d/120d 关注度 regime，score=-(rank(att)-0.5)（空拥挤）；
- 成分 D1 activity_burst：trade_count 20d 变异系数，score=-(rank(burst)-0.5)（空阵发）；
- factor = 0.5*score_C1 + 0.5*score_D1，shift(1) 广播分钟级。

日频改写说明（judgment calls）：
- 日聚合恒等：xbinance_taker_buy_quote_volume 日求和 = taker_buy_quote_volume，
  xbinance_trade_count 日求和 = trade_count，直接使用日频字段。
- data_needed 去掉 close（原版仅用其分钟索引做日聚合锚点，日频下不再需要）。
- 删除 factor_daily.shift(1)（仅为执行延迟；框架按次日开盘执行）。
- 删除分钟级 reindex 广播与分块列循环，直接返回日频矩阵。
- 20d/120d 均值比与 20d CV 均为日频滚动窗口，与原版语义一致。

方向说明：direction=premium（taker 热度/彩票溢价），固定符号已写入两成分公式（负号），
高因子值=冷清+平稳币为多头腿，故 factor_direction=1。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "activity_burst+volume_surge", "statement": "compose: C1 买盘美元额 20/120 regime(注意力) + D1 trade_count 20d 变异系数(活动阵发)"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "persistence", "statement": "双长窗结构(20/120 比、20d CV)均为慢变持续特征"},
    ],
    "direction": {"id": "premium", "statement": "taker 热度/彩票溢价：固定符号空拥挤+空阵发，多冷清+多平稳"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "C1 与 D1 互补结构实测：C1 强尾部变现(demeaned 组0独输)但 IC 弱(0.0156)、ρ0.62 边缘；"
                 "D1 全截面 IC 最强(0.0221)、ρ0.54 冗余逃逸，但信息在中段、组合吃尾部变现效率低(IR 0.66)。"
                 "等权 rank 融合：C1 出尾部、D1 出广度和增量，同一'taker 热度/彩票空头'资产的两个"
                 "正交切面(量的趋势 vs 活动的二阶结构)。",
    "fields":    ["xbinance_taker_buy_quote_volume", "xbinance_trade_count", "close"],
    "expected_horizon": "1d",
    "invalidation": "(1) ls_netir < 1.04（不优于最好成分 C1，compose 无增量，直接留 C1）；"
                    "(2) ls_netret_ann < 0.164（净收益不如单用 C1）；"
                    "(3) prod_corr >= 0.62（冗余较 C1 无改善）；"
                    "(4) turnover > 0.25；"
                    "(5) coverage < 0.9",
    "search_mode": "exploit",
    "semantic_key": "activity_burst+volume_surge|-|persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "compose_factor",
    "factor_family": "orderflow",
    "parent_factor_id": "crypto_alpha_taker_attention_premium_20_120",
    "semantic_mutation": "volume_surge|-|persistence|premium|rank + activity_burst|-|premium|rank -> "
                         "activity_burst+volume_surge|-|persistence|premium|rank",
    "hypothesis": "compose 互补假设：C1(16.4%/1.04/ρ0.62)出尾部变现，D1(10.3%/0.66/ρ0.54)出全截面 IC "
                  "与低相关增量。若两成分真的正交互补，融合后 IR 应超过两者最大值(1.04)；"
                  "若 IR 介于两者之间，说明共享成分主导，compose 只是稀释。",
    "change": "两成分各自构造不变（rank 域分数），factor = 0.5*score_C1 + 0.5*score_D1，shift(1) 广播。",
    "expected": "ls_netret_ann 0.16~0.22, ls_netir 1.0~1.3, |RankIC| 0.018~0.024, "
                "turnover ~0.08, prod_corr 0.5~0.6, coverage>0.92。",
    "result": "REJECTED (STATISTICAL_REJECT, 但 compose 结构性改善): |RankIC|=0.0229/|ICIR|=3.81(超两成分), "
              "ls_netir=1.0085 过 1.0 门, 净+15.7%/yr, turnover=0.077, maxdd=0.189, prod_corr=0.5754 PASS"
              "(较 C1 的 0.62 改善); 分年全正(2022 +2.0% 修复了 C1 的 -2.9%, 2024 +30.4/2025 +35.5), "
              "仅 2026 半年 -2.4%。预注册增量判据严格读: IR 1.0085 < C1 的 1.04、净 15.7% < 16.4%, "
              "IR/净无增量但 IC(+47%)/ICIR(+41%)/ρ(-0.045)/分年稳健性全面改善, demeaned 组0 -1.6 "
              "比两成分各自更干净(C1 -1.2/D1 组1 -0.9)=尾部短腿被融合强化。死因仍是净<0.2 门。"
              "家族 plateau 确认: taker 热度/彩票空头溢价各表达收敛于 ~16% 净/IR~1.0/ρ~0.58-0.62, "
              "weight 微调属于过拟合不再迭代。最终留档: 本 comp(稳健首选) + C1(IR/净最高)。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_comp_attention_burst_premium",
    "author": "wesleywu",
    "level": "daily",
    "category": "orderflow",
    "description": "0.5*(-(rank(taker买盘20d/120d)-0.5)) + 0.5*(-(rank(trade_count 20d CV)-0.5))，空热度+空阵发",
}

SETTING = {
    "data_needed": ["taker_buy_quote_volume", "trade_count"],
    "universe": "historical_top50",
    # 最长窗口 att_slow_days=120，加缓冲。
    "warmup_bars": 130,
    "preprocessing": "mad_rank",
    "params": {"att_fast_days": 20, "att_slow_days": 120, "burst_days": 20},
    "factor_direction": 1,
}

EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily attention+burst premium matrix (date x instrument)."""

    params = SETTING["params"]
    att_fast_days = params["att_fast_days"]
    att_slow_days = params["att_slow_days"]
    burst_days = params["burst_days"]

    taker_daily = data_ctx["taker_buy_quote_volume"].astype("float64").clip(lower=0)
    tc_daily = data_ctx["trade_count"].astype("float64").clip(lower=0)

    # 成分 C1 volume_surge: 买盘美元额 20d/120d 关注度 regime，空拥挤
    att_fast = taker_daily.rolling(att_fast_days, min_periods=att_fast_days).mean()
    att_slow = taker_daily.rolling(att_slow_days, min_periods=att_slow_days).mean()
    att = att_fast / (att_slow + EPS)
    score_c1 = -(att.rank(axis=1, pct=True) - 0.5)

    # 成分 D1 activity_burst: 活动 20d 变异系数，空阵发
    tc_mean = tc_daily.rolling(burst_days, min_periods=burst_days).mean()
    tc_std = tc_daily.rolling(burst_days, min_periods=burst_days).std()
    burst = tc_std / (tc_mean + EPS)
    score_d1 = -(burst.rank(axis=1, pct=True) - 0.5)

    # compose: 等权 rank 域融合
    factor = 0.5 * score_c1 + 0.5 * score_d1
    return factor.replace([np.inf, -np.inf], np.nan)
