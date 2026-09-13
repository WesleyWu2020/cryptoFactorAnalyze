import os

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "positioning_buildup+positioning_chase", "statement": "OI 慢速水平抬升（20d/120d，杠杆拥挤 regime）+ OI 快变确认追仓（5d ΔOI 同向 mark 确认，情绪单建仓脉冲）"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "mark_premium_confirm", "statement": "chase 成分：mark/close 偏离与收益同号才计为有效追仓"},
        {"id": "persistence", "statement": "buildup 成分：双长窗均值比=持续数周的持仓堆积趋势"},
    ],
    "direction": {"id": "premium", "statement": "杠杆拥挤溢价：固定符号空拥挤（慢 regime + 快脉冲）多出清/平静，无方向预测；两成分均为 premium|rank，满足 compose 一致性"},
    "output":    {"id": "rank", "statement": "两成分截面 rank 等权平均"},
    "mechanism": "IS 探针（delay=1 口径）：chase_int 与 M1 因子值 Spearman ρ=-0.024（近零正交），"
                 "等权 blend IC=0.0276 ≥ 两成分各自（0.0161/0.0270）——唯一不被稀释的配对"
                 "（chase+D1/C1 预检均被稀释否决）。结构性互补：(a) M1 连续慢变 rank 填平 chase 的"
                 "零值并列块，修复轮2 long_nums≈0 的空长腿结构病；(b) 分年 regime 互补——"
                 "chase 2022 熊市 +92.3% 正是 M1 bleed 的年份（-5.3%），M1 2024 +24.9% 正是 chase 弱年。"
                 "风险预注册：两成分同为 OI 字段+空尾主导，可能共享暴露集中（comp 前科三次失败的元教训）。",
    "fields":    ["open_interest", "close", "mark_close"],
    "expected_horizon": "1d",
    "invalidation": "(1) ls_netir < 0.898（不超 M1 standalone=compose 无增量，重复 comp1 失败）；"
                    "(2) |RankIC| < 0.020（blend 预检 0.0276 的七成）；"
                    "(3) prod_corr >= 0.6（成分 0.46/0.15，blend 应介于其间）；"
                    "(4) turnover > 0.6；(5) 2022 或 2023 ls_netret < 0（regime 互补假设的核心检验）",
    "search_mode": "exploit",
    "semantic_key": "positioning_buildup+positioning_chase|-|mark_premium_confirm+persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "compose_factor",
    "factor_family": "positioning",
    "parent_factor_id": "crypto_alpha_oi_chase_intensity_premium_5d",
    "parent_iter": None,
    "semantic_mutation": None,
    "hypothesis": "持仓层快慢双信号等权组合：M1（慢水平，IC 0.0263/4.89，败因 2022-23 长腿 bleed）"
                  "+ chase_int（快确认流量，空尾真实但长腿空桶）。IS 预检：ρ=-0.024 近零正交，"
                  "blend IC 0.0276 超两成分；M1 连续 rank 修复 chase 空长腿，chase 2022 +92% 补 M1 弱年。",
    "reasoning": "用户指定 chase_intensity 组合方向。预检否决了 D1/C1 配对（等权 blend 被强成分稀释，"
                 "重复 comp1/comp2 失败模式），只有 M1 配对 blend ≥ max(成分)——且 regime 互补证据最强。"
                 "悲观先验同预注册：OI 字段内 compose 可能共享空尾暴露，增量可能边际（0.0276 vs 0.0270），"
                 "组合端六图（尤其 demeaned 长腿与分年）才是判决依据。",
    "change": "新 compose 因子。buildup=mean(oi,20)/mean(oi,120)；chase=doi5.clip(0)*mark确认；"
              "factor=-( (rank(buildup)+rank(chase))/2 - 0.5 )，shift(1) 广播分钟级。",
    "expected": "|RankIC| 0.024~0.030（预检 0.0276），ls_netir 0.7~1.1（M1 0.898 之上才有增量），"
                "turnover 0.15~0.40（M1 0.063 与 chase 0.509 之间），coverage>0.9，"
                "2022/2023 双年为正（regime 互补核心检验），prod_corr<0.6。",
    "result": "REJECTED (STATISTICAL_REJECT, IS 2022~2024): IC 层完全兑现(|RankIC|=0.0273/|ICIR|=4.87 "
              "≈预检 0.0276, 全程最稳 IC 曲线), 结构修复成功(long_nums 17→55 正常填充, 轮2空长腿病愈; "
              "turnover 0.44/maxdd 0.20 过门, prod_corr=0.4692 PASS); 但 payoff 未回: ls_netir=-0.43 "
              "<< M1 standalone 0.898=compose 负增量(预注册 invalidation#1 触发); 毛仅+7%/yr 盖不住 "
              "12%/yr 成本; 2022 仍 -5.4%=regime 互补证伪(chase 的 2022+92% 来自集中空尾, rank 等权平均"
              "把尾部 alpha 稀释没); demeaned 组1 -0.45 最差/组5,6 最优/组9 平盘负=长腿仍无 payoff。"
              "元规则第四次复现: 分数正交≠payoff 正交, IC blend 增益≠组合增益, 尾部型 alpha 不可 rank 平均。"
              "持仓层 5 次 IS 失败(M1x2+chase x2+comp x1), standalone 与等权 compose 双路线关闭; "
              "chase 的归宿维持原判: universe 层短黑名单素材。",
    "oos_result": None,
}


TYPE = "regular"

BUILDUP_FAST_DAYS = 20
BUILDUP_SLOW_DAYS = 120
CHASE_HORIZON_DAYS = 5
MINUTES_PER_DAY = 1440
WARMUP_BARS = (BUILDUP_SLOW_DAYS + BUILDUP_FAST_DAYS + 10) * MINUTES_PER_DAY
CHUNK_SIZE = int(os.getenv("FACTOR_CHUNK_SIZE", "8"))
EPS = 1e-12

META = {
    "factor_name": "crypto_alpha_comp_oi_buildup_chase_int",
    "author": "wesleywu",
    "level": "minutes",
    "tag": "comp_oi_buildup_chase",
    "category": "positioning",
}

SETTING = {
    "data_needed": ["open_interest", "close", "mark_close"],
    "universe": "tradable_mask",
    "pasteurization": True,
    "warmup_bars": WARMUP_BARS,
}


def _chunk_columns(columns: list[str], chunk_size: int):
    chunk_size = max(int(chunk_size), 1)
    for i in range(0, len(columns), chunk_size):
        yield columns[i:i + chunk_size]


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    oi = data_ctx["open_interest"]
    close = data_ctx["close"]
    mark = data_ctx["mark_close"]
    day = close.index.normalize()

    # 分块分钟级日聚合（内存控制），截面操作在小型日频帧上统一做
    oi_parts, mark_parts, close_parts = [], [], []
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        oi_parts.append(oi.loc[:, cols].clip(lower=0).groupby(day).mean())
        mark_parts.append(mark.loc[:, cols].groupby(day).last())
        close_parts.append(close.loc[:, cols].groupby(day).last())

    oi_daily = pd.concat(oi_parts, axis=1)[close.columns]
    mark_daily = pd.concat(mark_parts, axis=1)[close.columns]
    close_daily = pd.concat(close_parts, axis=1)[close.columns]
    del oi_parts, mark_parts, close_parts

    # 成分1 positioning_buildup: OI 慢速水平 = 20d/120d 双窗比
    buildup = (oi_daily.rolling(BUILDUP_FAST_DAYS, min_periods=BUILDUP_FAST_DAYS).mean()
               / (oi_daily.rolling(BUILDUP_SLOW_DAYS, min_periods=BUILDUP_SLOW_DAYS).mean() + EPS))

    # 成分2 positioning_chase（unsigned）: 5d 确认追仓强度
    ret5 = close_daily.pct_change(CHASE_HORIZON_DAYS, fill_method=None)
    doi5 = oi_daily.pct_change(CHASE_HORIZON_DAYS, fill_method=None).clip(lower=0)
    prem5 = (mark_daily / close_daily - 1).rolling(CHASE_HORIZON_DAYS, min_periods=CHASE_HORIZON_DAYS).mean()
    chase = doi5.where(np.sign(prem5) == np.sign(ret5), 0.0)

    # direction premium: 空拥挤多出清/平静，两成分 rank 等权平均
    blend_rank = (buildup.rank(axis=1, pct=True) + chase.rank(axis=1, pct=True)) / 2
    factor_daily = -(blend_rank - 0.5)
    factor_daily = factor_daily.shift(1)  # 只用已完成交易日

    # 分块广播回分钟级
    factor = pd.DataFrame(index=close.index, columns=close.columns, dtype=np.float32)
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        part = factor_daily[cols].reindex(day)
        part.index = close.index
        factor.loc[:, cols] = part.astype(np.float32)

    return factor.replace([np.inf, -np.inf], np.nan)
