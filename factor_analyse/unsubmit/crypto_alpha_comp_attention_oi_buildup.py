import os

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "positioning_buildup+volume_surge", "statement": "compose: C1 币安 taker 买盘美元额 20/120 regime(投机流量) + M1 Bybit OI 20/120 regime(杠杆存量)"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "persistence", "statement": "双长窗结构（20/120 比）均为慢变持续特征"},
    ],
    "direction": {"id": "premium", "statement": "投机热度+杠杆拥挤双重溢价：固定符号空升温+空堆积，多冷清+多出清"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "C1(流量热度)与 M1(存量杠杆)是'投机过度定价'资产的两个正交切面：逐日截面相关实测 "
                 "mean ρ=0.194(p95=0.386)——场地(币安 vs Bybit)、口径(流量 vs 存量)都不同。"
                 "C1 尾部变现强(16.4%/1.04)但 ρ0.62 贴边；M1 IC 全样本稳定(0.0263/4.89)、ρ0.4574 干净, "
                 "但 2022-23 熊市 bleed。等权 rank 融合：若正交互补成立, IR 应超 C1 的 1.04 且 ρ 显著稀释; "
                 "若只是平均化, IR 居中=compose 无真增量。",
    "fields":    ["xbinance_taker_buy_quote_volume", "open_interest", "close"],
    "expected_horizon": "1d",
    "invalidation": "(1) ls_netir <= 1.04（不优于最好成分 C1，compose 无增量，直接留 C1/comp1）；"
                    "(2) prod_corr >= 0.62（冗余较 C1 无改善）；"
                    "(3) ls_netret_ann < 0.164；"
                    "(4) turnover > 0.25；"
                    "(5) coverage < 0.75（M1 覆盖 0.81，融合后预期 ~0.8）；"
                    "(6) 2022/2023 仍双负且幅度 >5%（regime 缺陷未被分散化缓解=两成分共享同一熊市暴露, 正交性存疑）",
    "search_mode": "exploit",
    "semantic_key": "positioning_buildup+volume_surge|-|persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "compose_factor",
    "factor_family": "positioning",
    "parent_factor_id": "crypto_alpha_oi_buildup_premium_20_120",
    "semantic_mutation": "volume_surge|-|persistence|premium|rank + positioning_buildup|-|persistence|premium|rank -> "
                         "positioning_buildup+volume_surge|-|persistence|premium|rank",
    "hypothesis": "C1↔M1 逐日截面相关 0.194：流量热度与存量杠杆是真正交切面。融合后 ρ vs 库应从 0.62 稀释到 "
                  "~0.55，IC 叠加(0.0156+0.0263 的秩平均 ~0.022)，IR 若超 1.04=真互补。",
    "change": "两成分各自构造不变（20/120 双窗比、rank 域分数、shift(1)），factor = 0.5*score_C1 + 0.5*score_M1。"
              "评估走 pkl 预计算路径（矿工 OOM 风波期间瘦身），逻辑与本文件逐行一致。",
    "expected": "ls_netret_ann 0.14~0.20, ls_netir 0.9~1.3, |RankIC| 0.020~0.026, turnover ~0.07, "
                "prod_corr 0.52~0.60, coverage ~0.8。坦诚预期: C1 与 M1 的 2022-23 都弱(C1 2022 -2.9%, "
                "M1 -5.3/-3.7%), compose 大概率仍熊负——若如此但 IR/ρ/IC 全面改善, 判'compose 有效但 "
                "regime 天花板'，家族封顶。",
    "result": "REJECTED (STATISTICAL_REJECT, compose 无增量判据触发): ls_netir=0.9121 < 1.04(C1), "
              "净 11.48% < 16.4%, IC 0.0175 较 M1 的 0.0263 被稀释——rank 等权融合是平均化而非放大。"
              "改善面: 2023 转正(+3.4%, M1 为 -3.7%), prod_corr 0.5970 较 C1 的 0.62 略改善, "
              "2025 +32.6% 为家族最强年份。但 2022 -6.9% 三者最差(C1 -2.9/M1 -5.3)——"
              "关键教训: 分数正交(ρ0.194)≠ payoff 正交, 两成分通过同一 regime 暴露(熊市 bleed)相关, "
              "分散化救不了 2022。六图: demeaned 组6 +1.05 最高、组9 仅 ~0——长腿 payoff 在中段而非尾部, "
              "十分位长腿次优。家族 plateau 二次确认: 投机溢价资产全部表达收敛于 11~16% 净/IR 0.9~1.04, "
              "2022 熊市 bleed 在所有变体中存活(仅 comp1 的 D1 成分曾修复)。"
              "(pkl 预计算路径; m1 日级得分 pkl 损坏为 pandas datetime pickle quirk, 已改 parquet。)",
}


TYPE = "regular"

ATT_FAST_DAYS = 20
ATT_SLOW_DAYS = 120
MINUTES_PER_DAY = 1440
WARMUP_BARS = (ATT_SLOW_DAYS + ATT_FAST_DAYS + 10) * MINUTES_PER_DAY
CHUNK_SIZE = int(os.getenv("FACTOR_CHUNK_SIZE", "8"))
EPS = 1e-12

META = {
    "factor_name": "crypto_alpha_comp_attention_oi_buildup",
    "author": "wesleywu",
    "level": "minutes",
    "tag": "comp_attention_oi_buildup",
    "category": "positioning",
}

SETTING = {
    "data_needed": ["xbinance_taker_buy_quote_volume", "open_interest", "close"],
    "universe": "tradable_mask",
    "pasteurization": True,
    "warmup_bars": WARMUP_BARS,
}


def _chunk_columns(columns: list[str], chunk_size: int):
    chunk_size = max(int(chunk_size), 1)
    for i in range(0, len(columns), chunk_size):
        yield columns[i:i + chunk_size]


def _buildup_score(daily: pd.DataFrame) -> pd.DataFrame:
    """20d/120d 双窗比 -> premium 方向 rank 域分数 -> shift(1)"""
    fast = daily.rolling(ATT_FAST_DAYS, min_periods=ATT_FAST_DAYS).mean()
    slow = daily.rolling(ATT_SLOW_DAYS, min_periods=ATT_SLOW_DAYS).mean()
    ratio = fast / (slow + EPS)
    return -(ratio.rank(axis=1, pct=True) - 0.5).shift(1)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    taker = data_ctx["xbinance_taker_buy_quote_volume"]
    oi = data_ctx["open_interest"]
    close = data_ctx["close"]
    day = close.index.normalize()

    # 分块做分钟级日聚合（内存控制）：流量用 sum、存量用 mean
    taker_parts, oi_parts = [], []
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        taker_parts.append(taker.loc[:, cols].clip(lower=0).groupby(day).sum())
        oi_parts.append(oi.loc[:, cols].clip(lower=0).groupby(day).mean())

    taker_daily = pd.concat(taker_parts, axis=1)[close.columns]
    oi_daily = pd.concat(oi_parts, axis=1)[close.columns]
    del taker_parts, oi_parts

    score_c1 = _buildup_score(taker_daily)   # 投机流量热度
    score_m1 = _buildup_score(oi_daily)      # 杠杆存量拥挤

    # compose: 等权 rank 域融合
    factor_daily = 0.5 * score_c1 + 0.5 * score_m1

    # 分块广播回分钟级
    factor = pd.DataFrame(index=close.index, columns=close.columns, dtype=np.float32)
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        part = factor_daily[cols].reindex(day)
        part.index = close.index
        factor.loc[:, cols] = part.astype(np.float32)

    return factor.replace([np.inf, -np.inf], np.nan)
