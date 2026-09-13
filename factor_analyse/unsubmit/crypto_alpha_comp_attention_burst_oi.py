import os

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "activity_burst+positioning_buildup+volume_surge", "statement": "compose: C1 taker 买盘美元额 20/120(投机流量) + D1 trade_count 20d 变异系数(活动阵发) + M1 OI 20/120(杠杆存量)"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "persistence", "statement": "三成分均为慢变持续结构（双窗比/长窗 CV)"},
    ],
    "direction": {"id": "premium", "statement": "投机过度定价三重溢价：空升温+空阵发+空堆积，多冷清+多平稳+多出清"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "comp1(C1+D1)修复了 2022(+2.0%)但 2026 弱(-2.4%);M1 带来持仓层增量(ρ0.4574)和 "
                 "2024-26 强度(+24.9/+20.8/+13.2%)但 2022-23 bleed。三者两两测量正交(C1↔M1 ρ=0.194), "
                 "互补结构: D1 补 2022、M1 补 2024-26、C1 出尾部。C2 教训已内化：分数正交≠payoff 正交, "
                 "本轮预注册直接把'分年全正'设为硬判据——若三合一仍不能同时修 2022 和 2026, "
                 "证明该资产的 regime 暴露不可通过成分分散化消除，家族封顶。",
    "fields":    ["xbinance_taker_buy_quote_volume", "xbinance_trade_count", "open_interest", "close"],
    "expected_horizon": "1d",
    "invalidation": "(1) ls_netir <= 1.04（不优于最好成分 C1，compose 无增量）；"
                    "(2) 任一完整年份 ls_netret <= 0（含 2022 与 2026  partial——分年全正为本轮硬判据, 不过=regime 暴露不可分散, 家族封顶）；"
                    "(3) prod_corr >= 0.60；"
                    "(4) turnover > 0.25；"
                    "(5) coverage < 0.75",
    "search_mode": "exploit",
    "semantic_key": "activity_burst+positioning_buildup+volume_surge|-|persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "compose_factor",
    "factor_family": "composite",
    "parent_factor_id": "crypto_alpha_comp_attention_burst_premium",
    "semantic_mutation": "activity_burst+volume_surge + positioning_buildup -> "
                         "activity_burst+positioning_buildup+volume_surge(三等权)",
    "hypothesis": "comp1(15.7%/1.01/2022 +2.0/2026 -2.4) + M1(10.35%/0.90/2024-26 强) 三等权融合, "
                  "赌互补结构：D1 修 2022、M1 修 2024-26。若 IR>1.04 且分年全正=真三正交互补; "
                  "否则 regime 暴露不可分散，家族封顶。",
    "change": "三成分各自构造不变（rank 域分数），factor = (score_C1+score_D1+score_M1)/3，统一 shift(1)。"
              "评估走 pkl 预计算路径（串行单字段加载，峰值 ~15G)。",
    "expected": "ls_netret_ann 0.15~0.22, ls_netir 1.0~1.3, |RankIC| 0.020~0.025, turnover ~0.08, "
                "prod_corr 0.55~0.60, coverage ~0.8, 2022 与 2026 转正是核心悬念。",
    "result": "REJECTED (STATISTICAL_REJECT, 家族封顶): 双硬判据全灭——ls_netir=0.4451 << 1.04, "
              "2022 -11.7% 为全家族最深(comp1 的 D1 修复力被 M1 稀释, 2022 反而恶化); "
              "净 6.1%/yr, prod_corr 0.6052 WARN。三元相关图谱: C1↔D1=0.888(高度同源!), "
              "C1↔M1=0.203, D1↔M1=0.082——M1 是唯一独立信息源, 但三等权把 C1/D1 的重复资产"
              "稀释到 2/3 权重, 放大的恰是共享的 2022 regime 暴露。六图: NAV 2022 下凹至 0.75, "
              "demeaned 组6-7 最高而组9 仅 0.3(长腿 payoff 中段化在三合一上重现)。"
              "最终结论: 投机过度定价资产的单因子/compose 表达全部收敛于 6~16% 净、IR 0.45~1.04, "
              "2022 熊市 bleed 为结构性、不可通过成分分散化消除——家族正式封顶。"
              "留档: C1(IR/净最高) + comp1(分年最稳)。M1 的 OI 层增量(ρ0.46)为真实资产, "
              "未来若做多因子组合层的 regime 中性化可复活, 单因子口径下不再迭代。",
}


TYPE = "regular"

ATT_FAST_DAYS = 20
ATT_SLOW_DAYS = 120
BURST_DAYS = 20
MINUTES_PER_DAY = 1440
WARMUP_BARS = (ATT_SLOW_DAYS + ATT_FAST_DAYS + 10) * MINUTES_PER_DAY
CHUNK_SIZE = int(os.getenv("FACTOR_CHUNK_SIZE", "8"))
EPS = 1e-12

META = {
    "factor_name": "crypto_alpha_comp_attention_burst_oi",
    "author": "wesleywu",
    "level": "minutes",
    "tag": "comp_attention_burst_oi",
    "category": "composite",
}

SETTING = {
    "data_needed": ["xbinance_taker_buy_quote_volume", "xbinance_trade_count", "open_interest", "close"],
    "universe": "tradable_mask",
    "pasteurization": True,
    "warmup_bars": WARMUP_BARS,
}


def _chunk_columns(columns: list[str], chunk_size: int):
    chunk_size = max(int(chunk_size), 1)
    for i in range(0, len(columns), chunk_size):
        yield columns[i:i + chunk_size]


def _buildup_ratio(daily: pd.DataFrame) -> pd.DataFrame:
    """20d/120d 双窗比 -> premium 方向 rank 域分数（未 shift)"""
    fast = daily.rolling(ATT_FAST_DAYS, min_periods=ATT_FAST_DAYS).mean()
    slow = daily.rolling(ATT_SLOW_DAYS, min_periods=ATT_SLOW_DAYS).mean()
    ratio = fast / (slow + EPS)
    return -(ratio.rank(axis=1, pct=True) - 0.5)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    taker = data_ctx["xbinance_taker_buy_quote_volume"]
    trade_count = data_ctx["xbinance_trade_count"]
    oi = data_ctx["open_interest"]
    close = data_ctx["close"]
    day = close.index.normalize()

    # 分块做分钟级日聚合（内存控制）：流量/活动用 sum、存量用 mean
    taker_parts, tc_parts, oi_parts = [], [], []
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        taker_parts.append(taker.loc[:, cols].clip(lower=0).groupby(day).sum())
        tc_parts.append(trade_count.loc[:, cols].clip(lower=0).groupby(day).sum())
        oi_parts.append(oi.loc[:, cols].clip(lower=0).groupby(day).mean())

    taker_daily = pd.concat(taker_parts, axis=1)[close.columns]
    tc_daily = pd.concat(tc_parts, axis=1)[close.columns]
    oi_daily = pd.concat(oi_parts, axis=1)[close.columns]
    del taker_parts, tc_parts, oi_parts

    # 成分 C1 volume_surge: 买盘美元额 20/120 关注度 regime，空拥挤
    score_c1 = _buildup_ratio(taker_daily)

    # 成分 D1 activity_burst: 活动 20d 变异系数，空阵发
    tc_mean = tc_daily.rolling(BURST_DAYS, min_periods=BURST_DAYS).mean()
    tc_std = tc_daily.rolling(BURST_DAYS, min_periods=BURST_DAYS).std()
    burst = tc_std / (tc_mean + EPS)
    score_d1 = -(burst.rank(axis=1, pct=True) - 0.5)

    # 成分 M1 positioning_buildup: OI 20/120 杠杆拥挤 regime，空堆积
    score_m1 = _buildup_ratio(oi_daily)

    # compose: 三等权 rank 域融合
    factor_daily = (score_c1 + score_d1 + score_m1) / 3.0
    factor_daily = factor_daily.shift(1)  # 只用已完成交易日

    # 分块广播回分钟级
    factor = pd.DataFrame(index=close.index, columns=close.columns, dtype=np.float32)
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        part = factor_daily[cols].reindex(day)
        part.index = close.index
        factor.loc[:, cols] = part.astype(np.float32)

    return factor.replace([np.inf, -np.inf], np.nan)
