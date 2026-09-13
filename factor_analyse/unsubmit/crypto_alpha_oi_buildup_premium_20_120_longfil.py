import os

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "positioning_buildup", "statement": "open_interest（币单位持仓量）20d/120d 慢速抬升=杠杆持仓堆积、拥挤度 regime 上行"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "persistence", "statement": "双长窗均值比（20d/120d）本身是持续数周的持仓堆积趋势，非单日脉冲"},
        {"id": "outlier_filter", "statement": "OI 极端崩塌尾部（buildup 截面 rank 最低 10%）置 NaN 排除——长腿从崩塌币换成温和出清币"},
    ],
    "direction": {"id": "premium", "statement": "杠杆拥挤溢价：固定符号空堆积（拥挤=脆弱=未来跑输）多出清（neglect），无方向预测"},
    "output":    {"id": "rank", "statement": "截面 rank 打分（剔除崩塌尾部后的子集内 rank）"},
    "mechanism": "M1 证明持仓层增量维度成立（ρ0.4574 逃逸、IC 0.0263/4.89 全样本稳定、组0 极端堆积可空），"
                 "唯一败因是长腿：OI 极端崩塌币在 2022-23 熊市 bleed——极端 OI 崩塌≠健康出清，"
                 "而是濒死币（LUNA/FTX 式连环清算后无人再接），long 它们=接飞刀。"
                 "把 buildup 最低 10%（与组合十分位对齐）置 NaN，长腿自动变成 10~20% 带的温和出清币；"
                 "空腿（alpha 源）完全不动。",
    "fields":    ["open_interest", "close"],
    "expected_horizon": "1d",
    "invalidation": "(1) 2022 或 2023 ls_netret 仍为负（第一判据：濒死币污染假说证伪，长腿问题不在崩塌尾部，M1 家族归档）；"
                    "(2) ls_netret_ann < 0.20 且 ls_netir < 1.0（修复后仍差临门=该维度单因子上限如此，转 compose 素材归档）；"
                    "(3) |RankIC| < 0.02（剔除尾部毁掉信号本身）；"
                    "(4) turnover(组合换手) > 0.25；"
                    "(5) prod_corr >= 0.85 vs M1 母体或库内（纯子集无新意）",
    "search_mode": "exploit",
    "semantic_key": "positioning_buildup|-|outlier_filter+persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "modify_factor",
    "factor_family": "positioning",
    "parent_factor_id": "crypto_alpha_oi_buildup_premium_20_120",
    "semantic_mutation": "qualities: persistence → outlier_filter+persistence（buildup 最低 10% NaN 排除）",
    "hypothesis": "M1 的 2022(-5.3%)/2023(-3.7%) 熊市 bleed 来自长腿：极端 OI 崩塌币=濒死币污染。"
                  "剔除 buildup 最低 10% 后长腿换成温和出清币，分年应转正；"
                  "组0 空腿（alpha 源）不动，IC 资产保留。",
    "change": "母体打分逻辑不变；buildup 截面 pct rank < 0.10 的币置 NaN 后再 rank，其余同 M1。"
              "阈值 10% 一次预注册不扫参（与组合十分位粒度对齐）。",
    "expected": "第一判据=分年全正（2022/2023 ≥ 0）；|RankIC| 0.02~0.026（略降）；"
                "ls_netret 10%→15%+；换手 0.06 不变；coverage 0.81→~0.73（剔 10% 后，预注册线降至 0.6 同 eval 门）。",
    "result": "REJECTED (STATISTICAL_REJECT, 濒死币污染假说证伪): 第一判据失败——2022 -4.75%/2023 -6.04% "
              "仍为负(2023 比 M1 更差), 且整体恶化: ls_netir 0.898→0.573, 净 10.35%→6.75%/yr; "
              "2024-26 好年份也被削弱(+24.9→+22.4, +20.8→+12.7, +13.2→+8.7)——崩塌尾部在所有年份都含"
              "正收益(崩盘后反弹币), 一刀切剔除=全年份失血。IC 反而升至 0.0297/5.46: 尾部剔除改善了"
              "截面排序但伤害尾部组合, 又一个'IC 与组合打架'案例。结论: M1 的 2022-23 bleed 不是"
              "长腿崩塌尾部污染, 而是 OI 出清币在熊市的广谱跑输——premium 方向在持仓层是 regime 条件化的, "
              "而市场级择时门已被 #32 判废, 单因子结构性不可修复。M1 家族 standalone 归档; "
              "M1 本体(ρ0.4574 增量、IC 全样本稳定)留作 compose 素材。"
              "(评估走 pkl 预计算路径绕过矿工 OOM 风波, 逻辑与 .py 逐行一致。)",
}


TYPE = "regular"

BUILDUP_FAST_DAYS = 20
BUILDUP_SLOW_DAYS = 120
COLLAPSE_EXCLUDE_PCT = 0.10   # OI 极端崩塌尾部（buildup 截面 rank 最低 10%）置 NaN
MINUTES_PER_DAY = 1440
WARMUP_BARS = (BUILDUP_SLOW_DAYS + BUILDUP_FAST_DAYS + 10) * MINUTES_PER_DAY
CHUNK_SIZE = int(os.getenv("FACTOR_CHUNK_SIZE", "8"))
EPS = 1e-12

META = {
    "factor_name": "crypto_alpha_oi_buildup_premium_20_120_longfil",
    "author": "wesleywu",
    "level": "minutes",
    "tag": "oi_buildup_premium_longfil",
    "category": "positioning",
}

SETTING = {
    "data_needed": ["open_interest", "close"],
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
    day = close.index.normalize()

    # 分块做分钟级日聚合（内存控制），截面 rank 在小型日频帧上统一做
    oi_parts = []
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        oi_parts.append(
            oi.loc[:, cols].clip(lower=0).groupby(day).mean()
        )

    oi_daily = pd.concat(oi_parts, axis=1)[close.columns]
    del oi_parts

    # event positioning_buildup: OI 杠杆拥挤 regime = 20d/120d 双窗比
    buildup_fast = oi_daily.rolling(BUILDUP_FAST_DAYS, min_periods=BUILDUP_FAST_DAYS).mean()
    buildup_slow = oi_daily.rolling(BUILDUP_SLOW_DAYS, min_periods=BUILDUP_SLOW_DAYS).mean()
    buildup = buildup_fast / (buildup_slow + EPS)

    # quality outlier_filter: OI 极端崩塌尾部（buildup 最低 10%）置 NaN——濒死币污染排除
    buildup_pct = buildup.rank(axis=1, pct=True)
    buildup_fil = buildup.where(buildup_pct >= COLLAPSE_EXCLUDE_PCT)

    # direction premium: 空堆积（杠杆拥挤=脆弱）多出清（neglect），固定符号；子集内 rank
    factor_daily = -(buildup_fil.rank(axis=1, pct=True) - 0.5)
    factor_daily = factor_daily.shift(1)  # 只用已完成交易日

    # 分块广播回分钟级
    factor = pd.DataFrame(index=close.index, columns=close.columns, dtype=np.float32)
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        part = factor_daily[cols].reindex(day)
        part.index = close.index
        factor.loc[:, cols] = part.astype(np.float32)

    return factor.replace([np.inf, -np.inf], np.nan)
