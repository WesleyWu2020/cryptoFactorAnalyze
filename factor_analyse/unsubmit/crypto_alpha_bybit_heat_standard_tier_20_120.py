import os

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "volume_surge", "statement": "turnover（Bybit 侧活动）20d/120d 慢速抬升=该场地投机热度 regime 上行"},
    "context":   {"id": "listing_tier", "statement": "仅在 standards==True 标准区内激活打分，非标准/创新区置 NaN 排除出截面"},
    "qualities": [
        {"id": "persistence", "statement": "双长窗均值比（20d/120d）本身是持续数周的热度趋势，非单日脉冲"},
    ],
    "direction": {"id": "premium", "statement": "投机热度溢价：固定符号空升温（拥挤=过度定价）多降温（冷清=neglect），无方向预测"},
    "output":    {"id": "rank", "statement": "截面 rank 打分（在标准区子集内 rank）"},
    "mechanism": "Q1 证明 Bybit 热度信号 IC 极强（0.0393）但组合颗粒无收：组0 极端热币是 squeeze 火箭独赢 +0.9，"
                 "炸掉空腿，组1-8 服从热度空头逻辑。先验：火箭集中在非标准/创新区（meme 币），标准区（主流通行证）"
                 "的上市审核滤掉纯投机壳，极端热更可能是真实事件驱动而非轧空——若标准区内热度空头干净，"
                 "组合应能把 IC 资产变现。这是把'极端热尾部=空头死亡区'元规则在 universe 层的变现实验："
                 "不做择时门（已判废），只做逐币静态分层排除。",
    "fields":    ["turnover", "close", "standards"],
    "expected_horizon": "1d",
    "invalidation": "(1) ls_netret_ann < 0.20（分层后组合仍无法变现=火箭不在非标准区，分层假说证伪，此方向关闭）；"
                    "(2) 原始 RankIC 掉到 |RankIC| < 0.015（分层毁掉信号本身，Q1 的 IC 来自被排除的币）；"
                    "(3) turnover(组合换手) > 0.25；"
                    "(4) coverage < 0.5（standards~71% True，预注册线从 0.9 降到 0.5）；"
                    "(5) prod_corr >= 0.85 vs Q1（若恰好相关性爆表且绩效近似=纯子集无新意）",
    "search_mode": "exploit",
    "semantic_key": "volume_surge|listing_tier|persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "modify_factor",
    "factor_family": "sentiment",
    "parent_factor_id": "crypto_alpha_bybit_heat_premium_20_120",
    "semantic_mutation": "context: - → listing_tier（新增词条：standards 静态分层掩码作 context）",
    "hypothesis": "Q1 的 IC 资产（0.0393/6.54）被组0 squeeze 火箭独赢吃掉，中段 1-8 组服从热度空头逻辑。"
                  "若火箭集中于非标准/创新区，则仅在标准区激活打分可避开毒尾，组合变现 IC。",
    "change": "母体打分逻辑不变；standards 日级静态掩码（groupby(day).last()>0.5），"
              "非标准区币置 NaN 后再做截面 pct rank（子集内排序），其余同 Q1。",
    "expected": "coverage ~0.6-0.7（静态掩码 ~71% × 原覆盖）；|RankIC| 0.02-0.04；"
                "核心判据 ls_netret_ann：>0.20=分层救活，<0.20=火箭在标准区或分布无关，方向关闭。",
    "result": "REJECTED (IMPLEMENTATION_INVALID, 操纵空操作): 全部指标与 Q1 逐位一致"
              "(|RankIC|=0.0393/|ICIR|=6.5382/ls_netir=0.0195/coverage=0.9253/prod_corr=0.5615 vs 同一 GP)。"
              "根因(三层探针实证): standards(=listed_mask) 在全 539 币上 38% True(198 全 True/324 全 False/17 混合), "
              "但 tradable_mask 宇宙(~265 币/日)内 standards=True 占比 1643 天全部 100.0%——非标准区整个在"
              "可投宇宙之外, 分层 mask 在宇宙内是完美空操作。推论: Q1 的组0 squeeze 火箭就在标准区主流币里, "
              "'火箭集中在创新区'的先验被间接证伪; listing_tier 维度在本宇宙内零变异, 无信息增量, 方向关闭。"
              "教训: 用静态掩码做 context 前, 先验证掩码在 universe 内的方差(本次 universe≈265 与全样本 539 "
              "的差异被忽视, 预注册 coverage 0.5 线建立在不成立的前提上)。",
}


TYPE = "regular"

HEAT_FAST_DAYS = 20
HEAT_SLOW_DAYS = 120
MINUTES_PER_DAY = 1440
WARMUP_BARS = (HEAT_SLOW_DAYS + HEAT_FAST_DAYS + 10) * MINUTES_PER_DAY
CHUNK_SIZE = int(os.getenv("FACTOR_CHUNK_SIZE", "8"))
EPS = 1e-12

META = {
    "factor_name": "crypto_alpha_bybit_heat_standard_tier_20_120",
    "author": "wesleywu",
    "level": "minutes",
    "tag": "bybit_heat_standard_tier",
    "category": "sentiment",
}

SETTING = {
    "data_needed": ["turnover", "close", "standards"],
    "universe": "tradable_mask",
    "pasteurization": True,
    "warmup_bars": WARMUP_BARS,
}


def _chunk_columns(columns: list[str], chunk_size: int):
    chunk_size = max(int(chunk_size), 1)
    for i in range(0, len(columns), chunk_size):
        yield columns[i:i + chunk_size]


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    turnover = data_ctx["turnover"]
    close = data_ctx["close"]
    standards = data_ctx["standards"]
    day = close.index.normalize()

    # 分块做分钟级日聚合（内存控制），截面 rank 在小型日频帧上统一做
    to_parts = []
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        to_parts.append(
            turnover.loc[:, cols].clip(lower=0).groupby(day).sum()
        )

    to_daily = pd.concat(to_parts, axis=1)[close.columns]
    del to_parts

    # event volume_surge（慢速版）: Bybit 活动热度 regime = 20d/120d 双窗比
    heat_fast = to_daily.rolling(HEAT_FAST_DAYS, min_periods=HEAT_FAST_DAYS).mean()
    heat_slow = to_daily.rolling(HEAT_SLOW_DAYS, min_periods=HEAT_SLOW_DAYS).mean()
    heat = heat_fast / (heat_slow + EPS)

    # context listing_tier: standards 静态分层掩码，仅标准区（True）保留，非标准区置 NaN
    std_daily = standards.astype(float).groupby(day).last().reindex(heat.index).ffill()
    std_mask = std_daily > 0.5
    heat_tiered = heat.where(std_mask)

    # direction premium: 空升温（投机拥挤）多降温（neglect），固定符号；子集内 rank
    factor_daily = -(heat_tiered.rank(axis=1, pct=True) - 0.5)
    factor_daily = factor_daily.shift(1)  # 只用已完成交易日

    # 分块广播回分钟级
    factor = pd.DataFrame(index=close.index, columns=close.columns, dtype=np.float32)
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        part = factor_daily[cols].reindex(day)
        part.index = close.index
        factor.loc[:, cols] = part.astype(np.float32)

    return factor.replace([np.inf, -np.inf], np.nan)
