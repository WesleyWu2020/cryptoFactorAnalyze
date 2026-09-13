import os

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "basis_dislocation", "statement": "mark_close/index_close-1 基差水平高=perp 被买盘推离指数公允价（盘口失衡流的日度留影），低/负基差=被砸离公允价"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [],
    "direction": {"id": "mean_reversion", "statement": "基差向公允价收敛：高基差币未来跑输（价格收敛+空头收 funding 双重收益），低基差币跑赢"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "T3 基差层首次直接挖掘（词表 basis_dislocation 零实验记录）。"
                 "IS 探针（2022~2024，4h 抽样）：-basis 日级 RankIC=+0.0249（日 t≈6.6），"
                 "强于 20d 中位数偏离（+0.0212）与 premium_close 偏离（+0.0071）——"
                 "信号在基差水平本身，慢 norm 稀释信号。"
                 "机制：高基差=主动买盘把 perp 推离指数，造临时高估；随后 (a) 套利盘压回收敛、"
                 "(b) 高基差→高 funding，空头持仓还收资金费。与 funding_extreme 不同质："
                 "funding 是 8h 离散费率的极端化，basis 是连续价格偏离本身。",
    "fields":    ["mark_close", "index_close"],
    "expected_horizon": "1d",
    "invalidation": "(1) RankIC 符号为负（探针 +0.0249，若引擎口径翻负=实现错误）；"
                    "(2) |RankIC| < 0.012（探针值的一半，低于此=日聚合损失过半信号）；"
                    "(3) prod_corr >= 0.85——最大风险：高基差≈当日跑赢≈短期反转换皮，GP 反转类因子密集；"
                    "(4) ls_netir < 0.3；(5) turnover > 1.5（日频 rank 无平滑，换手是已知风险，超限则下轮转平滑表达）",
    "search_mode": "explore",
    "semantic_key": "basis_dislocation|-|-|mean_reversion|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "basis",
    "parent_factor_id": None,
    "semantic_mutation": None,
    "hypothesis": "基差均值回归：IS 探针 -basis RankIC=+0.0249 且逐日稳定；"
                  "premium_close 作慢 norm 的构造被 IS 证据否决（IC 稀释到 0.0071），"
                  "故用最简表达：截面 rank 当日基差水平，取负。",
    "reasoning": "轨迹审查后选 T3 基差层：positioning 层 4 次 IS 失败已关闭、taker/波动 premium 家族 "
                 "plateau、volume_instability 封顶；basis_dislocation 是词表上从未实验的条目，"
                 "符合 factor-families 决策树'优先 Tier 3 未占用信息源'。"
                 "字段选择 mark_close/index_close/premium_close 三选层，探针淘汰 premium_close 构造。",
    "change": "新因子。basis=mark_close/index_close-1（日级 last）；"
              "factor=-(rank(basis)-0.5)，shift(1) 广播分钟级。无窗口、无平滑、无过滤——最简机制测试。",
    "expected": "|RankIC| 0.020~0.030（探针 0.0249），ls_netir 0.3~1.0，"
                "turnover 0.5~1.2（日频 rank 无平滑，主风险项；超限→下轮转 EMA 平滑单点改动），"
                "coverage ~1.0。核心悬念=prod_corr：basis 日 rank 与短期反转 GP 的语义距离待测。",
    "result": "REJECTED (STATISTICAL_REJECT, IS 2022~2024): delay=1 |RankIC|=0.0043 未过门——"
              "但 delay=0 对照组 |RankIC|=0.0209/|ICIR|=4.40: 基差反转 edge 半衰期<1天, "
              "79% IC 在一根 bar 内衰减完, 探针 +0.0249 本质是 trade-at-close 幻觉(探针口径≈delay=0)。"
              "prod_corr=0.3231 PASS(与反转GP有距离, 非换皮)。毛 ls +10%/yr 存在但 turnover 1.44 "
              "年拖累~39% 吃成净-29.7%。demeaned 组1(高基差)-0.55 全场最差=高基差毒性真实, "
              "但组9平盘=低基差无溢价(同轮2签名: 只有空尾)。宪法禁止换 perp_8h 救活→"
              "快基差线在日频关闭; 慢基差水平≈funding代理(funding分支已关闭), trio 语义穷尽。"
              "元教训: IS 探针的 fwd 收益必须用 delay=1 一致口径(shift(-2)), 否则快信号全高估。",
    "oos_result": None,
}


TYPE = "regular"

MINUTES_PER_DAY = 1440
WARMUP_BARS = 3 * MINUTES_PER_DAY
CHUNK_SIZE = int(os.getenv("FACTOR_CHUNK_SIZE", "8"))

META = {
    "factor_name": "crypto_alpha_basis_meanrev_1d",
    "author": "wesleywu",
    "level": "minutes",
    "tag": "basis_meanrev",
    "category": "basis",
}

SETTING = {
    "data_needed": ["mark_close", "index_close"],
    "universe": "tradable_mask",
    "pasteurization": True,
    "warmup_bars": WARMUP_BARS,
}


def _chunk_columns(columns: list[str], chunk_size: int):
    chunk_size = max(int(chunk_size), 1)
    for i in range(0, len(columns), chunk_size):
        yield columns[i:i + chunk_size]


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mark = data_ctx["mark_close"]
    indx = data_ctx["index_close"]
    day = mark.index.normalize()

    # 分块分钟级日聚合（内存控制）
    mark_parts, indx_parts = [], []
    for cols in _chunk_columns(list(mark.columns), CHUNK_SIZE):
        mark_parts.append(mark.loc[:, cols].groupby(day).last())
        indx_parts.append(indx.loc[:, cols].groupby(day).last())

    mark_daily = pd.concat(mark_parts, axis=1)[mark.columns]
    indx_daily = pd.concat(indx_parts, axis=1)[mark.columns]
    del mark_parts, indx_parts

    # event basis_dislocation: perp 相对指数公允价的偏离水平
    basis = mark_daily / indx_daily - 1

    # direction mean_reversion: 空高基差（高估）多低基差（低估），截面 rank
    factor_daily = -(basis.rank(axis=1, pct=True) - 0.5)
    factor_daily = factor_daily.shift(1)  # 只用已完成交易日

    # 分块广播回分钟级
    factor = pd.DataFrame(index=mark.index, columns=mark.columns, dtype=np.float32)
    for cols in _chunk_columns(list(mark.columns), CHUNK_SIZE):
        part = factor_daily[cols].reindex(day)
        part.index = mark.index
        factor.loc[:, cols] = part.astype(np.float32)

    return factor.replace([np.inf, -np.inf], np.nan)
