import os

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "positioning_chase", "statement": "ΔOI(5d)>0 且有价格同向确认=杠杆资金顺势追仓；取幅度（unsigned 追仓拥挤度），ΔOI≤0 或无确认为中性 0"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "mark_premium_confirm", "statement": "mark_close/close-1 的 5d 均值与收益同号才计为有效追仓——成交价被推离标记价=主动情绪单驱动，过滤被动漂移"},
    ],
    "direction": {"id": "premium", "statement": "追仓拥挤溢价：固定符号空高追仓强度（任何方向的确认追仓=脆弱）多平静币（neglect），无方向预测"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "轮1（oi_chase_reversal_5d）的 demeaned 倒 U 证明：信号的经济内容是 unsigned 追仓拥挤度——"
                 "组0（追多拥挤）-0.6 与组9（追空拥挤）-1.2 双尾同毒，中段平静币全正；"
                 "signed reversal 装反了尾部方向（追空尾是清算级联延续不是 squeeze 反弹）。"
                 "本轮丢掉符号：空双尾合并的确认追仓币，多平静币块。",
    "fields":    ["open_interest", "close", "mark_close"],
    "expected_horizon": "1d",
    "invalidation": "(1) ls 毛收益仍 < 0（双尾合并做空仍不赚=倒 U 诊断错误）；"
                    "(2) |RankIC| < 0.008；(3) prod_corr >= 0.85；"
                    "(4) ls_netir < 0.3；(5) turnover > 1.5（本轮不治理换手，但失控到翻倍说明表达有结构性病）",
    "search_mode": "mutate",
    "semantic_key": "positioning_chase|-|mark_premium_confirm|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "positioning",
    "parent_factor_id": "crypto_alpha_oi_chase_reversal_5d",
    "parent_iter": None,
    "semantic_mutation": "positioning_chase|-|mark_premium_confirm|reversal|rank -> positioning_chase|-|mark_premium_confirm|premium|rank",
    "hypothesis": "轮1 六图证据：demeaned 组0(-0.6)/组9(-1.2)双尾同毒、中段组2~8全正=信号真身是"
                  "unsigned 追仓拥挤度，非 signed 反转。改为 premium 表达后：空腿=双 Toxic 尾合并"
                  "（轮1 组0+组9 demeaned 合计约 -0.9 均），长腿=平静币块（轮1 中段 +0.2~+0.4），"
                  "毛端应转正。轮1 prod_corr=0.20 已证维度增量，本轮沿用同字段同窗口，方向/符号是唯一点改动。",
    "reasoning": "轮1 Return NAV 毛端 -15%/yr=尾部方向装反（非成本问题，成本只解释净毛差 ~34%）；"
                 "单点改动只修方向，换手治理（平滑/滞回）明确留给后轮，避免一轮两改无法归因。",
    "change": "chase 去掉 sign(ret5)：intensity=doi5.clip(lower=0)*confirmed（unsigned 强度）；"
              "factor=-(rank(intensity)-0.5)，其余（5d 窗、mark 确认、shift(1) 广播）与轮1 逐行一致。",
    "expected": "ls 毛收益转正（+10~25%/yr）；|RankIC| 持平或升（0.012~0.02）；"
                "turnover 仍 0.8~1.3（本轮不治理）；ls_netir 毛口径转正、净口径仍可能为负——"
                "本轮验收标准是毛端转正+分组单调性修复，净门槛等换手治理后再过。",
    "result": "REJECTED (STATISTICAL_REJECT, IS 2022~2024): |RankIC|=0.0164/|ICIR|=3.48 双过门且强于轮1, "
              "turnover 0.509 意外过门(去符号后排名自稳), prod_corr=0.1527 PASS(更低); "
              "空腿假设验证(demeaned 组0/1=-0.35/-0.45 全场最差, 追仓拥挤币确实跑输), "
              "但长腿假设证伪: 组9平静币 demeaned≈0 平盘(平静≠neglect溢价, 最优是中段组5), "
              "且零值并列块(~70%币)pct-rank平均秩~0.3 被十分位切分整体排除→long_nums全期≈0, "
              "长腿结构性无法填充(零值膨胀因子×十分位LS引擎错配); 强regime依赖: 2022熊市+92.3% vs "
              "2023/2024牛市-47~-49%(牛市空追仓尾=空火箭)。结论: 可变现内容=空追仓拥挤尾, "
              "归宿是universe层短黑名单/compose素材(与bybit_heat元规则第三次复现), standalone LS路线关闭。",
    "oos_result": None,
}


TYPE = "regular"

HORIZON_DAYS = 5
MINUTES_PER_DAY = 1440
WARMUP_BARS = (HORIZON_DAYS * 3 + 5) * MINUTES_PER_DAY
CHUNK_SIZE = int(os.getenv("FACTOR_CHUNK_SIZE", "8"))

META = {
    "factor_name": "crypto_alpha_oi_chase_intensity_premium_5d",
    "author": "wesleywu",
    "level": "minutes",
    "tag": "oi_chase_intensity_premium",
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

    # event positioning_chase（unsigned 幅度用法）: 追仓拥挤度 = max(doi5, 0)
    ret5 = close_daily.pct_change(HORIZON_DAYS, fill_method=None)
    doi5 = oi_daily.pct_change(HORIZON_DAYS, fill_method=None).clip(lower=0)

    # quality mark_premium_confirm: prem5 与 ret5 同号才保留，否则中性 0
    prem5 = (mark_daily / close_daily - 1).rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS).mean()
    confirmed = np.sign(prem5) == np.sign(ret5)
    intensity = doi5.where(confirmed, 0.0)

    # direction premium: 空高追仓强度（任何方向的确认追仓=脆弱），多平静币，固定符号
    factor_daily = -(intensity.rank(axis=1, pct=True) - 0.5)
    factor_daily = factor_daily.shift(1)  # 只用已完成交易日

    # 分块广播回分钟级
    factor = pd.DataFrame(index=close.index, columns=close.columns, dtype=np.float32)
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        part = factor_daily[cols].reindex(day)
        part.index = close.index
        factor.loc[:, cols] = part.astype(np.float32)

    return factor.replace([np.inf, -np.inf], np.nan)
