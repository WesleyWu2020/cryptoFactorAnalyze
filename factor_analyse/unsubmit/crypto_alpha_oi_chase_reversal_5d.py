import os

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "positioning_chase", "statement": "ΔOI(5d)>0 且与 5d 收益同号=杠杆资金顺势追仓拥挤；OI 出清视为中性不计"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "mark_premium_confirm", "statement": "mark_close/close-1 的 5d 均值与收益同号才计为有效追仓——成交价被推离标记价=主动情绪单驱动，过滤被动漂移"},
    ],
    "direction": {"id": "reversal", "statement": "追多拥挤→回落（空），追空拥挤→反弹（多）；事件方向反转"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "M1（oi_buildup_premium）证明持仓层是真增量维度（ρ0.4574），但其 unsigned 慢变水平用法"
                 "standalone 两次 STATISTICAL_REJECT 已关闭。本轮换 signed 快变流量：OI 象限逻辑——"
                 "价格上涨+OI 增=新多追仓（拥挤多头，对清算级联脆弱，未来回落）；"
                 "价格下跌+OI 增=新空追仓（拥挤空头，squeeze 燃料，未来反弹）。"
                 "mark premium 确认追仓由主动情绪单驱动（成交价偏离标记价）而非被动漂移。",
    "fields":    ["open_interest", "close", "mark_close"],
    "expected_horizon": "1d",
    "invalidation": "(1) RankIC 显著为负（|IC|>0.01 且符号反）=反转假设证伪、追仓实为延续；"
                    "(2) |RankIC| < 0.008 =signed 流量无截面信息；"
                    "(3) prod_corr >= 0.85（vs gp_auto_20260608_145002_seed430_r34 等 OI/价 GP）；"
                    "(4) ls_netir < 0.3；(5) turnover > 0.8（5d 快信号换手天然高于 M1 的慢水平）",
    "search_mode": "explore",
    "semantic_key": "positioning_chase|-|mark_premium_confirm|reversal|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "positioning",
    "parent_factor_id": None,
    "semantic_mutation": None,
    "hypothesis": "OI 象限经典逻辑在 crypto perp 的 signed 表达：顺势追仓拥挤→反转。"
                  "IS 探针（2022~2024，4h 抽样）：象限分布均衡（0.20~0.29），doi5 IQR ±8~9% 有真实变异，"
                  "mark/close 偏离 p5/p95=-8.9/+14.0bp，prem5 与 ret5 同号率 0.437（确认项有约 56% 过滤力）。"
                  "与已关闭的 M1 不同质：M1=unsigned 慢变水平（premium），本轮=signed 快变流量（reversal）。",
    "change": "新因子。日级：ret5=close.pct_change(5)；doi5=oi.pct_change(5).clip(lower=0)；"
              "prem5=(mark/close-1).rolling(5).mean()；chase=sign(ret5)*doi5，仅 sign(prem5)==sign(ret5) 保留，否则置 0；"
              "factor=-(rank(chase)-0.5)，shift(1) 广播分钟级。",
    "expected": "|RankIC| 0.010~0.025, ls_netir 0.3~0.8, turnover 0.2~0.6（5d rank 快于 M1 的 0.063）, "
                "coverage>0.9（三字段宇宙内覆盖均 ~1.0）。核心悬念=prod_corr：GP 已及 OI+价字段，"
                "但 signed 象限结构与 GP 的量价/水平表达应有距离（M1 已证持仓层 ρ<0.5）。",
    "result": "REJECTED (STATISTICAL_REJECT, IS 2022~2024): |RankIC|=0.0127/|ICIR|=2.61 双过门, "
              "prod_corr=0.2000 PASS（signed 持仓流量与 GP 库几乎零撞车，增量维度二次确认）; "
              "败因: signed reversal 方向被六图证伪 + 换手失控——demeaned 倒 U: 组0(追多拥挤)-0.6、"
              "组9(追空拥挤)-1.2 双尾同毒, 追空拥挤无反弹=清算级联延续非 squeeze, "
              "IC 全来自'平静币>任何确认追仓币'的中段梯度; ls 毛-15%/净-48.7%/yr, 三年全负, maxdd 0.89; "
              "turnover 1.23(基线1~2+spike 5~18, 硬过滤0值并列块日洗十分位), 年成本拖累~34%。"
              "下一轮: 方向 reversal→premium（unsigned 追仓强度，空双尾多平静币），换手治理再下一轮。",
    "oos_result": None,
}


TYPE = "regular"

HORIZON_DAYS = 5
MINUTES_PER_DAY = 1440
WARMUP_BARS = (HORIZON_DAYS * 3 + 5) * MINUTES_PER_DAY
CHUNK_SIZE = int(os.getenv("FACTOR_CHUNK_SIZE", "8"))

META = {
    "factor_name": "crypto_alpha_oi_chase_reversal_5d",
    "author": "wesleywu",
    "level": "minutes",
    "tag": "oi_chase_reversal",
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

    # event positioning_chase: 顺势追仓 = sign(ret5) * max(doi5, 0)
    ret5 = close_daily.pct_change(HORIZON_DAYS, fill_method=None)
    doi5 = oi_daily.pct_change(HORIZON_DAYS, fill_method=None).clip(lower=0)

    # quality mark_premium_confirm: prem5 与 ret5 同号才保留，否则中性 0
    prem5 = (mark_daily / close_daily - 1).rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS).mean()
    confirmed = np.sign(prem5) == np.sign(ret5)
    chase = (np.sign(ret5) * doi5).where(confirmed, 0.0)

    # direction reversal: 空追多拥挤（chase>0），多追空拥挤（chase<0），固定符号
    factor_daily = -(chase.rank(axis=1, pct=True) - 0.5)
    factor_daily = factor_daily.shift(1)  # 只用已完成交易日

    # 分块广播回分钟级
    factor = pd.DataFrame(index=close.index, columns=close.columns, dtype=np.float32)
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        part = factor_daily[cols].reindex(day)
        part.index = close.index
        factor.loc[:, cols] = part.astype(np.float32)

    return factor.replace([np.inf, -np.inf], np.nan)
