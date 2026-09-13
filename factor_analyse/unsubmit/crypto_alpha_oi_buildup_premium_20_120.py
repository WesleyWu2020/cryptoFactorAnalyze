import os

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "positioning_buildup", "statement": "open_interest（币单位持仓量）20d/120d 慢速抬升=杠杆持仓堆积、拥挤度 regime 上行"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "persistence", "statement": "双长窗均值比（20d/120d）本身是持续数周的持仓堆积趋势，非单日脉冲"},
    ],
    "direction": {"id": "premium", "statement": "杠杆拥挤溢价：固定符号空堆积（拥挤=脆弱=未来跑输）多出清（neglect），无方向预测"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "49 轮挖掘从未触碰 open_interest（持仓/杠杆层，与量价活动是完全不同的信息源）。"
                 "复用 C1 验证过的'慢变 unsigned 特征 + premium'低换手范式（C1 换手 0.05、净≈毛）："
                 "OI 慢速堆积=该币杠杆拥挤度上行，拥挤的多空双方使价格对清算级联脆弱，未来截面跑输；"
                 "OI 出清=持仓 neglect，未来跑赢。与 C1 的投机活动（taker 买盘流量）经济内容不同质："
                 "活动是流量、持仓是存量——若 prod_corr 低=持仓层是真增量维度。",
    "fields":    ["open_interest", "close"],
    "expected_horizon": "1d",
    "invalidation": "(1) prod_corr >= 0.6（OI 与库内 GP 撞车=GP 已占用持仓层，增量不存在）；"
                    "(2) |RankIC| < 0.01 或符号为负（先验：堆积币跑输，factor 已内嵌负号，IC 应为正）；"
                    "(3) ls_netir < 0.5；"
                    "(4) turnover(组合换手) > 0.25（范式预期 0.04~0.10）；"
                    "(5) coverage < 0.9（OI 宇宙内日覆盖已验证 ~1.000）",
    "search_mode": "explore",
    "semantic_key": "positioning_buildup|-|persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "positioning",
    "parent_factor_id": None,
    "semantic_mutation": None,
    "hypothesis": "OI 慢速堆积=杠杆拥挤=未来截面跑输；出清=跑赢。单位校验：oi 为币单位持仓"
                  "（oi*close/1m dollar_volume 中位 3145≈2.2 天成交量，合理杠杆比），"
                  "宇宙内日覆盖率 ~1.000，日变化率 p5/p95=±12~15% 有真实变异。",
    "change": "新因子。oi_daily=日级 last；buildup=mean(oi,20d)/mean(oi,120d)；"
              "factor=-(rank(buildup)-0.5)，shift(1) 广播分钟级。",
    "expected": "|RankIC| 0.01~0.03, ls_netir 0.4~1.0, 组合换手 0.04~0.10, coverage>0.9, "
                "核心悬念=prod_corr（GP 矿机同 H5 可及 OI，撞车风险真实存在；"
                "但已撞车的全是量价表达，持仓层逃逸机会大）。",
    "result": "REJECTED (STATISTICAL_REJECT, 但持仓层增量维度确立): |RankIC|=0.0263/|ICIR|=4.89, "
              "prod_corr=0.4574 vs gp_auto_20260608_145002_seed430_r34 PASS(冗余逃逸成功——GP 没占用 OI 层); "
              "换手 0.063/maxdd 0.169 全过; 败因: ls_netir=0.898/净+10.35%/yr 差临门一脚 + 2022(-5.3%)/2023(-3.7%) "
              "熊市年为负, 2024-2026 强(+24.9/+20.8/+13.2%)。六图: RankIC CumSum 全样本斜率稳定(含 2022-23)"
              "——信号本身无 regime 问题; demeaned 组0(极端堆积, 空腿)全程单调崩到 -1.2, 组1-9 挤作一团"
              "=尾部选择因子, alpha 几乎全在空极端 OI 堆积; 长腿(OI 出清) 2022-23 熊市 bleed——"
              "OI 崩塌币含濒死币污染(LUNA/FTX 连环)。重要对照: 极端 OI 堆积尾部可空(与 activity/vol 家族的"
              "'极端热尾部=squeeze 火箭死亡区'相反)——premium 方向在持仓层成立。",
}


TYPE = "regular"

BUILDUP_FAST_DAYS = 20
BUILDUP_SLOW_DAYS = 120
MINUTES_PER_DAY = 1440
WARMUP_BARS = (BUILDUP_SLOW_DAYS + BUILDUP_FAST_DAYS + 10) * MINUTES_PER_DAY
CHUNK_SIZE = int(os.getenv("FACTOR_CHUNK_SIZE", "8"))
EPS = 1e-12

META = {
    "factor_name": "crypto_alpha_oi_buildup_premium_20_120",
    "author": "wesleywu",
    "level": "minutes",
    "tag": "oi_buildup_premium",
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

    # direction premium: 空堆积（杠杆拥挤=脆弱）多出清（neglect），固定符号
    factor_daily = -(buildup.rank(axis=1, pct=True) - 0.5)
    factor_daily = factor_daily.shift(1)  # 只用已完成交易日

    # 分块广播回分钟级
    factor = pd.DataFrame(index=close.index, columns=close.columns, dtype=np.float32)
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        part = factor_daily[cols].reindex(day)
        part.index = close.index
        factor.loc[:, cols] = part.astype(np.float32)

    return factor.replace([np.inf, -np.inf], np.nan)
