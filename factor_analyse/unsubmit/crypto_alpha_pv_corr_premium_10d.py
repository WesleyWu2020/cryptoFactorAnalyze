"""crypto_alpha_pv_corr_premium_10d 因子（factor_common 日频契约版）。

原始定义：
    pvc = corr(daily_close, log(daily_quote_volume + 1), 10d)
    factor = -(rank_cs(pvc) - 0.5)
    空高相关（追涨同步过热），多背离（吸筹/出清）。

分钟->日频改写判断（记录在案）：
  - 原实现 turnover 为分钟成交额，按日 sum 聚合；日频契约下直接等价于 quote_volume
    字段（分钟成交额日求和 == 日 quote_volume），data_needed 用 close + quote_volume。
  - 原实现 close 按日 last 聚合；日频契约下 close 直接为日收盘。
  - 原实现 factor_daily.shift(1) 后广播回分钟索引，属于执行延迟（只用已完成交易日）；
    框架以次开盘价执行，删除该 shift，date t 因子可直接使用 t 日完整数据。
  - CHUNK_SIZE 分块循环纯为内存控制，日频数据量小，删除。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "pv_diverge", "statement": "corr(close, turnover, 10d) 高=价量同步追涨/杀跌（过热定价）；低或负=量价背离（缩量涨=吸筹、放量跌=出清）"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [],
    "direction": {"id": "premium", "statement": "量价同步溢价：固定符号空高相关（追涨盘主导=过度定价）多背离（吸筹/出清=低估），无方向预测"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "pvc 族日频经典表达首次直接测试（pv_diverge 前 2 次实验都是分钟时段结构，与本构造不同质）。"
                 "IS 探针（2022~2024，delay=1 一致口径 T+1→T+2，已应用 basis_meanrev 教训）："
                 "-corr(close,turnover,10d) RankIC=+0.0371、日ICIR=0.35（t≈9.6），10d 优于 20d，"
                 "close×dv 优于 ret×Δdv。机制：价量高度同步=交易由追涨/恐慌盘单边驱动，"
                 "价格被推离均衡；量价背离=知情资金逆势吸筹或出清完成，后续修复。",
    "fields":    ["close", "turnover"],
    "expected_horizon": "1d",
    "invalidation": "(1) |RankIC| < 0.024（探针值的 2/3，低于此=日聚合/口径损失过多）；"
                    "(2) prod_corr >= 0.85（GP 在同 H5 上密集挖掘量价，撞车是核心悬念；0.6~0.85 需 ITER_NOTE 论证增量）；"
                    "(3) ls_netir < 0.5；(4) turnover > 1.0；"
                    "(5) demeaned 组9平盘/组0独毒=又是空尾-only 签名（出现则按元规则转黑名单素材，不再迭代表达）",
    "search_mode": "explore",
    "semantic_key": "pv_diverge|-|-|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "pvc",
    "parent_factor_id": None,
    "semantic_mutation": None,
    "hypothesis": "量价同步过热溢价：-corr(close,turnover,10d) IS 探针 IC +0.0371/日ICIR 0.35，"
                  "方向与 Bali/CPV 文献先验一致（高相关=彩票式追涨需求集中→负溢价）。",
    "reasoning": "轨迹审查四问：volume_surge 14 次/breakout 10 次过度集中，positioning 层 5 败关闭；"
                 "pv_diverge 仅 2 次分钟时段实验，日频经典 pvc 表达是词表上真正的空白；"
                 "pvc 是 factor-families Tier1 明列族但零占位。探针已用 delay=1 一致口径，"
                 "排除 basis_meanrev 式 trade-at-close 幻觉。",
    "change": "新因子。close_d=日 last；dv_d=turnover 日 sum；pvc=close_d.rolling(10).corr(log(dv_d+1))；"
              "factor=-(rank(pvc)-0.5)，shift(1) 广播分钟级。",
    "expected": "|RankIC| 0.030~0.037（探针 0.0371），ls_netir 0.5~1.2，turnover 0.3~0.7"
                "（10d 滚动窗口重叠=中速），coverage>0.9。核心悬念=prod_corr vs 量价/反转类 GP。",
    "result": "REJECTED (STATISTICAL_REJECT, IS 2022~2024, 但发现项目最强 IC 资产): "
              "|RankIC|=0.0454/|ICIR|=7.53(69轮最强, 近直线 IC 曲线, delay=1 保住 delay=0 的 89%=非快衰减), "
              "prod_corr=0.4416 PASS, turnover=0.488/coverage=0.978 过门; "
              "死因=分位错序: demeaned 教科书 regime_split 形态——组0(极端高相关)+0.55(2024后火箭延续), "
              "组1/2(温和高相关)-0.5/-0.55 稳定出血=真毒区, 组8 +0.65 最优而组9平盘; "
              "毛仅+6%/yr 被成本13%/yr 吃成净-7.4%, 三年全负。"
              "'中段反转/尾部延续撕裂'第10次复现, 但这次底层 IC 厚度史无前例——"
              "下一轮: 表达 premium→regime_split(极端尾 rank>0.9 翻多接火箭, 中段保持空), "
              "词表 regime_split 零实验, 此图即其定义插图。",
    "oos_result": None,
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_pv_corr_premium_10d",
    "author": "wesleywu",
    "level": "daily",
    "category": "pvc",
    "description": "量价同步过热溢价：-(rank(corr(close, log(quote_volume+1), 10d)) - 0.5)",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {"corr_days": 10},
    "factor_direction": 1,  # 高因子值 = 量价背离（低相关），ITER_NOTE 假设背离侧为多头
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily pv-corr-premium matrix (date x instrument)."""

    corr_days = SETTING["params"]["corr_days"]

    close = data_ctx["close"].astype("float64")
    dv = data_ctx["quote_volume"].astype("float64").clip(lower=0.0)

    # event pv_diverge: 日频价量滚动相关（log 量抑制量纲右偏）
    pvc = close.rolling(corr_days, min_periods=corr_days).corr(np.log(dv + 1.0))

    # direction premium: 空高相关（追涨同步过热），多背离（吸筹/出清），固定符号
    factor = -(pvc.rank(axis=1, pct=True) - 0.5)
    return factor.replace([np.inf, -np.inf], np.nan)
