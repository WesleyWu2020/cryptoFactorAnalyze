"""crypto_alpha_pv_corr_regimesplit_10d 因子（factor_common 日频契约版）。

原始定义：
    pvc = corr(daily_close, log(daily_quote_volume + 1), 10d)
    r = rank_cs(pvc)
    base = -(r - 0.5)
    factor = 0.5 if r > 0.9 else base
    极端高相关尾 continuation 翻多，中段高相关保持空，低相关背离保持多。

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
    "event":     {"id": "pv_diverge", "statement": "corr(close, turnover, 10d) 高=价量同步追涨/杀跌（过热定价）；低或负=量价背离（吸筹/出清）"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [],
    "direction": {"id": "regime_split", "statement": "极端高相关尾（rank>0.9）走 continuation（火箭延续=多），中段高相关区走 reversal/premium（温和过热=空），低相关背离区保持多——同因子双机制"},
    "output":    {"id": "rank", "statement": "截面 rank 打分（经非单调变换）"},
    "mechanism": "轮5（pv_corr_premium）的 demeaned 是教科书 regime_split 形态："
                 "组0（极端高相关）+0.55=极端追涨同步是 meme 火箭发射台（延续不可空，range_level/bybit_heat 同判三次），"
                 "组1/2（温和高相关）-0.5/-0.55 稳定出血=真正可收割的过热毒区，组8/9（背离）+0.65/平。"
                 "单调 premium 表达把火箭装进空腿（自杀），把最优组8 稀释在多腿中段。"
                 "轮5 IC 0.0454/7.53 是 69 轮最厚 IC 资产——值得为它定制表达。",
    "fields":    ["close", "turnover"],
    "expected_horizon": "1d",
    "invalidation": "(1) ls_netir < 0（双机制仍不赚=火箭延续不稳定或组1-2 bleed 不可收割）；"
                    "(2) |RankIC| < 0.01 且 ls 亏损（非单调变换摧毁全部信息）；"
                    "(3) turnover > 1.2（0.9 边界跳跃 churn 失控）；"
                    "(4) 2023 与 2024 双年仍负（regime 依赖未修复）；"
                    "(5) prod_corr >= 0.6",
    "search_mode": "mutate",
    "semantic_key": "pv_diverge|-|-|regime_split|rank",
}

ITER_NOTE: dict = {
    "op_type": "modify_factor",
    "factor_family": "pvc",
    "parent_factor_id": "crypto_alpha_pv_corr_premium_10d",
    "parent_iter": None,
    "semantic_mutation": "pv_diverge|-|-|premium|rank -> pv_diverge|-|-|regime_split|rank",
    "hypothesis": "轮5 demeaned 证据：极端高相关尾（rank>0.9）是延续型火箭（组0 +0.55），"
                  "温和高相关区（rank 0.5~0.9）是反转型毒区（组1/2 -0.5）。"
                  "把极端尾从空腿翻到多腿：空腿改吃组1/2 的稳定 bleed，多腿=火箭+背离双源。",
    "reasoning": "词表 regime_split（极端区 continuation/中段 reversal）零实验记录，"
                 "轮5 demeaned 即其定义插图；IC 资产 0.0454/7.53 为项目最厚，"
                 "值得承受一次表达改动（单点：仅改 rank→factor 的映射，pvc 计算逐行不动）。",
    "change": "factor 映射非单调化：base=-(rank(pvc)-0.5)；rank(pvc)>0.9 的币 factor 置 +0.5（翻多），"
              "其余保持 base。pvc=corr(close,dv,10) 计算与轮5 完全一致。",
    "expected": "|RankIC| 降至 0.020~0.035（非单调变换牺牲全截面梯度，预注册可接受——判决指标是组合端）；"
                "ls_netir 0.3~0.9，ls 净转正（空腿吃组1/2 bleed -0.5 demeaned，多腿组0火箭+0.55+组8 +0.65）；"
                "turnover 0.5~0.9（0.9 边界 churn）；2024 修复（火箭多在身）。",
    "result": "REJECTED (STATISTICAL_REJECT, IS 2022~2024, 但 regime_split 经济学验证成功): "
              "毛 ls +18~19%/yr(NAV 1.0→1.7 稳定爬升, bleed 修复), demeaned 近单调: "
              "组9(火箭多腿)+0.7 最优/组8+0.45/组0(温和高相关)-0.75 垫底——方向判断正确; "
              "2023 +4.4%/2024 +1.4% 转正, maxdd 0.24 过门, prod_corr=0.2495。死因二重: "
              "(1) RankIC 0.0018 崩塌——火箭收益来自右偏肥尾(偶发爆发日), 组合均值吃偏度而 "
              "Spearman 按秩全盲=IC/组合口径在偏度资产上结构性脱钩; "
              "(2) 算术封顶: 毛~19%/yr < 净门 20%/yr, 零成本也不够, turnover 压到 0.25 净仅~12%——"
              "十分位 LS 引擎×此资产毛厚度=天花板在门槛下, 与 volume_instability(0.15-0.19)/"
              "venue_share(10-12%) 同归宿。pvc 线关闭: 资产真实(0.0454/7.53 IC)但 decile-LS 口径"
              "不可变现, 留档组合层素材(偏度采收需非对称工具, 超出本引擎)。",
    "oos_result": None,
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_pv_corr_regimesplit_10d",
    "author": "wesleywu",
    "level": "daily",
    "category": "pvc",
    "description": "pv_corr regime_split：rank(corr(close,log(qv+1),10d))>0.9 翻多，其余按 -(rank-0.5)",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 20,
    "preprocessing": "mad_rank",
    "params": {"corr_days": 10, "tail_q": 0.9, "tail_value": 0.5},
    "factor_direction": 1,  # 高因子值 = 火箭延续尾/量价背离，ITER_NOTE 假设该侧为多头
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily pv-corr-regime-split matrix (date x instrument)."""

    params = SETTING["params"]
    corr_days = params["corr_days"]
    tail_q = params["tail_q"]
    tail_value = params["tail_value"]

    close = data_ctx["close"].astype("float64")
    dv = data_ctx["quote_volume"].astype("float64").clip(lower=0.0)

    # event pv_diverge: 日频价量滚动相关（log 量抑制量纲右偏）
    pvc = close.rolling(corr_days, min_periods=corr_days).corr(np.log(dv + 1.0))

    # direction regime_split: 极端高相关尾（rank>tail_q）continuation 翻多（火箭延续）；
    # 中段高相关 reversal 保持空（温和过热 bleed）；低相关背离保持多
    r = pvc.rank(axis=1, pct=True)
    base = -(r - 0.5)
    factor = base.where(r <= tail_q, tail_value)
    return factor.replace([np.inf, -np.inf], np.nan)
