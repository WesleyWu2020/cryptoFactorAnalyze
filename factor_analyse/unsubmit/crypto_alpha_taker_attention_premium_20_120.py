"""crypto_alpha_taker_attention_premium_20_120 因子（factor_common 日频契约版）。

原始定义（分钟版改造）：
    att = mean(taker_buy_quote_volume, 20d) / mean(taker_buy_quote_volume, 120d)
    factor = -(rank_cs(att) - 0.5)（空拥挤多冷清，关注度溢价收割）

日频改写判断：
    - 分钟聚合（taker=sum）在日频输入下为恒等，直接用日频 taker_buy_quote_volume。
    - 分钟版 data_needed 里的 close 只用于取 index/columns 骨架，日频版不再需要。
    - 删除 factor_daily.shift(1)（纯执行延迟，框架按次日开盘执行）。
    - 删除分钟 index 广播与分块循环。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "volume_surge", "statement": "taker 主动买盘美元额 20d/120d 慢速抬升=关注度 regime 上行（相对自身长期水平，非绝对量）"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "persistence", "statement": "双长窗均值比（20d/120d）本身就是持续数周的关注度趋势，非单日脉冲"},
    ],
    "direction": {"id": "premium", "statement": "关注度溢价收割：固定符号空拥挤（attention 上行）多冷清（attention 下行）， Barber-Odean 先验，无方向预测"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "墓园规则#1 封死了 taker 快信号（半衰期<1d, #1/#2 双杀），本轮只做慢结构："
                 "散户追逐热点→主动买盘趋势性抬升→短期超买定价→随后跑输；被冷落币享受 neglect 溢价。"
                 "用 20d/120d 双窗比把信号做成慢变截面特征（预估换手 <0.2），绕开成本墙；"
                 "相对自身长期水平归一，与 amihud/size 等绝对水平类 GP（ρ0.97 陷阱）不同源。"
                 "G1 已验证 premium+慢变特征范式在本框架可行（净≈毛），本轮为该范式找第二个特征。",
    "fields":    ["xbinance_taker_buy_quote_volume", "close"],
    "expected_horizon": "1d",
    "invalidation": "(1) RankIC > 0.01 且符号为高 att 涨（先验反，拥挤=动量确认而非溢价，转符号检验）；"
                    "(2) ls_netir < 0.5（溢价太薄）；"
                    "(3) prod_corr >= 0.6（空关注度≈空近期赢家，与 reversal/mom 类 GP 冗余是最大风险）；"
                    "(4) turnover > 0.25（慢变假设破产）；"
                    "(5) coverage < 0.9（意外择时化）",
    "search_mode": "explore",
    "semantic_key": "volume_surge|-|persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "sentiment",
    "parent_factor_id": None,
    "semantic_mutation": None,
    "hypothesis": "关注度溢价：taker 主动买盘的周尺度趋势性抬升=散户拥挤入场，未来截面跑输；"
                  "关注度枯竭=neglect 溢价，未来跑赢。G3 元结论指导下：premium 方向+慢变特征"
                  "是当前唯一验证可行的范式，且必须先想清冗余源——本因子最大冗余风险是"
                  "'空关注度≈空近期赢家'与 reversal/mom GP 撞车，prod_corr 是核心观察点。",
    "change": "新因子。att = mean(taker_buy_quote_volume日聚合,20d)/mean(...,120d)，"
              "factor = -(rank_cs(att)-0.5)（空拥挤多冷清），无额外平滑（双窗比本身慢变），"
              "shift(1) 广播分钟级。",
    "expected": "turnover 0.05~0.20, |RankIC| 0.005~0.02（溢价类弱而稳）, ls_netir 0.3~1.0, "
                "ls_netret_ann 0~+0.15, coverage>0.9, prod_corr<0.6（核心悬念）。",
    "result": "REJECTED (STATISTICAL_REJECT, 但为 37 轮最强非冗余候选): |RankIC|=0.0156/|ICIR|=2.70 PASS, "
              "ls_netir=1.0407 过 1.0 门(史上第二个), turnover=0.051 慢变假设成立, coverage=0.925, "
              "maxdd=0.191; 死因: ls_netret_ann=+16.4% < 0.2 门 + 2022 -2.9%/2026 -5.2%; "
              "prod_corr=0.6206 vs gp_auto_20260607_101002_seed421_r57 [WARN](预注册 0.6 线边缘触发)。"
              "六图: 净≈毛(成本~2.8%/yr 可忽略); demeaned 组0(最拥挤)-1.2 全场最差=alpha 几乎全在"
              "空腿, 组5/6(温和冷清)+1.3 最优而组9(极端冷清)~0=多腿非单调噪声。学到: (1) 关注度溢价"
              "真实且以短腿为主; (2) 冗余源='空关注度≈空近期赢家'与 mom/reversal GP 部分重叠; "
              "下一步方向=收益中性化(att 对 20d 收益截面回归取残差)纯化空腿并降 rho。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_taker_attention_premium_20_120",
    "author": "wesleywu",
    "level": "daily",
    "category": "sentiment",
    "description": "-(rank_cs(mean(taker_buy_qv,20d)/mean(taker_buy_qv,120d)) - 0.5): 空拥挤多冷清，关注度溢价",
}

SETTING = {
    "data_needed": ["taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 150,
    "preprocessing": "mad_rank",
    "params": {"att_fast_days": 20, "att_slow_days": 120},
    "factor_direction": 1,  # 因子已带固定符号：高值=关注度枯竭（neglect 溢价多头）
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily taker-attention-premium matrix (date x instrument)."""

    att_fast_days = SETTING["params"]["att_fast_days"]
    att_slow_days = SETTING["params"]["att_slow_days"]

    taker_daily = data_ctx["taker_buy_quote_volume"].astype("float64").clip(lower=0)

    # event volume_surge（慢速版）: 关注度 regime = 20d/120d 双窗均值比，慢变
    att_fast = taker_daily.rolling(att_fast_days, min_periods=att_fast_days).mean()
    att_slow = taker_daily.rolling(att_slow_days, min_periods=att_slow_days).mean()
    att = att_fast / (att_slow + _EPS)

    # direction premium: 空拥挤（高 att）多冷清（低 att），固定符号
    factor = -(att.rank(axis=1, pct=True) - 0.5)
    return factor.replace([np.inf, -np.inf], np.nan)
