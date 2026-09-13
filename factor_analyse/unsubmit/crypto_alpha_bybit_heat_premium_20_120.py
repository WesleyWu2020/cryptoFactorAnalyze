"""crypto_alpha_bybit_heat_premium_20_120 因子（factor_common 日频契约版）。

原始定义（分钟版）：heat=mean(turnover 日聚合, 20d)/mean(..., 120d)；
factor=-(rank(heat)-0.5)，shift(1) 广播分钟级。

日频改写说明（judgment calls）：
- 日聚合恒等：turnover 日求和 = quote_volume，直接使用日频字段。
- data_needed 去掉 close（原版仅用其分钟索引做日聚合锚点，日频下不再需要）。
- 删除 factor_daily.shift(1)（仅为执行延迟；框架按次日开盘执行）。
- 删除分钟级 reindex 广播与分块列循环，直接返回日频矩阵。
- 20d/120d 双窗均值比为日频滚动窗口，与原版语义一致。

方向说明：direction=premium，空升温（投机拥挤=过度定价）多降温（neglect），
固定符号已写入公式（负号），高因子值=降温币为多头腿，故 factor_direction=1。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "volume_surge", "statement": "turnover（Bybit 侧活动）20d/120d 慢速抬升=该场地投机热度 regime 上行"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "persistence", "statement": "双长窗均值比（20d/120d）本身是持续数周的热度趋势，非单日脉冲"},
    ],
    "direction": {"id": "premium", "statement": "投机热度溢价：固定符号空升温（拥挤=过度定价）多降温（冷清=neglect），无方向预测"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "C1 在币安 taker 买盘美元额上验证了热度空头溢价（16.4%/1.04），本轮换场地换口径："
                 "Bybit 侧总活动（双向、含做市）的慢速升温=该场地散户投机热度，与币安主动买盘"
                 "是相关但不同的测量（场地结构、用户群、口径都不同）。若 prod_corr 低=场地维度增量；"
                 "若 ρ 高=热度资产与场地无关（C1 已是最优表达，此方向关闭）。",
    "fields":    ["turnover", "close"],
    "expected_horizon": "1d",
    "invalidation": "(1) prod_corr >= 0.6 vs C1 撞的 GP 族（场地维度无增量，热度资产与场地无关）；"
                    "(2) 原始 RankIC > -0.01 即符号为非负（先验反：升温币跑赢取绝对值判断, 转翻转检验）；"
                    "(3) ls_netir < 0.5；"
                    "(4) turnover(组合换手) > 0.25；"
                    "(5) coverage < 0.9",
    "search_mode": "explore",
    "semantic_key": "volume_surge|-|persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "sentiment",
    "parent_factor_id": None,
    "semantic_mutation": None,
    "hypothesis": "Bybit 活动升温=投机热度=未来截面跑输。单位校验确认 turnover 为 Bybit 侧活动字段"
                  "（与 xbinance_quote_volume 比值中位 0.18 且分散），与 C1 的币安 taker 买盘不同场地不同口径。",
    "change": "新因子。heat=mean(turnover日聚合,20d)/mean(...,120d)；factor=-(rank(heat)-0.5)，"
              "shift(1) 广播分钟级。",
    "expected": "|RankIC| 0.01~0.02, ls_netir 0.4~1.0, 组合换手 0.05~0.15, coverage>0.9, "
                "prod_corr<0.6（核心悬念：与 C1 的 gp_auto_20260607 撞车率）。",
    "result": "REJECTED (STATISTICAL_REJECT, 尾部挤压第三次确认): |RankIC|=0.0393/|ICIR|=6.54(全场第二强 IC, "
              "符号正确=空热度对), prod_corr=0.5615 PASS(场地维度冗余逃逸成功——Bybit 活动与币安 taker 买盘"
              "不是同一测量); 但 ls_netir=0.02/净+0.3%/yr=组合颗粒无收。六图: RankIC CumSum 五年稳定正; "
              "demeaned 组0(最热, 空腿)+0.9 独赢(2025 冲 1.0)=极端热币是 squeeze 火箭, 空腿被炸; "
              "组1 -0.5 最差、组2-8 服从热度空头逻辑——IC 巨强组合为零的中段/尾部打架, 与 P2 同构。"
              "分年交替: 2022 +20.1/2025 +19.0(熊市震荡年空头赢) vs 2023 -18.2/2024 -16.5(牛市火箭年)."
              "翻转算术已排除(翻转后复合 -14%)。尾部剔除(NaN 最热 10%)预期毛~+12%/yr 仍低于 0.2 门, 不迭代。"
              "元规则(三次确认): crypto perp 极端热/高波尾部持续跑赢(squeeze 机制), 中段服从过度定价逻辑——"
              "此类因子的正确归宿是空头黑名单(universe 层), 非多空因子。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_bybit_heat_premium_20_120",
    "author": "wesleywu",
    "level": "daily",
    "category": "sentiment",
    "description": "heat=mean(quote_volume,20d)/mean(...,120d)；factor=-(rank(heat)-0.5)，空投机热度升温",
}

SETTING = {
    "data_needed": ["quote_volume"],
    "universe": "historical_top50",
    # 最长窗口 heat_slow_days=120，加缓冲。
    "warmup_bars": 130,
    "preprocessing": "mad_rank",
    "params": {"heat_fast_days": 20, "heat_slow_days": 120},
    "factor_direction": 1,
}

EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily heat-premium matrix (date x instrument)."""

    heat_fast_days = SETTING["params"]["heat_fast_days"]
    heat_slow_days = SETTING["params"]["heat_slow_days"]

    to_daily = data_ctx["quote_volume"].astype("float64").clip(lower=0)

    # event volume_surge（慢速版）: 活动热度 regime = 20d/120d 双窗比
    heat_fast = to_daily.rolling(heat_fast_days, min_periods=heat_fast_days).mean()
    heat_slow = to_daily.rolling(heat_slow_days, min_periods=heat_slow_days).mean()
    heat = heat_fast / (heat_slow + EPS)

    # direction premium: 空升温（投机拥挤）多降温（neglect），固定符号
    factor = -(heat.rank(axis=1, pct=True) - 0.5)
    return factor.replace([np.inf, -np.inf], np.nan)
