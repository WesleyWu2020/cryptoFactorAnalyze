"""crypto_alpha_range_level_premium_20d 因子（factor_common 日频契约版）。

原始定义（分钟版改造）：
    range_daily = (high - low) / close
    range20 = rolling_mean(range_daily, 20d)
    factor = -(rank_cs(range20) - 0.5)（空高振幅多低振幅，彩票/低波溢价）

日频改写判断：
    - 分钟聚合（close=last, high=max, low=min）在日频输入下为恒等，直接用日频字段。
    - 删除 factor_daily.shift(1)（纯执行延迟，框架按次日开盘执行）。
    - 删除分钟 index 广播与分块循环。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "range_level", "statement": "(H-L)/C 日内振幅 20d 均值=已实现波动/彩票性水平"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [],
    "direction": {"id": "premium", "statement": "低波/彩票溢价：固定符号空高振幅（彩票需求=过度定价）多低振幅，无方向预测"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "彩票溢价先验（Bali）：高日内振幅币提供类彩票收益分布，散户博彩需求使其系统性高估，"
                 "未来截面跑输；低振幅币无聊被忽视，享受 neglect 溢价。与 vol_spike/vol_squeeze 不同质："
                 "那些是波动'变化'事件（已全灭），本因子是波动'水平'的 unsigned 慢特征。"
                 "close/high/low 三字段覆盖自查中最后一个未碰的独立维度。",
    "fields":    ["close", "high", "low"],
    "expected_horizon": "1d",
    "invalidation": "(1) 原始 RankIC < -0.01（高振幅未来跑赢，彩票先验反，转翻转检验）；"
                    "(2) ls_netir < 0.5；"
                    "(3) prod_corr >= 0.6（vol/彩票类 GP 同源是最大风险，vol 家族语义近邻）；"
                    "(4) turnover > 0.25；"
                    "(5) coverage < 0.9",
    "search_mode": "explore",
    "semantic_key": "range_level|-|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "price_action",
    "parent_factor_id": None,
    "semantic_mutation": None,
    "hypothesis": "空高振幅多低振幅收割彩票/低波溢价。20d 均值慢变，换手预估 <0.1。"
                  "2022 附带观察：熊市高波币崩得更狠，空头腿应在 2022 反而走强"
                  "（与 taker 热度家族 2022 弱窗结构不同）。",
    "change": "新因子。range_daily=(high-low)/close 日聚合；range20=rolling_mean(20d)；"
              "factor=-(rank(range20)-0.5)（空高振幅），shift(1) 广播分钟级。",
    "expected": "|RankIC| 0.01~0.025, ls_netir 0.4~1.0, turnover 0.03~0.10, "
                "coverage>0.9, prod_corr 0.4~0.7（核心悬念：vol 类 GP 撞车概率不低）。",
    "result": "REJECTED (REDUNDANCY_REJECT, 全场最强 IC 撞尾部挤压墙): |RankIC|=0.0735/|ICIR|=6.10——"
              "47 轮最强 IC(CumSum 斜率离谱地稳), 但 prod_corr=0.7682 vs gp_auto_20260605_112001_seed402_r44 "
              "(与 C1a/C1c 撞同一个 GP, 触发 ρ 判废线); ls_netir=-0.80/净-19.6%/yr。六图定位: "
              "demeaned 组0(最高振幅, 空腿)+1.9 独赢——极端高波尾是 meme 火箭/squeeze 发射台, "
              "空它等于接火箭; 组1-8 则完美服从彩票溢价(高波跑输), 故全截面 IC 巨强而组合深亏。"
              "turnover=0.10 无成本问题, 死因纯结构性尾部反转。2022 +8.1% 唯一为正(预测命中: "
              "熊市空头腿走强)。元结论: 极端波动尾=空头死亡区(breakout 家族'毒尾=收益源'的镜像确认), "
              "价格层三个慢维度(CLV/极端度/振幅)IC 全部真实, 但要么尾部反转要么库内持有, 不可变现。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_range_level_premium_20d",
    "author": "wesleywu",
    "level": "daily",
    "category": "price_action",
    "description": "-(rank_cs(mean((H-L)/C, 20d)) - 0.5): 空高振幅多低振幅，彩票/低波溢价",
}

SETTING = {
    "data_needed": ["close", "high", "low"],
    "universe": "historical_top50",
    "warmup_bars": 30,
    "preprocessing": "mad_rank",
    "params": {"range_days": 20},
    "factor_direction": 1,  # 因子已带固定符号：高值=低振幅（neglect 溢价多头）
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily range-level-premium matrix (date x instrument)."""

    range_days = SETTING["params"]["range_days"]

    close = data_ctx["close"].astype("float64")
    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")

    # event range_level: 日内振幅 20d 均值（慢变）
    range_daily = (high - low) / (close + _EPS)
    range20 = range_daily.rolling(range_days, min_periods=range_days).mean()

    # direction premium: 空高振幅（彩票/投机）多低振幅（neglect），固定符号
    factor = -(range20.rank(axis=1, pct=True) - 0.5)
    return factor.replace([np.inf, -np.inf], np.nan)
