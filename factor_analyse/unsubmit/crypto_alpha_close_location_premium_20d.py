"""crypto_alpha_close_location_premium_20d 因子（factor_common 日频契约版）。

原始定义（分钟版）：clv_daily=(2C-H-L)/(H-L)，clv20=rolling_mean(clv,20d)；
factor=rank(clv20)-0.5（多持续收高空持续收低），shift(1) 广播分钟级。

日频改写说明（judgment calls）：
- 日聚合恒等：close=日内最后价、high=日内最高、low=日内最低，直接使用日频字段，
  CLV 日频直接计算（原 ITER_NOTE 已注明 T1 日级字段无需分钟聚合）。
- 删除 factor_daily.shift(1)（仅为执行延迟；框架按次日开盘执行）。
- 删除分钟级 reindex 广播与分块列循环，直接返回日频矩阵。

方向说明：direction=premium，多持续收高（尾盘承接/吸筹）空持续收低（派发），
高因子值为多头腿，故 factor_direction=1。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "close_location", "statement": "CLV=(2C-H-L)/(H-L) 日内收盘位置，20d 均值=持续尾盘承接/派发行为"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "persistence", "statement": "20d 均值的慢变行为特征，非单日位置"},
    ],
    "direction": {"id": "premium", "statement": "尾盘承接溢价：固定符号多持续收高（吸筹痕迹）空持续收低（派发痕迹），无方向预测"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "知情资金倾向于在收盘附近建仓（流动性最好、信息最充分），持续收在日内高位=系统性承接；"
                 "持续收在日内低位=系统性派发/上方抛压。与 breakout 不同质：breakout 是 N 日区间的"
                 "极端位置（快、事件性），close_location 是每日区间位置的长期平均（慢、行为性）。"
                 "价格层覆盖自查：该维度在 44 轮中未被任何已挖因子表达。",
    "fields":    ["close", "high", "low"],
    "expected_horizon": "1d",
    "invalidation": "(1) 原始 RankIC < -0.01（持续收高未来跑输，先验反，转翻转检验）；"
                    "(2) ls_netir < 0.5；"
                    "(3) prod_corr >= 0.6（与 mom 族 GP 共享'收高=近期涨'成分）；"
                    "(4) turnover > 0.3；"
                    "(5) coverage < 0.9",
    "search_mode": "explore",
    "semantic_key": "close_location|-|persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "price_action",
    "parent_factor_id": None,
    "semantic_mutation": None,
    "hypothesis": "持续尾盘承接=吸筹，未来截面跑赢。冗余风险预注册：CLV 与近期收益正相关"
                  "（收高往往=当天涨），可能撞上 mom 族 GP——若 ρ 高且残差化杀 IC，"
                  "则同 C1a 判例（共享暴露非独立信息）。",
    "change": "新因子。clv_daily=(2*close-high-low)/(high-low+EPS) 日频直接算（T1 日级字段无需分钟聚合）；"
              "clv20=rolling_mean(clv,20)；factor=rank(clv20)-0.5（多持续收高），shift(1) 广播分钟级。",
    "expected": "|RankIC| 0.008~0.02, ls_netir 0.4~1.0, turnover 0.08~0.20, "
                "coverage>0.9, prod_corr 0.4~0.6（mom 共享是核心悬念）。",
    "result": "REJECTED (STATISTICAL_REJECT, IC 全场最强但组合不可变现): |RankIC|=0.0282/|ICIR|=4.09 "
              "(44 轮最强 IC, CumSum 五年教科书级正斜率), prod_corr=0.5408 PASS(非冗余); "
              "但 ls_netir=-0.52/净-8.1%/yr——IC/组合严重背离。六图定位: demeaned 呈倒 U——"
              "组5(行为中性)+0.8 独赢, 组0(持续收低)与组9(持续收高)双双 -0.5 垫底: "
              "CLV 极端化的币(无论方向)都是有毒的(操纵/抛物线/单边市), 信息全在中段梯度; "
              "组合吃尾部=两头有毒, 毛 ls~+8%/yr 弱, turnover=0.30(慢家族 3-6 倍, CLV 日噪声"
              "致 rank 重洗)成本~16%/yr 吃成负。学到: (1) close_location 维度信息真实且非冗余, "
              "但正确表达是'行为极端化'空头而非方向性 premium; (2) IC 强+组合弱+倒 U demeaned "
              "=中段因子的识别签名。下一步: P1a factor=-|clv20_rank-0.5|(空极端行为多中性行为)。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_close_location_premium_20d",
    "author": "wesleywu",
    "level": "daily",
    "category": "price_action",
    "description": "clv20=mean((2C-H-L)/(H-L),20d)；factor=rank(clv20)-0.5，多持续收高空持续收低",
}

SETTING = {
    "data_needed": ["close", "high", "low"],
    "universe": "historical_top50",
    # 最长窗口 clv_days=20，加缓冲。
    "warmup_bars": 26,
    "preprocessing": "mad_rank",
    "params": {"clv_days": 20},
    "factor_direction": 1,
}

EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily close-location-premium matrix (date x instrument)."""

    clv_days = SETTING["params"]["clv_days"]

    close_daily = data_ctx["close"].astype("float64")
    high_daily = data_ctx["high"].astype("float64")
    low_daily = data_ctx["low"].astype("float64")

    # event close_location: CLV 日内收盘位置 [-1,1]，20d 均值（慢变行为特征）
    clv_daily = (2.0 * close_daily - high_daily - low_daily) / ((high_daily - low_daily) + EPS)
    clv20 = clv_daily.rolling(clv_days, min_periods=clv_days).mean()

    # direction premium: 多持续收高（尾盘承接/吸筹）空持续收低（派发），固定符号
    factor = clv20.rank(axis=1, pct=True) - 0.5
    return factor.replace([np.inf, -np.inf], np.nan)
