"""crypto_alpha_close_location_extremeshort_20d 因子（factor_common 日频契约版）。

原始定义（分钟版）：clv_daily=(2C-H-L)/(H-L)，clv20=rolling_mean(clv,20d)；
factor=-|rank(clv20)-0.5|（空行为极端多行为中性），shift(1) 广播分钟级。

日频改写说明（judgment calls）：
- 日聚合恒等：close=日内最后价、high=日内最高、low=日内最低，直接使用日频字段，
  CLV 日频直接计算（原 ITER_NOTE 已注明 T1 日级字段无需分钟聚合）。
- 删除 factor_daily.shift(1)（仅为执行延迟；框架按次日开盘执行）。
- 删除分钟级 reindex 广播与分块列循环，直接返回日频矩阵。

方向说明：direction=premium（极端度版），固定符号已写入公式（-|rank-0.5|），
高因子值=行为中性币为多头腿，故 factor_direction=1。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "close_location", "statement": "CLV=(2C-H-L)/(H-L) 日内收盘位置 20d 均值；本轮取其截面极端度 |rank-0.5|"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "persistence", "statement": "20d 均值的慢变行为特征，非单日位置"},
    ],
    "direction": {"id": "premium", "statement": "行为极端化溢价：固定符号空极端（持续收高或收低=操纵/抛物线/单边市）多中性（供需平衡），无方向预测"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "P1 六图定位：CLV 的信息是倒 U——组5(中性)+0.8 独赢、组0/组9(双向极端)双双 -0.5 垫底，"
                 "方向性 premium 把两个毒尾同时装进组合所以失败。本轮按'极端即毒'重写表达："
                 "持续收在日内极端位置(无论上下)=价格发现失效(单边做市/操纵/抛物线行情末端)，"
                 "空之；收盘位置围绕中位波动=正常双向供需，多之。",
    "fields":    ["close", "high", "low"],
    "expected_horizon": "1d",
    "invalidation": "(1) ls_netir < 0.8（倒 U 价差无法被极端度表达收割）；"
                    "(2) |RankIC| < 0.008（hump 变换把中段 IC 也毁掉）；"
                    "(3) prod_corr >= 0.6；"
                    "(4) turnover > 0.35；"
                    "(5) 2023 ls_netret 仍明显为负（P1 最烂年 -21% 未修复）；"
                    "(6) coverage < 0.9",
    "search_mode": "exploit",
    "semantic_key": "close_location|-|persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "modify_factor",
    "factor_family": "price_action",
    "parent_factor_id": "crypto_alpha_close_location_premium_20d",
    "semantic_mutation": "semantic_key 不变；方向性 premium -> 极端度 premium：factor=-|clv20_rank-0.5|",
    "hypothesis": "P1 的倒 U 价差（组5 vs 组0/9 约 1.3）是真实毛资产，但被方向性表达浪费。"
                  "极端度变换后，组合多腿=中性币(+0.8)、空腿=双向极端币(-0.5)，"
                  "毛 ls 应从 ~+8%/yr 抬到 ~15-25%/yr；同时极端度是 unsigned 特征，"
                  "rank 重洗噪声应低于 signed CLV，turnover 有望从 0.30 下降。",
    "change": "clv20 构造不变；factor 由 rank(clv20)-0.5 改为 -|rank(clv20)-0.5|（空极端多中性），"
              "shift(1) 广播。",
    "expected": "ls_netret_ann +0.05~+0.20（倒 U 价差减成本）, ls_netir 0.5~1.2, "
                "|RankIC| 0.008~0.02, turnover 0.15~0.30, prod_corr<0.6, coverage>0.9, "
                "2023 分年转正或接近零。",
    "result": "REJECTED (STATISTICAL_REJECT, gross good net bad): 极端化方向验证正确——毛 ls 约 +17%/yr, "
              "demeaned 组0(最极端)-0.95 独输(极端币有毒确认), RankIC +0.0119/ICIR +2.36 保持正向, "
              "prod_corr=0.5047 PASS; 但 turnover=0.573(较 P1 翻倍, 尖峰>1.0)致成本~31%/yr > 毛 17%, "
              "净 -14.6%/yr 逐年全负。根因=rank 密度陷阱: hump 变换后多腿是'最接近中位'的币, "
              "中位稠密区 CLV 微小噪声即重洗排名——极端度因子的换手治理需分箱滞回而非平滑。"
              "算术判决: 即使换手压到 0.2, 净~+6% 仍不过 0.2 门。另注意 demeaned 组9(最中性)仅+0.1 "
              "而非最优(组5/6 +0.5)=二阶非单调, 多腿也非理想。close_location 分支关闭: "
              "IC 资产真实(P1 0.028/4.09)但方向性/极端度两种组合表达均不可变现。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_close_location_extremeshort_20d",
    "author": "wesleywu",
    "level": "daily",
    "category": "price_action",
    "description": "clv20=mean((2C-H-L)/(H-L),20d)；factor=-|rank(clv20)-0.5|，空行为极端多中性",
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
    """Return the raw daily close-location-extremeness matrix (date x instrument)."""

    clv_days = SETTING["params"]["clv_days"]

    close_daily = data_ctx["close"].astype("float64")
    high_daily = data_ctx["high"].astype("float64")
    low_daily = data_ctx["low"].astype("float64")

    # event close_location: CLV 日内收盘位置 [-1,1]，20d 均值
    clv_daily = (2.0 * close_daily - high_daily - low_daily) / ((high_daily - low_daily) + EPS)
    clv20 = clv_daily.rolling(clv_days, min_periods=clv_days).mean()

    # direction premium（极端度版）: 空行为极端（双向）多行为中性
    factor = -(clv20.rank(axis=1, pct=True) - 0.5).abs()
    return factor.replace([np.inf, -np.inf], np.nan)
