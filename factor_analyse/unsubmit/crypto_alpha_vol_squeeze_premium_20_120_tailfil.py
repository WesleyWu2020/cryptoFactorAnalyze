"""crypto_alpha_vol_squeeze_premium_20_120_tailfil 因子（factor_common 日频契约版）。

原始定义（SEMANTIC_PLAN/ITER_NOTE）：
    vol20 = std(daily_return, 20d)
    position = (vol20 - min(vol20, 120d)) / (max(vol20, 120d) - min(vol20, 120d))
    factor = -position（低波溢价：越压缩值越高）
    outlier_filter: 每日截面因子值 top 5%（最极端压缩，波动性死亡/地雷区）置 NaN

分钟->日频改写说明：
    - 原实现对分钟 close 取日内 last 得到日频收盘；日频契约下 close 直接为日频矩阵。
    - 删除原实现对日频结果的 .shift(1)（执行延迟）与 reindex 回分钟索引步骤；
      框架按次日开盘执行，factor(t) 允许使用 t 日及以前数据。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "vol_squeeze", "statement": "20d 已实现波动率压缩至自身 120d 区间低位"},
    "context":   None,
    "qualities": [
        {"id": "outlier_filter", "statement": "每日截面剔除最极端压缩的 5%（波动死亡/地雷区），只做干净的低波溢价"},
        {"id": "persistence", "statement": "20d vol 窗口要求压缩持续，慢变量天然低换手"},
    ],
    "direction": {"id": "premium", "statement": "低波溢价是风险定价特征而非方向预测：做多稳定低波、做空高波 lottery，符号由先验固定"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "高波币被 lottery 偏好的散户资金定价过高，持续压缩的低波币被系统性低估——这是"
                 "溢价（谁赚），不是预测（往哪走）。组9（极端压缩）是'波动性死亡'：做市撤出/项目停更/"
                 "崩盘前夜的流动性真空，必须按因子值截面截尾剔除。",
    "fields":    ["close"],
    "expected_horizon": "1d",
    "search_mode": "exploit",
    "semantic_key": "vol_squeeze|-|outlier_filter+persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "vol",
    "hypothesis": "母因子留下最强 IC 资产（|RankIC|=0.0247/|ICIR|=4.14）但被组9 毒性拖死。"
                  "把'压缩延续'方向框架换成'低波溢价'特征框架（premium），用 outlier_filter "
                  "截掉极端压缩 5% 地雷区，保留组4~8 的单调溢价。",
    "result": "REJECTED (STATISTICAL_REJECT, invalidation#1 触发: 毒性归因二次证伪): "
              "IC 资产完好但 ls_netret_ann=-16.1%。驼峰不是尾部问题是整体结构，"
              "vol_squeeze 族永久关闭。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_vol_squeeze_premium_20_120_tailfil",
    "author": "wesleywu",
    "level": "daily",
    "category": "vol",
    "description": "low-vol premium: -position of 20d vol within its 120d range, "
                   "cross-sectionally tail-cut at top 5% (volatility-death zone)",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    # 1 (pct_change) + 20 (vol) + 120 (range) + buffer
    "warmup_bars": 150,
    "preprocessing": "mad_rank",
    "params": {
        "vol_days": 20,
        "range_days": 120,
        "tail_cut_pct": 0.95,
    },
    # 低波溢价：factor = -position，值越高 = 越压缩 = 做多低波，方向为正
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily vol-squeeze-premium matrix (date x instrument)."""

    vol_days = SETTING["params"]["vol_days"]
    range_days = SETTING["params"]["range_days"]
    tail_cut_pct = SETTING["params"]["tail_cut_pct"]

    close = data_ctx["close"].astype("float64")

    ret = close.pct_change(fill_method=None)
    vol20 = ret.rolling(vol_days, min_periods=vol_days).std()

    # 当前 vol 在自身 range_days 区间中的位置，0=最压缩，1=最扩张；取负=越压缩值越高
    lo = vol20.rolling(range_days, min_periods=range_days).min()
    hi = vol20.rolling(range_days, min_periods=range_days).max()
    position = (vol20 - lo) / ((hi - lo) + _EPS)
    factor = -position

    # quality outlier_filter: 截面最极端压缩 top 5% 是波动性死亡/地雷区，置 NaN
    cutoff = factor.quantile(tail_cut_pct, axis=1)
    factor = factor.where(factor.le(cutoff, axis=0))

    return factor.replace([np.inf, -np.inf], np.nan)
