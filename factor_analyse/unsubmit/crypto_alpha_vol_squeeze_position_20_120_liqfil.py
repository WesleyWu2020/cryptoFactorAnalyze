"""crypto_alpha_vol_squeeze_position_20_120_liqfil 因子（factor_common 日频契约版）。

原始定义（SEMANTIC_PLAN/ITER_NOTE）：
    vol20 = std(daily_return, 20d)
    position = (vol20 - min(vol20, 120d)) / (max(vol20, 120d) - min(vol20, 120d))
    liquidity_filter: 20d 日均成交额截面 rank <= 20% 的标的置 NaN（剔除死币/枯竭币）
    factor = -position（越压缩值越高），截面 rank 打分

分钟->日频改写说明：
    - 原实现对分钟 close 取日内 last、对分钟 turnover 求和得到日频序列；日频契约下
      close 与 quote_volume（成交额）直接就是日频矩阵，聚合步骤为恒等。
    - 删除原实现对日频结果的 .shift(1)（执行延迟）与 reindex 回分钟索引步骤；
      框架按次日开盘执行，factor(t) 允许使用 t 日及以前数据。
    - 删除 CHUNK_SIZE 分块循环，整表向量化计算。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "vol_squeeze", "statement": "20d 已实现波动率压缩至自身 120d 区间低位"},
    "context":   None,
    "qualities": [
        {"id": "persistence", "statement": "20d vol 窗口要求压缩持续，慢变量天然低换手"},
        {"id": "liquidity_filter", "statement": "剔除 20d 美元成交额截面最低 20% 的死币/枯竭币，"
                                               "防止极端压缩尾部被病危币污染"},
    ],
    "direction": {"id": "continuation", "statement": "压缩状态的相对收益优势延续（低波异象 / lottery 规避）"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "高波动币被 lottery 偏好定价过高；持续压缩且流动性正常的低波币被系统性低估；"
                 "但压缩到接近零 vol 的死币是病危信号而非低波溢价，必须滤除",
    "fields":    ["close", "turnover"],
    "expected_horizon": "1d",
    "invalidation": "加 liquidity_filter 后组 9 毒性不消失（ls_netret 仍 < -5%）=> 尾部毒性非死币归因，"
                    "分支关闭；或与库内 vol 族 GP 因子 max|rho| >= 0.85",
    "search_mode": "mutate",
    "semantic_key": "vol_squeeze|-|liquidity_filter+persistence|continuation|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "vol",
    "hypothesis": "F0004 (vol_squeeze_position_20_120) IC 端全 PASS 但 ls 亏 14%/yr："
                  "组 9（最压缩）最毒，归因为死币/流动性枯竭币。加截面流动性过滤"
                  "（剔除 20d 美元成交额最低 20%）应能清除组 9 毒性，保留中段单调排序关系。",
    "result": "REJECTED (STATISTICAL_REJECT, 归因证伪): 指标与 F0004 几乎重合，组9毒性未消失。"
              "demeaned 收益结构本质上是驼峰形，十分位多空组合结构性无法变现中段 IC。"
              "vol_squeeze 分支按预定纪律关闭。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_vol_squeeze_position_20_120_liqfil",
    "author": "wesleywu",
    "level": "daily",
    "category": "vol",
    "description": "-position of 20d vol within its 120d range, excluding the bottom-20% "
                   "20d-quote-volume names (dead/illiquid tail filter)",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    # 1 (pct_change) + 20 (vol) + 120 (range) + buffer
    "warmup_bars": 150,
    "preprocessing": "mad_rank",
    "params": {
        "vol_days": 20,
        "range_days": 120,
        "liq_days": 20,
        "liq_exclude_pct": 0.20,
    },
    # factor = -position：值越高 = 波动越压缩 = 做多低波，方向为正
    "factor_direction": 1,
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily vol-squeeze-position matrix (date x instrument)."""

    vol_days = SETTING["params"]["vol_days"]
    range_days = SETTING["params"]["range_days"]
    liq_days = SETTING["params"]["liq_days"]
    liq_exclude_pct = SETTING["params"]["liq_exclude_pct"]

    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")

    ret = close.pct_change(fill_method=None)
    vol20 = ret.rolling(vol_days, min_periods=vol_days).std()

    # 当前 vol 在自身 range_days 区间中的位置，0=最压缩，1=最扩张
    lo = vol20.rolling(range_days, min_periods=range_days).min()
    hi = vol20.rolling(range_days, min_periods=range_days).max()
    position = (vol20 - lo) / ((hi - lo) + _EPS)

    # liquidity_filter: liq_days 日均成交额处于当日截面最低 liq_exclude_pct -> 剔除
    adv = quote_volume.clip(lower=0).rolling(liq_days, min_periods=liq_days).mean()
    adv_rank = adv.rank(axis=1, pct=True)
    illiquid = adv_rank <= liq_exclude_pct

    factor = (-position).where(~illiquid)
    return factor.replace([np.inf, -np.inf], np.nan)
