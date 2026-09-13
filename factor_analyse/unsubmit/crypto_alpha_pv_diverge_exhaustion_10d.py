"""crypto_alpha_pv_diverge_exhaustion_10d 因子（factor_common 日频契约版）。

原始定义（分钟版改造）：
    ret10 = close.pct_change(10d)
    vol_dry = clip(1 - mean(quote_volume, 5d) / mean(quote_volume, 20d), lower=0)
    tr_fast/tr_slow = taker 买占比 5d / 20d
    taker_fade = clip(-sign(ret10) * (tr_fast - tr_slow) / |tr_slow|, lower=0)
    exhaust = 0.5 * (vol_dry + taker_fade)
    factor = -(ret10 * exhaust)（衰竭漂移 -> 反向）

日频改写判断：
    - 分钟聚合（close=last, quote_volume/taker=sum）在日频输入下为恒等，直接用日频字段。
    - 删除 factor_daily.shift(1)（纯执行延迟，框架按次日开盘执行）。
    - 删除分钟 index 广播与分块循环。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "pv_diverge", "statement": "10 日价格漂移与量能趋势背离（价涨量缩/价跌量缩）"},
    "context":   None,
    "qualities": [{"id": "taker_confirm", "statement": "taker 主动买卖占比同向退潮，确认需求/抛压衰竭"}],
    "direction": {"id": "reversal", "statement": "量能与 taker 流同步退潮的漂移无承接 -> 反向"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "上涨但量缩 + taker 买退潮 = 需求衰竭 -> 回落；下跌但量缩 + taker 卖退潮 = 抛压衰竭 -> 反弹。"
                 "对称衰竭构造: -(ret10 * exhaust)，exhaust 由量缩幅度与 taker 退潮幅度合成",
    "fields":    ["close", "xbinance_quote_volume", "xbinance_taker_buy_quote_volume"],
    "expected_horizon": "1d",
    "invalidation": "|RankIC| < 0.01；或与 pvc 族 GP 因子 max|rho| >= 0.6（taker_confirm 未提供 T2 增量）",
    "search_mode": "explore",
    "semantic_key": "pv_diverge|-|taker_confirm|reversal|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "pvc",
    "hypothesis": "前五轮证伪/定位了 taker 延续、taker 反转、funding 回归、vol_squeeze 四个分支；"
                  "剩余候选中 pv_diverge+taker_confirm 是唯一 T1+T2 混合事件：纯 T1 量价背离"
                  "大概率与 GP 库同质，taker 退潮确认是区分度来源。机制：无承接的漂移反转。"
                  "构造沿用慢窗口（10d 漂移、5d vs 20d 量能/taker 趋势），换手预期 < 0.4。",
    "change": "新建因子：日频 ret10 × exhaust 取负；exhaust = 0.5*(量缩幅度 + taker 同向退潮幅度)，"
              "两项均 clip 下界 0（只计衰竭方向），shift(1) 广播到分钟 index。",
    "expected": "|RankIC| 0.012~0.025, |RankICIR| > 2, ls_netir > 1, turnover 0.15~0.4, "
                "coverage > 0.6; prod_corr vs pvc 族 GP max|rho| < 0.6（T2 成分应拉低相关）。",
    "result": "REJECTED (STATISTICAL_REJECT, 无信号+尾部反向): |RankIC|=0.0011 (delay=0 也仅 0.0093), "
              "|ICIR|=0.19, ls_netret_ann=-32.0%, turnover=0.66, maxdd=0.86; prod_corr=0.258 PASS。"
              "图④：组0（被做空的衰竭上涨尾部分位）demeaned +2.0 全场最强——衰竭漂移在尾部是延续"
              "而非反转，与 funding 轮的 continuation 结论互证（本宇宙尾部趋势、中段噪声）。"
              "但 |IC| 两个方向都过不了 0.01 门槛（中段无排序力），翻号也救不了，分支关闭。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_pv_diverge_exhaustion_10d",
    "author": "wesleywu",
    "level": "daily",
    "category": "pvc",
    "description": "-(ret10 * exhaust); exhaust = 0.5*(量缩幅度 + taker 同向退潮幅度), 衰竭漂移反转",
}

SETTING = {
    "data_needed": ["close", "quote_volume", "taker_buy_quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 35,
    "preprocessing": "mad_rank",
    "params": {"ret_days": 10, "fast_days": 5, "slow_days": 20},
    "factor_direction": 1,  # 默认：高因子值（衰竭下跌/未衰竭）优先；原始分钟版已被证伪（见 ITER_NOTE.result）
}

_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily pv-diverge-exhaustion matrix (date x instrument)."""

    ret_days = SETTING["params"]["ret_days"]
    fast_days = SETTING["params"]["fast_days"]
    slow_days = SETTING["params"]["slow_days"]

    close = data_ctx["close"].astype("float64")
    qv_daily = data_ctx["quote_volume"].astype("float64").clip(lower=0)
    taker_daily = data_ctx["taker_buy_quote_volume"].astype("float64").clip(lower=0)

    ret10 = close.pct_change(ret_days, fill_method=None)

    # 量能趋势：5d 均值相对 20d 均值的缩量幅度（>0 = 量在缩）
    qv_fast = qv_daily.rolling(fast_days, min_periods=fast_days).mean()
    qv_slow = qv_daily.rolling(slow_days, min_periods=slow_days).mean()
    vol_dry = (1.0 - qv_fast / (qv_slow + _EPS)).clip(lower=0)

    # taker 退潮：taker 买占比 5d vs 20d 的变化，逆着漂移方向计（>0 = 同向流在退）
    tr_fast = (taker_daily.rolling(fast_days, min_periods=fast_days).sum()
               / (qv_fast * fast_days + _EPS))
    tr_slow = (taker_daily.rolling(slow_days, min_periods=slow_days).sum()
               / (qv_slow * slow_days + _EPS))
    taker_fade = (-np.sign(ret10) * (tr_fast - tr_slow) / (tr_slow.abs() + _EPS)).clip(lower=0)

    exhaust = 0.5 * (vol_dry + taker_fade)

    # reversal: 衰竭漂移 -> 反向；ret10 与 exhaust 同号组合出对称多空
    factor = -(ret10 * exhaust)
    return factor.replace([np.inf, -np.inf], np.nan)
