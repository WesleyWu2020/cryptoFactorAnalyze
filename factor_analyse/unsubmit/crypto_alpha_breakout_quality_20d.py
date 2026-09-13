"""crypto_alpha_breakout_quality_20d 因子（factor_common 日频契约版）。

原始定义（分钟版）：20d 区间位置中心化到 [-1,1] × 20d Kaufman 效率比
× clip(5d/20d 成交额比, 0, 2)/2，shift(1) 后广播到分钟。

日频改写说明（judgment calls）：
- 日聚合恒等：close=日内最后价、turnover 日求和 = quote_volume，直接使用日频字段。
- 删除 factor_daily.shift(1)（仅为执行延迟；框架按次日开盘执行，t 日因子可用 <= t 数据）。
- 删除分钟级 reindex 广播与分块列循环，直接返回日频矩阵。
- 所有窗口（20d 区间、20d 效率比、5d/20d 成交额比）均为日频滚动窗口，与原版语义一致。

方向说明：原 ITER_NOTE 实测结构=尾部延续 + 中段反转（整体 RankIC 为负、组合毛正净负）。
延续假设指高因子值（放量干净向上突破）延续，故 factor_direction=1（默认延续方向；
注意实测中段反转意味着该因子历史上被 REJECTED，此处仅做契约转换，不改变语义）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "breakout", "statement": "价格处于 20 日区间极端位置（中心化 [-1,1]，两端=突破/跌破）"},
    "context":   None,
    "qualities": [
        {"id": "path_cleanliness", "statement": "Kaufman 效率比 |ret20|/sum|日收益|，趋势平滑无拉锯"},
        {"id": "volume_confirm", "statement": "5d/20d 成交额放大，确认突破有真实参与"},
    ],
    "direction": {"id": "continuation", "statement": "放量干净的突破在日频尺度延续（尾部惯性市）"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "放量 + 路径干净的突破 = 知情资金持续建仓而非噪声脉冲，趋势税在日频延续；"
                 "缩量/拉锯式突破被双重 quality 过滤为假突破",
    "fields":    ["close", "turnover"],
    "expected_horizon": "1d",
    "invalidation": "|RankIC| < 0.01；或与 mom 族 GP 因子 max|rho| >= 0.85（退化为裸动量换皮）",
    "search_mode": "explore",
    "semantic_key": "breakout|-|path_cleanliness+volume_confirm|continuation|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "mom",
    "hypothesis": "六轮元结论：三个独立信息源（funding 偏离、pv 衰竭尾部、taker FOMO）都显示"
                  "crypto perp 极端状态在日频是惯性市——反转类假设系统性错误，延续类有先验支持。"
                  "与库内 hypothesis_funding_level_20d（funding 延续）机制不同源：本轮用 T1 价格"
                  "突破本身，加 volume_confirm + path_cleanliness 过滤假突破，赚趋势质量溢价。"
                  "慢窗口构造（20d 区间/效率比），换手预期 < 0.4。",
    "change": "新建因子：日频区间位置 (close-min20)/(max20-min20) 中心化到 [-1,1]，"
              "× 20d Kaufman 效率比 × clip(5d/20d 成交额比, 0, 2)/2，shift(1) 广播到分钟。",
    "expected": "|RankIC| 0.015~0.035, |RankICIR| > 2.5, ls_netir > 1, turnover 0.2~0.4, "
                "coverage > 0.6, maxdd < 0.25; prod_corr vs mom 族 GP 预期 0.4~0.7（quality 过滤"
                "应提供与裸动量的区分度），>= 0.85 则按 REDUNDANCY_REJECT 记。",
    "result": "REJECTED (STATISTICAL_REJECT, 毛好净差+方向撕裂): RankIC=-0.0290（中段反转主导）, "
              "|ICIR|=4.15 PASS, prod_corr=0.271 PASS, coverage=0.96 PASS; 但 ls_netret_ann=-1.5%, "
              "ls_netir=-0.09, maxdd=0.29。关键结构：图④尾部延续（组9 +1.0 vs 组0 -1.1，demeaned "
              "价差 2.1）与中段反转（IC<0）并存；ls 毛收益 ~+13%/yr（绿线稳升），但 turnover=0.53 "
              "x 15bp 年拖累 ~14.5%，刚好吃光毛 edge。即使换手砍半净收益也仅 ~5%/yr，达不到 0.20 "
              "门槛——毛 edge 量级不足是硬约束，分支关闭。元结论修正：延续性只在极端十分位成立，"
              "中段截面是反转市。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_breakout_quality_20d",
    "author": "wesleywu",
    "level": "daily",
    "category": "mom",
    "description": "20d 区间位置([-1,1]) × 20d Kaufman 效率比 × 5d/20d 成交额确认，赚趋势质量溢价",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    # 最长链式窗口：shift(eff_days) + rolling(eff_days) = 40 天，加缓冲。
    "warmup_bars": 46,
    "preprocessing": "mad_rank",
    "params": {
        "range_days": 20,
        "eff_days": 20,
        "vol_fast_days": 5,
        "vol_slow_days": 20,
    },
    "factor_direction": 1,
}

EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily breakout-quality matrix (date x instrument)."""

    params = SETTING["params"]
    range_days = params["range_days"]
    eff_days = params["eff_days"]
    vol_fast_days = params["vol_fast_days"]
    vol_slow_days = params["vol_slow_days"]

    close_daily = data_ctx["close"].astype("float64")
    to_daily = data_ctx["quote_volume"].astype("float64").clip(lower=0)

    # event: 20d 区间位置，中心化到 [-1, 1]（+1=创 20d 新高，-1=新低）
    lo = close_daily.rolling(range_days, min_periods=range_days).min()
    hi = close_daily.rolling(range_days, min_periods=range_days).max()
    pos = 2.0 * (close_daily - lo) / ((hi - lo) + EPS) - 1.0

    # quality 1 path_cleanliness: Kaufman 效率比，|20d 净位移| / 20d 路径总长
    ret1 = close_daily.pct_change(fill_method=None)
    net_move = (close_daily - close_daily.shift(eff_days)).abs()
    path_len = ret1.abs().rolling(eff_days, min_periods=eff_days).sum() * close_daily.shift(eff_days)
    efficiency = (net_move / (path_len + EPS)).clip(0, 1)

    # quality 2 volume_confirm: 5d/20d 成交额比，封顶 2x 后归一到 [0,1]
    to_fast = to_daily.rolling(vol_fast_days, min_periods=vol_fast_days).mean()
    to_slow = to_daily.rolling(vol_slow_days, min_periods=vol_slow_days).mean()
    vol_conf = (to_fast / (to_slow + EPS)).clip(0, 2) / 2.0

    # continuation: 区间极端位置 × 路径干净 × 放量确认
    factor = pos * efficiency * vol_conf
    return factor.replace([np.inf, -np.inf], np.nan)
