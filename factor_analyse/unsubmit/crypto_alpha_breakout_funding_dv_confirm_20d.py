"""crypto_alpha_breakout_funding_dv_confirm_20d 因子（factor_common 日频契约版）。

原始定义（分钟版，见 ITER_NOTE）：
    pos        = 20d 区间位置，中心化到 [-1, 1]（两端=突破/跌破）
    efficiency = Kaufman 效率比 |close-close.shift(20)| / (sum(|ret1|, 20d) * close.shift(20))
    dv_conf    = clip(dollar_volume 5d 均值 / 20d 均值, 0, 2) / 2
    fund_mult  = 1 + 0.5 * sign(pos) * rank_cs(funding 5d 均值)*2-1  （同向拥挤 x1.5，反向 x0.5）
    factor     = pos * efficiency * dv_conf * fund_mult，ewm(span=5) 平滑

日频化改写说明（judgment calls）：
- data_ctx 已是日频矩阵：close=日末、quote_volume=日成交额（原 dollar_volume 分钟求和的
  忠实等价物）、funding=日均值，原分钟级 groupby(day) 聚合全部变成恒等，直接使用日频字段。
- 删除 factor_daily.shift(1)（仅为分钟广播实现的执行延迟；框架按次日开盘执行，
  不允许人为延迟），并删除分钟索引广播/分块逻辑，直接返回日频矩阵。
- 所有 rolling/ewm 窗口原本即按日聚合后计算（RANGE_DAYS=20 等），日频下语义不变。
- 计算只使用当日及历史数据，无未来函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "breakout", "statement": "价格处于 20 日区间极端位置（中心化 [-1,1]，两端=突破/跌破）"},
    "context":   {"id": "funding_crowding", "statement": "5d 均值 funding 的截面分位=多空杠杆拥挤度；与突破同向拥挤=杠杆资金确认趋势"},
    "qualities": [
        {"id": "path_cleanliness", "statement": "Kaufman 效率比 |ret20|/sum|日收益|，趋势平滑无拉锯"},
        {"id": "volume_confirm", "statement": "5d/20d quote_volume 美元成交额放大，真实资金参与确认"},
    ],
    "direction": {"id": "continuation", "statement": "放量干净 + 杠杆确认的突破在日频延续（尾部惯性市先验）"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "母因子 breakout_quality 证明：极端十分位的突破延续是真实毛资产；"
                 "quote_volume 做真实体量确认，funding_crowding 做杠杆资金确认，"
                 "ewm(span=5) 压换手。",
    "fields":    ["close", "quote_volume", "funding"],
    "expected_horizon": "1d",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "mom",
    "parent_factor_id": "crypto_alpha_breakout_quality_20d",
    "hypothesis": "以 breakout_quality 为母体，补 dollar_volume + funding 维度："
                  "funding_crowding context 做杠杆资金确认，quote_volume 做真实体量确认，"
                  "ewm(span=5) 压换手。",
    "result": "REJECTED 但 15 轮最强：|RankIC|=0.0237/|ICIR|=3.40，turnover=0.227，"
              "ls_netret_ann=+13.3%，ls_netir=0.80。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_breakout_funding_dv_confirm_20d",
    "author": "wesleywu",
    "level": "daily",
    "category": "mom",
    "description": "20d 区间突破位置 x Kaufman 效率 x quote_volume 确认 x funding 拥挤乘子，ewm5 平滑",
}

SETTING = {
    "data_needed": ["close", "quote_volume", "funding"],
    "universe": "historical_top50",
    "warmup_bars": 50,
    "preprocessing": "mad_rank",
    "params": {
        "range_days": 20,
        "eff_days": 20,
        "vol_fast_days": 5,
        "vol_slow_days": 20,
        "fund_days": 5,
        "smooth_span": 5,
    },
    "factor_direction": 1,
}

EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily breakout/funding/dv-confirm matrix (date x instrument)."""

    range_days = SETTING["params"]["range_days"]
    eff_days = SETTING["params"]["eff_days"]
    vol_fast_days = SETTING["params"]["vol_fast_days"]
    vol_slow_days = SETTING["params"]["vol_slow_days"]
    fund_days = SETTING["params"]["fund_days"]
    smooth_span = SETTING["params"]["smooth_span"]

    close = data_ctx["close"].astype("float64")
    dv_daily = data_ctx["quote_volume"].astype("float64").clip(lower=0)
    fund_daily = data_ctx["funding"].astype("float64")

    # event breakout: 20d 区间位置，中心化到 [-1, 1]
    lo = close.rolling(range_days, min_periods=range_days).min()
    hi = close.rolling(range_days, min_periods=range_days).max()
    pos = 2.0 * (close - lo) / ((hi - lo) + EPS) - 1.0

    # quality path_cleanliness: Kaufman 效率比（忠实保留原始量纲写法）
    ret1 = close.pct_change(fill_method=None)
    net_move = (close - close.shift(eff_days)).abs()
    path_len = ret1.abs().rolling(eff_days, min_periods=eff_days).sum() * close.shift(eff_days)
    efficiency = (net_move / (path_len + EPS)).clip(0, 1)

    # quality volume_confirm: quote_volume 5d/20d，封顶 2x 归一 [0,1]
    dv_fast = dv_daily.rolling(vol_fast_days, min_periods=vol_fast_days).mean()
    dv_slow = dv_daily.rolling(vol_slow_days, min_periods=vol_slow_days).mean()
    dv_conf = (dv_fast / (dv_slow + EPS)).clip(0, 2) / 2.0

    # context funding_crowding: 5d 均值 funding 截面分位中心化 [-1,1]，
    # 与突破方向同向拥挤=杠杆确认（x1.5），反向=无杠杆背书（x0.5）
    fund_ma = fund_daily.rolling(fund_days, min_periods=fund_days).mean()
    fund_rank_c = fund_ma.rank(axis=1, pct=True) * 2.0 - 1.0
    fund_mult = 1.0 + 0.5 * np.sign(pos) * fund_rank_c

    factor_daily = pos * efficiency * dv_conf * fund_mult

    # 换手控制：ewm 平滑（母因子 0.53 换手是死因）
    factor_daily = factor_daily.ewm(span=smooth_span, adjust=False).mean()

    return factor_daily.replace([np.inf, -np.inf], np.nan)
