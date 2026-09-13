"""crypto_alpha_breakout_funding_dv_confirm_20d_sqfil 因子（factor_common 日频契约版）。

在母体上新增 F1' vol_squeeze_filter 驼峰乘子：
    vol20        = 日收益 20d 已实现波动（std）
    squeeze_pos  = vol20 在自身 120d rolling min-max 区间中的位置（同 F0004 构造）
    squeeze_mult = 1 - 0.5 * |2*squeeze_pos - 1|（极端压缩=死币、极端扩张=彩票，中段=1.0）
    factor = pos * efficiency * dv_conf * fund_mult * squeeze_mult

日频化改写说明（judgment calls）：
- data_ctx 已是日频矩阵：quote_volume 替代原 dollar_volume、funding=日均值；
  原分钟级 groupby(day) 聚合全部删除，直接使用日频字段。
- 删除 factor_daily.shift(1)（仅为分钟广播实现的执行延迟；框架按次日开盘执行），
  并删除分钟索引广播/分块逻辑。
- vol20 原本即由日收益计算（ret1 来自日频 close），不是分钟级 realized vol，
  日频下完全忠实，无近似。
- 最长链路 = 20d vol std + 120d min-max 区间 = 140 天，warmup 取 150 天。
- 所有操作因果，无未来函数。
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
        {"id": "vol_squeeze_filter", "statement": "驼峰过滤：vol20 在自身 120d 区间的位置，"
                  "1-0.5*|2*pos-1|——极端压缩=死币、极端扩张=彩票（F0004 倒U+组9最毒），逐币降权有毒 vol 状态"},
    ],
    "direction": {"id": "continuation", "statement": "放量干净 + 杠杆确认的突破在日频延续（尾部惯性市先验）"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "F0004 解剖显示 vol_squeeze 呈倒 U——极端压缩=死币、极端扩张=彩票（组9最毒），"
                 "中段健康。驼峰乘子逐币压低有毒 vol 状态的仓位，不要求 squeeze 自身有组合级毛收益。",
    "fields":    ["close", "quote_volume", "funding"],
    "expected_horizon": "1d",
}

ITER_NOTE: dict = {
    "op_type": "modify_factor",
    "factor_family": "mom",
    "parent_factor_id": "crypto_alpha_breakout_funding_dv_confirm_20d",
    "hypothesis": "squeeze 的 IC 资产作为条件化工具变现——作 quality 过滤而非收益成分："
                  "压死极端压缩（死币）与极端扩张（彩票）仓位。",
    "result": "REJECTED（F1' 证伪，squeeze 资产彻底关闭）：净年化 13.3%→5.3%——母体尾部收益"
              "恰恰来自极端 vol 状态币，'毒尾'就是收益源。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_breakout_funding_dv_confirm_20d_sqfil",
    "author": "wesleywu",
    "level": "daily",
    "category": "mom",
    "description": "母体 + vol_squeeze 驼峰过滤（vol20 在自身 120d 区间位置，两端降权）",
}

SETTING = {
    "data_needed": ["close", "quote_volume", "funding"],
    "universe": "historical_top50",
    "warmup_bars": 150,
    "preprocessing": "mad_rank",
    "params": {
        "range_days": 20,
        "eff_days": 20,
        "vol_fast_days": 5,
        "vol_slow_days": 20,
        "fund_days": 5,
        "smooth_span": 5,
        "sq_vol_days": 20,
        "sq_range_days": 120,
    },
    "factor_direction": 1,
}

EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily matrix with the vol-squeeze hump filter."""

    range_days = SETTING["params"]["range_days"]
    eff_days = SETTING["params"]["eff_days"]
    vol_fast_days = SETTING["params"]["vol_fast_days"]
    vol_slow_days = SETTING["params"]["vol_slow_days"]
    fund_days = SETTING["params"]["fund_days"]
    smooth_span = SETTING["params"]["smooth_span"]
    sq_vol_days = SETTING["params"]["sq_vol_days"]
    sq_range_days = SETTING["params"]["sq_range_days"]

    close = data_ctx["close"].astype("float64")
    dv_daily = data_ctx["quote_volume"].astype("float64").clip(lower=0)
    fund_daily = data_ctx["funding"].astype("float64")

    # event breakout: 20d 区间位置，中心化到 [-1, 1]
    lo = close.rolling(range_days, min_periods=range_days).min()
    hi = close.rolling(range_days, min_periods=range_days).max()
    pos = 2.0 * (close - lo) / ((hi - lo) + EPS) - 1.0

    # quality path_cleanliness: Kaufman 效率比
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

    # quality vol_squeeze_filter: 驼峰乘子，逐币降权有毒 vol 状态
    # （极端压缩=死币/极端扩张=彩票，F0004 倒U+组9最毒；中段=1.0，两端=0.5）
    vol20 = ret1.rolling(sq_vol_days, min_periods=sq_vol_days).std()
    sq_lo = vol20.rolling(sq_range_days, min_periods=sq_range_days).min()
    sq_hi = vol20.rolling(sq_range_days, min_periods=sq_range_days).max()
    squeeze_pos = (vol20 - sq_lo) / ((sq_hi - sq_lo) + EPS)
    squeeze_mult = 1.0 - 0.5 * (2.0 * squeeze_pos - 1.0).abs()

    factor_daily = pos * efficiency * dv_conf * fund_mult * squeeze_mult

    # 换手控制：ewm 平滑（母因子 0.53 换手是死因）
    factor_daily = factor_daily.ewm(span=smooth_span, adjust=False).mean()

    return factor_daily.replace([np.inf, -np.inf], np.nan)
