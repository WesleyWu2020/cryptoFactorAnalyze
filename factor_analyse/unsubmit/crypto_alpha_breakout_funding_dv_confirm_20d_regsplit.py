"""crypto_alpha_breakout_funding_dv_confirm_20d_regsplit 因子（factor_common 日频契约版）。

在母体上做 D2 双 regime 编码：极端区（|pos|>=0.6）走 continuation（pos 原样），
中段走 reversal（-pos 翻转），funding 确认方向跟随翻转后的交易方向：
    pos_signed = pos * where(|pos|>=tail_th, 1, -1)
    factor = pos_signed * efficiency * dv_conf * fund_mult(sign 跟随 pos_signed)

日频化改写说明（judgment calls）：
- data_ctx 已是日频矩阵：quote_volume 替代原 dollar_volume、funding=日均值；
  原分钟级 groupby(day) 聚合全部删除，直接使用日频字段。
- 删除 factor_daily.shift(1)（仅为分钟广播实现的执行延迟；框架按次日开盘执行），
  并删除分钟索引广播/分块逻辑。
- 所有窗口原本即按日聚合后计算，日频下语义不变；无未来函数。
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
    "direction": {"id": "regime_split", "statement": "|pos|>=0.6 极端区走 continuation（尾部延续），中段走 reversal（均值回归）——16 轮元结论的显式编码"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "16 轮实验最强元结论：本宇宙极端十分位是延续市、中段是反转市。双 regime 编码"
                 "把两个方向写进同一因子：极端区 pos 原样（延续），中段 -pos 翻转（反转）。",
    "fields":    ["close", "quote_volume", "funding"],
    "expected_horizon": "1d",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "mom",
    "parent_factor_id": "crypto_alpha_breakout_funding_dv_confirm_20d",
    "hypothesis": "单假设测试：pos_signed = pos * where(|pos|>=0.6, 1, -1) 替代原 pos 进入乘积；"
                  "IC 若显著转正，中段反转是真实资产。母体 funding/dv 确认、ewm 平滑全部冻结。",
    "result": "REJECTED（D2 证伪）：RankIC -0.024→-0.0164 未转正，turnover +64%，"
              "中段为纯噪声区。本因子资产 100% 在尾部，D2 永久关闭。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_breakout_funding_dv_confirm_20d_regsplit",
    "author": "wesleywu",
    "level": "daily",
    "category": "mom",
    "description": "母体 + 双 regime 编码（|pos|>=0.6 延续、中段 -pos 反转）",
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
        "tail_th": 0.6,
    },
    "factor_direction": 1,
}

EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily regime-split breakout matrix (date x instrument)."""

    range_days = SETTING["params"]["range_days"]
    eff_days = SETTING["params"]["eff_days"]
    vol_fast_days = SETTING["params"]["vol_fast_days"]
    vol_slow_days = SETTING["params"]["vol_slow_days"]
    fund_days = SETTING["params"]["fund_days"]
    smooth_span = SETTING["params"]["smooth_span"]
    tail_th = SETTING["params"]["tail_th"]

    close = data_ctx["close"].astype("float64")
    dv_daily = data_ctx["quote_volume"].astype("float64").clip(lower=0)
    fund_daily = data_ctx["funding"].astype("float64")

    # event breakout: 20d 区间位置，中心化到 [-1, 1]
    lo = close.rolling(range_days, min_periods=range_days).min()
    hi = close.rolling(range_days, min_periods=range_days).max()
    pos = 2.0 * (close - lo) / ((hi - lo) + EPS) - 1.0

    # direction regime_split: 极端区(|pos|>=tail_th)延续，中段(-pos)反转
    pos_signed = pos * np.where(pos.abs() >= tail_th, 1.0, -1.0)

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
    # 确认方向跟随交易方向（中段反转后 sign 同步翻转）
    fund_ma = fund_daily.rolling(fund_days, min_periods=fund_days).mean()
    fund_rank_c = fund_ma.rank(axis=1, pct=True) * 2.0 - 1.0
    fund_mult = 1.0 + 0.5 * np.sign(pos_signed) * fund_rank_c

    factor_daily = pos_signed * efficiency * dv_conf * fund_mult

    # 换手控制：ewm 平滑（母因子 0.53 换手是死因）
    factor_daily = factor_daily.ewm(span=smooth_span, adjust=False).mean()

    return factor_daily.replace([np.inf, -np.inf], np.nan)
