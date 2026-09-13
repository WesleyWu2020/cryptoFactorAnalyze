"""crypto_alpha_breakout_funding_dv_confirm_20d_resid 因子（factor_common 日频契约版）。

在母体上做 N1 残差化输出层：原始分数（pos*efficiency*dv_conf*fund_mult）在 ewm 之前，
逐日截面对 60d 滚动 beta（vs 等权市场收益）回归取残差，剔除"高 beta 币牛市被顶入
top decile、熊市反向出血"的搭便车成分。

日频化改写说明（judgment calls）：
- data_ctx 已是日频矩阵：quote_volume 替代原 dollar_volume、funding=日均值；
  原分钟级 groupby(day) 聚合全部删除，直接使用日频字段。
- 删除 factor_daily.shift(1)（仅为分钟广播实现的执行延迟；框架按次日开盘执行），
  并删除分钟索引广播/分块逻辑。
- beta 用日收益对截面等权市场收益的 60d 滚动 cov/var（原本即在日频帧上计算，
  语义不变）；warmup 按最长链路 60d beta + 1d pct_change 取 75 天。
- 所有操作因果（rolling/cov 只用过去数据），无未来函数。
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
    "output":    {"id": "residualized", "statement": "原始分数逐日截面对 60d 滚动 beta（vs 等权市场）回归取残差，"
                  "剔除 beta 搭便车成分后再 ewm 平滑"},
    "mechanism": "母体多头腿搭市场便车（高 beta 币牛市被顶入 top decile、2022 熊市反向出血）。"
                 "对 beta 取截面残差后，因子只捕捉纯 idiosyncratic 突破强度，市场方向暴露归零。",
    "fields":    ["close", "quote_volume", "funding"],
    "expected_horizon": "1d",
}

ITER_NOTE: dict = {
    "op_type": "modify_factor",
    "factor_family": "mom",
    "parent_factor_id": "crypto_alpha_breakout_funding_dv_confirm_20d",
    "hypothesis": "原始分数里混入 beta 成分，逐日截面回归 raw ~ beta_60d 取残差后，"
                  "纯 idiosyncratic 突破强度的净年化应提升且 2022 修复。",
    "result": "REJECTED（residualized 方向关闭）：净年化 13.3%→+1.7%，2022 更差——母体 edge "
              "实质是系统性 beta/regime 暴露而非纯截面 alpha，'搭便车'就是车本身。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_breakout_funding_dv_confirm_20d_resid",
    "author": "wesleywu",
    "level": "daily",
    "category": "mom",
    "description": "母体原始分数对 60d 滚动 beta（vs 等权市场）逐日截面回归取残差后再 ewm5",
}

SETTING = {
    "data_needed": ["close", "quote_volume", "funding"],
    "universe": "historical_top50",
    "warmup_bars": 75,
    "preprocessing": "mad_rank",
    "params": {
        "range_days": 20,
        "eff_days": 20,
        "vol_fast_days": 5,
        "vol_slow_days": 20,
        "fund_days": 5,
        "smooth_span": 5,
        "beta_days": 60,
    },
    "factor_direction": 1,
}

EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the beta-residualized daily breakout matrix (date x instrument)."""

    range_days = SETTING["params"]["range_days"]
    eff_days = SETTING["params"]["eff_days"]
    vol_fast_days = SETTING["params"]["vol_fast_days"]
    vol_slow_days = SETTING["params"]["vol_slow_days"]
    fund_days = SETTING["params"]["fund_days"]
    smooth_span = SETTING["params"]["smooth_span"]
    beta_days = SETTING["params"]["beta_days"]

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

    factor_daily = pos * efficiency * dv_conf * fund_mult

    # output residualized: 逐日截面回归 raw ~ beta_60d（vs 等权市场）取残差，
    # 剔除"高 beta 币牛市被顶入 top decile、熊市反向出血"的搭便车成分
    mkt_ret = ret1.mean(axis=1)
    beta_cov = ret1.rolling(beta_days, min_periods=beta_days).cov(mkt_ret)
    beta_var = mkt_ret.rolling(beta_days, min_periods=beta_days).var()
    beta = beta_cov.div(beta_var + EPS, axis=0)
    beta_c = beta.sub(beta.mean(axis=1), axis=0)
    raw_c = factor_daily.sub(factor_daily.mean(axis=1), axis=0)
    slope = (raw_c * beta_c).sum(axis=1) / (beta_c.pow(2).sum(axis=1) + EPS)
    factor_daily = factor_daily - beta.mul(slope, axis=0)

    # 换手控制：ewm 平滑（母因子 0.53 换手是死因）
    factor_daily = factor_daily.ewm(span=smooth_span, adjust=False).mean()

    return factor_daily.replace([np.inf, -np.inf], np.nan)
