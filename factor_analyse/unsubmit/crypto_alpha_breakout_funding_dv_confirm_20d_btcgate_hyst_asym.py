"""crypto_alpha_breakout_funding_dv_confirm_20d_btcgate_hyst_asym 因子（factor_common 日频契约版）。

在 hyst 滞回门上做 E3 非对称：关门日只 NaN 多头侧（正分数，熊市向上突破=空头回补
假反弹），空头侧（负分数，恐慌延续）照常交易。门触发器（-3%/+2% 滞回）与母体其余
构造全部冻结。

日频化改写说明（judgment calls）：
- data_ctx 已是日频矩阵：quote_volume 替代原 dollar_volume、funding=日均值；
  原分钟级 groupby(day) 聚合全部删除，直接使用日频字段。
- 删除 factor_daily.shift(1)（仅为分钟广播实现的执行延迟；框架按次日开盘执行），
  并删除分钟索引广播/分块逻辑。
- 滞回门的 ffill 是因果操作，保留；初始 fillna(1.0)=默认开门，与原版一致。
- BTCUSDT 若不在矩阵列中，退化为等权市场指数代理（正常不会发生）。
- 所有窗口原本即按日聚合后计算，日频下语义不变；无未来函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "breakout", "statement": "价格处于 20 日区间极端位置（中心化 [-1,1]，两端=突破/跌破）"},
    "context":   {"id": "btc_trend", "statement": "BTC 20d 趋势滞回门：ret20<-3% 关门、>+2% 开门、"
                  "中间保持原状态；E3 非对称——关门日只 NaN 多头侧（熊市向上突破=空头回补假反弹），"
                  "空头侧继续交易（熊市恐慌延续是真信号）"},
    "qualities": [
        {"id": "funding_confirm", "statement": "5d 均值 funding 截面分位与突破同向=杠杆资金确认（x1.5），反向=无背书（x0.5）"},
        {"id": "path_cleanliness", "statement": "Kaufman 效率比 |ret20|/sum|日收益|，趋势平滑无拉锯"},
        {"id": "volume_confirm", "statement": "5d/20d quote_volume 美元成交额放大，真实资金参与确认"},
    ],
    "direction": {"id": "continuation", "statement": "放量干净 + 杠杆确认的突破在日频延续（尾部惯性市先验）"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "hyst 唯一 FAIL 是 coverage——'熊市整体空仓'与 coverage 门槛结构性矛盾；"
                 "机制假设 2022 熊市亏损几乎全部来自多头侧，非对称门只关多头侧。",
    "fields":    ["close", "quote_volume", "funding"],
    "expected_horizon": "1d",
}

ITER_NOTE: dict = {
    "op_type": "modify_factor",
    "factor_family": "mom",
    "parent_factor_id": "crypto_alpha_breakout_funding_dv_confirm_20d_btcgate_hyst",
    "hypothesis": "2022 熊市亏损几乎全部来自多头侧（向上突破=空头回补假反弹），空头侧"
                  "（向下突破=恐慌延续）在熊市反而是最强段。非对称门：关门日只 NaN 正分数币。",
    "result": "REJECTED（非对称假设证伪）：coverage 0.905 PASS 但 2022 -8.1%——熊市空头侧"
              "同样出血，2022 修复来自整体回避 regime 而非选边。全面被 hyst 支配，淘汰。",
}


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_breakout_funding_dv_confirm_20d_btcgate_hyst_asym",
    "author": "wesleywu",
    "level": "daily",
    "category": "mom",
    "description": "母体 + BTC 滞回门非对称版（关门日只 NaN 多头侧，空头侧照常交易）",
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
        "btc_trend_days": 20,
        "gate_shut_th": -0.03,
        "gate_open_th": 0.02,
    },
    "factor_direction": 1,
}

BTC_SYMBOL = "BTCUSDT"
EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily matrix with an asymmetric BTC-trend hysteresis gate."""

    range_days = SETTING["params"]["range_days"]
    eff_days = SETTING["params"]["eff_days"]
    vol_fast_days = SETTING["params"]["vol_fast_days"]
    vol_slow_days = SETTING["params"]["vol_slow_days"]
    fund_days = SETTING["params"]["fund_days"]
    smooth_span = SETTING["params"]["smooth_span"]
    btc_trend_days = SETTING["params"]["btc_trend_days"]
    gate_shut_th = SETTING["params"]["gate_shut_th"]
    gate_open_th = SETTING["params"]["gate_open_th"]

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

    # 换手控制：ewm 平滑（母因子 0.53 换手是死因）
    factor_daily = factor_daily.ewm(span=smooth_span, adjust=False).mean()

    # context btc_trend 滞回门：ret20<-3% 关门、>+2% 开门、中间保持原状态
    if BTC_SYMBOL in close.columns:
        btc_price = close[BTC_SYMBOL]
    else:
        # 退化代理：全列等权平均收益构建的市场指数（正常不会发生，top50 宇宙含 BTC）
        btc_price = (1.0 + ret1.mean(axis=1).fillna(0.0)).cumprod()
    btc_ret20 = btc_price.pct_change(btc_trend_days, fill_method=None)
    gate = pd.Series(np.nan, index=btc_ret20.index)
    gate[btc_ret20 > gate_open_th] = 1.0
    gate[btc_ret20 < gate_shut_th] = 0.0
    gate = gate.ffill().fillna(1.0)  # 因果 ffill 保持原状态；初始默认开门
    # E3 非对称：关门日只 NaN 多头侧（正分数），空头侧（恐慌延续）照常交易
    shut = gate.eq(0.0)
    kill_long = factor_daily.gt(0).to_numpy() & shut.to_numpy()[:, None]
    factor_daily = factor_daily.mask(kill_long)

    return factor_daily.replace([np.inf, -np.inf], np.nan)
