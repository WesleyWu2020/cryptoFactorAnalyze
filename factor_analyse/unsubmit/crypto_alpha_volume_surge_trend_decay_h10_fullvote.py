"""crypto_alpha_volume_surge_trend_decay_h10_fullvote 因子（factor_common 日频契约版）。

原始定义（h10 母因子的单点修改版）：
    surge = ma5(turnover) / ma60(turnover)，surge > 1.3 触发放量事件，
    强度 strength = clip(log(surge/1.3), 0, 1)；
    dir = mean(sign(ret5), sign(ret20), sign(ret60))（多尺度方向投票）；
    btc_align = 方向与 BTC 20d 收益同向取 1，否则 0.3；
    事件值 = dir × align × (0.5 + 0.5×strength)，
    激活额外要求 3/3 全票同号（|dir_score| == 1），过滤次极端假信号；
    触发日赋值，其后按半衰期 10d 指数衰减（event_decay），无事件不新开仓。

分钟 -> 日频改写说明：
    - turnover -> quote_volume（分钟 turnover 全日求和 == 日频 quote_volume），
      分钟 groupby(day) 聚合在日频输入下为恒等，直接使用日频字段。
    - 删除 factor_daily.shift(1) 与 reindex 回分钟索引：原 shift 仅实现执行延迟，
      框架已按次日开盘执行，t 日因子可用 <= t 数据。
    - BTC 趋势取自因子矩阵内的 BTCUSDT 列（universe 内）；若该列缺失，
      退化为全截面等权收益作为市场代理（语义最接近的全市场标量）。
    - 全票激活条件、触发比、半衰期、BTC dampen 与原实现完全一致。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_volume_surge_trend_decay_h10_fullvote",
    "author": "wesleywu",
    "level": "daily",
    "category": "pvc",
    "description": "放量事件 × 5/20/60d 收益 3/3 全票方向 × BTC regime，半衰期 10d 衰减",
}

SETTING = {
    "data_needed": ["close", "quote_volume"],
    "universe": "historical_top50",
    "warmup_bars": 70,
    "preprocessing": "mad_rank",
    "params": {
        "vol_fast_days": 5,
        "vol_slow_days": 60,
        "trigger_ratio": 1.3,
        "half_life_days": 10,
        "horizons": [5, 20, 60],
        "btc_trend_days": 20,
    },
    "factor_direction": 1,
}

_BTC_SYMBOL = "BTCUSDT"
_EPS = 1e-12


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily full-vote volume-surge trend-decay matrix (date x instrument)."""

    params = SETTING["params"]
    vol_fast_days = params["vol_fast_days"]
    vol_slow_days = params["vol_slow_days"]
    trigger_ratio = params["trigger_ratio"]
    half_life_days = params["half_life_days"]
    horizons = params["horizons"]
    btc_trend_days = params["btc_trend_days"]

    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64").clip(lower=0.0)

    # event volume_surge: 5d/60d 成交额比 > 1.3 触发，强度按 log 超出量封顶到 1
    to_fast = quote_volume.rolling(vol_fast_days, min_periods=vol_fast_days).mean()
    to_slow = quote_volume.rolling(vol_slow_days, min_periods=vol_slow_days).mean()
    surge = to_fast / (to_slow + _EPS)
    trigger = surge > trigger_ratio
    strength = np.log(surge / trigger_ratio).clip(0.0, 1.0)

    # quality multi_horizon_consistency: 5/20/60d 收益方向投票均值 ∈ {-1,-1/3,1/3,1}
    votes = [np.sign(close.pct_change(h, fill_method=None)) for h in horizons]
    dir_score = sum(votes) / len(votes)

    # context btc_trend: 信号方向与 BTC 20d 趋势同向取 1，反向降至 0.3
    if _BTC_SYMBOL in close.columns:
        btc_close = close[_BTC_SYMBOL]
    else:
        # BTC 不在 universe 内时退化为全截面等权市场收益
        btc_close = close.mean(axis=1)
    btc_ret = btc_close.pct_change(btc_trend_days, fill_method=None)
    btc_dir = np.sign(btc_ret).fillna(0.0)
    same = np.sign(dir_score.values) == btc_dir.values[:, None]
    align = pd.DataFrame(
        np.where(same, 1.0, 0.3), index=dir_score.index, columns=dir_score.columns
    )

    # 事件值：方向 × BTC regime × 强度调制；仅触发日有值
    # fullvote 单点修改：激活额外要求 3/3 全票同号（|dir_score|==1），过滤次极端假信号
    event_val = (dir_score * align * (0.5 + 0.5 * strength)).where(
        trigger & (dir_score.abs() > 0.99)
    )

    # output event_decay: 触发日赋值，其后按半衰期指数衰减
    vals = event_val.values
    state = np.full(vals.shape[1], np.nan)
    out = np.full_like(vals, np.nan)
    decay = 0.5 ** (1.0 / half_life_days)
    for t in range(vals.shape[0]):
        ev = vals[t]
        has = ~np.isnan(ev)
        state = np.where(has, ev, state * decay)
        out[t] = state

    factor = pd.DataFrame(out, index=close.index, columns=close.columns)
    return factor.replace([np.inf, -np.inf], np.nan)
