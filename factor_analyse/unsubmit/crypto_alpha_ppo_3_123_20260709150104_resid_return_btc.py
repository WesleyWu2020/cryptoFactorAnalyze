"""crypto_alpha_ppo_3_123_20260709150104_resid_return_btc 因子（factor_common 日频契约版）。

原始定义（PPO v18 seed123 影子候选）：
    候选表达式 EMA($resid_return_btc_20d, 10d)，score 0.018682
    计算步骤：
      1. 各币日收盘收益
      2. 相对 BTC 的 20 日滚动 beta
      3. 残差收益 = ret - beta * btc_ret
      4. resid_return_btc_20d = 20 日滚动残差和
      5. 对其做 10 日 EMA（AlphaGen 固定窗口加权）

分钟->日频改写判断（记录在案）：
  - 原实现对分钟 close groupby(day).last() 得日收盘；日频契约下直接用 daily close。
  - 原 data_needed 中的 open/high/low/volume/turnover 实际未被计算使用，精简为 close。
  - 删除把日因子广播回分钟索引的 reindex 步骤与 tradable_mask 掩码
    （框架按 universe 自动掩码）。
  - 原手写逐行 numpy 滚动循环改为 pandas rolling，语义一致（window=20, min_periods=8）。

注意：该 PPO 候选当初因 "train_rankic_spread_sign_mismatch" 被拒，
factor_direction 默认取 1（高残差动量为多头），使用前需自行验证符号。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_ppo_3_123_20260709150104_resid_return_btc",
    "author": "wesleywu",
    "level": "daily",
    "category": "single",
    "description": "EMA(20d BTC-残差收益和, 10d)：剔除 BTC beta 后的个体动量",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    # beta(20d) + resid_sum(20d) + EMA(10d) 链式最长约 50 天，加缓冲取 60
    "warmup_bars": 60,
    "preprocessing": "mad_rank",
    "params": {
        "beta_window": 20,
        "resid_window": 20,
        "ema_window": 10,
        "min_periods": 8,
        "ret_clip_low": -0.5,
        "ret_clip_high": 5.0,
    },
    "factor_direction": 1,  # 默认：高残差动量优先（原候选曾因符号不一致被拒，见 docstring）
}

_EPS = 1e-12


def _find_btc_column(columns: pd.Index) -> Optional[str]:
    for name in columns:
        s = str(name).upper()
        if s == "BTCUSDT" or s.startswith("BTC") or "BTC" in s:
            return str(name)
    return None


def _rolling_ema(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """AlphaGen EMA(x, window)：固定窗口加权平均（非递归 ewm）。"""
    alpha = 1.0 - 2.0 / (1.0 + float(window))
    weights = alpha ** np.arange(window, 0, -1, dtype=np.float64)
    weights /= weights.sum()

    def _ema(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        return float(np.dot(values, weights))

    return frame.rolling(window, min_periods=window).apply(_ema, raw=True)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily BTC-residual-momentum matrix (date x instrument)."""

    p = SETTING["params"]
    beta_window = p["beta_window"]
    resid_window = p["resid_window"]
    ema_window = p["ema_window"]
    min_periods = p["min_periods"]

    close = data_ctx["close"].astype("float64")

    btc_col = _find_btc_column(close.columns)
    if btc_col is None:
        # 无 BTC 基准可用；返回全 NaN。
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    # 日收益（保留原实现的截尾，抑制极端异动）
    daily_ret = close / (close.shift(1) + _EPS) - 1.0
    daily_ret = daily_ret.clip(lower=p["ret_clip_low"], upper=p["ret_clip_high"])

    btc_ret = daily_ret[btc_col]

    # 20 日滚动 beta vs BTC
    cov_sum = daily_ret.mul(btc_ret, axis=0).rolling(beta_window, min_periods=min_periods).sum()
    var_sum = btc_ret.pow(2).rolling(beta_window, min_periods=min_periods).sum()
    beta = cov_sum.div(var_sum + _EPS, axis=0)

    # 残差收益与 20 日滚动和
    resid = daily_ret.sub(beta.mul(btc_ret, axis=0))
    resid_sum = resid.rolling(resid_window, min_periods=min_periods).sum()

    # 10 日 EMA（AlphaGen 固定窗口）
    factor = _rolling_ema(resid_sum, ema_window)
    return factor.replace([np.inf, -np.inf], np.nan)
