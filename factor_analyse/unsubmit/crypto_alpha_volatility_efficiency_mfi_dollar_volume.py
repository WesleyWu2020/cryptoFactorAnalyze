"""crypto_alpha_volatility_efficiency_mfi_dollar_volume 因子（factor_common 日频契约版）。

原始定义（分钟版，两个日频子信号的截面合成）：
    vol_eff = (close / close.shift(20d) - 1) / rolling_mean((high-low)/prev_close, 20d)
    mfi     = 100 * sum(pos_money_flow, 120d) / sum(pos+neg money_flow, 120d)
              （typical_price = (H+L+C)/3，raw_money_flow = tp * 日成交额，
                按 tp.diff() 符号拆正/负流）
    flow_score       = zscore(mfi - 50)
    efficiency_score = zscore(vol_eff)
    agreement_gate   = 0.5 + 0.5 * rank_pct(flow_score * efficiency_score)
    factor = flow_score * agreement_gate

分钟->日频改写说明（语义判断）：
    - 原实现对分钟数据 resample("1440min") 得到日频 H/L/C 与成交额（dollar_volume
      日内求和 == 日频 quote_volume）；日频契约下这些字段直接为日频矩阵。
    - vol_eff 分母原来是分钟级 range 均值，日频下按公式意图改写为 20 个"日"的
      (daily_high - daily_low)/prev_daily_close 均值（与 crypto_alpha_volatility_efficiency
      的日频版一致）。
    - 删除两个子信号上的 .shift(1)（执行延迟）与 broadcast 回分钟索引步骤；
      框架按次日开盘执行，factor(t) 允许使用 t 日及以前数据。
    - 删除 CHUNK_SIZE 分块循环，整表向量化计算。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_volatility_efficiency_mfi_dollar_volume",
    "author": "wesleywu",
    "level": "daily",
    "category": "volatility_volume_price",
    "description": "cross-sectional zscore(MFI-50) gated by its agreement with "
                   "volatility efficiency (20d ROC / 20d mean daily range)",
}

SETTING = {
    "data_needed": ["high", "low", "close", "quote_volume"],
    "universe": "historical_top50",
    # 120 (mfi) + 20 (vol_eff roc+range) + buffer
    "warmup_bars": 150,
    "preprocessing": "mad_rank",
    "params": {"vol_window_days": 20, "mfi_window_days": 120},
    # 无 ITER_NOTE 方向记录；资金流为正且与波动效率共振 => 高值做多，默认 1
    "factor_direction": 1,
}

_EPS = 1e-12


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily combined matrix (date x instrument)."""

    vol_window = SETTING["params"]["vol_window_days"]
    mfi_window = SETTING["params"]["mfi_window_days"]

    high = data_ctx["high"].astype("float64")
    low = data_ctx["low"].astype("float64")
    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")

    # --- volatility efficiency (daily equivalent) ---
    base_close = close.shift(vol_window)
    roc = close / base_close - 1.0
    range_pct = ((high - low) / close.shift(1)).rolling(
        vol_window,
        min_periods=vol_window,
    ).mean()
    vol_eff = roc / (range_pct + _EPS)

    # --- MFI on daily typical price x quote volume ---
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * quote_volume.clip(lower=0)
    typ_diff = typical_price.diff()

    positive_flow = raw_money_flow.where(typ_diff > 0, 0.0).where(typ_diff.notna())
    negative_flow = raw_money_flow.where(typ_diff < 0, 0.0).where(typ_diff.notna())
    positive_sum = positive_flow.rolling(mfi_window, min_periods=mfi_window).sum()
    negative_sum = negative_flow.rolling(mfi_window, min_periods=mfi_window).sum()
    total_sum = positive_sum + negative_sum
    mfi = 100.0 * positive_sum / total_sum.replace(0.0, np.nan)

    # --- cross-sectional combination ---
    flow_score = _cross_sectional_zscore(mfi - 50.0)
    efficiency_score = _cross_sectional_zscore(vol_eff)
    agreement_gate = 0.5 + 0.5 * (flow_score * efficiency_score).rank(axis=1, pct=True)
    factor = flow_score * agreement_gate

    return factor.replace([np.inf, -np.inf], np.nan)
