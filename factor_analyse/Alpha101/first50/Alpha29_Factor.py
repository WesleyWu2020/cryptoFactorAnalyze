"""Alpha101 Alpha#29 因子（factor_common 契约版）。

原始定义:
    Alpha#29 = min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5)
               + ts_rank(delay((-1 * returns), 6), 5)

本实现保留原脚本“币圈7×24h优化版”的实际计算结构:
    左支: delta(close, delta_window)
          -> -1 * rank -> rank -> rank
          -> ts_min(..., ts_min_window)
          -> log(|x| + 1)
          -> TS z-score（rolling scale_window 均值/标准差；旧代码的 scale 实现）
          -> 两次 rank（product(..., 1) 为恒等，省略）
          -> ts_min(..., min_window)
    右支: ts_rank(delay(-returns, delay_window), ts_rank_window)
    factor = 左支 + 右支
参数取原脚本 __main__ 实际调用值 ts_rank_window=5, delay_window=6，
其余窗口沿用代码内硬编码（delta_window=5, ts_min_window=2, scale_window=20, min_window=5；
rebalance_period 仅用于旧版 future_ret，丢弃）。

与旧脚本的一处必要偏离（修复未来函数）:
    旧脚本中所有 rank(...) 均为“单 symbol 全历史时间序列秩”，t 时点取值依赖未来数据，
    属于前视偏差。这里按经典 Alpha#29 公式语义统一改为当日横截面 pct 秩（axis=1）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
横截面去极值与秩归一化由框架 preprocessing="mad_rank" 完成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "Alpha29_Factor",
    "author": "local",
    "level": "daily",
    "category": "alpha101",
    "description": "Alpha101 #29: ts_min(rank(scale(log(ts_min(rank(-rank(delta(close,5))),2))+1)),5) + ts_rank(delay(-returns,6),5)",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 40,
    "preprocessing": "mad_rank",
    "params": {
        "delta_window": 5,
        "ts_min_window": 2,
        "scale_window": 20,
        "min_window": 5,
        "ts_rank_window": 5,
        "delay_window": 6,
    },
    "factor_direction": 1,
}

_EPSILON = 1e-8


def _ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series pct rank of the latest value within the window, causal."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda arr: pd.Series(arr).rank(pct=True).iloc[-1], raw=False
    )


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily Alpha#29 matrix (date x instrument)."""

    params = SETTING["params"]
    delta_window = params["delta_window"]
    ts_min_window = params["ts_min_window"]
    scale_window = params["scale_window"]
    min_window = params["min_window"]
    ts_rank_window = params["ts_rank_window"]
    delay_window = params["delay_window"]
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"SETTING.params.{name} must be an integer >= 2")

    close = data_ctx["close"].astype("float64")

    # === 左支：价格变动的多层秩 + 极值 + 非线性变换 ===
    # 1. delta((close - 1), 5)；常数平移不影响 diff，等价于 close.diff(5)
    delta_close = close.diff(delta_window)

    # 2-3. -1 * rank -> rank -> rank（横截面 pct 秩，替代旧脚本非因果的全历史 TS 秩）
    neg_rank_delta = -1.0 * delta_close.rank(axis=1, pct=True)
    double_rank = neg_rank_delta.rank(axis=1, pct=True).rank(axis=1, pct=True)

    # 4. ts_min(..., ts_min_window)
    ts_min_double_rank = double_rank.rolling(window=ts_min_window, min_periods=ts_min_window).min()

    # 5. log(|x| + 1)（旧代码加 1 避免 log(0)；sum(..., 1) 为恒等，省略）
    log_ts_min = np.log(ts_min_double_rank.abs() + 1.0)

    # 6. scale：旧代码实现为 rolling scale_window 的时间序列 z-score
    roll_mean = log_ts_min.rolling(window=scale_window, min_periods=scale_window).mean()
    roll_std = log_ts_min.rolling(window=scale_window, min_periods=scale_window).std()
    scaled_log = (log_ts_min - roll_mean) / (roll_std + _EPSILON)

    # 7-8. rank(rank(...))；product(..., 1) 为恒等，省略
    rank_rank_scaled = scaled_log.rank(axis=1, pct=True).rank(axis=1, pct=True)

    # 9. min(product(...), min_window)
    min_product = rank_rank_scaled.rolling(window=min_window, min_periods=min_window).min()

    # === 右支：历史负收益的时间序列秩 ===
    # 10. delay(-returns, delay_window)
    returns = close.pct_change()
    neg_returns_delay = (-1.0 * returns).shift(delay_window)

    # 11. ts_rank(..., ts_rank_window)
    ts_rank_delay = _ts_rank(neg_returns_delay, ts_rank_window)

    # 12. 合成
    return min_product + ts_rank_delay
