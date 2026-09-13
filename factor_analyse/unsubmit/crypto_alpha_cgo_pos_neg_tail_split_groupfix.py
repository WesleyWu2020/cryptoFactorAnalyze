"""crypto_alpha_cgo_pos_neg_tail_split_groupfix 因子（factor_common 日频契约版）。

原始定义（分钟版）：15m bar 上的 CGO（capital gain outstanding）。每根 bar 用
volume/rolling_mean(volume, 60d) 映射出自适应换手率
turnover = base_rate + sensitivity * tanh((volume_ratio - 1) / 2)（再经 2 日 EMA、
floor/cap 截断），参考价按 ref = ref_prev*(1-t) + price_prev*t 递推，
cgo = clip((price - ref)/ref, -1, 1)。

日频改写说明（judgment calls）：
- 分钟版换手率参数按 per-bar 缩放（base_rate=0.03/96 等）；日频版本将 base_rate、
  sensitivity、floor、cap 还原为每日量纲（0.03、0.08、0.01、0.25），与
  crypto_alpha_cgo_volume_ratio_turnover_conservative 的 DAILY_* 常量一致。
- volume_ratio 用日频 volume / rolling_mean(volume, 60d)（原版为 15m bar 对 60d
  bar 均值，日频下的忠实对应物）。
- EMA span 取 turnover_ema_days=2 天（原版 2 日 * 96 bars 的 15m EMA，日频对应 span=2）。
- min_valid_days=10：窗口内有效观测不足 10 天不出信号（原版 10 日 * 96 bars）。
- 状态递推（EMA、参考价）与原版逐 bar 递推逻辑一致；日频矩阵规模小，
  用纯 Python 逐列循环实现（本环境无 numba），删除分块列循环封装，直接返回日频矩阵。
- 原版无显式 shift(1) 执行延迟（递推本身只用 <= t 数据），无需删除。

方向说明：原模块未注明方向，按默认 factor_direction=1（CGO 高=浮盈大，动量延续解读）。

计算只使用当日及历史数据，无未来函数。FactorManager 只调用下面的标准模块接口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name": "crypto_alpha_cgo_pos_neg_tail_split_groupfix",
    "author": "wesleywu",
    "level": "daily",
    "category": "behavior",
    "description": "CGO：量比率自适应换手率递推参考价，cgo=clip((price-ref)/ref,-1,1)（日频版）",
}

SETTING = {
    "data_needed": ["close", "volume"],
    "universe": "historical_top50",
    # 最长窗口 turnover_window_days=60，加缓冲。
    "warmup_bars": 70,
    "preprocessing": "mad_rank",
    "params": {
        "turnover_window_days": 60,
        "turnover_ema_days": 2,
        "min_valid_days": 10,
        "base_rate": 0.03,
        "sensitivity": 0.08,
        "turnover_floor": 0.01,
        "turnover_cap": 0.25,
    },
    "factor_direction": 1,
}

EPS = 1e-12


def _calc_cgo_np(
    close_np: np.ndarray,
    volume_np: np.ndarray,
    window_days: int,
    ema_alpha: float,
    min_valid_days: int,
    base_rate: float,
    sensitivity: float,
    turnover_floor: float,
    turnover_cap: float,
) -> np.ndarray:
    """Per-column sequential CGO recursion on daily bars (same logic as the minute version)."""

    n_rows, n_cols = close_np.shape
    out_np = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    for col in range(n_cols):
        volume_buffer = np.full(window_days, np.nan, dtype=np.float64)

        volume_sum = 0.0
        volume_count = 0
        ema_turnover = np.nan
        prev_turnover = np.nan
        prev_ref = np.nan
        prev_price = np.nan

        for row in range(n_rows):
            price = close_np[row, col]
            if not np.isfinite(price) or price <= 0.0:
                price = np.nan

            volume = volume_np[row, col]
            if not np.isfinite(volume):
                volume = np.nan
            elif volume < 0.0:
                volume = 0.0

            slot = row % window_days
            old_volume = volume_buffer[slot]
            if not np.isnan(old_volume):
                volume_sum -= old_volume
                volume_count -= 1

            volume_buffer[slot] = volume
            if not np.isnan(volume):
                volume_sum += volume
                volume_count += 1

            turnover = np.nan
            if volume_count >= min_valid_days and not np.isnan(volume):
                avg_volume = volume_sum / volume_count
                volume_ratio = volume / (avg_volume + EPS)
                raw_turnover = base_rate + sensitivity * np.tanh((volume_ratio - 1.0) / 2.0)

                if np.isnan(ema_turnover):
                    ema_turnover = raw_turnover
                else:
                    ema_turnover = (1.0 - ema_alpha) * ema_turnover + ema_alpha * raw_turnover

                turnover = ema_turnover
                if turnover < turnover_floor:
                    turnover = turnover_floor
                elif turnover > turnover_cap:
                    turnover = turnover_cap

            if row == 0:
                ref_price = price
            else:
                applied_turnover = prev_turnover
                if np.isnan(applied_turnover):
                    applied_turnover = 0.0
                elif applied_turnover < 0.0:
                    applied_turnover = 0.0
                elif applied_turnover > 1.0:
                    applied_turnover = 1.0

                base_ref = prev_ref
                if np.isnan(base_ref):
                    base_ref = prev_price

                ref_price = base_ref * (1.0 - applied_turnover) + prev_price * applied_turnover
                if np.isnan(prev_price):
                    ref_price = base_ref

            if np.isfinite(price) and np.isfinite(ref_price) and volume_count >= min_valid_days:
                cgo = (price - ref_price) / (ref_price + EPS)
                if cgo < -1.0:
                    cgo = -1.0
                elif cgo > 1.0:
                    cgo = 1.0
                out_np[row, col] = cgo

            prev_turnover = turnover
            prev_ref = ref_price
            prev_price = price

    return out_np


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the raw daily CGO matrix (date x instrument)."""

    params = SETTING["params"]
    ema_alpha = 2.0 / (params["turnover_ema_days"] + 1.0)

    close_daily = data_ctx["close"].astype("float64")
    volume_daily = data_ctx["volume"].astype("float64")

    factor_np = _calc_cgo_np(
        close_daily.to_numpy(dtype=np.float64, copy=False),
        volume_daily.to_numpy(dtype=np.float64, copy=False),
        params["turnover_window_days"],
        ema_alpha,
        params["min_valid_days"],
        params["base_rate"],
        params["sensitivity"],
        params["turnover_floor"],
        params["turnover_cap"],
    )

    factor = pd.DataFrame(factor_np, index=close_daily.index, columns=close_daily.columns)
    return factor.replace([np.inf, -np.inf], np.nan)
