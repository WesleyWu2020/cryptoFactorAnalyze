"""散户拥挤下的流动性耗散因子（Retail Friction / Illiquidity）。

公式（与旧版 CSV 管线 ``compute_one`` 逐点等价）：

    ret_1d_abs      = ABS(PCT_CHANGE(close, 1))
    friction_raw_1d = (ret_1d_abs / (quote_volume + eps)) * trade_count
    factor          = log1p(TS_MEAN(friction_raw_1d, window))

即“单位成交额对应的绝对收益率 × 成交笔数”的 N 日均值，再做 log1p 压缩右尾。
计算只使用当日及历史数据，无未来函数。截面 MAD 去极值与按日 rank 由
factor_common 框架的 ``mad_rank`` 预处理完成，这里返回原始因子值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ITER_NOTE: dict = {
    "op_type":    "modify_factor",
    "hypothesis": "IS 实测（本会话筛选）：本因子在 rb=3 下 net +40.7%/yr、sharpe 1.37、"
                  "|rank IC| 0.087、turnover 0.091，唯一 FAIL 是 maxdd 0.324（>0.25）；"
                  "rb=1 同样只差 maxdd（0.357）。friction 是慢变溢价水平，20d 估计窗在"
                  "波动 regime 切换时截面排名跳变、拖累回撤；拉长到 60d 平滑估计，"
                  "水平效应应保持（IC 微降可接受），排序稳定性上升 → maxdd 下降。",
    "change":     "SETTING['params']['window'] 20 → 60；warmup_bars 21 → 70 同步；"
                  "公式与方向不动。",
    "expected":   "rb=3：maxdd 0.324 → 0.25±0.05，net +40% 持平（±8%），"
                  "sharpe ≥1.2 保持，|rank IC| 0.087 → 0.08±0.015；"
                  "rb=1 主口径：maxdd 0.357 → 0.28±0.05。",
    "parent_iter": 13,
    "reasoning":  "Iter13 跨族 compose 证明 term-spread 相对本因子无增量；"
                  "真正的最强 IS 候选是本因子自身在 rb=3 的形态，唯一短板 maxdd，"
                  "单点变量=估计窗口。",
    "old_param":  20,
    "new_param":  60,
    "oos_result": "pass: IS rankIC -0.066 → OOS -0.043（同号，衰减 34% ≤50%）；"
                  "OOS net +44.8%/yr > 0（sharpe 1.12）；all_costs OOS +20.3%/yr > 0，"
                  "非成本敏感。ACCEPTED（2026-09-13）。",
    "corr_note":  "与库内及 unsubmit 缓存 max|ρ|=0.777（crypto_alpha_illiq_premium_20d，"
                  "同属非流动性溢价族，WARN 区）；独立信息论证：本因子摩擦项含 trade_count"
                  " 结构（笔数加权），与 illiq_premium_20d 的纯 amihud |ret|/额 口径不同，"
                  "且 rb=1 IS sharpe 1.10 vs 对方未过成本门；库内（factor_mining）"
                  "max|ρ|=0.674（retail_activity_divergence_factor），同族正常邻域。",
}


TYPE = "regular"

META = {
    "factor_name": "Retail_Friction_Illiquidity_Factor",
    "author": "local",
    "level": "daily",
    "category": "volume",
    "description": "N-day mean of |ret| / quote_volume * trade_count (log1p)",
}

SETTING = {
    "data_needed": ["close", "quote_volume", "trade_count"],
    "universe": "historical_top50",
    "warmup_bars": 70,
    "preprocessing": "mad_rank",
    "params": {"window": 60, "eps": 1e-5},
    "factor_direction": -1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回逐日 Retail Friction / Illiquidity 原始因子矩阵。

    ``data_ctx`` 为 date(升序) × instrument 的矩阵字典；``rolling`` 与
    ``pct_change`` 沿时间轴（axis 0）进行，每个值只依赖当日及之前的数据。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("SETTING.params.window must be a positive integer")
    eps = SETTING["params"]["eps"]

    close = data_ctx["close"].astype("float64")
    quote_volume = data_ctx["quote_volume"].astype("float64")
    trade_count = data_ctx["trade_count"].astype("float64")

    # PCT_CHANGE(close, 1)：旧版 roc 语义 close / close.shift(1) - 1
    ret_1d_abs = close.pct_change(periods=1).abs()
    friction_raw_1d = (ret_1d_abs / (quote_volume + eps)) * trade_count
    # TS_MEAN 旧版默认 min_periods = window
    factor_raw = friction_raw_1d.rolling(window=window, min_periods=window).mean()
    # log1p 变换与旧管线一致（在 winsorize/rank 之前，单调且只用当日值）
    return np.log1p(factor_raw)
