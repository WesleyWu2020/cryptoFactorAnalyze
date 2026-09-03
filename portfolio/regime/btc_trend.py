"""BTC trend score T in [-1, +1] — pure backward-looking calculation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_btc_trend_score(
    kline: pd.DataFrame,
    ema_short: int = 50,
    ema_long: int = 200,
    zscore_window: int = 120,
    ema_ratio_saturation: float = 0.15,
    zscore_saturation: float = 2.0,
    smoothing_span: int = 5,
    btc_symbol: str = "BTCUSDT",
) -> pd.Series:
    """Compute daily BTC trend score T in [-1, +1].

    T = EMA(0.5*tanh((EMA_short/EMA_long - 1) / sat1) +
           0.5*tanh((close - MA_long) / std_long / sat2),
           span=smoothing_span)

    All operations are strictly backward-looking (no future leak).

    Args:
        kline: DataFrame with [symbol, date, close]
        btc_symbol: Filter to this symbol; raises if not found
    Returns:
        pd.Series indexed by date with T values.
    """
    df = kline[kline["symbol"].str.upper() == btc_symbol.upper()].copy()
    if df.empty:
        raise ValueError(f"No rows for symbol {btc_symbol}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    close = df["close"].astype(float)

    ema_s = close.ewm(span=ema_short, adjust=False, min_periods=ema_short).mean()
    ema_l = close.ewm(span=ema_long, adjust=False, min_periods=ema_long).mean()
    ma_l = close.rolling(ema_long, min_periods=ema_long).mean()
    std_l = close.rolling(zscore_window, min_periods=zscore_window).std(ddof=1)

    raw_ratio = (ema_s / ema_l) - 1.0
    raw_z = (close - ma_l) / std_l

    T1 = np.tanh(raw_ratio / ema_ratio_saturation)
    T2 = np.tanh(raw_z / zscore_saturation)
    T_raw = 0.5 * T1 + 0.5 * T2

    T = T_raw.ewm(span=smoothing_span, adjust=False, min_periods=smoothing_span).mean()
    T = T.clip(-1.0, 1.0)
    T.name = "btc_trend_T"
    return T
