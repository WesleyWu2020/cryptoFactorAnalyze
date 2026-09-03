"""Compute unified forward returns from kline at a given horizon."""
from __future__ import annotations

import pandas as pd


def build_future_ret(kline: pd.DataFrame, period_days: int) -> pd.DataFrame:
    if period_days <= 0:
        raise ValueError("period_days must be >= 1")
    df = kline[["date", "symbol", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["symbol", "date"])
    df["future_close"] = df.groupby("symbol")["close"].shift(-period_days)
    df["future_ret"] = df["future_close"] / df["close"] - 1.0
    df = df.rename(columns={"symbol": "instrument"})
    return df[["date", "instrument", "future_ret"]].reset_index(drop=True)
