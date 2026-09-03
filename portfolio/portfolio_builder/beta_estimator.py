"""Rolling Beta estimation: each coin vs BTC, and portfolio-level aggregate."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_coin_betas(
    kline: pd.DataFrame,
    window: int = 60,
    btc_symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """Compute rolling Beta of each coin vs BTC using daily log returns.

    beta_t = cov(ret_coin, ret_btc, window) / var(ret_btc, window)

    Args:
        kline: [symbol, date, close]
        window: rolling window size (default 60)

    Returns:
        Long-format DataFrame [date, instrument, beta].
        NaN for dates with <window obs.
    """
    df = kline.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["symbol", "date"])
    df["ret"] = df.groupby("symbol")["close"].pct_change()

    wide = df.pivot_table(index="date", columns="symbol", values="ret")
    if btc_symbol not in wide.columns:
        raise ValueError(f"No {btc_symbol} in kline")
    btc = wide[btc_symbol]
    btc_var = btc.rolling(window, min_periods=window).var(ddof=1)

    rows = []
    for col in wide.columns:
        if col == btc_symbol:
            continue
        coin = wide[col]
        cov = coin.rolling(window, min_periods=window).cov(btc)
        beta = cov / btc_var
        for d, b in beta.items():
            if pd.notna(b):
                rows.append({"date": d, "instrument": col, "beta": float(b)})
    return pd.DataFrame(rows)


def compute_portfolio_beta(
    betas_df: pd.DataFrame,
    holdings: dict[str, float],
    as_of_date: pd.Timestamp,
    beta_prior: float = 1.3,
) -> float:
    """Weighted average beta of current holdings as of date.

    Uses each coin's most recent available beta <= as_of_date.
    If a coin has no beta history, falls back to beta_prior.
    If holdings is empty, returns beta_prior.
    """
    if not holdings:
        return beta_prior

    total_w = sum(holdings.values())
    if total_w < 1e-12:
        return beta_prior

    as_of = pd.Timestamp(as_of_date).normalize()
    beta_sum = 0.0
    for inst, w in holdings.items():
        sub = betas_df[(betas_df["instrument"] == inst) &
                       (betas_df["date"] <= as_of)]
        if sub.empty:
            coin_beta = beta_prior
        else:
            coin_beta = float(sub.sort_values("date").iloc[-1]["beta"])
            if not np.isfinite(coin_beta):
                coin_beta = beta_prior
        beta_sum += (w / total_w) * coin_beta
    return beta_sum
