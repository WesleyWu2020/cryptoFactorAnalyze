"""Funding rate P80 filter for short-side eligibility.

Given per-instrument 8h funding rates, compute for each date/instrument a rolling-window P80
(80th percentile) of recent funding rates. Mark allowed_short=True if current funding is BELOW
this threshold (cheaper to short than the historical 80th-percentile cost).

Rationale: On Binance USDT-M, funding is exchanged every 8h. Positive funding = longs pay shorts
→ shorter benefits. But extreme negative funding = shorts pay longs → expensive to short.
Filter out instruments whose recent funding is in the top 20% most-expensive-to-short tail.
"""

import pandas as pd
import numpy as np


def compute_funding_allowed_mask(
    funding_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    lookback_days: int = 30,
    percentile: float = 0.80,
) -> pd.DataFrame:
    """Compute rolling P80 funding rate and mark short eligibility.
    
    Args:
        funding_df: DataFrame with columns [symbol, fundingTime, fundingRate, fundingTimeMs].
                   symbol like "BTCUSDT"; fundingTime is ISO UTC string; 8h cadence.
        target_dates: List of pd.Timestamp or date-like at daily cadence (portfolio decision days).
        lookback_days: Rolling window size in days (default 30).
        percentile: Percentile threshold (default 0.80 for P80).
    
    Returns:
        DataFrame with columns [date, instrument, allowed_short, funding_p80, funding_recent].
        - date: Portfolio decision date.
        - instrument: Symbol stripped of 'USDT' suffix (e.g., 'BTC' from 'BTCUSDT').
        - allowed_short: bool. True if recent funding is BELOW the P80 threshold (cheaper to short).
        - funding_p80: 80th percentile of funding rates in the lookback window.
        - funding_recent: Mean funding rate on the target date, or last entry <= target date if none that day.
    
    Notes:
        - For each (date d, instrument i):
          - window = funding entries with fundingTime in (d - lookback_days, d] (strictly past + today's known events).
          - funding_p80 = np.nanpercentile(window_rates, percentile*100).
          - funding_recent = daily-mean of funding entries on date d itself (or last entry <= d if none that day).
          - allowed_short = funding_recent < funding_p80.
        - If window has fewer than 3 entries: allowed_short = False (insufficient history — defensive).
        - No lookahead: For date d, only use fundingTime <= end-of-day(d).
    """
    df = funding_df.copy()
    
    # Handle empty dataframe
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "instrument", "allowed_short", "funding_p80", "funding_recent"])
    
    # Parse fundingTime to UTC datetime, then convert to naive UTC for comparison
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], utc=True).dt.tz_convert(None)
    
    # Extract instrument from symbol (remove 'USDT' suffix)
    df["instrument"] = df["symbol"].str.replace("USDT", "", regex=False)
    
    # Convert fundingRate to numeric, coerce errors to NaN
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    
    # Drop rows with missing fundingRate
    df = df.dropna(subset=["fundingRate"])
    
    # Normalize target_dates to midnight UTC
    target_dates = pd.DatetimeIndex(target_dates).normalize()
    
    rows = []
    
    # Process each instrument
    for instr, g in df.groupby("instrument"):
        g = g.sort_values("fundingTime")
        
        # Process each target date
        for d in target_dates:
            # Define end-of-day (just before midnight of next day)
            eod = d + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
            
            # Define window start (strictly past, not inclusive)
            window_start = d - pd.Timedelta(days=lookback_days)
            
            # Select funding entries in the window: fundingTime > window_start AND fundingTime <= eod
            w = g[(g["fundingTime"] > window_start) & (g["fundingTime"] <= eod)]
            
            # If insufficient history, mark as not allowed to short
            if len(w) < 3:
                rows.append({
                    "date": d,
                    "instrument": instr,
                    "allowed_short": False,
                    "funding_p80": np.nan,
                    "funding_recent": np.nan,
                })
                continue
            
            # Compute P80 of the window
            p80 = np.nanpercentile(w["fundingRate"].values, percentile * 100)
            
            # Get the most recent funding rate for this date
            # Prefer mean of same-day rates; if none, use last rate in window
            same_day = w[w["fundingTime"].dt.normalize() == d]
            if len(same_day) > 0:
                recent = same_day["fundingRate"].mean()
            else:
                recent = w["fundingRate"].iloc[-1]
            
            # allowed_short = True if recent is BELOW the expensive tail (P80)
            rows.append({
                "date": d,
                "instrument": instr,
                "allowed_short": bool(recent < p80),
                "funding_p80": p80,
                "funding_recent": recent,
            })
    
    return pd.DataFrame(rows)
