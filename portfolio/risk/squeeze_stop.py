"""Short-squeeze stop-loss flag detector.

Flags daily OHLCV rows where a short-side position should be forcibly closed
due to price/volume squeeze. Pure function with no lookahead bias.

The squeeze_flag is True on date t iff:
  - Single-day return > ret_threshold
  - AND volume_t > vol_multiple * rolling_mean(volume, vol_window, shift=1)

Both conditions use only data available at time t or earlier (no future leak).
"""
import pandas as pd


def compute_squeeze_flags(
    kline: pd.DataFrame,
    ret_threshold: float = 0.15,    # single-day return > this triggers flag
    vol_multiple: float = 3.0,      # volume > vol_multiple * 20d_avg also required
    vol_window: int = 20,
) -> pd.DataFrame:
    """Compute squeeze flags for daily OHLCV data.

    Args:
        kline: DataFrame with columns [date, instrument, close, volume].
               Must be sorted by instrument and date.
        ret_threshold: Return threshold (default 0.15 = 15%).
        vol_multiple: Volume multiplier for rolling average (default 3.0).
        vol_window: Window size for volume rolling average (default 20).

    Returns:
        DataFrame with columns [date, instrument, squeeze_flag], sorted by
        instrument then date. squeeze_flag is bool.

    No lookahead bias:
        - ret = close.pct_change() uses only (close_t / close_{t-1}) - 1
        - vol_ma = volume.shift(1).rolling(...) uses only data <= t-1
        - Both conditions use only historical data, no forward-looking.
    """
    # Ensure input has required columns
    required_cols = {"date", "instrument", "close", "volume"}
    if not required_cols.issubset(kline.columns):
        raise ValueError(f"kline must contain columns: {required_cols}")

    # Group by instrument and process independently
    results = []
    
    for instrument, group in kline.groupby("instrument", sort=False):
        # Sort by date (assume already sorted but be explicit)
        group = group.sort_values("date").reset_index(drop=True)
        
        # Compute return: pct_change() uses only t-1 and t
        ret = group["close"].pct_change()
        
        # Compute volume MA with shift(1) to ensure no lookahead
        # shift(1) moves data forward 1 row, effectively using only data <= t-1
        vol_ma = (
            group["volume"]
            .shift(1)
            .rolling(window=vol_window, min_periods=vol_window)
            .mean()
        )
        
        # Flag: both return AND volume spike required
        squeeze_flag = (ret > ret_threshold) & (group["volume"] > vol_multiple * vol_ma)
        
        # Fill NaN (first vol_window rows) with False
        squeeze_flag = squeeze_flag.fillna(False).astype(bool)
        
        # Add to group and collect
        group["squeeze_flag"] = squeeze_flag
        results.append(group[["date", "instrument", "squeeze_flag"]])
    
    # Concatenate all instruments and sort by instrument then date
    output = pd.concat(results, ignore_index=True)
    output = output.sort_values(["instrument", "date"]).reset_index(drop=True)
    
    return output
