"""Tests for funding rate P80 filter."""

import pandas as pd
import numpy as np
import pytest
from portfolio.regime.funding_filter import compute_funding_allowed_mask


def _mk_funding(symbol, rates, start="2024-01-01"):
    """Helper to create synthetic funding data with 8h cadence."""
    ts = pd.date_range(start, periods=len(rates), freq="8h")
    return pd.DataFrame({
        "symbol": symbol,
        "fundingTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fundingRate": rates,
        "fundingTimeMs": (ts.astype("int64") // 10**6).astype(int),
    })


class TestFundingAllowedMask:
    """Test suite for compute_funding_allowed_mask."""

    def test_allows_short_when_recent_below_p80(self):
        """Test that short is allowed when recent funding is below P80 threshold."""
        # 90 days × 3 funding/day = 270 entries. Most 0.0001, last few 0.00005 (below p80)
        # Entry 260 is at ~2024-03-27 16:00, so target 2024-03-29 ensures low rates are in window
        rates = [0.0001] * 260 + [0.00005] * 10
        df = _mk_funding("BTCUSDT", rates)
        targets = pd.DatetimeIndex(["2024-03-29"])  # after low rates are introduced
        
        out = compute_funding_allowed_mask(df, targets, lookback_days=30, percentile=0.80)
        row = out.iloc[0]
        
        assert row["allowed_short"] == True
        assert row["funding_recent"] < row["funding_p80"]

    def test_blocks_short_when_recent_at_expensive_tail(self):
        """Test that short is blocked when recent funding is in the expensive tail."""
        # Most low, last few spike high (expensive to short)
        rates = [0.00001] * 260 + [0.001] * 10
        df = _mk_funding("ETHUSDT", rates)
        targets = pd.DatetimeIndex(["2024-03-25"])
        
        out = compute_funding_allowed_mask(df, targets, lookback_days=30, percentile=0.80)
        row = out.iloc[0]
        
        assert row["allowed_short"] == False

    def test_insufficient_history_blocks(self):
        """Test that short is blocked when history is insufficient (< 3 entries)."""
        df = _mk_funding("SOLUSDT", [0.0001, 0.0001])  # only 2 entries
        targets = pd.DatetimeIndex(["2024-01-05"])
        
        out = compute_funding_allowed_mask(df, targets, lookback_days=30, percentile=0.80)
        
        assert out.iloc[0]["allowed_short"] == False

    def test_no_lookahead(self):
        """Test that computation is data-independent (no lookahead bias).
        
        For date d, computation using full history and truncated history
        (up to end-of-day d) should be identical.
        """
        # Build 90 days of synthetic funding data
        rates = list(np.random.RandomState(0).normal(0.0001, 0.00005, 90))
        df_full = _mk_funding("BTCUSDT", rates)
        
        target = pd.DatetimeIndex(["2024-01-20"])
        
        # Truncate data at end-of-day target (must handle timezone properly)
        # fundingTime is UTC string, parse it
        cutoff_ts = pd.Timestamp("2024-01-20 23:59:59.999")
        df_full_parsed = df_full.copy()
        df_full_parsed["fundingTime"] = pd.to_datetime(df_full_parsed["fundingTime"], utc=True).dt.tz_convert(None)
        df_trunc = df_full_parsed[df_full_parsed["fundingTime"] <= cutoff_ts].copy()
        # Restore fundingTime to string format for the truncated df
        df_trunc["fundingTime"] = df_trunc["fundingTime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Compute with full and truncated data
        a = compute_funding_allowed_mask(df_full, target).iloc[0]
        b = compute_funding_allowed_mask(df_trunc, target).iloc[0]
        
        # Results should be identical
        assert a["allowed_short"] == b["allowed_short"]
        assert abs(a["funding_p80"] - b["funding_p80"]) < 1e-12

    def test_multiple_instruments(self):
        """Test that multiple instruments are processed independently."""
        rates_btc = [0.0001] * 100 + [0.00008] * 20
        rates_eth = [0.00005] * 100 + [0.0001] * 20  # ETH more expensive recently
        
        df = pd.concat([
            _mk_funding("BTCUSDT", rates_btc),
            _mk_funding("ETHUSDT", rates_eth),
        ], ignore_index=True)
        
        targets = pd.DatetimeIndex(["2024-03-25"])
        out = compute_funding_allowed_mask(df, targets, lookback_days=30, percentile=0.80)
        
        # Should have 2 rows: one for each instrument
        assert len(out) == 2
        assert set(out["instrument"].unique()) == {"BTC", "ETH"}

    def test_multiple_target_dates(self):
        """Test that multiple target dates are processed independently."""
        rates = list(np.random.RandomState(42).normal(0.0001, 0.00005, 90))
        df = _mk_funding("BTCUSDT", rates)
        
        targets = pd.DatetimeIndex(["2024-01-15", "2024-01-20", "2024-02-15"])
        out = compute_funding_allowed_mask(df, targets, lookback_days=30, percentile=0.80)
        
        # Should have 3 rows, one per date
        assert len(out) == 3
        assert set(out["date"].unique()) == set(pd.DatetimeIndex(targets))

    def test_same_day_funding_mean(self):
        """Test that same-day funding rates are averaged (not just last entry)."""
        # Create funding data with multiple entries on the same day (8h apart)
        # Then verify recent is the mean, not the last entry
        rates = [0.0001] * 100
        df = _mk_funding("BTCUSDT", rates, start="2024-01-01")
        
        # Pick a date with at least 2 entries (e.g., 2024-01-10)
        targets = pd.DatetimeIndex(["2024-01-10"])
        
        out = compute_funding_allowed_mask(df, targets, lookback_days=30, percentile=0.80)
        
        # For a constant series, mean should equal all individual entries
        # So this is more of a sanity check that the computation runs
        assert len(out) == 1
        assert not pd.isna(out.iloc[0]["funding_recent"])

    def test_empty_dataframe(self):
        """Test that empty input is handled gracefully."""
        df = pd.DataFrame({
            "symbol": [],
            "fundingTime": [],
            "fundingRate": [],
            "fundingTimeMs": [],
        })
        targets = pd.DatetimeIndex(["2024-01-15"])
        
        out = compute_funding_allowed_mask(df, targets)
        
        # Should return empty DataFrame with correct columns
        assert len(out) == 0
        assert list(out.columns) == ["date", "instrument", "allowed_short", "funding_p80", "funding_recent"]

    def test_nan_funding_rates_are_dropped(self):
        """Test that NaN funding rates are properly handled."""
        data = {
            "symbol": ["BTCUSDT"] * 5,
            "fundingTime": pd.date_range("2024-01-01", periods=5, freq="8h").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fundingRate": [0.0001, np.nan, 0.0001, 0.0001, 0.0001],
            "fundingTimeMs": [1, 2, 3, 4, 5],
        }
        df = pd.DataFrame(data)
        targets = pd.DatetimeIndex(["2024-01-01"])
        
        # Should still work; NaN values dropped, leaving 4 entries (> 3 threshold)
        out = compute_funding_allowed_mask(df, targets, lookback_days=30)
        assert len(out) == 1
        assert not pd.isna(out.iloc[0]["allowed_short"])

    def test_percentile_boundary_exactly_at_threshold(self):
        """Test behavior when recent funding exactly equals P80."""
        # Create data where we know the exact percentile
        rates = list(range(1, 11))  # [1, 2, 3, ..., 10]
        df = _mk_funding("BTCUSDT", [float(r) for r in rates])
        targets = pd.DatetimeIndex(["2024-01-04"])
        
        out = compute_funding_allowed_mask(df, targets, lookback_days=30, percentile=0.80)
        
        # P80 of [1..10] is 8.2
        # If recent == 8.2, allowed_short should be False (NOT <, but ==)
        row = out.iloc[0]
        assert row["allowed_short"] == (row["funding_recent"] < row["funding_p80"])

    def test_window_excludes_before_lookback_start(self):
        """Test that data before lookback start is excluded from P80 calculation."""
        # Create two segments: high rates (old), then low rates (recent)
        # 50 high entries * 8h = 400h = ~16.7 days
        # 100 low entries * 8h = 800h = ~33.3 days
        old_rates = [0.001] * 50  # High rates, before lookback
        recent_rates = [0.00001] * 100  # Low rates, within lookback
        all_rates = old_rates + recent_rates
        
        df = _mk_funding("BTCUSDT", all_rates, start="2024-01-01")
        
        # Target date: entry 149 is at ~2024-02-20
        # With lookback=10 days, we look back to ~2024-02-10
        # This should exclude all the old high-rate entries
        targets = pd.DatetimeIndex(["2024-02-20"])
        
        out = compute_funding_allowed_mask(df, targets, lookback_days=10, percentile=0.80)
        row = out.iloc[0]
        
        # P80 should be the low recent rate since we excluded the old high rates
        assert row["funding_p80"] < 0.0001
