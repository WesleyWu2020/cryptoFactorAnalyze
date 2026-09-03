"""Tests for BTC core sizer."""

import pandas as pd
import numpy as np
import pytest

from portfolio.sizing.btc_core_sizer import btc_core_weight


class TestBTCCoreSizer:
    """Test suite for btc_core_weight function."""

    def test_maps_extremes(self):
        """Test that T ∈ {-1, 0, +1} map to correct weights."""
        T = pd.Series([-1.0, 0.0, 1.0], index=pd.date_range("2024-01-01", periods=3))
        w = btc_core_weight(T, w_min=0.4, w_max=0.7)
        
        assert abs(w.iloc[0] - 0.4) < 1e-9, f"T=-1 should map to w_min=0.4, got {w.iloc[0]}"
        assert abs(w.iloc[1] - 0.55) < 1e-9, f"T=0 should map to 0.55, got {w.iloc[1]}"
        assert abs(w.iloc[2] - 0.7) < 1e-9, f"T=+1 should map to w_max=0.7, got {w.iloc[2]}"

    def test_clips_out_of_range(self):
        """Test that out-of-range T values are clipped to [w_min, w_max]."""
        T = pd.Series([-2.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
        w = btc_core_weight(T, w_min=0.4, w_max=0.7)
        
        assert w.iloc[0] == 0.4, f"T=-2 should clip to w_min=0.4, got {w.iloc[0]}"
        assert w.iloc[1] == 0.7, f"T=+2 should clip to w_max=0.7, got {w.iloc[1]}"

    def test_nan_defaults_to_wmin(self):
        """Test that NaN T values default to w_min."""
        T = pd.Series([np.nan, 0.5], index=pd.date_range("2024-01-01", periods=2))
        w = btc_core_weight(T, w_min=0.4, w_max=0.7)
        
        assert w.iloc[0] == 0.4, f"NaN T should map to w_min=0.4, got {w.iloc[0]}"
        assert abs(w.iloc[1] - 0.625) < 1e-9, f"T=0.5 should map to 0.625, got {w.iloc[1]}"

    def test_preserves_index(self):
        """Test that output Series preserves input index."""
        dates = pd.date_range("2024-01-01", periods=5)
        T = pd.Series([-0.5, 0.0, 0.5, 1.0, -1.0], index=dates)
        w = btc_core_weight(T)
        
        assert (w.index == T.index).all(), "Output index should match input index"

    def test_custom_bounds(self):
        """Test custom w_min and w_max bounds."""
        T = pd.Series([0.0], index=pd.date_range("2024-01-01", periods=1))
        
        # Test with different bounds
        w = btc_core_weight(T, w_min=0.3, w_max=0.8)
        expected = 0.3 + (0.8 - 0.3) * 1 / 2  # = 0.55
        assert abs(w.iloc[0] - expected) < 1e-9, f"Expected {expected}, got {w.iloc[0]}"

    def test_all_nan_series(self):
        """Test with a Series of all NaN values."""
        T = pd.Series([np.nan, np.nan, np.nan], index=pd.date_range("2024-01-01", periods=3))
        w = btc_core_weight(T, w_min=0.4, w_max=0.7)
        
        assert (w == 0.4).all(), "All NaN input should yield all w_min outputs"
