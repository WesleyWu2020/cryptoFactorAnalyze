"""Tests for short-squeeze stop-loss detector."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolio.risk.squeeze_stop import compute_squeeze_flags


def _mk(symbol, closes, vols):
    """Helper to create test DataFrame."""
    dates = pd.date_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({
        "date": dates,
        "instrument": symbol,
        "close": closes,
        "volume": vols
    })


def test_flag_triggered_on_combined_spike():
    """Flag should trigger when both return and volume spike."""
    closes = [100.0] * 25 + [130.0]  # +30% jump on day 26
    vols = [1.0] * 25 + [100.0]      # huge volume
    df = _mk("X", closes, vols)
    out = compute_squeeze_flags(df, ret_threshold=0.15, vol_multiple=3.0, vol_window=20)
    assert out.iloc[-1]["squeeze_flag"] == True, "Should flag when both return and volume spike"
    assert out.iloc[-2]["squeeze_flag"] == False, "Previous day should not be flagged"


def test_no_flag_if_only_return_spike():
    """Flag should not trigger with return spike but no volume spike."""
    closes = [100.0] * 25 + [130.0]
    vols = [1.0] * 26  # no volume spike
    df = _mk("X", closes, vols)
    out = compute_squeeze_flags(df, 0.15, 3.0, 20)
    assert out.iloc[-1]["squeeze_flag"] == False, "Should not flag without volume spike"


def test_no_flag_if_only_volume_spike():
    """Flag should not trigger with volume spike but no return spike."""
    closes = [100.0] * 26  # no price spike
    vols = [1.0] * 25 + [100.0]
    df = _mk("X", closes, vols)
    out = compute_squeeze_flags(df, 0.15, 3.0, 20)
    assert out.iloc[-1]["squeeze_flag"] == False, "Should not flag without return spike"


def test_no_lookahead_first_window_false():
    """Flags should be False for first vol_window rows (insufficient history)."""
    closes = list(np.linspace(100, 200, 10))  # short series
    vols = [1.0] * 10
    df = _mk("X", closes, vols)
    out = compute_squeeze_flags(df, 0.15, 3.0, 20)
    assert out["squeeze_flag"].sum() == 0, "Not enough history, all should be False"


def test_multi_instrument_independent():
    """Different instruments should be evaluated independently."""
    df1 = _mk("A", [100.0] * 25 + [130.0], [1.0] * 25 + [100.0])
    df2 = _mk("B", [100.0] * 26, [1.0] * 26)
    df = pd.concat([df1, df2], ignore_index=True)
    out = compute_squeeze_flags(df, 0.15, 3.0, 20)
    a_last = out[(out.instrument == "A")].iloc[-1]
    b_last = out[(out.instrument == "B")].iloc[-1]
    assert a_last["squeeze_flag"] == True, "Instrument A should be flagged"
    assert b_last["squeeze_flag"] == False, "Instrument B should not be flagged"


def test_return_threshold_boundary():
    """Test return threshold at exact boundary."""
    closes = [100.0] * 25 + [115.0]  # exactly +15% (threshold)
    vols = [1.0] * 25 + [100.0]       # volume spike
    df = _mk("X", closes, vols)
    out = compute_squeeze_flags(df, ret_threshold=0.15, vol_multiple=3.0, vol_window=20)
    # At exactly threshold, should not trigger (> not >=)
    assert out.iloc[-1]["squeeze_flag"] == False, "Should not flag at exact threshold"

    # Slightly above should trigger
    closes2 = [100.0] * 25 + [115.1]  # +15.1%
    df2 = _mk("X", closes2, vols)
    out2 = compute_squeeze_flags(df2, ret_threshold=0.15, vol_multiple=3.0, vol_window=20)
    assert out2.iloc[-1]["squeeze_flag"] == True, "Should flag above threshold"


def test_volume_threshold_boundary():
    """Test volume threshold at exact boundary."""
    closes = [100.0] * 25 + [130.0]   # +30% return
    # Average vol for first 25 is 1.0, so at exactly 3.0 should not trigger
    vols = [1.0] * 25 + [3.0]
    df = _mk("X", closes, vols)
    out = compute_squeeze_flags(df, ret_threshold=0.15, vol_multiple=3.0, vol_window=20)
    assert out.iloc[-1]["squeeze_flag"] == False, "Should not flag at exact volume multiple"

    # Slightly above should trigger
    vols2 = [1.0] * 25 + [3.01]
    df2 = _mk("X", closes, vols2)
    out2 = compute_squeeze_flags(df2, ret_threshold=0.15, vol_multiple=3.0, vol_window=20)
    assert out2.iloc[-1]["squeeze_flag"] == True, "Should flag above volume threshold"


def test_output_columns_and_ordering():
    """Test that output has correct columns and ordering."""
    df1 = _mk("B", [100.0] * 10, [1.0] * 10)
    df2 = _mk("A", [100.0] * 10, [1.0] * 10)
    df = pd.concat([df1, df2], ignore_index=True)
    out = compute_squeeze_flags(df, 0.15, 3.0, 20)
    
    # Check columns
    assert list(out.columns) == ["date", "instrument", "squeeze_flag"]
    
    # Check ordering (should be sorted by instrument then date)
    assert list(out.instrument.unique()) == ["A", "B"]


def test_nan_handling():
    """Test handling of NaN values in flag computation."""
    closes = [100.0] * 25 + [130.0]
    vols = [1.0] * 25 + [100.0]
    df = _mk("X", closes, vols)
    out = compute_squeeze_flags(df, 0.15, 3.0, 20)
    
    # Should have no NaN values in output
    assert not out["squeeze_flag"].isna().any(), "Flag should not have NaN values"


if __name__ == "__main__":
    # Run all tests
    import pytest
    pytest.main([__file__, "-v"])
