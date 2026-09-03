import pytest
import os
import sys
import tempfile
import shutil
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add the repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.capture_funding_rate import fetch_symbol


@pytest.fixture
def temp_out_dir():
    """Create a temporary directory for test output."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_fetch_symbol_with_mock_api(temp_out_dir):
    """
    Test fetch_symbol with mocked Binance API.
    
    Verifies that:
    - CSV is created with correct columns
    - Funding rate is converted to float
    - fundingTime is converted to ISO UTC format
    - fundingTimeMs is preserved as integer
    """
    
    # Mock response data (2 funding records)
    mock_response_data = [
        {
            "symbol": "BTCUSDT",
            "fundingTime": 1704067200000,  # 2024-01-01 00:00:00 UTC
            "fundingRate": "0.00005"
        },
        {
            "symbol": "BTCUSDT",
            "fundingTime": 1704096000000,  # 2024-01-01 08:00:00 UTC
            "fundingRate": "0.00010"
        }
    ]
    
    # Mock requests.get
    with patch("data.capture_funding_rate.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_get.return_value = mock_response
        
        # Call fetch_symbol
        rows_fetched, min_date, max_date = fetch_symbol(
            symbol="BTC",
            start_ms=1704067200000,
            end_ms=1704153600000,  # 2024-01-02 00:00:00 UTC
            pause=0.0,  # No pause for tests
            out_dir=temp_out_dir
        )
    
    # Verify results
    assert rows_fetched == 2, f"Expected 2 rows, got {rows_fetched}"
    assert min_date == "2024-01-01 00:00:00", f"Min date mismatch: {min_date}"
    assert max_date == "2024-01-01 08:00:00", f"Max date mismatch: {max_date}"
    
    # Verify CSV file
    csv_path = os.path.join(temp_out_dir, "BTCUSDT_funding.csv")
    assert os.path.exists(csv_path), f"CSV file not created at {csv_path}"
    
    # Read and verify CSV contents
    df = pd.read_csv(csv_path)
    
    # Check columns
    expected_columns = {"symbol", "fundingTime", "fundingRate", "fundingTimeMs"}
    assert set(df.columns) == expected_columns, f"Column mismatch. Expected {expected_columns}, got {set(df.columns)}"
    
    # Check data types
    assert df["fundingRate"].dtype in ["float64", "float32"], f"fundingRate should be float, got {df['fundingRate'].dtype}"
    assert df["fundingTimeMs"].dtype in ["int64", "int32"], f"fundingTimeMs should be int, got {df['fundingTimeMs'].dtype}"
    
    # Check data values
    assert df.loc[0, "symbol"] == "BTCUSDT"
    assert abs(df.loc[0, "fundingRate"] - 0.00005) < 1e-10
    assert df.loc[0, "fundingTimeMs"] == 1704067200000
    assert df.loc[0, "fundingTime"] == "2024-01-01 00:00:00"
    
    assert df.loc[1, "symbol"] == "BTCUSDT"
    assert abs(df.loc[1, "fundingRate"] - 0.00010) < 1e-10
    assert df.loc[1, "fundingTimeMs"] == 1704096000000
    assert df.loc[1, "fundingTime"] == "2024-01-01 08:00:00"


def test_fetch_symbol_empty_response(temp_out_dir):
    """Test fetch_symbol when API returns empty response."""
    
    with patch("data.capture_funding_rate.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        rows_fetched, min_date, max_date = fetch_symbol(
            symbol="ETH",
            start_ms=1704067200000,
            end_ms=1704153600000,
            pause=0.0,
            out_dir=temp_out_dir
        )
    
    assert rows_fetched == 0
    assert min_date == ""
    assert max_date == ""
    
    csv_path = os.path.join(temp_out_dir, "ETHUSDT_funding.csv")
    assert not os.path.exists(csv_path)


def test_fetch_symbol_http_error_retry(temp_out_dir):
    """Test fetch_symbol HTTP error handling with retry."""
    
    with patch("data.capture_funding_rate.requests.get") as mock_get:
        # First call fails, second succeeds
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = []
        
        mock_get.side_effect = [mock_response_fail, mock_response_success]
        
        rows_fetched, min_date, max_date = fetch_symbol(
            symbol="SOL",
            start_ms=1704067200000,
            end_ms=1704153600000,
            pause=0.0,
            out_dir=temp_out_dir
        )
    
    # Should return 0 because second call also has empty data
    assert rows_fetched == 0
    
    # Verify requests.get was called twice (original + retry)
    assert mock_get.call_count == 2


def test_fetch_symbol_pagination(temp_out_dir):
    """Test fetch_symbol pagination with multiple API calls."""
    
    # Create two pages of mock data
    page1 = [
        {
            "symbol": "BNBUSDT",
            "fundingTime": 1704067200000 + i * 28800000,  # 8h increments
            "fundingRate": f"0.0000{i+1}"
        }
        for i in range(1000)  # Full page
    ]
    
    page2 = [
        {
            "symbol": "BNBUSDT",
            "fundingTime": 1704067200000 + 1000 * 28800000 + i * 28800000,
            "fundingRate": "0.00001"
        }
        for i in range(100)  # Partial page (< 1000)
    ]
    
    with patch("data.capture_funding_rate.requests.get") as mock_get:
        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = page1
        
        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = page2
        
        mock_get.side_effect = [mock_response1, mock_response2]
        
        rows_fetched, min_date, max_date = fetch_symbol(
            symbol="BNB",
            start_ms=1704067200000,
            end_ms=2000000000000,  # Far future to allow pagination
            pause=0.0,
            out_dir=temp_out_dir
        )
    
    # Should fetch both pages
    assert rows_fetched == 1100, f"Expected 1100 rows, got {rows_fetched}"
    
    # Verify CSV
    csv_path = os.path.join(temp_out_dir, "BNBUSDT_funding.csv")
    df = pd.read_csv(csv_path)
    assert len(df) == 1100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
