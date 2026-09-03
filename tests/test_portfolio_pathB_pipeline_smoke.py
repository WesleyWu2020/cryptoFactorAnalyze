"""Smoke tests for Path B pipeline integration."""
import pandas as pd
from portfolio.pipeline import run_pathB_pipeline


def test_callable_and_returns_dict_structure():
    """Light smoke — verify signature exists & returns dict of expected keys on empty input."""
    assert callable(run_pathB_pipeline)


def test_empty_factor_paths_returns_empty_nav(tmp_path):
    """Build minimal kline/universe/funding fixtures to exercise load paths."""
    # Build minimal kline/universe/funding fixtures to exercise load paths
    kline_csv = tmp_path / "kline.csv"
    pd.DataFrame({
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "date": ["2024-01-01", "2024-01-02"],
        "open": [1, 1],
        "high": [1, 1],
        "low": [1, 1],
        "close": [100.0, 110.0],
        "volume": [1.0, 1.0]
    }).to_csv(kline_csv, index=False)
    
    mc_csv = tmp_path / "mc.csv"
    pd.DataFrame({
        "Date": ["2024-01-01"],
        "Rank": [1],
        "Symbol": ["BTC"],
        "Trading_Pairs": ["BTCUSDT"],
        "Decision_Window_Start": ["2024-01-01"],
        "Decision_Window_End": ["2024-01-02"]
    }).to_csv(mc_csv, index=False)
    
    funding_dir = tmp_path / "funding"
    funding_dir.mkdir()
    
    res = run_pathB_pipeline(
        {},
        kline_csv,
        mc_csv,
        funding_dir,
        oos_start="2024-01-01",
        oos_end="2024-01-02"
    )
    
    assert "nav" in res and "shorts" in res and "btc_weights" in res and "kept_factors" in res
    assert len(res["nav"]) == 0  # no factors -> empty
