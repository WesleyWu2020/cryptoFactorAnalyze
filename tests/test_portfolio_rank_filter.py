import pandas as pd
from portfolio.universe.rank_filter import filter_universe_by_rank


def test_filter_keeps_only_mid_ranks():
    mc = pd.DataFrame({
        "Date": ["2024-01-01"] * 5,
        "Rank": [1, 15, 50, 80, 120],
        "Symbol": ["BTC", "ETH", "SOL", "DOGE", "X"],
        "Trading_Pairs": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XUSDT"],
    })
    out = filter_universe_by_rank(mc, rank_min=30, rank_max=100)
    assert set(out["Symbol"]) == {"SOL", "DOGE"}


def test_missing_rank_dropped():
    mc = pd.DataFrame({
        "Date": ["2024-01-01"],
        "Rank": [None],
        "Symbol": ["X"],
        "Trading_Pairs": ["XUSDT"]
    })
    assert filter_universe_by_rank(mc, 30, 100).empty
