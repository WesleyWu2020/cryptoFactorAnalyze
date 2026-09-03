"""Rank-based universe filter: keep coins whose current Rank falls in [rank_min, rank_max]."""
import pandas as pd


def filter_universe_by_rank(market_cap_df: pd.DataFrame, rank_min: int, rank_max: int) -> pd.DataFrame:
    """Filter universe by market cap rank.
    
    Args:
        market_cap_df: DataFrame with at least a "Rank" column
        rank_min: Minimum rank (inclusive)
        rank_max: Maximum rank (inclusive)
    
    Returns:
        Filtered DataFrame containing only rows with Rank in [rank_min, rank_max]
    """
    df = market_cap_df.copy()
    df = df.dropna(subset=["Rank"])
    df["Rank"] = df["Rank"].astype(int)
    mask = (df["Rank"] >= rank_min) & (df["Rank"] <= rank_max)
    return df.loc[mask].reset_index(drop=True)
