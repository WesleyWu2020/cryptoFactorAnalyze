"""Build investable universe per date from Binance/CoinGecko market-cap CSV."""
from __future__ import annotations

import pathlib
from typing import Union

import pandas as pd

from portfolio.portfolio_builder.constraints import DEFAULT_BLACKLIST


def build_universe(
    csv_path: Union[str, pathlib.Path],
    blacklist: frozenset[str] | None = None,
) -> dict[pd.Timestamp, set[str]]:
    """Load market-cap CSV and return {date -> set[instrument]} mapping.

    Each date maps to the set of tradeable instruments for that window,
    with blacklisted tokens removed.

    Args:
        csv_path: Path to binance_coingecko_top100_marketcap_historical.csv
        blacklist: Tokens to exclude. Defaults to DEFAULT_BLACKLIST.

    Returns:
        dict mapping each calendar date (Timestamp) in a window to its instruments.
    """
    if blacklist is None:
        blacklist = DEFAULT_BLACKLIST

    df = pd.read_csv(csv_path)
    df["Decision_Window_Start"] = pd.to_datetime(df["Decision_Window_Start"])
    df["Decision_Window_End"] = pd.to_datetime(df["Decision_Window_End"])

    universe: dict[pd.Timestamp, set[str]] = {}

    for _, row in df.iterrows():
        raw_pairs = str(row["Trading_Pairs"])
        instruments = {
            p.strip() for p in raw_pairs.split(",")
            if p.strip() and p.strip() not in blacklist
        }
        start = row["Decision_Window_Start"]
        end = row["Decision_Window_End"]
        for date in pd.date_range(start, end, freq="D"):
            if date not in universe:
                universe[date] = set()
            universe[date] |= instruments

    return universe
