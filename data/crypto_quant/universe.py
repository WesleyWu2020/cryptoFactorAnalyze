"""Point-in-time monthly market-cap universe construction."""
from __future__ import annotations

from datetime import date

import pandas as pd


class UniverseBuildError(ValueError):
    """Raised when a monthly universe cannot satisfy its requested size."""


UNIVERSE_COLUMNS = [
    "decision_date",
    "effective_date",
    "effective_end_date",
    "cmc_id",
    "cmc_symbol",
    "binance_symbol",
    "weight",
    "market_cap_rank",
]


def _timestamp_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce").dt.normalize()


def _month_starts(start: date, end: date) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        return []
    first = start_ts.to_period("M").start_time
    return [timestamp for timestamp in pd.date_range(first, end_ts, freq="MS") if timestamp >= start_ts]


def build_monthly_universe(
    constituents: pd.DataFrame,
    mappings: pd.DataFrame,
    klines: pd.DataFrame,
    start: date,
    end: date,
    top_n: int = 50,
    current_trading_symbols: set[str] | None = None,
    current_observed_date: date | None = None,
) -> pd.DataFrame:
    """Build monthly accepted memberships using only information known at decision time."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if constituents.empty:
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)

    member_data = constituents.copy()
    member_data["_date"] = _timestamp_column(member_data, "date")
    mapping_data = mappings.copy()
    mapping_data["_valid_from"] = _timestamp_column(mapping_data, "valid_from")
    mapping_data["_valid_to"] = _timestamp_column(mapping_data, "valid_to")
    kline_data = klines.copy()
    kline_data["_date"] = _timestamp_column(kline_data, "date")
    if "completed" in kline_data:
        kline_data = kline_data[kline_data["completed"].fillna(False).astype(bool)]
    completed_keys = kline_data[["_date", "symbol"]].drop_duplicates()

    accepted_months: list[pd.DataFrame] = []
    for decision_date in _month_starts(start, end):
        snapshot = member_data[member_data["_date"] == decision_date].copy()
        if snapshot.empty:
            eligible = snapshot
        else:
            eligible = snapshot.merge(mapping_data, on="cmc_id", how="inner", suffixes=("", "_mapping"))
            eligible = eligible[
                eligible["_valid_from"].isna() | (eligible["_valid_from"] <= decision_date)
            ]
            eligible = eligible[
                eligible["_valid_to"].isna() | (eligible["_valid_to"] >= decision_date)
            ]
            prior_date = decision_date - pd.Timedelta(days=1)
            eligible = eligible.merge(
                completed_keys[completed_keys["_date"] == prior_date][["symbol"]].rename(columns={"symbol": "binance_symbol"}),
                on="binance_symbol",
                how="inner",
            )
            observed_date = (
                pd.Timestamp(current_observed_date).normalize()
                if current_observed_date is not None
                else None
            )
            if observed_date is not None and decision_date == observed_date:
                if current_trading_symbols is None:
                    raise ValueError("current_trading_symbols is required for current_observed_date")
                eligible = eligible[eligible["binance_symbol"].isin(current_trading_symbols)]

        if len(eligible) < top_n:
            raise UniverseBuildError(
                f"decision date {decision_date.date()} eligible contracts: {len(eligible)}"
            )
        eligible = eligible.sort_values(["weight", "cmc_id"], ascending=[False, True], kind="stable").head(top_n).copy()
        eligible["decision_date"] = decision_date
        eligible["effective_date"] = decision_date + pd.Timedelta(days=1)
        eligible["market_cap_rank"] = range(1, top_n + 1)
        accepted_months.append(eligible)

    if not accepted_months:
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)
    output = pd.concat(accepted_months, ignore_index=True)
    output = output.sort_values(["decision_date", "market_cap_rank", "cmc_id"], kind="stable").reset_index(drop=True)
    output["effective_end_date"] = pd.NaT
    decisions = output["decision_date"].drop_duplicates().sort_values().tolist()
    for current, following in zip(decisions, decisions[1:]):
        mask = output["decision_date"] == current
        output.loc[mask, "effective_end_date"] = following - pd.Timedelta(days=1)
    return output[UNIVERSE_COLUMNS]
