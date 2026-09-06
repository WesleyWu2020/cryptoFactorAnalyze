"""Calendar-safe access to the CryptoQuant HDF5 store for factor research."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from data.crypto_quant.panel import annotate_funding_prices
from data.crypto_quant.reader import load_market_history, load_membership_history
from data.crypto_quant.schemas import TABLE_SPECS
from data.crypto_quant.store import CryptoQuantStore


MARKET_FIELDS = tuple(
    column
    for column in TABLE_SPECS["klines_daily"].columns
    if column not in {"date", "symbol"}
)
QUALITY_FIELDS = (
    "has_complete_kline",
    "has_complete_funding",
    "has_placeholder_kline",
    "funding_coverage_status",
    "funding_invalid_price_count",
)
FUNDING_FIELDS = (
    "funding_time",
    "instrument",
    "funding_rate",
    "mark_price",
    "rate_type",
    "mark_price_valid",
)


def _normalized_day(value: date | str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def _requested_range(
    start: date | str | pd.Timestamp,
    end: date | str | pd.Timestamp,
    as_of: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_day, end_day = _normalized_day(start), _normalized_day(end)
    if start_day > end_day:
        raise ValueError("start must be on or before end")
    if as_of is not None:
        end_day = min(end_day, as_of)
    return start_day, end_day


def _symbol_list(symbols: Iterable[str] | str) -> list[str]:
    if isinstance(symbols, str):
        symbols = [symbols]
    return list(dict.fromkeys(str(symbol) for symbol in symbols))


class DataProvider:
    """Expose raw market/events and point-in-time quality on daily axes.

    Market and universe methods return a ``date`` index with symbol columns.
    Funding preserves one row per raw settlement event. Quality returns one row
    per requested date/instrument pair, retaining unknown values when an older
    panel schema does not carry the corresponding quality flag.
    """

    def __init__(self, path: str | Path, *, as_of: date | str | pd.Timestamp | None = None):
        self.path = Path(path)
        self._as_of = _normalized_day(as_of) if as_of is not None else None
        self._store = CryptoQuantStore(self.path)
        market = self._store.read("klines_daily")
        if "date" in market.columns and self._as_of is not None:
            dates = pd.to_datetime(market["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
            market = market[dates <= self._as_of]
        market_symbols = set(market["symbol"].astype(str)) if "symbol" in market else set()
        universe = self._store.read("universe_monthly")
        if "binance_symbol" in universe.columns:
            if self._as_of is not None and "decision_date" in universe.columns:
                decision_dates = pd.to_datetime(
                    universe["decision_date"], errors="coerce", utc=True
                ).dt.tz_localize(None).dt.normalize()
                universe = universe[decision_dates <= self._as_of]
            universe_symbols = set(universe["binance_symbol"].astype(str))
        else:
            universe_symbols = set()
        self._symbols = tuple(sorted(market_symbols | universe_symbols))

    def list_datas(self) -> list[str]:
        """List supported daily market fields using the HDF5 schema names."""
        return list(MARKET_FIELDS)

    def get_time_range(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        market = self._store.read("klines_daily")
        if market.empty or "date" not in market:
            return None, None
        dates = pd.to_datetime(market["date"], errors="raise", utc=True).dt.tz_localize(None).dt.normalize()
        if self._as_of is not None:
            dates = dates[dates <= self._as_of]
        if dates.empty:
            return None, None
        return dates.min(), dates.max()

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def _calendar(self, start, end) -> pd.DatetimeIndex:
        start_day, end_day = _requested_range(start, end, self._as_of)
        if end_day < start_day:
            return pd.DatetimeIndex([], name="date")
        return pd.date_range(start_day, end_day, freq="D", name="date")

    def get_single_data(self, field: str, *, start, end) -> pd.DataFrame:
        if field not in MARKET_FIELDS:
            raise ValueError(f"unknown market field: {field}")
        calendar = self._calendar(start, end)
        history = load_market_history(
            self.path,
            start,
            end,
            lookback_days=0,
            symbols=self.symbols,
            as_of=self._as_of,
        )
        if history.empty:
            result = pd.DataFrame(index=calendar, columns=self.symbols, dtype="float64")
        else:
            history["date"] = pd.to_datetime(history["date"]).dt.normalize()
            result = history.pivot(index="date", columns="instrument", values=field)
            result = result.reindex(index=calendar, columns=self.symbols)
        result.index.name = "date"
        result.columns.name = None
        return result

    def get_universe(self, *, start, end) -> pd.DataFrame:
        calendar = self._calendar(start, end)
        result = pd.DataFrame(False, index=calendar, columns=self.symbols, dtype=bool)
        memberships = load_membership_history(
            self.path,
            start,
            end,
            as_of=self._as_of,
        )
        for day, members in memberships.items():
            if day in result.index:
                result.loc[day, [symbol for symbol in self.symbols if symbol in members]] = True
        result.index.name = "date"
        result.columns.name = None
        return result

    def get_funding(self, *, start, end, symbols) -> pd.DataFrame:
        requested = _symbol_list(symbols)
        start_day, end_day = _requested_range(start, end, self._as_of)
        if end_day < start_day or not requested:
            return pd.DataFrame(columns=FUNDING_FIELDS)
        events = self._store.read("funding_events")
        if events.empty:
            return pd.DataFrame(columns=FUNDING_FIELDS)
        required = {"funding_time", "symbol", "funding_rate", "mark_price", "rate_type"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"funding_events missing columns: {sorted(missing)}")
        events = annotate_funding_prices(events)
        events["funding_time"] = pd.to_datetime(events["funding_time"], errors="raise", utc=True)
        event_dates = events["funding_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        events = events[
            event_dates.between(start_day, end_day)
            & events["symbol"].astype(str).isin(requested)
        ].copy()
        events = events.rename(columns={"symbol": "instrument"})
        return events.loc[:, FUNDING_FIELDS].sort_values(
            ["funding_time", "instrument"], kind="mergesort"
        ).reset_index(drop=True)

    def get_quality(self, *, start, end, symbols) -> pd.DataFrame:
        requested = _symbol_list(symbols)
        calendar = self._calendar(start, end)
        index = pd.MultiIndex.from_product(
            [calendar, requested], names=["date", "instrument"]
        )
        result = pd.DataFrame(index=index, columns=QUALITY_FIELDS)
        result["funding_coverage_status"] = "unknown"
        panel = self._store.read("research_panel_daily")
        if panel.empty or not {"date", "binance_symbol"}.issubset(panel.columns):
            return result
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"], errors="raise", utc=True).dt.tz_localize(None).dt.normalize()
        panel["instrument"] = panel["binance_symbol"].astype(str)
        panel = panel[panel["date"].isin(calendar) & panel["instrument"].isin(requested)]
        if panel.empty:
            return result
        key_columns = ["date", "instrument"]
        if panel.duplicated(key_columns).any():
            raise ValueError("duplicate quality key: date/instrument")
        panel = panel.set_index(key_columns)
        for field in QUALITY_FIELDS:
            if field in panel.columns:
                result.loc[panel.index, field] = panel[field]

        # A position can survive one execution boundary after its last
        # membership day while it is being closed. The research panel only
        # contains active-universe rows, so use the prior complete panel row
        # plus the current raw event cadence for that exit day. The prior
        # day's event times must all reappear today (+-1s, matching the
        # pipeline's drift tolerance): extra events (a funding-cadence
        # increase mid-day) are observed data and resolvable, while a missing
        # prior-cadence event means an unobserved settlement and stays
        # unknown. This fallback never fills an active-universe gap and never
        # treats an empty event day as complete.
        accepted = {"complete", "not_applicable"}
        universe = self.get_universe(start=start, end=end)
        event_data = self.get_funding(start=start, end=end, symbols=requested)
        if not event_data.empty:
            event_data["_time"] = (
                pd.to_datetime(event_data["funding_time"], utc=True)
                .dt.tz_convert("UTC").dt.tz_localize(None)
            )
            event_data["_date"] = event_data["_time"].dt.normalize()
            event_times: dict[tuple[pd.Timestamp, str], list[pd.Timestamp]] = {}
            for (day, instrument), group in event_data.groupby(["_date", "instrument"]):
                event_times[(day, str(instrument))] = sorted(group["_time"].tolist())
            for (day, instrument), events in event_data.groupby(["_date", "instrument"]):
                key = (day, str(instrument))
                if key not in result.index or bool(universe.loc[day, str(instrument)]):
                    continue
                previous = (day - pd.Timedelta(days=1), str(instrument))
                if previous in panel.index:
                    previous_status = panel.loc[previous, "funding_coverage_status"]
                elif previous in result.index:
                    previous_status = result.loc[previous, "funding_coverage_status"]
                else:
                    continue
                previous_times = event_times.get(previous, [])
                actual_hours = [t - t.normalize() for t in events["_time"].tolist()]
                cadence_kept = bool(previous_times) and all(
                    any(
                        abs((a - a.normalize()) - b) <= pd.Timedelta(seconds=1)
                        for b in actual_hours
                    )
                    for a in previous_times
                )
                valid_prices = events["mark_price_valid"].fillna(False).astype(bool).all()
                if previous_status in accepted and cadence_kept and valid_prices:
                    result.loc[key, "funding_coverage_status"] = "complete"
        return result


__all__ = ["DataProvider", "FUNDING_FIELDS", "MARKET_FIELDS", "QUALITY_FIELDS"]
