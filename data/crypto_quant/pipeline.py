"""Orchestration for reproducible crypto-quant backfills and updates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable, Protocol, Sequence

import pandas as pd

from .binance import DAY_MS
from .config import PipelineConfig
from .mapping import build_contract_mappings, load_mapping_rules
from .panel import build_research_panel, last_complete_panel_date
from .schemas import TABLE_SPECS
from .store import CryptoQuantStore, staged_store
from .universe import build_monthly_universe
from .validation import validate_store


class CmcSource(Protocol):
    def fetch_history(self, start: date, end: date, on_page: Callable[..., None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]: ...


class BinanceSource(Protocol):
    def fetch_exchange_info(self) -> pd.DataFrame: ...
    def fetch_klines(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...
    def fetch_funding(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame: ...


@dataclass(frozen=True)
class RunSummary:
    mode: str
    as_of_utc: datetime
    published_path: Path
    last_complete_kline_date: date
    last_complete_panel_date: date | None
    row_counts: dict[str, int]
    warnings: Sequence[str]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _day(value: object) -> date | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


class CryptoQuantPipeline:
    def __init__(self, config: PipelineConfig, cmc_source: CmcSource, binance_source: BinanceSource):
        self.config = config
        self.cmc_source = cmc_source
        self.binance_source = binance_source

    def backfill(self, as_of: datetime, *, reset_staging: bool = False) -> RunSummary:
        return self._run("backfill", as_of, reset_staging=reset_staging)

    def update(self, as_of: datetime, *, reset_staging: bool = False) -> RunSummary:
        return self._run("update", as_of, reset_staging=reset_staging)

    def rebuild_derived(self, as_of: datetime, *, reset_staging: bool = False) -> RunSummary:
        return self._run("rebuild", as_of, reset_staging=reset_staging, derived_only=True)

    def _fingerprint(self, mode: str) -> str:
        payload = {"mode": mode, "schema_version": 1, "rules_version": load_mapping_rules(self.config.rules_path).version, "target": str(self.config.store_path)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _run(self, mode: str, as_of: datetime, *, reset_staging: bool, derived_only: bool = False) -> RunSummary:
        run_at = _utc(as_of)
        cmc_end = run_at.date()
        kline_end = cmc_end - timedelta(days=1)
        funding_end_ms = int(run_at.timestamp() * 1000)
        fingerprint = self._fingerprint(mode)
        with staged_store(self.config.store_path, self.config.staging_path, fingerprint, reset_staging=reset_staging, lock_path=self.config.lock_path) as store:
            prior_target = store.read_metadata().get("run_target_as_of")
            if prior_target is not None and run_at.isoformat() < str(prior_target) and not reset_staging:
                from .store import StagingMismatchError
                raise StagingMismatchError(f"staging target {prior_target!r} is newer than {run_at.isoformat()!r}")
            store.write_metadata({"run_target_as_of": run_at.isoformat(), "mode": mode})

            kline_complete = funding_complete = False
            cmc_watermark = None
            if not derived_only:
                cmc_watermark = self._fetch_cmc(store, mode, cmc_end)
                exchange = self.binance_source.fetch_exchange_info()
                store.replace("futures_contracts", self._map_contracts(store, exchange))
                if mode != "rebuild":
                    kline_complete, funding_complete = self._fetch_binance(store, mode, cmc_end, kline_end, funding_end_ms)
            else:
                exchange = store.read("futures_contracts")

            for name, spec in TABLE_SPECS.items():
                if name not in store.keys():
                    store.replace(name, self._empty_frame(name))
                    if name not in store.keys():
                        # PyTables drops zero-row tables; seed and remove one typed row
                        # so the required empty table still has a stable schema.
                        seed = self._seed_row(name)
                        store.replace(name, pd.DataFrame([seed]))
                        with pd.HDFStore(store.path, mode="a") as hdf:
                            hdf.remove(name, start=0, stop=1)
            self._build_derived(store, cmc_end, kline_end)
            metadata = self._metadata(store, run_at, cmc_end, kline_end, funding_end_ms, cmc_watermark, kline_complete, funding_complete)
            store.write_metadata(metadata)
            report = validate_store(self.config.staging_path)
            report.raise_for_errors()
            counts = {name: len(store.read(name)) for name in TABLE_SPECS}
            panel = store.read("research_panel_daily")
            complete = last_complete_panel_date(panel)
            return RunSummary(mode, run_at, self.config.store_path, kline_end, complete.date() if complete is not None else None, counts, tuple())

    @staticmethod
    def _empty_frame(name: str) -> pd.DataFrame:
        frame = pd.DataFrame(columns=TABLE_SPECS[name].columns)
        for column in frame.columns:
            if column in {"cmc_id", "trade_count", "funding_event_count", "market_cap_rank"}:
                frame[column] = pd.Series(dtype="int64")
            elif column in {"has_complete_kline", "has_complete_funding"}:
                frame[column] = pd.Series(dtype="bool")
            elif column.endswith("_time") or column in {"date", "decision_date", "effective_date", "effective_end_date", "onboard_date", "valid_from", "valid_to"}:
                frame[column] = pd.Series(dtype="datetime64[ns]")
            elif column not in {"symbol", "binance_symbol", "cmc_symbol", "base_asset", "quote_asset", "contract_type", "status", "mapping_source", "rate_type", "name"}:
                frame[column] = pd.Series(dtype="float64")
        return frame

    @staticmethod
    def _seed_row(name: str) -> dict[str, object]:
        values: dict[str, object] = {}
        for column in TABLE_SPECS[name].columns:
            if column in {"date", "decision_date", "effective_date", "effective_end_date", "onboard_date", "valid_from", "valid_to"}:
                values[column] = date(1970, 1, 1)
            elif column.endswith("_time"):
                values[column] = pd.Timestamp("1970-01-01", tz="UTC")
            elif column in {"cmc_id", "trade_count", "funding_event_count", "market_cap_rank"}:
                values[column] = 0
            elif column in {"has_complete_kline", "has_complete_funding"}:
                values[column] = False
            elif column in {"symbol", "binance_symbol", "cmc_symbol", "base_asset", "quote_asset", "contract_type", "status", "mapping_source", "rate_type", "name"}:
                values[column] = "__EMPTY__"
            else:
                values[column] = 0.0
        return values

    def _fetch_cmc(self, store: CryptoQuantStore, mode: str, end: date) -> date | None:
        existing = store.read("cmc100_daily")
        last = _day(existing["date"].max()) if not existing.empty else None
        start = self.config.universe_start if mode == "backfill" or last is None else max(self.config.universe_start, last - timedelta(days=self.config.cmc_overlap_days - 1))
        self._replace_cmc_range(store, "cmc100_daily", start, end)
        self._replace_cmc_range(store, "cmc100_constituents", start, end)

        def on_page(daily: pd.DataFrame, members: pd.DataFrame, through: date) -> None:
            daily = self._clip_date_frame(daily, "date", start, end)
            members = self._clip_date_frame(members, "date", start, end)
            if not daily.empty:
                self._replace_cmc_dates(store, "cmc100_daily", daily)
            if not members.empty:
                self._replace_cmc_dates(store, "cmc100_constituents", members)
            prefix = self._cmc_common_prefix(store, end)
            if prefix is not None:
                store.write_metadata({"checkpoint.cmc_through": prefix.isoformat()})

        daily, members = self.cmc_source.fetch_history(start, end, on_page=on_page)
        daily = self._clip_date_frame(daily, "date", start, end)
        members = self._clip_date_frame(members, "date", start, end)
        if not daily.empty:
            self._replace_cmc_dates(store, "cmc100_daily", daily)
        if not members.empty:
            self._replace_cmc_dates(store, "cmc100_constituents", members)
        return self._cmc_common_prefix(store, end)

    @staticmethod
    def _replace_cmc_dates(store: CryptoQuantStore, name: str, incoming: pd.DataFrame) -> None:
        if incoming.empty:
            return
        incoming = incoming.copy()
        for column in ("index_value", "weight"):
            if column in incoming:
                incoming[column] = pd.to_numeric(incoming[column], errors="raise").astype("float64")
        if "cmc_id" in incoming:
            incoming["cmc_id"] = pd.to_numeric(incoming["cmc_id"], errors="raise").astype("int64")
        dates = set(pd.to_datetime(incoming["date"], errors="coerce", utc=True).dt.date.dropna())
        existing = store.read(name)
        if not existing.empty:
            old_dates = pd.to_datetime(existing["date"], errors="coerce", utc=True).dt.date
            existing = existing.loc[~old_dates.isin(dates)]
        combined = incoming if existing.empty else pd.concat([existing, incoming], ignore_index=True)
        store.replace(name, combined)

    @classmethod
    def _replace_cmc_range(cls, store: CryptoQuantStore, name: str, start: date, end: date) -> None:
        existing = store.read(name)
        if existing.empty:
            return
        values = pd.to_datetime(existing["date"], errors="coerce", utc=True).dt.date
        store.replace(name, existing.loc[~values.between(start, end)])

    def _map_contracts(self, store: CryptoQuantStore, exchange: pd.DataFrame) -> pd.DataFrame:
        constituents = store.read("cmc100_constituents")
        rules = load_mapping_rules(self.config.rules_path)
        def probe(symbol: str, when: date) -> bool:
            return not self.binance_source.fetch_klines(symbol, when, when).empty
        mappings, _issues = build_contract_mappings(constituents, exchange, rules, probe)
        if mappings.empty:
            return pd.DataFrame(columns=TABLE_SPECS["futures_contracts"].columns)
        return mappings.reindex(columns=TABLE_SPECS["futures_contracts"].columns)

    @staticmethod
    def _clip_date_frame(frame: pd.DataFrame, column: str, start: date, end: date) -> pd.DataFrame:
        if frame.empty or column not in frame:
            return frame.copy()
        values = pd.to_datetime(frame[column], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        return frame.loc[values.between(pd.Timestamp(start), pd.Timestamp(end))].copy()

    @staticmethod
    def _clip_funding(frame: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
        if frame.empty or "funding_time" not in frame:
            return frame.copy()
        values = pd.to_datetime(frame["funding_time"], errors="coerce", utc=True)
        start = pd.to_datetime(start_ms, unit="ms", utc=True)
        end = pd.to_datetime(end_ms, unit="ms", utc=True)
        return frame.loc[values.between(start, end)].copy()

    @staticmethod
    def _longest_daily_prefix(frame: pd.DataFrame, start: date, end: date) -> date | None:
        if frame.empty or "date" not in frame:
            return None
        observed = set(pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.date.dropna())
        current = start
        while current <= end and current in observed:
            current += timedelta(days=1)
        return current - timedelta(days=1) if current > start else None

    def _cmc_common_prefix(self, store: CryptoQuantStore, end: date) -> date | None:
        daily = store.read("cmc100_daily")
        members = store.read("cmc100_constituents")
        if daily.empty or members.empty or "date" not in daily or "date" not in members:
            return None
        daily_dates = set(pd.to_datetime(daily["date"], errors="coerce", utc=True).dt.date.dropna())
        member_dates = pd.to_datetime(members["date"], errors="coerce", utc=True).dt.date
        required = {"date", "cmc_id", "symbol", "name", "weight"}
        if not required.issubset(members.columns):
            return None
        current = self.config.universe_start
        while current <= end:
            if current not in daily_dates:
                break
            snapshot = members.loc[member_dates == current]
            valid = len(snapshot) == 100
            valid &= not snapshot.duplicated(["date", "cmc_id"]).any()
            valid &= not snapshot[list(required)].isna().any().any()
            if not valid:
                break
            current += timedelta(days=1)
        return current - timedelta(days=1) if current > self.config.universe_start else None

    @staticmethod
    def _funding_complete_end(end_ms: int) -> date:
        end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
        return end.date() if end.time() != datetime.min.time() else end.date() - timedelta(days=1)

    def _fetch_binance(self, store: CryptoQuantStore, mode: str, cmc_end: date, kline_end: date, funding_end_ms: int) -> tuple[bool, bool]:
        mappings = store.read("futures_contracts")
        constituents = store.read("cmc100_constituents")
        old_klines = store.read("klines_daily")
        old_funding = store.read("funding_events")
        kline_batches: list[pd.DataFrame] = []
        funding_batches: list[pd.DataFrame] = []
        coverage_starts: dict[str, date] = {}
        kline_batch_symbols: set[str] = set()
        funding_batch_symbols: set[str] = set()
        all_kline_complete = bool(len(mappings))
        all_funding_complete = bool(len(mappings))
        for row in mappings.sort_values("binance_symbol").itertuples():
            symbol = row.binance_symbol
            observed = constituents.loc[constituents.cmc_id == row.cmc_id, "date"]
            first = _day(observed.min()) or self.config.universe_start
            support_start = first - timedelta(days=self.config.warmup_days)
            onboard = _day(getattr(row, "onboard_date", None))
            if onboard is not None:
                support_start = max(support_start, onboard)
            if mode == "backfill" or old_klines.empty:
                kline_start = support_start
            else:
                dates = old_klines.loc[old_klines.symbol == symbol, "date"]
                kline_start = max(support_start, (_day(dates.max()) or support_start) - timedelta(days=self.config.binance_overlap_days - 1))
            klines = self.binance_source.fetch_klines(symbol, kline_start, kline_end)
            klines = self._clip_date_frame(klines, "date", kline_start, kline_end)
            coverage_starts[symbol] = support_start
            if not klines.empty:
                kline_batches.append(klines)
                kline_batch_symbols.add(symbol)
            else:
                all_kline_complete = False

            if mode == "backfill" or old_funding.empty:
                funding_start = int(pd.Timestamp(support_start, tz="UTC").timestamp() * 1000)
            else:
                times = old_funding.loc[old_funding.symbol == symbol, "funding_time"]
                funding_start = max(int(pd.Timestamp(support_start, tz="UTC").timestamp() * 1000), int(pd.Timestamp(times.max()).timestamp() * 1000) - 7 * DAY_MS if not times.empty else 0)
            funding = self.binance_source.fetch_funding(symbol, funding_start, funding_end_ms)
            funding = self._clip_funding(funding, funding_start, funding_end_ms)
            if not funding.empty:
                funding_batches.append(funding)
                funding_batch_symbols.add(symbol)
            else:
                all_funding_complete = False
        if kline_batches:
            store.upsert("klines_daily", pd.concat(kline_batches, ignore_index=True))
        if funding_batches:
            store.upsert("funding_events", pd.concat(funding_batches, ignore_index=True))
        for symbol, coverage_start in coverage_starts.items():
            if symbol not in kline_batch_symbols and symbol not in funding_batch_symbols:
                continue
            full_klines = store.read("klines_daily")
            full_klines = full_klines[full_klines["symbol"] == symbol]
            expected = set(pd.date_range(coverage_start, kline_end, freq="D").date)
            observed = set(pd.to_datetime(full_klines["date"], errors="coerce").dt.date.dropna())
            missing_count = len(expected - observed)
            actual_start = _day(full_klines["date"].min()) if not full_klines.empty else None
            actual_end = _day(full_klines["date"].max()) if not full_klines.empty else None
            complete = actual_start == coverage_start and actual_end == kline_end and missing_count == 0
            all_kline_complete &= complete
            if symbol in kline_batch_symbols:
                store.write_metadata({f"checkpoint.klines.{symbol}": {"requested_start": coverage_start.isoformat(), "requested_end": kline_end.isoformat(), "actual_start": actual_start.isoformat() if actual_start else None, "actual_end": actual_end.isoformat() if actual_end else None, "missing_date_count": missing_count, "complete": complete}})
            full_funding = store.read("funding_events")
            full_funding = full_funding[full_funding["symbol"] == symbol]
            expected_days = set(pd.date_range(coverage_start, self._funding_complete_end(funding_end_ms), freq="D").date)
            observed_days = set(pd.to_datetime(full_funding["funding_time"], errors="coerce", utc=True).dt.date.dropna())
            missing_count = len(expected_days - observed_days)
            complete = expected_days.issubset(observed_days)
            all_funding_complete &= complete
            actual_start_ms = int(pd.Timestamp(full_funding["funding_time"].min()).timestamp() * 1000) if not full_funding.empty else None
            actual_end_ms = int(pd.Timestamp(full_funding["funding_time"].max()).timestamp() * 1000) if not full_funding.empty else None
            if symbol in funding_batch_symbols:
                store.write_metadata({f"checkpoint.funding.{symbol}": {"requested_start_ms": int(pd.Timestamp(coverage_start, tz="UTC").timestamp() * 1000), "requested_end_ms": funding_end_ms, "actual_start_ms": actual_start_ms, "actual_end_ms": actual_end_ms, "missing_date_count": missing_count, "complete": complete}})
        return all_kline_complete, all_funding_complete

    def _build_derived(self, store: CryptoQuantStore, cmc_end: date, panel_end: date) -> None:
        constituents = store.read("cmc100_constituents")
        mappings = store.read("futures_contracts")
        klines = store.read("klines_daily")
        current_symbols = None
        current_observed = None
        today = datetime.now(timezone.utc).date()
        latest_decision = pd.Timestamp(cmc_end).to_period("M").start_time.date()
        if cmc_end == today and latest_decision == cmc_end:
            current_symbols = set(mappings.loc[mappings["status"].astype(str).str.upper() == "TRADING", "binance_symbol"])
            current_observed = cmc_end
        universe = build_monthly_universe(
            constituents, mappings, klines, self.config.universe_start, cmc_end,
            top_n=self.config.top_n, current_trading_symbols=current_symbols,
            current_observed_date=current_observed,
        )
        universe = universe.rename(columns={"weight": "cmc_weight"})
        store.replace("universe_monthly", universe)
        panel_klines = klines.rename(columns={"quote_volume": "quote_asset_volume"}) if "quote_volume" in klines else klines
        panel = build_research_panel(universe, panel_klines, store.read("funding_events"), panel_end, panel_end)
        panel = panel.merge(
            universe[["effective_date", "binance_symbol", "decision_date", "market_cap_rank"]],
            left_on=["universe_effective_date", "binance_symbol"],
            right_on=["effective_date", "binance_symbol"], how="left",
        ).drop(columns=["effective_date"])
        if "quote_asset_volume" in panel:
            panel = panel.rename(columns={"quote_asset_volume": "quote_volume"})
        if "open_time" not in panel:
            panel["open_time"] = pd.NaT
        for column in ("funding_rate_sum", "funding_rate_mean", "funding_rate_last"):
            panel[column] = pd.to_numeric(panel[column], errors="coerce").astype("float64")
        panel = panel.reindex(columns=TABLE_SPECS["research_panel_daily"].columns)
        store.replace("research_panel_daily", panel)

    def _metadata(self, store: CryptoQuantStore, run_at: datetime, cmc_end: date, kline_end: date, funding_end_ms: int, cmc_watermark: date | None, kline_complete: bool, funding_complete: bool) -> dict[str, object]:
        prior = store.read_metadata()
        ranges: dict[str, object] = {}
        for name, spec in TABLE_SPECS.items():
            frame = store.read(name)
            dates = [c for c in spec.columns if c in {"date", "decision_date", "effective_date", "funding_time"}]
            if frame.empty or not dates:
                ranges[name] = None
            else:
                column = dates[0]
                ranges[name] = {"min": str(frame[column].min()), "max": str(frame[column].max())}
        return {
            "schema_version": 1, "pipeline_version": "task10", "rules_version": load_mapping_rules(self.config.rules_path).version,
            "source_urls": {"cmc": self.config.cmc_url, "exchange_info": self.config.exchange_info_url, "klines": self.config.klines_url, "funding": self.config.funding_url},
            "last_successful_cmc_date": cmc_watermark.isoformat() if cmc_watermark is not None else prior.get("last_successful_cmc_date"),
            "last_successful_kline_date": kline_end.isoformat() if kline_complete else prior.get("last_successful_kline_date"),
            "last_successful_funding_time": funding_end_ms if funding_complete else prior.get("last_successful_funding_time"),
            "last_complete_panel_date": str(last_complete_panel_date(store.read("research_panel_daily"))),
            "table_row_counts": {name: len(store.read(name)) for name in TABLE_SPECS}, "table_date_ranges": ranges,
            "created_at_utc": store.read_metadata().get("created_at_utc", run_at.isoformat()), "updated_at_utc": run_at.isoformat(),
        }


__all__ = ["BinanceSource", "CmcSource", "CryptoQuantPipeline", "RunSummary"]
