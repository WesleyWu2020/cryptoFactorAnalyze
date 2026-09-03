"""Orchestration for reproducible crypto-quant backfills and updates."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
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
    symbol_status: Mapping[str, object] = field(default_factory=dict)


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
        payload = {"mode": mode, "schema_version": 1, "rules_version": load_mapping_rules(self.config.rules_path).version, "target": str(Path(self.config.store_path).expanduser().resolve())}
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
                mappings, mapping_issues = self._map_contracts(store, exchange)
                store.replace("futures_contracts", mappings)
                store.write_metadata({"mapping_issues": self._serialize_mapping_issues(mapping_issues)})
                if mode != "rebuild":
                    kline_complete, funding_complete = self._fetch_binance(store, mode, cmc_end, kline_end, funding_end_ms)
            else:
                exchange = store.read("futures_contracts")
                funding_complete = self._funding_checkpoints_complete(store, funding_end_ms)

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
            self._build_derived(store, cmc_end, kline_end, funding_complete)
            metadata = self._metadata(store, run_at, cmc_end, kline_end, funding_end_ms, cmc_watermark, kline_complete, funding_complete)
            store.write_metadata(metadata)
            report = validate_store(self.config.staging_path)
            report.raise_for_errors()
            counts = {name: len(store.read(name)) for name in TABLE_SPECS}
            panel = store.read("research_panel_daily")
            complete = last_complete_panel_date(panel)
            warnings_list = [f"{issue.code}: {issue.detail}" for issue in report.issues if issue.level == "warning"]
            for issue in store.read_metadata().get("mapping_issues", []):
                warnings_list.append(f"mapping_{issue.get('issue', 'unknown')}: cmc_id={issue.get('cmc_id')} symbol={issue.get('cmc_symbol')}")
            warnings = tuple(warnings_list)
            symbol_status = self._symbol_status(store)
            store.write_metadata({"validation_warnings": list(warnings), "symbol_status": symbol_status})
            last_kline = _day(metadata.get("last_successful_kline_date"))
            return RunSummary(mode, run_at, self.config.store_path, last_kline, complete.date() if complete is not None else None, counts, warnings, symbol_status)

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
        last = self._cmc_common_prefix(store, end) if not existing.empty else None
        start = self.config.universe_start if mode == "backfill" or last is None else max(self.config.universe_start, last - timedelta(days=self.config.cmc_overlap_days - 1))

        def on_page(daily: pd.DataFrame, members: pd.DataFrame, through: date) -> None:
            daily = self._clip_date_frame(daily, "date", start, end)
            members = self._clip_date_frame(members, "date", start, end)
            valid_dates = self._valid_cmc_dates(daily, members)
            daily = daily[pd.to_datetime(daily["date"], errors="coerce").dt.date.isin(valid_dates)]
            members = members[pd.to_datetime(members["date"], errors="coerce").dt.date.isin(valid_dates)]
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
        valid_dates = self._valid_cmc_dates(daily, members)
        daily = daily[pd.to_datetime(daily["date"], errors="coerce").dt.date.isin(valid_dates)]
        members = members[pd.to_datetime(members["date"], errors="coerce").dt.date.isin(valid_dates)]
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

    @staticmethod
    def _valid_cmc_dates(daily: pd.DataFrame, members: pd.DataFrame) -> set[date]:
        required = {"date", "cmc_id", "symbol", "name", "weight"}
        if daily.empty or members.empty or not required.issubset(members.columns):
            return set()
        daily_dates = pd.to_datetime(daily["date"], errors="coerce", utc=True).dt.date
        member_dates = pd.to_datetime(members["date"], errors="coerce", utc=True).dt.date
        valid: set[date] = set()
        for current in set(daily_dates.dropna()):
            daily_snapshot = daily.loc[daily_dates == current]
            member_snapshot = members.loc[member_dates == current]
            if len(daily_snapshot) != 1 or not CryptoQuantPipeline._is_complete_cmc_snapshot(member_snapshot):
                continue
            valid.add(current)
        return valid

    @staticmethod
    def _is_complete_cmc_snapshot(snapshot: pd.DataFrame) -> bool:
        required = {"date", "cmc_id", "symbol", "name", "weight"}
        if len(snapshot) != 100 or not required.issubset(snapshot.columns):
            return False
        normalized_dates = pd.to_datetime(snapshot["date"], errors="coerce", utc=True).dt.date
        if normalized_dates.isna().any() or normalized_dates.nunique() != 1:
            return False
        if snapshot[list(required)].isna().any().any():
            return False
        if any(
            isinstance(value, (bool, np.bool_))
            or not (
                isinstance(value, Integral)
                or (
                    isinstance(value, Real)
                    and math.isfinite(float(value))
                    and float(value).is_integer()
                )
            )
            for value in snapshot["cmc_id"]
        ):
            return False
        cmc_ids = pd.to_numeric(snapshot["cmc_id"], errors="coerce")
        return not cmc_ids.isna().any() and (cmc_ids > 0).all() and cmc_ids.nunique() == 100

    def _map_contracts(self, store: CryptoQuantStore, exchange: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        constituents = store.read("cmc100_constituents")
        rules = load_mapping_rules(self.config.rules_path)
        def probe(symbol: str, when: date) -> bool:
            return not self.binance_source.fetch_klines(symbol, when, when).empty
        mappings, issues = build_contract_mappings(constituents, exchange, rules, probe)
        if mappings.empty:
            mappings = pd.DataFrame(columns=TABLE_SPECS["futures_contracts"].columns)
        else:
            mappings = mappings.reindex(columns=TABLE_SPECS["futures_contracts"].columns)
        return mappings, issues

    @staticmethod
    def _serialize_mapping_issues(issues: pd.DataFrame) -> list[dict[str, object]]:
        if issues.empty:
            return []
        records = issues.to_dict("records")
        for record in records:
            for key, value in record.items():
                if isinstance(value, (date, datetime, pd.Timestamp)):
                    record[key] = pd.Timestamp(value).date().isoformat()
                elif pd.isna(value):
                    record[key] = None
        return records

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
        current = self.config.universe_start
        while current <= end:
            if current not in daily_dates:
                break
            snapshot = members.loc[member_dates == current]
            if not self._is_complete_cmc_snapshot(snapshot):
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
        working_klines = old_klines.copy()
        working_funding = old_funding.copy()
        kline_dirty = funding_dirty = False

        for name, frame in (("klines_daily", working_klines), ("funding_events", working_funding)):
            clean = frame.drop_duplicates(TABLE_SPECS[name].key, keep="last").reset_index(drop=True)
            if len(clean) != len(frame):
                store.replace(name, clean)
                if name == "klines_daily":
                    working_klines = clean
                else:
                    working_funding = clean

        def append_new(name: str, current: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, bool, bool]:
            keys = TABLE_SPECS[name].key
            incoming = incoming.drop_duplicates(keys, keep="last").reset_index(drop=True)
            if current.empty:
                new_rows = incoming
                revised = False
            else:
                existing_keys = {tuple(row) for row in current[list(keys)].astype(str).itertuples(index=False, name=None)}
                new_rows = incoming.loc[
                    [tuple(row) not in existing_keys for row in incoming[list(keys)].astype(str).itertuples(index=False, name=None)]
                ].reset_index(drop=True)
                revised = False
                business = [column for column in TABLE_SPECS[name].columns if column not in keys]
                for row in incoming.itertuples(index=False):
                    key = tuple(str(getattr(row, column)) for column in keys)
                    if key not in existing_keys:
                        continue
                    key_mask = pd.Series(True, index=current.index)
                    for column, value in zip(keys, key):
                        key_mask &= current[column].astype(str).eq(value)
                    old = current.loc[key_mask].iloc[0]
                    for column in business:
                        old_value, new_value = old[column], getattr(row, column)
                        if pd.isna(old_value) and pd.isna(new_value):
                            continue
                        if old_value != new_value:
                            revised = True
                            break
                    if revised:
                        break
            return merge_working(current, incoming, name), not new_rows.empty, revised

        def merge_working(current: pd.DataFrame, incoming: pd.DataFrame, name: str) -> pd.DataFrame:
            combined = pd.concat([current, incoming], ignore_index=True)
            combined = combined.drop_duplicates(TABLE_SPECS[name].key, keep="last").reset_index(drop=True)
            numeric_columns = {
                "klines_daily": ("open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume"),
                "funding_events": ("funding_rate", "mark_price"),
            }.get(name, ())
            for column in numeric_columns:
                if column in combined:
                    combined[column] = pd.to_numeric(combined[column], errors="raise")
            return combined

        kline_batch_symbols: set[str] = set()
        funding_batch_symbols: set[str] = set()
        all_kline_complete = bool(len(mappings))
        all_funding_complete = bool(len(mappings))
        prior_metadata = store.read_metadata()

        def record_coverage(symbol: str, coverage_start: date, kline_received: bool, funding_received: bool, empty_funding_response: bool) -> tuple[bool, bool]:
            full_klines = working_klines[working_klines["symbol"] == symbol]
            expected_klines = set(pd.date_range(coverage_start, kline_end, freq="D").date)
            observed_klines = set(pd.to_datetime(full_klines["date"], errors="coerce").dt.date.dropna())
            kline_missing = len(expected_klines - observed_klines)
            kline_missing_dates = sorted(expected_klines - observed_klines)
            actual_start = _day(full_klines["date"].min()) if not full_klines.empty else None
            actual_end = _day(full_klines["date"].max()) if not full_klines.empty else None
            kline_complete = actual_start == coverage_start and actual_end == kline_end and kline_missing == 0
            if kline_received:
                store.write_metadata({f"checkpoint.klines.{symbol}": {"requested_start": coverage_start.isoformat(), "requested_end": kline_end.isoformat(), "actual_start": actual_start.isoformat() if actual_start else None, "actual_end": actual_end.isoformat() if actual_end else None, "missing_date_count": kline_missing, "missing_dates": [value.isoformat() for value in kline_missing_dates], "complete": kline_complete}})

            full_funding = working_funding[working_funding["symbol"] == symbol]
            prior_funding_checkpoint = prior_metadata.get(f"checkpoint.funding.{symbol}")
            can_confirm_empty_funding = empty_funding_response and prior_funding_checkpoint is None and full_funding.empty
            if can_confirm_empty_funding:
                funding_missing = 0
                funding_complete = True
            else:
                expected_funding = set(pd.date_range(coverage_start, self._funding_complete_end(funding_end_ms), freq="D").date)
                observed_funding = set(pd.to_datetime(full_funding["funding_time"], errors="coerce", utc=True).dt.date.dropna())
                funding_missing_dates = sorted(expected_funding - observed_funding)
                funding_missing = len(funding_missing_dates)
                funding_complete = expected_funding.issubset(observed_funding)
            if empty_funding_response and funding_complete:
                funding_missing_dates = []
            elif empty_funding_response:
                funding_missing_dates = sorted(expected_funding - observed_funding)
            actual_start_ms = int(pd.Timestamp(full_funding["funding_time"].min()).timestamp() * 1000) if not full_funding.empty else None
            actual_end_ms = int(pd.Timestamp(full_funding["funding_time"].max()).timestamp() * 1000) if not full_funding.empty else None
            if funding_received:
                store.write_metadata({f"checkpoint.funding.{symbol}": {"requested_start_ms": int(pd.Timestamp(coverage_start, tz="UTC").timestamp() * 1000), "requested_end_ms": funding_end_ms, "actual_start_ms": actual_start_ms, "actual_end_ms": actual_end_ms, "missing_date_count": funding_missing, "missing_dates": [value.isoformat() for value in funding_missing_dates], "empty_result": empty_funding_response, "complete": funding_complete}})
            return kline_complete, funding_complete

        for row in mappings.sort_values("binance_symbol").itertuples():
            symbol = row.binance_symbol
            kline_checkpoint = prior_metadata.get(f"checkpoint.klines.{symbol}", {})
            funding_checkpoint = prior_metadata.get(f"checkpoint.funding.{symbol}", {})
            observed = constituents.loc[constituents.cmc_id == row.cmc_id, "date"]
            first = _day(observed.min()) or self.config.universe_start
            support_start = first - timedelta(days=self.config.warmup_days)
            onboard = _day(getattr(row, "onboard_date", None))
            if onboard is not None:
                support_start = max(support_start, onboard)
            kline_checkpoint_end = _day(kline_checkpoint.get("requested_end"))
            kline_skip = kline_checkpoint.get("complete") is True and kline_checkpoint_end is not None and kline_checkpoint_end >= kline_end
            if kline_skip:
                kline_start = None
            elif mode == "backfill" or old_klines.empty:
                kline_start = support_start
            else:
                dates = old_klines.loc[old_klines.symbol == symbol, "date"]
                missing_start = _day(min(kline_checkpoint.get("missing_dates", []), default=None))
                base = missing_start or _day(dates.max()) or support_start
                kline_start = max(support_start, base - timedelta(days=self.config.binance_overlap_days - 1))
            if kline_skip:
                all_kline_complete &= True
            else:
                klines = self.binance_source.fetch_klines(symbol, kline_start, kline_end)
                klines = self._clip_date_frame(klines, "date", kline_start, kline_end)
                if not klines.empty:
                    working_klines, has_new_rows, has_revisions = append_new("klines_daily", working_klines, klines)
                    if has_new_rows:
                        store.append("klines_daily", klines)
                        kline_dirty = True
                    if has_revisions:
                        store.replace("klines_daily", working_klines)
                        kline_dirty = True
                    kline_batch_symbols.add(symbol)
                    kline_status, _ = record_coverage(symbol, support_start, True, False, False)
                    all_kline_complete &= kline_status
                else:
                    all_kline_complete = False

            funding_checkpoint_end = funding_checkpoint.get("requested_end_ms")
            funding_skip = funding_checkpoint.get("complete") is True and funding_checkpoint_end is not None and int(funding_checkpoint_end) >= funding_end_ms
            if funding_skip:
                funding_start = None
            elif mode == "backfill" or old_funding.empty:
                funding_start = int(pd.Timestamp(support_start, tz="UTC").timestamp() * 1000)
            else:
                times = old_funding.loc[old_funding.symbol == symbol, "funding_time"]
                missing_start = _day(min(funding_checkpoint.get("missing_dates", []), default=None))
                base = int(pd.Timestamp(missing_start, tz="UTC").timestamp() * 1000) if missing_start else int(pd.Timestamp(times.max()).timestamp() * 1000) - 7 * DAY_MS if not times.empty else 0
                funding_start = max(int(pd.Timestamp(support_start, tz="UTC").timestamp() * 1000), base)
            empty_funding_response = False
            if funding_skip:
                all_funding_complete &= True
            else:
                funding = self.binance_source.fetch_funding(symbol, funding_start, funding_end_ms)
                funding = self._clip_funding(funding, funding_start, funding_end_ms)
                if not funding.empty:
                    working_funding, has_new_rows, has_revisions = append_new("funding_events", working_funding, funding)
                    if has_new_rows:
                        store.append("funding_events", funding)
                        funding_dirty = True
                    if has_revisions:
                        store.replace("funding_events", working_funding)
                        funding_dirty = True
                    funding_batch_symbols.add(symbol)
                elif {"funding_time", "symbol", "funding_rate", "mark_price", "rate_type"}.issubset(funding.columns):
                    funding_batch_symbols.add(symbol)
                    empty_funding_response = True
                else:
                    all_funding_complete = False
                if symbol in funding_batch_symbols:
                    _, funding_status = record_coverage(symbol, support_start, False, True, empty_funding_response)
                    all_funding_complete &= funding_status
        if kline_dirty:
            store.replace("klines_daily", working_klines)
        if funding_dirty:
            store.replace("funding_events", working_funding)
        return all_kline_complete, all_funding_complete

    @staticmethod
    def _funding_checkpoints_complete(store: CryptoQuantStore, target_end_ms: int) -> bool:
        mappings = store.read("futures_contracts")
        if mappings.empty:
            return False
        metadata = store.read_metadata()
        return all(
            metadata.get(f"checkpoint.funding.{symbol}", {}).get("complete") is True
            and int(metadata.get(f"checkpoint.funding.{symbol}", {}).get("requested_end_ms", -1)) >= target_end_ms
            for symbol in mappings["binance_symbol"].dropna().astype(str)
        )

    @staticmethod
    def _symbol_status(store: CryptoQuantStore) -> dict[str, object]:
        metadata = store.read_metadata()
        status: dict[str, dict[str, object]] = {}
        for key, value in metadata.items():
            if key.startswith("checkpoint.klines."):
                status.setdefault(key.removeprefix("checkpoint.klines."), {})["kline"] = value
            elif key.startswith("checkpoint.funding."):
                status.setdefault(key.removeprefix("checkpoint.funding."), {})["funding"] = value
        return dict(sorted(status.items()))

    def _build_derived(self, store: CryptoQuantStore, cmc_end: date, panel_end: date, funding_complete: bool) -> None:
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
        funding_through = panel_end if funding_complete else None
        panel = build_research_panel(universe, panel_klines, store.read("funding_events"), panel_end, funding_through)
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
        panel_complete = last_complete_panel_date(store.read("research_panel_daily"))
        return {
            "schema_version": 1, "pipeline_version": "task10", "rules_version": load_mapping_rules(self.config.rules_path).version,
            "source_urls": {"cmc": self.config.cmc_url, "exchange_info": self.config.exchange_info_url, "klines": self.config.klines_url, "funding": self.config.funding_url},
            "last_successful_cmc_date": cmc_watermark.isoformat() if cmc_watermark is not None else prior.get("last_successful_cmc_date"),
            "last_successful_kline_date": kline_end.isoformat() if kline_complete else prior.get("last_successful_kline_date"),
            "last_successful_funding_time": funding_end_ms if funding_complete else prior.get("last_successful_funding_time"),
            "last_complete_panel_date": panel_complete.date().isoformat() if panel_complete is not None else None,
            "table_row_counts": {name: len(store.read(name)) for name in TABLE_SPECS}, "table_date_ranges": ranges,
            "created_at_utc": store.read_metadata().get("created_at_utc", run_at.isoformat()), "updated_at_utc": run_at.isoformat(),
        }


__all__ = ["BinanceSource", "CmcSource", "CryptoQuantPipeline", "RunSummary"]
