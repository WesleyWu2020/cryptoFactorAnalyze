from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib

import pandas as pd
import pytest

from data.crypto_quant.config import PipelineConfig
from data.crypto_quant.pipeline import CryptoQuantPipeline, RunSummary
from data.crypto_quant.store import CryptoQuantStore
from data.crypto_quant.universe import UniverseBuildError


SYMBOLS = [f"C{i:02d}USDT" for i in range(50)]


def _cmc_frame(start=date(2024, 1, 1), end=date(2024, 2, 25), extra=False):
    days = pd.date_range(start, end, freq="D")
    daily = pd.DataFrame({"date": days, "index_value": 100.0, "source_update_time": days, "fetched_at_utc": pd.Timestamp("2024-02-25", tz="UTC")})
    if extra:
        daily = pd.concat([daily, pd.DataFrame([{"date": pd.Timestamp("2024-07-01"), "index_value": 100.0, "source_update_time": pd.Timestamp("2024-07-01"), "fetched_at_utc": pd.Timestamp("2024-02-25", tz="UTC")}])], ignore_index=True)
    rows = []
    for day in days:
        for i in range(100):
            rows.append({"date": day, "cmc_id": i + 1, "symbol": f"C{i:02d}", "name": f"Coin {i:02d}", "weight": 50 - i})
    if extra:
        for i in range(99):
            rows.append({"date": pd.Timestamp("2024-07-01"), "cmc_id": i + 1, "symbol": f"C{i:02d}", "name": f"Coin {i:02d}", "weight": 50 - i})
        rows.append({"date": pd.Timestamp("2024-07-01"), "cmc_id": 101, "symbol": "NEW", "name": "New", "weight": 1})
    return daily, pd.DataFrame(rows)


def _klines(symbols=SYMBOLS, start=date(2023, 12, 31), end=date(2024, 2, 24)):
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame([{
        "date": d, "symbol": s, "open_time": d, "close_time": d + pd.Timedelta(hours=23, minutes=59),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0,
        "quote_volume": 100000.0, "trade_count": 10, "taker_buy_base_volume": 500.0,
        "taker_buy_quote_volume": 50000.0,
    } for d in dates for s in symbols])


class FakeCmc:
    def __init__(self, extra=False):
        self.daily, self.members = _cmc_frame(extra=extra)
        self.ranges = []
        self.max_return_date = None

    def fetch_history(self, start, end, on_page=None):
        self.ranges.append((start, end))
        daily = self.daily[(pd.to_datetime(self.daily.date).dt.date >= start) & (pd.to_datetime(self.daily.date).dt.date <= end)].copy()
        members = self.members[(pd.to_datetime(self.members.date).dt.date >= start) & (pd.to_datetime(self.members.date).dt.date <= end)].copy()
        if self.max_return_date is not None:
            daily = daily[pd.to_datetime(daily.date).dt.date <= self.max_return_date]
            members = members[pd.to_datetime(members.date).dt.date <= self.max_return_date]
        if on_page:
            on_page(daily, members, end)
        return daily, members


class FakeBinance:
    def __init__(self, fail_symbol=None, only_49=False):
        self.fail_symbol = fail_symbol
        self.only_49 = only_49
        self.kline_requests = []
        self.funding_requests = []

    def fetch_exchange_info(self):
        count = 49 if self.only_49 else 50
        return pd.DataFrame([{
            "binance_symbol": SYMBOLS[i], "base_asset": f"C{i:02d}", "quote_asset": "USDT",
            "contract_type": "PERPETUAL", "onboard_date": pd.Timestamp("2020-01-01"),
            "status": "TRADING", "fetched_at_utc": pd.Timestamp("2024-02-25", tz="UTC"),
        } for i in range(count)])

    def fetch_klines(self, symbol, start, end):
        self.kline_requests.append((symbol, start, end))
        if symbol == self.fail_symbol:
            raise RuntimeError("selected symbol failure")
        if symbol not in SYMBOLS and symbol != "NEWUSDT":
            return pd.DataFrame()
        if self.only_49 and symbol == SYMBOLS[-1]:
            return pd.DataFrame()
        return _klines([symbol], start, end)

    def fetch_funding(self, symbol, start_ms, end_ms):
        self.funding_requests.append((symbol, start_ms, end_ms))
        return pd.DataFrame(columns=["funding_time", "symbol", "funding_rate", "mark_price", "rate_type"])


def _config(tmp_path):
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    return PipelineConfig(
        repo_root=tmp_path,
        store_path=root / "crypto_quant.h5",
        staging_path=root / ".crypto_quant.staging.h5",
        lock_path=root / ".crypto_quant.lock",
        rules_path=__import__("pathlib").Path("data/crypto_quant_rules.json").resolve(),
    )


def _run(tmp_path, cmc=None, binance=None, as_of=datetime(2024, 2, 25, 12, tzinfo=timezone.utc)):
    return CryptoQuantPipeline(_config(tmp_path), cmc or FakeCmc(), binance or FakeBinance()).backfill(as_of)


def test_backfill_writes_all_seven_tables_and_metadata(tmp_path):
    summary = _run(tmp_path)
    store = CryptoQuantStore(tmp_path / "data" / "crypto_quant.h5")
    assert store.keys() == {"_metadata", "cmc100_daily", "cmc100_constituents", "futures_contracts", "klines_daily", "funding_events", "universe_monthly", "research_panel_daily"}
    metadata = store.read_metadata()
    assert {"schema_version", "pipeline_version", "rules_version", "source_urls", "last_successful_cmc_date", "last_successful_kline_date", "last_successful_funding_time", "last_complete_panel_date", "table_row_counts", "table_date_ranges", "created_at_utc", "updated_at_utc"}.issubset(metadata)
    assert summary.symbol_status and metadata["symbol_status"] == summary.symbol_status


def test_complete_empty_funding_response_is_published_as_complete(tmp_path):
    _run(tmp_path)
    store = CryptoQuantStore(_config(tmp_path).store_path)
    metadata = store.read_metadata()
    checkpoint = metadata["checkpoint.funding.C00USDT"]
    assert checkpoint["complete"] is True
    assert checkpoint["empty_result"] is True
    assert metadata["last_successful_funding_time"] == int(datetime(2024, 2, 25, 12, tzinfo=timezone.utc).timestamp() * 1000)
    panel = store.read("research_panel_daily")
    assert panel["funding_event_count"].eq(0).all()
    assert panel["funding_rate_mean"].isna().all()
    assert panel["has_complete_funding"].all()


def test_completed_funding_checkpoint_does_not_hide_empty_extension(tmp_path):
    _run(tmp_path)
    config = _config(tmp_path)
    before = CryptoQuantStore(config.store_path).read_metadata()

    class EmptyFunding(FakeBinance):
        def fetch_funding(self, symbol, start_ms, end_ms):
            return pd.DataFrame(columns=["funding_time", "symbol", "funding_rate", "mark_price", "rate_type"])

    CryptoQuantPipeline(config, FakeCmc(), EmptyFunding()).update(
        datetime(2024, 2, 26, 12, tzinfo=timezone.utc)
    )
    store = CryptoQuantStore(config.store_path)
    metadata = store.read_metadata()
    checkpoint = metadata["checkpoint.funding.C00USDT"]
    assert checkpoint["complete"] is False
    assert checkpoint["missing_date_count"] > 0
    assert metadata["last_successful_funding_time"] == before["last_successful_funding_time"]
    assert store.read("research_panel_daily")["has_complete_funding"].eq(False).all()


def test_summary_does_not_report_kline_end_when_kline_coverage_is_incomplete(tmp_path):
    class PartialKline(FakeBinance):
        def fetch_klines(self, symbol, start, end):
            if symbol == SYMBOLS[0]:
                self.kline_requests.append((symbol, start, end))
                out = super().fetch_klines(symbol, start, end)
                return out[out.date != pd.Timestamp("2024-02-24")]
            return super().fetch_klines(symbol, start, end)

    summary = _run(tmp_path, binance=PartialKline())
    assert summary.last_complete_kline_date is None
    assert summary.warnings
    metadata = CryptoQuantStore(_config(tmp_path).store_path).read_metadata()
    assert metadata["validation_warnings"]
    assert metadata["last_complete_panel_date"] != "None"
    assert summary.last_complete_kline_date is None or isinstance(summary.last_complete_kline_date, date)


def test_pipeline_incremental_persistence_uses_bounded_table_reads(tmp_path, monkeypatch):
    reads = {"klines_daily": 0, "funding_events": 0}
    original_read = CryptoQuantStore.read

    def counted_read(self, name, where=None):
        if name in reads:
            reads[name] += 1
        return original_read(self, name, where)

    def fail_upsert(*args, **kwargs):
        raise AssertionError("incremental pipeline must not upsert full tables per symbol")

    monkeypatch.setattr(CryptoQuantStore, "read", counted_read)
    monkeypatch.setattr(CryptoQuantStore, "upsert", fail_upsert)
    _run(tmp_path)
    assert reads["klines_daily"] < 20
    assert reads["funding_events"] < 20


def test_incremental_symbol_lookups_are_pre_grouped_once(tmp_path, monkeypatch):
    """Symbol growth must not add full-frame boolean scans inside the loop."""
    groupby_columns = []
    original_groupby = pd.DataFrame.groupby

    def counted_groupby(self, by=None, *args, **kwargs):
        if by == "symbol":
            groupby_columns.append(tuple(self.columns))
        return original_groupby(self, by=by, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", counted_groupby)
    _run(tmp_path)

    assert sum("symbol" in columns for columns in groupby_columns) >= 2
    assert len(groupby_columns) <= 2


def test_mapping_issues_are_retained_in_metadata(tmp_path):
    summary = _run(tmp_path)
    metadata = CryptoQuantStore(_config(tmp_path).store_path).read_metadata()
    issues = metadata["mapping_issues"]
    assert issues
    assert {"cmc_id", "cmc_symbol", "decision_date", "issue"}.issubset(issues[0])
    assert any(issue["issue"] == "unresolved" for issue in issues)
    assert any("unresolved" in warning for warning in summary.warnings)
    assert any("unresolved" in warning for warning in metadata["validation_warnings"])


def test_rebuild_derived_does_not_reuse_expired_funding_checkpoint(tmp_path):
    _run(tmp_path)
    config = _config(tmp_path)
    summary = CryptoQuantPipeline(config, FakeCmc(), FakeBinance()).rebuild_derived(
        datetime(2024, 2, 26, 12, tzinfo=timezone.utc)
    )
    panel = CryptoQuantStore(config.store_path).read("research_panel_daily")
    assert not panel["has_complete_funding"].any()
    assert summary.last_complete_panel_date is not None


def test_incremental_revision_of_existing_kline_key_is_persisted(tmp_path):
    _run(tmp_path)

    class Revised(FakeBinance):
        def fetch_klines(self, symbol, start, end):
            out = super().fetch_klines(symbol, start, end)
            if out.empty:
                return out
            mask = out.date == pd.Timestamp("2024-02-20")
            out.loc[mask, "close"] = 100.75
            return out

    config = _config(tmp_path)
    CryptoQuantPipeline(config, FakeCmc(), Revised()).update(
        datetime(2024, 2, 26, 12, tzinfo=timezone.utc)
    )
    stored = CryptoQuantStore(config.store_path).read("klines_daily")
    assert stored.loc[(stored.symbol == SYMBOLS[0]) & (stored.date == pd.Timestamp("2024-02-20")), "close"].iloc[0] == 100.75


def test_incremental_revision_of_existing_funding_key_is_persisted(tmp_path):
    class FundingEvents(FakeBinance):
        def fetch_funding(self, symbol, start_ms, end_ms):
            start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
            end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
            dates = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
            return pd.DataFrame([{"funding_time": value + pd.Timedelta(hours=8), "symbol": symbol, "funding_rate": 0.001, "mark_price": 1.0, "rate_type": "Regular"} for value in dates if start <= value + pd.Timedelta(hours=8) <= end])

    _run(tmp_path, binance=FundingEvents())

    class RevisedFunding(FundingEvents):
        def fetch_funding(self, symbol, start_ms, end_ms):
            out = super().fetch_funding(symbol, start_ms, end_ms)
            out.loc[out.funding_time == pd.Timestamp("2024-02-20 08:00", tz="UTC"), "funding_rate"] = 0.009
            return out

    config = _config(tmp_path)
    CryptoQuantPipeline(config, FakeCmc(), RevisedFunding()).update(datetime(2024, 2, 26, 12, tzinfo=timezone.utc))
    stored = CryptoQuantStore(config.store_path).read("funding_events")
    assert stored.loc[(stored.symbol == SYMBOLS[0]) & (stored.funding_time == pd.Timestamp("2024-02-20 08:00", tz="UTC")), "funding_rate"].iloc[0] == 0.009


def test_fingerprint_is_stable_for_relative_and_absolute_target_paths(tmp_path):
    absolute = _config(tmp_path)
    relative = _config(tmp_path)
    target = __import__("pathlib").Path.cwd() / "fingerprint-target.h5"
    object.__setattr__(absolute, "store_path", target)
    object.__setattr__(relative, "store_path", __import__("pathlib").Path("fingerprint-target.h5"))
    assert CryptoQuantPipeline(absolute, FakeCmc(), FakeBinance())._fingerprint("backfill") == CryptoQuantPipeline(relative, FakeCmc(), FakeBinance())._fingerprint("backfill")


def test_second_identical_update_is_idempotent(tmp_path):
    _run(tmp_path)
    active = tmp_path / "data" / "crypto_quant.h5"
    first = CryptoQuantStore(active)
    frames = {name: first.read(name) for name in first.keys() if name != "_metadata"}
    meta1 = first.read_metadata()
    _run(tmp_path)
    second = CryptoQuantStore(active)
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(frame, second.read(name))
    meta2 = second.read_metadata()
    for key in meta1:
        if key not in {"created_at_utc", "updated_at_utc", "run_attempts"}:
            assert meta1[key] == meta2[key]


def test_update_refetches_ten_cmc_days_and_seven_binance_days(tmp_path):
    cmc = FakeCmc()
    binance = FakeBinance()
    cmc.max_return_date = date(2024, 2, 20)
    _run(tmp_path, cmc, binance, datetime(2024, 2, 21, tzinfo=timezone.utc))
    cmc.ranges.clear(); binance.kline_requests.clear()
    pipeline = CryptoQuantPipeline(_config(tmp_path), cmc, binance)
    pipeline.update(datetime(2024, 2, 25, 0, 20, tzinfo=timezone.utc))
    assert cmc.ranges[-1][0] == date(2024, 2, 11)
    assert min(start for _, start, end in binance.kline_requests if start < end) == date(2024, 2, 14)


def test_new_candidate_receives_180_day_support_backfill(tmp_path):
    cmc = FakeCmc(extra=True)
    binance = FakeBinance()
    pipeline = CryptoQuantPipeline(_config(tmp_path), cmc, binance)
    with pytest.raises(UniverseBuildError):
        pipeline.backfill(datetime(2024, 7, 2, tzinfo=timezone.utc))
    new = [item for item in binance.kline_requests if item[0] == "NEWUSDT"]
    assert new and min(item[1] for item in new) == date(2024, 1, 3)


def test_first_decision_uses_2023_12_31_kline_and_activates_2024_01_02(tmp_path):
    binance = FakeBinance()
    _run(tmp_path, binance=binance)
    assert any(symbol == SYMBOLS[0] and start <= date(2023, 12, 31) <= end for symbol, start, end in binance.kline_requests)
    universe = CryptoQuantStore(tmp_path / "data" / "crypto_quant.h5").read("universe_monthly")
    assert universe["effective_date"].min() == pd.Timestamp("2024-01-02")


def test_month_start_creates_t_plus_one_membership(tmp_path):
    _run(tmp_path, as_of=datetime(2024, 2, 1, 12, tzinfo=timezone.utc))
    universe = CryptoQuantStore(tmp_path / "data" / "crypto_quant.h5").read("universe_monthly")
    feb = universe[universe.decision_date == pd.Timestamp("2024-02-01")]
    assert len(feb) == 50 and feb.effective_date.eq(pd.Timestamp("2024-02-02")).all()


def test_api_failure_keeps_active_sha256_unchanged_and_staging_resumable(tmp_path):
    _run(tmp_path)
    active = tmp_path / "data" / "crypto_quant.h5"
    before = hashlib.sha256(active.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError):
        CryptoQuantPipeline(_config(tmp_path), FakeCmc(), FakeBinance(fail_symbol=SYMBOLS[2])).update(datetime(2024, 2, 25, tzinfo=timezone.utc))
    assert hashlib.sha256(active.read_bytes()).hexdigest() == before
    assert CryptoQuantStore(tmp_path / "data" / ".crypto_quant.staging.h5").read_metadata().get("checkpoint.klines.C00USDT") is not None


def test_fresh_backfill_failure_persists_completed_symbols_in_staging(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(RuntimeError):
        CryptoQuantPipeline(config, FakeCmc(), FakeBinance(fail_symbol=SYMBOLS[2])).backfill(
            datetime(2024, 2, 25, tzinfo=timezone.utc)
        )
    assert not config.store_path.exists()
    staging = CryptoQuantStore(config.staging_path)
    klines = staging.read("klines_daily")
    assert not klines[klines.symbol == SYMBOLS[0]].empty
    assert not klines[klines.symbol == SYMBOLS[1]].empty
    assert staging.read_metadata().get("checkpoint.klines.C00USDT") is not None
    assert staging.read_metadata().get("checkpoint.klines.C01USDT") is not None


def test_resume_consumes_symbol_checkpoints_and_deduplicates_retry(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(RuntimeError):
        CryptoQuantPipeline(config, FakeCmc(), FakeBinance(fail_symbol=SYMBOLS[2])).backfill(
            datetime(2024, 2, 25, tzinfo=timezone.utc)
        )
    resumed = FakeBinance()
    summary = CryptoQuantPipeline(config, FakeCmc(), resumed).backfill(datetime(2024, 2, 25, tzinfo=timezone.utc))
    resumed_symbols = {request[0] for request in resumed.kline_requests}
    assert SYMBOLS[0] not in resumed_symbols and SYMBOLS[1] not in resumed_symbols
    assert SYMBOLS[2] in resumed_symbols
    store = CryptoQuantStore(config.store_path)
    for name in ("klines_daily", "funding_events"):
        frame = store.read(name)
        assert not frame.duplicated([column for column in ("date", "funding_time") if column in frame] + ["symbol"]).any()
    assert summary.symbol_status[SYMBOLS[2]]["kline"]["complete"] is True


def test_funding_failure_keeps_current_symbol_kline_checkpoint_in_staging(tmp_path):
    class FundingFailure(FakeBinance):
        def fetch_funding(self, symbol, start_ms, end_ms):
            if symbol == SYMBOLS[2]:
                raise RuntimeError("selected funding failure")
            return super().fetch_funding(symbol, start_ms, end_ms)

    config = _config(tmp_path)
    with pytest.raises(RuntimeError):
        CryptoQuantPipeline(config, FakeCmc(), FundingFailure()).backfill(
            datetime(2024, 2, 25, tzinfo=timezone.utc)
        )
    assert not config.store_path.exists()
    staging = CryptoQuantStore(config.staging_path)
    klines = staging.read("klines_daily")
    assert not klines[klines.symbol == SYMBOLS[2]].empty
    checkpoint = staging.read_metadata().get("checkpoint.klines.C02USDT")
    assert checkpoint is not None and checkpoint["complete"] is True
    assert "checkpoint.funding.C02USDT" not in staging.read_metadata()
    assert "last_successful_funding_time" not in staging.read_metadata()


def test_validation_failure_never_publishes_staging(tmp_path):
    _run(tmp_path)
    active = tmp_path / "data" / "crypto_quant.h5"
    before = hashlib.sha256(active.read_bytes()).hexdigest()
    with pytest.raises(UniverseBuildError):
        CryptoQuantPipeline(_config(tmp_path), FakeCmc(), FakeBinance(only_49=True)).update(datetime(2024, 2, 25, tzinfo=timezone.utc), reset_staging=True)
    assert hashlib.sha256(active.read_bytes()).hexdigest() == before


def test_empty_incremental_sources_do_not_advance_watermarks_or_checkpoints(tmp_path):
    _run(tmp_path)
    config = _config(tmp_path)
    before = CryptoQuantStore(config.store_path).read_metadata()

    class Empty(FakeBinance):
        def fetch_klines(self, symbol, start, end):
            self.kline_requests.append((symbol, start, end))
            return pd.DataFrame()

        def fetch_funding(self, symbol, start_ms, end_ms):
            self.funding_requests.append((symbol, start_ms, end_ms))
            return pd.DataFrame()

    CryptoQuantPipeline(config, FakeCmc(), Empty()).update(datetime(2024, 2, 26, tzinfo=timezone.utc))
    after = CryptoQuantStore(config.store_path).read_metadata()
    assert after["last_successful_kline_date"] == before["last_successful_kline_date"]
    assert after["last_successful_funding_time"] == before["last_successful_funding_time"]
    assert after["checkpoint.klines.C00USDT"] == before["checkpoint.klines.C00USDT"]
    assert after["checkpoint.funding.C00USDT"] == before["checkpoint.funding.C00USDT"]


def test_pipeline_clips_adapter_rows_to_requested_ranges(tmp_path):
    class Leaky(FakeBinance):
        def fetch_klines(self, symbol, start, end):
            out = super().fetch_klines(symbol, start, end)
            if out.empty:
                return out
            extra = out.iloc[[0]].copy()
            extra["date"] = pd.Timestamp(start) - pd.Timedelta(days=1)
            return pd.concat([out, extra], ignore_index=True)

        def fetch_funding(self, symbol, start_ms, end_ms):
            return pd.DataFrame([
                {"funding_time": pd.Timestamp(start_ms, unit="ms") - pd.Timedelta(milliseconds=1), "symbol": symbol, "funding_rate": 1.0, "mark_price": 1.0, "rate_type": "Regular"},
                {"funding_time": pd.Timestamp(start_ms, unit="ms"), "symbol": symbol, "funding_rate": 1.0, "mark_price": 1.0, "rate_type": "Regular"},
                {"funding_time": pd.Timestamp(end_ms, unit="ms") + pd.Timedelta(milliseconds=1), "symbol": symbol, "funding_rate": 1.0, "mark_price": 1.0, "rate_type": "Regular"},
            ])

    source = Leaky()
    _run(tmp_path, binance=source)
    store = CryptoQuantStore(_config(tmp_path).store_path)
    klines = store.read("klines_daily")
    assert all((klines.loc[klines.symbol == symbol, "date"] >= pd.Timestamp(start)).all() and (klines.loc[klines.symbol == symbol, "date"] <= pd.Timestamp(end)).all() for symbol, start, end in source.kline_requests)
    funding = store.read("funding_events")
    assert not funding.empty
    assert funding["funding_time"].min() >= pd.Timestamp("2020-01-01", tz="UTC")


def test_pipeline_cutoff_replay_matches_full_raw_and_derived_prefix(tmp_path):
    full_dir, cutoff_dir = tmp_path / "full", tmp_path / "cutoff"
    class Bounded(FakeBinance):
        def __init__(self, max_date=None):
            super().__init__()
            self.max_date = max_date

        def fetch_klines(self, symbol, start, end):
            out = super().fetch_klines(symbol, start, end)
            if self.max_date is not None:
                if out.empty:
                    return out
                out = out[pd.to_datetime(out.date).dt.date <= self.max_date]
            return out

        def fetch_funding(self, symbol, start_ms, end_ms):
            start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
            end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
            dates = [value + pd.Timedelta(hours=8) for value in pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")]
            if self.max_date is not None:
                dates = [value for value in dates if value.date() <= self.max_date]
            return pd.DataFrame([{"funding_time": value, "symbol": symbol, "funding_rate": 0.0, "mark_price": 1.0, "rate_type": "Regular"} for value in dates if start <= value <= end])

    _run(full_dir, binance=Bounded())
    cmc = FakeCmc()
    cmc.max_return_date = date(2024, 2, 20)
    _run(cutoff_dir, cmc=cmc, binance=Bounded(date(2024, 2, 20)))
    full = CryptoQuantStore(_config(full_dir).store_path)
    cut = CryptoQuantStore(_config(cutoff_dir).store_path)
    assert full.read("cmc100_constituents")["date"].max() > pd.Timestamp("2024-02-20")
    assert full.read("klines_daily")["date"].max() > pd.Timestamp("2024-02-20")
    assert full.read("funding_events")["funding_time"].max() > pd.Timestamp("2024-02-20", tz="UTC")
    assert cut.read("cmc100_constituents")["date"].max() <= pd.Timestamp("2024-02-20")
    assert cut.read("klines_daily")["date"].max() <= pd.Timestamp("2024-02-20")
    assert cut.read("funding_events")["funding_time"].max() <= pd.Timestamp("2024-02-20 23:59:59", tz="UTC")
    prefixes = {
        "cmc100_daily": ("date", pd.Timestamp("2024-02-20")),
        "cmc100_constituents": ("date", pd.Timestamp("2024-02-20")),
        "futures_contracts": ("valid_from", pd.Timestamp("2024-02-20")),
        "klines_daily": ("date", pd.Timestamp("2024-02-19")),
        "funding_events": ("funding_time", pd.Timestamp("2024-02-20 23:59:59", tz="UTC")),
        "universe_monthly": ("effective_date", pd.Timestamp("2024-02-20")),
        "research_panel_daily": ("date", pd.Timestamp("2024-02-19")),
    }
    for name, (column, cutoff) in prefixes.items():
        left = full.read(name); right = cut.read(name)
        left_values = pd.to_datetime(left[column], utc=True)
        right_values = pd.to_datetime(right[column], utc=True)
        left = left[left_values <= cutoff.tz_localize("UTC") if cutoff.tzinfo is None else left_values <= cutoff].reset_index(drop=True)
        right = right[right_values <= cutoff.tz_localize("UTC") if cutoff.tzinfo is None else right_values <= cutoff].reset_index(drop=True)
        if name == "futures_contracts":
            for frame in (left, right):
                assert frame["valid_from"].notna().all()
                assert frame["valid_to"].isna().all()
        numeric = left.select_dtypes(include="number").columns.intersection(right.select_dtypes(include="number").columns)
        max_abs_diff = max((left[column].fillna(0).to_numpy() - right[column].fillna(0).to_numpy()).__abs__().max(initial=0.0) for column in numeric) if len(numeric) else 0.0
        assert max_abs_diff == 0.0
        if name == "research_panel_daily":
            assert left["has_complete_funding"].all()
            assert not right["has_complete_funding"].any()
            left = left.drop(columns=["has_complete_funding"])
            right = right.drop(columns=["has_complete_funding"])
        pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_cmc_checkpoint_uses_actual_clipped_maximum(tmp_path):
    cmc = FakeCmc()
    cmc.max_return_date = date(2024, 2, 20)
    _run(tmp_path, cmc=cmc)
    metadata = CryptoQuantStore(_config(tmp_path).store_path).read_metadata()
    assert metadata["checkpoint.cmc_through"] == "2024-02-20"
    assert metadata["last_successful_cmc_date"] == "2024-02-20"


def test_kline_gap_is_not_marked_complete_or_advanced_globally(tmp_path):
    class Gap(FakeBinance):
        def fetch_klines(self, symbol, start, end):
            out = super().fetch_klines(symbol, start, end)
            if symbol == SYMBOLS[0] and len(out) > 4:
                out = out.drop(out.index[len(out) // 2])
            return out

    _run(tmp_path, binance=Gap())
    metadata = CryptoQuantStore(_config(tmp_path).store_path).read_metadata()
    checkpoint = metadata["checkpoint.klines.C00USDT"]
    assert checkpoint["complete"] is False
    assert checkpoint["missing_date_count"] > 0
    assert metadata["last_successful_kline_date"] is None


def test_funding_watermark_uses_completed_natural_day_not_exact_event_end(tmp_path):
    class DailyFunding(FakeBinance):
        def fetch_funding(self, symbol, start_ms, end_ms):
            start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
            end_day = pd.Timestamp(end_ms, unit="ms", tz="UTC").normalize()
            dates = [value + pd.Timedelta(hours=8) for value in pd.date_range(start.normalize(), end_day, freq="D")]
            return pd.DataFrame([{"funding_time": value, "symbol": symbol, "funding_rate": 0.0, "mark_price": 1.0, "rate_type": "Regular"} for value in dates])

    _run(tmp_path, binance=DailyFunding())
    metadata = CryptoQuantStore(_config(tmp_path).store_path).read_metadata()
    assert metadata["last_successful_funding_time"] == int(datetime(2024, 2, 25, 12, tzinfo=timezone.utc).timestamp() * 1000)


def test_cmc_internal_gap_only_advances_longest_contiguous_prefix(tmp_path):
    class GapCmc(FakeCmc):
        def fetch_history(self, start, end, on_page=None):
            daily, members = super().fetch_history(start, end, on_page=None)
            daily = daily[daily.date != pd.Timestamp("2024-01-10")]
            members = members[members.date != pd.Timestamp("2024-01-10")]
            if on_page:
                on_page(daily, members, end)
            return daily, members

    _run(tmp_path, cmc=GapCmc())
    metadata = CryptoQuantStore(_config(tmp_path).store_path).read_metadata()
    assert metadata["checkpoint.cmc_through"] == "2024-01-09"
    assert metadata["last_successful_cmc_date"] == "2024-01-09"


@pytest.mark.parametrize("missing_table", ["daily", "members"])
def test_cmc_asymmetric_table_gap_blocks_common_watermark_prefix(tmp_path, missing_table):
    class AsymmetricGap(FakeCmc):
        def fetch_history(self, start, end, on_page=None):
            daily, members = super().fetch_history(start, end, on_page=None)
            if missing_table == "daily":
                daily = daily[daily.date != pd.Timestamp("2024-01-10")]
            else:
                members = members[members.date != pd.Timestamp("2024-01-10")]
            if on_page:
                on_page(daily, members, end)
            return daily, members

    _run(tmp_path, cmc=AsymmetricGap())
    metadata = CryptoQuantStore(_config(tmp_path).store_path).read_metadata()
    assert metadata["checkpoint.cmc_through"] == "2024-01-09"
    assert metadata["last_successful_cmc_date"] == "2024-01-09"


def test_cmc_partial_snapshot_truncates_common_watermark_prefix(tmp_path):
    class PartialSnapshot(FakeCmc):
        def fetch_history(self, start, end, on_page=None):
            daily, members = super().fetch_history(start, end, on_page=None)
            members = members[~((members.date == pd.Timestamp("2024-01-10")) & (members.cmc_id == 100))]
            if on_page:
                on_page(daily, members, end)
            return daily, members

    _run(tmp_path, cmc=PartialSnapshot())
    metadata = CryptoQuantStore(_config(tmp_path).store_path).read_metadata()
    assert metadata["checkpoint.cmc_through"] == "2024-01-09"
    assert metadata["last_successful_cmc_date"] == "2024-01-09"


def test_cmc_incremental_partial_snapshot_replaces_old_date_before_validation(tmp_path):
    _run(tmp_path)
    before = CryptoQuantStore(_config(tmp_path).store_path).read_metadata()
    class PartialUpdate(FakeCmc):
        def fetch_history(self, start, end, on_page=None):
            daily, members = super().fetch_history(start, end, on_page=None)
            members = members[~((members.date == pd.Timestamp("2024-02-20")) & (members.cmc_id == 100))]
            if on_page:
                on_page(daily, members, end)
            return daily, members

    config = _config(tmp_path)
    CryptoQuantPipeline(config, PartialUpdate(), FakeBinance()).update(datetime(2024, 2, 26, tzinfo=timezone.utc))
    metadata = CryptoQuantStore(config.store_path).read_metadata()
    assert metadata["last_successful_cmc_date"] == before["last_successful_cmc_date"]
    assert metadata["checkpoint.cmc_through"] == before["checkpoint.cmc_through"]
    members = CryptoQuantStore(config.store_path).read("cmc100_constituents")
    assert len(members[members.date == pd.Timestamp("2024-02-20")]) == 100


def test_empty_cmc_incremental_response_preserves_existing_history(tmp_path):
    _run(tmp_path)
    config = _config(tmp_path)
    before = CryptoQuantStore(config.store_path)
    before_daily = before.read("cmc100_daily")
    before_members = before.read("cmc100_constituents")

    class EmptyCmc(FakeCmc):
        def fetch_history(self, start, end, on_page=None):
            daily = self.daily.iloc[0:0].copy()
            members = self.members.iloc[0:0].copy()
            if on_page:
                on_page(daily, members, end)
            return daily, members

    CryptoQuantPipeline(config, EmptyCmc(), FakeBinance()).update(datetime(2024, 2, 26, tzinfo=timezone.utc))
    after = CryptoQuantStore(config.store_path)
    pd.testing.assert_frame_equal(before_daily, after.read("cmc100_daily"))
    pd.testing.assert_frame_equal(before_members, after.read("cmc100_constituents"))


def test_cmc_snapshot_with_101_unique_members_is_not_complete(tmp_path):
    class OversizedSnapshot(FakeCmc):
        def fetch_history(self, start, end, on_page=None):
            daily, members = super().fetch_history(start, end, on_page=None)
            extra = members[members.date == pd.Timestamp("2024-01-10")].iloc[[0]].copy()
            extra["cmc_id"] = 1001
            members = pd.concat([members, extra], ignore_index=True)
            if on_page:
                on_page(daily, members, end)
            return daily, members

    _run(tmp_path, cmc=OversizedSnapshot())
    metadata = CryptoQuantStore(_config(tmp_path).store_path).read_metadata()
    assert metadata["checkpoint.cmc_through"] == "2024-01-09"
    assert metadata["last_successful_cmc_date"] == "2024-01-09"


def test_cmc_snapshot_rejects_duplicate_id_across_same_utc_day_timestamps(tmp_path):
    class TimestampDuplicate(FakeCmc):
        def fetch_history(self, start, end, on_page=None):
            daily, members = super().fetch_history(start, end, on_page=None)
            target = (members.date == pd.Timestamp("2024-01-10"))
            duplicate = members[target & members.cmc_id.eq(1)].copy()
            duplicate["date"] = pd.Timestamp("2024-01-10 23:00", tz="UTC")
            members = pd.concat([members, duplicate], ignore_index=True)
            members = members[
                ~((members.date == pd.Timestamp("2024-01-10")) & members.cmc_id.eq(100))
            ]
            if on_page:
                on_page(daily, members, end)
            return daily, members

    _run(tmp_path, cmc=TimestampDuplicate())
    store = CryptoQuantStore(_config(tmp_path).store_path)
    metadata = store.read_metadata()
    assert metadata["checkpoint.cmc_through"] == "2024-01-09"
    assert metadata["last_successful_cmc_date"] == "2024-01-09"
    stored = store.read("cmc100_constituents")
    normalized = pd.to_datetime(stored["date"], utc=True).dt.date
    assert date(2024, 1, 10) not in set(normalized)


@pytest.mark.parametrize("invalid_id", [0, -1, 1.9, True])
def test_cmc_snapshot_rejects_non_positive_or_non_integer_cmc_id_before_cast(tmp_path, invalid_id):
    class InvalidId(FakeCmc):
        def fetch_history(self, start, end, on_page=None):
            daily, members = super().fetch_history(start, end, on_page=None)
            target = (members.date == pd.Timestamp("2024-01-10"))
            members["cmc_id"] = members["cmc_id"].astype(object)
            members.loc[target & members.cmc_id.eq(1), "cmc_id"] = invalid_id
            members = members[
                ~((members.date == pd.Timestamp("2024-01-10")) & members.cmc_id.eq(100))
            ]
            if on_page:
                on_page(daily, members, end)
            return daily, members

    _run(tmp_path, cmc=InvalidId())
    store = CryptoQuantStore(_config(tmp_path).store_path)
    metadata = store.read_metadata()
    assert metadata["checkpoint.cmc_through"] == "2024-01-09"
    assert metadata["last_successful_cmc_date"] == "2024-01-09"
    stored = store.read("cmc100_constituents")
    normalized = pd.to_datetime(stored["date"], utc=True).dt.date
    assert date(2024, 1, 10) not in set(normalized)


def test_funding_internal_natural_day_gap_blocks_completion(tmp_path):
    class GapFunding(FakeBinance):
        def fetch_funding(self, symbol, start_ms, end_ms):
            start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
            end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
            dates = [value + pd.Timedelta(hours=8) for value in pd.date_range(start.normalize(), end.normalize(), freq="D") if value.date() != date(2024, 1, 10)]
            return pd.DataFrame([{"funding_time": value, "symbol": symbol, "funding_rate": 0.0, "mark_price": 1.0, "rate_type": "Regular"} for value in dates])

    _run(tmp_path, binance=GapFunding())
    metadata = CryptoQuantStore(_config(tmp_path).store_path).read_metadata()
    checkpoint = metadata["checkpoint.funding.C00USDT"]
    assert checkpoint["complete"] is False
    assert checkpoint["missing_date_count"] > 0
    assert metadata["last_successful_funding_time"] is None
    panel = CryptoQuantStore(_config(tmp_path).store_path).read("research_panel_daily")
    assert panel["has_complete_funding"].eq(False).all()


def test_cmc_historical_gap_survives_incremental_overlap(tmp_path):
    class HistoricalGap(FakeCmc):
        def fetch_history(self, start, end, on_page=None):
            daily, members = super().fetch_history(start, end, on_page=None)
            daily = daily[daily.date != pd.Timestamp("2024-01-10")]
            members = members[members.date != pd.Timestamp("2024-01-10")]
            if on_page:
                on_page(daily, members, end)
            return daily, members

    cmc = HistoricalGap()
    _run(tmp_path, cmc=cmc)
    config = _config(tmp_path)
    CryptoQuantPipeline(config, cmc, FakeBinance()).update(datetime(2024, 2, 26, tzinfo=timezone.utc))
    metadata = CryptoQuantStore(config.store_path).read_metadata()
    assert cmc.ranges[-1][0] == date(2024, 1, 1)
    assert metadata["last_successful_cmc_date"] == "2024-01-09"
    assert metadata["checkpoint.cmc_through"] == "2024-01-09"


def test_kline_historical_gap_survives_incremental_overlap(tmp_path):
    class HistoricalGap(FakeBinance):
        def fetch_klines(self, symbol, start, end):
            out = super().fetch_klines(symbol, start, end)
            if symbol == SYMBOLS[0] and start == date(2023, 7, 5):
                out = out[out.date != pd.Timestamp("2024-01-10")]
            return out

    _run(tmp_path, binance=HistoricalGap())
    config = _config(tmp_path)
    repaired = FakeBinance()
    CryptoQuantPipeline(config, FakeCmc(), repaired).update(datetime(2024, 2, 26, tzinfo=timezone.utc))
    metadata = CryptoQuantStore(config.store_path).read_metadata()
    assert metadata["last_successful_kline_date"] == "2024-02-25"
    assert metadata["checkpoint.klines.C00USDT"]["complete"] is True
    assert next(start for symbol, start, _ in repaired.kline_requests if symbol == SYMBOLS[0]) == date(2024, 1, 4)


def test_funding_historical_gap_survives_incremental_overlap(tmp_path):
    class HistoricalGap(FakeBinance):
        def fetch_funding(self, symbol, start_ms, end_ms):
            start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
            end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
            dates = [value + pd.Timedelta(hours=8) for value in pd.date_range(start.normalize(), end.normalize(), freq="D") if value.date() != date(2024, 1, 10)]
            return pd.DataFrame([{"funding_time": value, "symbol": symbol, "funding_rate": 0.0, "mark_price": 1.0, "rate_type": "Regular"} for value in dates])

    class CompleteFunding(FakeBinance):
        def fetch_funding(self, symbol, start_ms, end_ms):
            start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
            end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
            dates = [value + pd.Timedelta(hours=8) for value in pd.date_range(start.normalize(), end.normalize(), freq="D")]
            return pd.DataFrame([{"funding_time": value, "symbol": symbol, "funding_rate": 0.0, "mark_price": 1.0, "rate_type": "Regular"} for value in dates])

    _run(tmp_path, binance=HistoricalGap())
    config = _config(tmp_path)
    CryptoQuantPipeline(config, FakeCmc(), CompleteFunding()).update(datetime(2024, 2, 26, tzinfo=timezone.utc))
    metadata = CryptoQuantStore(config.store_path).read_metadata()
    assert metadata["last_successful_funding_time"] == int(datetime(2024, 2, 26, tzinfo=timezone.utc).timestamp() * 1000)
    assert metadata["checkpoint.funding.C00USDT"]["complete"] is True


def test_empty_funding_response_does_not_hide_historical_gap(tmp_path):
    class GapFunding(FakeBinance):
        def fetch_funding(self, symbol, start_ms, end_ms):
            start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
            end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
            dates = [value + pd.Timedelta(hours=8) for value in pd.date_range(start.normalize(), end.normalize(), freq="D") if value.date() != date(2024, 1, 10)]
            return pd.DataFrame([{"funding_time": value, "symbol": symbol, "funding_rate": 0.0, "mark_price": 1.0, "rate_type": "Regular"} for value in dates])

    class EmptyFunding(FakeBinance):
        def fetch_funding(self, symbol, start_ms, end_ms):
            return pd.DataFrame(columns=["funding_time", "symbol", "funding_rate", "mark_price", "rate_type"])

    _run(tmp_path, binance=GapFunding())
    config = _config(tmp_path)
    CryptoQuantPipeline(config, FakeCmc(), EmptyFunding()).update(datetime(2024, 2, 26, tzinfo=timezone.utc))
    store = CryptoQuantStore(config.store_path)
    checkpoint = store.read_metadata()["checkpoint.funding.C00USDT"]
    assert checkpoint["complete"] is False
    assert checkpoint["empty_result"] is True
    assert checkpoint["missing_date_count"] > 0
    assert store.read_metadata()["last_successful_funding_time"] is None
    assert store.read("research_panel_daily")["has_complete_funding"].eq(False).all()
