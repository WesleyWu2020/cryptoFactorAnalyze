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
    rows = []
    for day in days:
        for i in range(50):
            rows.append({"date": day, "cmc_id": i + 1, "symbol": f"C{i:02d}", "name": f"Coin {i:02d}", "weight": 50 - i})
    if extra:
        rows.append({"date": pd.Timestamp("2024-07-01"), "cmc_id": 51, "symbol": "NEW", "name": "New", "weight": 1})
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
    _run(tmp_path)
    store = CryptoQuantStore(tmp_path / "data" / "crypto_quant.h5")
    assert store.keys() == {"_metadata", "cmc100_daily", "cmc100_constituents", "futures_contracts", "klines_daily", "funding_events", "universe_monthly", "research_panel_daily"}
    assert {"schema_version", "pipeline_version", "rules_version", "source_urls", "last_successful_cmc_date", "last_successful_kline_date", "last_successful_funding_time", "last_complete_panel_date", "table_row_counts", "table_date_ranges", "created_at_utc", "updated_at_utc"}.issubset(store.read_metadata())


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
    assert min(start for _, start, _ in binance.kline_requests) == date(2024, 2, 14)


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
    assert "checkpoint.funding.C00USDT" not in after


def test_pipeline_clips_adapter_rows_to_requested_ranges(tmp_path):
    class Leaky(FakeBinance):
        def fetch_klines(self, symbol, start, end):
            out = super().fetch_klines(symbol, start, end)
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


def test_cmc_historical_gap_survives_incremental_overlap(tmp_path):
    class HistoricalGap(FakeCmc):
        def fetch_history(self, start, end, on_page=None):
            daily, members = super().fetch_history(start, end, on_page=None)
            daily = daily[daily.date != pd.Timestamp("2024-01-10")]
            members = members[members.date != pd.Timestamp("2024-01-10")]
            if on_page:
                on_page(daily, members, end)
            return daily, members

    _run(tmp_path, cmc=HistoricalGap())
    config = _config(tmp_path)
    CryptoQuantPipeline(config, FakeCmc(), FakeBinance()).update(datetime(2024, 2, 26, tzinfo=timezone.utc))
    metadata = CryptoQuantStore(config.store_path).read_metadata()
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
    CryptoQuantPipeline(config, FakeCmc(), FakeBinance()).update(datetime(2024, 2, 26, tzinfo=timezone.utc))
    metadata = CryptoQuantStore(config.store_path).read_metadata()
    assert metadata["last_successful_kline_date"] is None
    assert metadata["checkpoint.klines.C00USDT"]["complete"] is False


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
    assert metadata["last_successful_funding_time"] is None
    assert metadata["checkpoint.funding.C00USDT"]["complete"] is False
