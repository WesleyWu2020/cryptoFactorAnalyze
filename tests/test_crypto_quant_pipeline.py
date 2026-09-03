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
    root.mkdir(exist_ok=True)
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
