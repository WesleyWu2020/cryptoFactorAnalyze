import hashlib
from datetime import date

import pandas as pd
import pytest

from data.crypto_quant.schemas import TABLE_SPECS, normalize_table
from data.crypto_quant.store import (
    CryptoQuantStore,
    StagingMismatchError,
    StoreLockedError,
    single_writer_lock,
    staged_store,
)


EXPECTED_KEYS = {
    "cmc100_daily": ("date",),
    "cmc100_constituents": ("date", "cmc_id"),
    "futures_contracts": ("cmc_id", "valid_from"),
    "klines_daily": ("date", "symbol"),
    "funding_events": ("funding_time", "symbol", "rate_type"),
    "universe_monthly": ("effective_date", "binance_symbol"),
    "research_panel_daily": ("date", "binance_symbol"),
}

EXPECTED_COLUMNS = {
    "cmc100_daily": "date index_value source_update_time fetched_at_utc",
    "cmc100_constituents": "date cmc_id symbol name weight",
    "futures_contracts": "cmc_id cmc_symbol binance_symbol base_asset quote_asset contract_type onboard_date status mapping_source valid_from valid_to",
    "klines_daily": "date symbol open_time close_time open high low close volume quote_volume trade_count taker_buy_base_volume taker_buy_quote_volume",
    "funding_events": "funding_time symbol funding_rate mark_price rate_type",
    "universe_monthly": "decision_date effective_date effective_end_date cmc_id cmc_symbol binance_symbol market_cap_rank cmc_weight",
    "research_panel_daily": "date binance_symbol open_time close_time open high low close volume quote_volume trade_count taker_buy_base_volume taker_buy_quote_volume decision_date universe_effective_date market_cap_rank cmc_weight_at_decision funding_rate_sum funding_rate_mean funding_rate_last funding_event_count has_complete_kline has_complete_funding",
}


def _cmc_row(value=100.0, fetched="2026-09-03T00:00:00Z"):
    return pd.DataFrame([{
        "date": date(2026, 9, 1), "index_value": value,
        "source_update_time": pd.Timestamp("2026-09-01", tz="UTC"),
        "fetched_at_utc": pd.Timestamp(fetched),
    }])


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_table_specs_have_required_keys_columns_and_version():
    assert {name: spec.key for name, spec in TABLE_SPECS.items()} == EXPECTED_KEYS
    assert {name: tuple(spec.columns) for name, spec in TABLE_SPECS.items()} == {
        name: tuple(columns.split()) for name, columns in EXPECTED_COLUMNS.items()
    }
    assert all(spec.schema_version == 1 for spec in TABLE_SPECS.values())


def test_normalize_table_requires_columns_and_sorts_by_primary_key():
    frame = _cmc_row(1).copy()
    later = _cmc_row(2)
    later["date"] = date(2026, 9, 2)
    frame = pd.concat([frame, later], ignore_index=True)
    normalized = normalize_table("cmc100_daily", frame.iloc[::-1])
    assert list(normalized.columns) == list(TABLE_SPECS["cmc100_daily"].columns)
    assert normalized.iloc[0]["date"] == pd.Timestamp("2026-09-01")
    with pytest.raises(ValueError, match="missing columns"):
        normalize_table("cmc100_daily", frame.drop(columns=["index_value"]))


def test_normalize_table_rejects_null_and_conflicting_duplicate_keys():
    null_key = _cmc_row().assign(date=pd.NaT)
    with pytest.raises(ValueError, match="null primary-key"):
        normalize_table("cmc100_daily", null_key)
    conflict = pd.concat([_cmc_row(1), _cmc_row(2)], ignore_index=True)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        normalize_table("cmc100_daily", conflict)


def test_upsert_is_idempotent_and_new_rows_win(tmp_path):
    store = CryptoQuantStore(tmp_path / "active.h5")
    store.upsert("cmc100_daily", _cmc_row(100))
    store.upsert("cmc100_daily", _cmc_row(100, "2026-09-04T00:00:00Z"))
    assert store.read("cmc100_daily").iloc[0]["fetched_at_utc"] == pd.Timestamp("2026-09-03", tz="UTC")
    store.upsert("cmc100_daily", _cmc_row(101, "2026-09-04T00:00:00Z"))
    row = store.read("cmc100_daily").iloc[0]
    assert len(store.read("cmc100_daily")) == 1
    assert row["index_value"] == 101
    assert row["fetched_at_utc"] == pd.Timestamp("2026-09-04", tz="UTC")


def test_metadata_round_trips_json_values(tmp_path):
    store = CryptoQuantStore(tmp_path / "active.h5")
    value = {"cmc100_daily": {"rows": 3, "dates": ["2026-09-01"]}}
    store.write_metadata(value)
    assert store.read_metadata() == value


def test_second_writer_fails_without_waiting(tmp_path):
    lock = tmp_path / "store.lock"
    with single_writer_lock(lock):
        with pytest.raises(StoreLockedError):
            with single_writer_lock(lock):
                pass


def test_failed_staging_context_keeps_active_bytes_unchanged(tmp_path):
    active, staging, lock = tmp_path / "active.h5", tmp_path / "stage.h5", tmp_path / "store.lock"
    CryptoQuantStore(active).replace("cmc100_daily", _cmc_row())
    before = _sha256(active)
    with pytest.raises(RuntimeError, match="forced failure"):
        with staged_store(active, staging, "fp", lock_path=lock):
            CryptoQuantStore(staging).replace("cmc100_daily", _cmc_row(999))
            raise RuntimeError("forced failure")
    assert _sha256(active) == before


def test_successful_publish_replaces_active_atomically(tmp_path):
    active, staging, lock = tmp_path / "active.h5", tmp_path / "stage.h5", tmp_path / "store.lock"
    CryptoQuantStore(active).replace("cmc100_daily", _cmc_row(100))
    with staged_store(active, staging, "fp", lock_path=lock) as store:
        store.upsert("cmc100_daily", _cmc_row(101))
    assert not staging.exists()
    assert CryptoQuantStore(active).read("cmc100_daily").iloc[0]["index_value"] == 101


def test_resume_reuses_matching_staging_fingerprint(tmp_path):
    active, staging, lock = tmp_path / "active.h5", tmp_path / "stage.h5", tmp_path / "store.lock"
    with pytest.raises(RuntimeError):
        with staged_store(active, staging, "fp", lock_path=lock) as store:
            store.write_metadata({"checkpoint": {"rows": 7}})
            raise RuntimeError("stop")
    with staged_store(active, staging, "fp", lock_path=lock) as store:
        assert store.read_metadata()["checkpoint"] == {"rows": 7}


def test_mismatched_staging_fingerprint_requires_reset(tmp_path):
    active, staging, lock = tmp_path / "active.h5", tmp_path / "stage.h5", tmp_path / "store.lock"
    with pytest.raises(RuntimeError):
        with staged_store(active, staging, "old", lock_path=lock):
            raise RuntimeError("stop")
    with pytest.raises(StagingMismatchError):
        with staged_store(active, staging, "new", lock_path=lock):
            pass
    with staged_store(active, staging, "new", lock_path=lock, reset_staging=True) as store:
        assert store.read_metadata() == {"run_fingerprint": "new"}
