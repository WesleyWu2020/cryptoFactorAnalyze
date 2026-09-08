from __future__ import annotations

import hashlib

import pandas as pd

from Genetic_Algorithm.config import STAGES
from Genetic_Algorithm.data import load_stage


def test_future_row_mutation_does_not_change_train_stage_fingerprint(gp_h5):
    before = load_stage(gp_h5, STAGES["train"], 180, ["close"])
    # The point-in-time train loader must never include validation/test values.
    from data.crypto_quant.store import CryptoQuantStore
    store = CryptoQuantStore(gp_h5)
    table = store.read("klines_daily")
    table.loc[table["date"] >= pd.Timestamp("2025-01-01"), "close"] += 100_000
    store.replace("klines_daily", table)
    after = load_stage(gp_h5, STAGES["train"], 180, ["close"])
    assert before.fingerprint == after.fingerprint
    assert before.features["close"].equals(after.features["close"])
    assert "crypto_quant.h5" not in str(before.audit["provenance"]).lower()


def test_stage_provenance_is_content_scoped_not_full_file_hash(gp_h5):
    loaded = load_stage(gp_h5, STAGES["validation"], 180, ["close"])
    serialized = repr(loaded.audit["provenance"])
    assert hashlib.sha256(gp_h5.read_bytes()).hexdigest() not in serialized


def test_train_loader_records_hdf_upper_bounds_before_validation_rows(gp_h5, monkeypatch):
    """A read spy proves the bounded query, rather than trusting equal values."""
    from data.crypto_quant.store import CryptoQuantStore

    calls = []
    original = CryptoQuantStore.read

    def recording_read(store, name, where=None):
        calls.append((name, where))
        return original(store, name, where)

    monkeypatch.setattr(CryptoQuantStore, "read", recording_read)
    load_stage(gp_h5, STAGES["train"], 180, ["close"])
    bounded = [(name, where) for name, where in calls if where]
    assert bounded
    assert any(name == "klines_daily" and "2024-12-31" in where for name, where in bounded)
    assert any(name == "universe_monthly" and "2024-12-31" in where for name, where in bounded)
    # HDF range predicates may contain a lower history bound, but their
    # upper date literals cannot exceed the frozen training end.
    import re
    dates = [pd.Timestamp(value) for _, where in bounded for value in re.findall(r"\d{4}-\d{2}-\d{2}", where)]
    # Funding event ranges are half-open and therefore spell the next midnight
    # as an exclusive bound; it is not a readable future daily row.
    assert dates and max(dates) <= STAGES["train"].end + pd.Timedelta(days=1)
