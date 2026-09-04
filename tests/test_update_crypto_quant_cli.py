from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from data.crypto_quant.store import CryptoQuantStore
from data.crypto_quant.validation import ValidationIssue, ValidationReport
from data import update_crypto_quant


class FakePipeline:
    def __init__(self, report=None):
        self.calls = []
        self.report = report
        self.closed = 0

    def close(self):
        self.closed += 1

    def update(self, as_of, *, reset_staging=False):
        self.calls.append(("update", as_of, reset_staging))
        return type("Summary", (), {"mode": "update", "as_of_utc": as_of, "row_counts": {"research_panel_daily": 3}, "published_path": Path("store.h5"), "last_complete_panel_date": None})()

    def backfill(self, as_of, *, reset_staging=False):
        self.calls.append(("backfill", as_of, reset_staging))
        return type("Summary", (), {"mode": "backfill", "as_of_utc": as_of, "row_counts": {}, "published_path": Path("store.h5"), "last_complete_panel_date": None})()

    def rebuild_derived(self, as_of, *, reset_staging=False):
        self.calls.append(("rebuild_derived", as_of, reset_staging))
        return type("Summary", (), {"mode": "rebuild", "as_of_utc": as_of, "row_counts": {}, "published_path": Path("store.h5"), "last_complete_panel_date": None})()


def test_update_command_uses_explicit_utc_as_of(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(update_crypto_quant, "build_pipeline", lambda *args, **kwargs: fake)

    assert update_crypto_quant.main(["update", "--as-of", "2026-09-03T00:20:00Z"]) == 0

    _, as_of, reset_staging = fake.calls[0]
    assert as_of.tzinfo == timezone.utc
    assert as_of.isoformat() == "2026-09-03T00:20:00+00:00"
    assert reset_staging is False


def test_backfill_command_forwards_reset_staging(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(update_crypto_quant, "build_pipeline", lambda *args, **kwargs: fake)

    assert update_crypto_quant.main(["backfill", "--reset-staging"]) == 0
    assert fake.calls[0][0] == "backfill"
    assert fake.calls[0][2] is True


def test_update_command_supports_explicit_cmc_keyless_mode(monkeypatch):
    fake = FakePipeline()
    build_calls = []

    def build(*args, **kwargs):
        build_calls.append((args, kwargs))
        return fake

    monkeypatch.setattr(update_crypto_quant, "build_pipeline", build)

    assert update_crypto_quant.main(["update", "--cmc-keyless"]) == 0
    assert build_calls[0][1] == {"cmc_keyless": True}


def test_rebuild_derived_dispatches_to_pipeline(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(update_crypto_quant, "build_pipeline", lambda *args, **kwargs: fake)

    assert update_crypto_quant.main(["rebuild-derived", "--as-of", "2026-09-03T00:20:00Z"]) == 0
    assert fake.calls[0][0] == "rebuild_derived"
    assert fake.closed == 1


def test_default_as_of_uses_injected_utc_clock(monkeypatch):
    fake = FakePipeline()
    fixed_now = datetime(2026, 9, 4, 1, 2, 3, 456789, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(update_crypto_quant, "datetime", FixedDateTime)
    monkeypatch.setattr(update_crypto_quant, "build_pipeline", lambda *args, **kwargs: fake)

    assert update_crypto_quant.main(["update"]) == 0
    as_of = fake.calls[0][1]
    assert as_of == fixed_now


@pytest.mark.parametrize("component", ["JsonHttpClient", "_CmcSource", "_BinanceSource", "CryptoQuantPipeline"])
def test_build_pipeline_closes_session_when_construction_fails(monkeypatch, tmp_path, component):
    class FakeSession:
        def __init__(self):
            self.headers = {}
            self.closed = 0

        def close(self):
            self.closed += 1

    session = FakeSession()

    def fail(*args, **kwargs):
        raise RuntimeError(f"{component} failed")

    monkeypatch.setattr(update_crypto_quant.requests, "Session", lambda: session)
    monkeypatch.setattr(update_crypto_quant, component, fail)

    with pytest.raises(RuntimeError, match="failed"):
        update_crypto_quant.build_pipeline(tmp_path / "store.h5")
    assert session.closed == 1


def test_runtime_exception_returns_one_and_closes_pipeline(monkeypatch, capsys):
    fake = FakePipeline()

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    fake.update = fail
    monkeypatch.setattr(update_crypto_quant, "build_pipeline", lambda *args, **kwargs: fake)

    assert update_crypto_quant.main(["update"]) == 1
    assert fake.closed == 1
    assert "RuntimeError" in capsys.readouterr().err


def test_build_pipeline_exposes_explicit_session_close(monkeypatch, tmp_path):
    class FakeSession:
        def __init__(self):
            self.headers = {}
            self.closed = 0

        def close(self):
            self.closed += 1

    session = FakeSession()
    class FakePipelineObject:
        pass

    pipeline = FakePipelineObject()
    monkeypatch.setattr(update_crypto_quant.requests, "Session", lambda: session)
    monkeypatch.setattr(update_crypto_quant, "CryptoQuantPipeline", lambda *args: pipeline)

    built = update_crypto_quant.build_pipeline(tmp_path / "store.h5")
    assert built is pipeline
    built.close()
    assert session.closed == 1


def test_build_pipeline_keyless_mode_does_not_send_env_api_key(
    monkeypatch, tmp_path
):
    class FakeSession:
        def __init__(self):
            self.headers = {}

        def close(self):
            pass

    session = FakeSession()
    class FakePipelineObject:
        pass

    pipeline = FakePipelineObject()
    monkeypatch.setenv("CMC_API_KEY", "invalid-key-from-env")
    monkeypatch.setattr(update_crypto_quant.requests, "Session", lambda: session)
    monkeypatch.setattr(update_crypto_quant, "CryptoQuantPipeline", lambda *args: pipeline)

    built = update_crypto_quant.build_pipeline(tmp_path / "store.h5", cmc_keyless=True)

    assert built is pipeline
    assert session.headers == {}


def test_validate_command_exits_zero_for_valid_store(monkeypatch, tmp_path):
    monkeypatch.setattr(update_crypto_quant, "validate_store", lambda path: ValidationReport(()))
    assert update_crypto_quant.main(["validate", "--store", str(tmp_path / "store.h5")]) == 0


def test_validate_command_exits_two_for_invalid_store(monkeypatch, tmp_path):
    report = ValidationReport((ValidationIssue("error", "bad", "invalid"),))
    monkeypatch.setattr(update_crypto_quant, "validate_store", lambda path: report)
    assert update_crypto_quant.main(["validate", "--store", str(tmp_path / "store.h5")]) == 2


def test_inspect_prints_row_counts_ranges_and_last_complete_date(capsys, tmp_path):
    store_path = tmp_path / "store.h5"
    store = CryptoQuantStore(store_path)
    store.write_metadata(
        {
            "table_row_counts": {"research_panel_daily": 42},
            "table_date_ranges": {"research_panel_daily": {"min": "2026-08-01", "max": "2026-09-02"}},
            "last_complete_panel_date": "2026-09-02",
        }
    )

    assert update_crypto_quant.main(["inspect", "--store", str(store_path)]) == 0
    output = capsys.readouterr().out
    assert "row_counts" in output and "42" in output
    assert "2026-08-01" in output and "2026-09-02" in output
    assert "last_complete_date" in output


def test_inspect_missing_store_returns_one_without_creating_file(tmp_path):
    store_path = tmp_path / "missing.h5"

    assert update_crypto_quant.main(["inspect", "--store", str(store_path)]) == 1
    assert not store_path.exists()


def test_inspect_corrupt_store_returns_one_without_replacing_file(tmp_path):
    store_path = tmp_path / "corrupt.h5"
    original = b"not an HDF5 file"
    store_path.write_bytes(original)

    assert update_crypto_quant.main(["inspect", "--store", str(store_path)]) == 1
    assert store_path.read_bytes() == original


def test_as_of_rejects_timezone_free_timestamp():
    with pytest.raises(SystemExit) as exc_info:
        update_crypto_quant.main(["update", "--as-of", "2026-09-03T00:20:00"])
    assert exc_info.value.code == 2
