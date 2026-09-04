from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pytest

from data.crypto_quant.store import CryptoQuantStore
from data.crypto_quant.validation import ValidationIssue, ValidationReport
from data import update_crypto_quant


class FakePipeline:
    def __init__(self, report=None):
        self.calls = []
        self.report = report

    def update(self, as_of, *, reset_staging=False):
        self.calls.append(("update", as_of, reset_staging))
        return type("Summary", (), {"mode": "update", "as_of_utc": as_of, "row_counts": {"research_panel_daily": 3}, "published_path": Path("store.h5"), "last_complete_panel_date": None})()

    def backfill(self, as_of, *, reset_staging=False):
        self.calls.append(("backfill", as_of, reset_staging))
        return type("Summary", (), {"mode": "backfill", "as_of_utc": as_of, "row_counts": {}, "published_path": Path("store.h5"), "last_complete_panel_date": None})()


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


def test_as_of_rejects_timezone_free_timestamp():
    with pytest.raises(SystemExit) as exc_info:
        update_crypto_quant.main(["update", "--as-of", "2026-09-03T00:20:00"])
    assert exc_info.value.code == 2
