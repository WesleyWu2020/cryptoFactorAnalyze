from __future__ import annotations

import pandas as pd
import pytest

from Genetic_Algorithm.config import STAGES, Stage
from tests.factor_common.conftest import write_h5_fixture


@pytest.fixture
def stage_fixture(tmp_path):
    return write_h5_fixture(
        tmp_path / "stage.h5",
        funding_schedule=pd.DataFrame(),
    )


def test_load_stage_is_cut_off_and_preserves_daily_axes(stage_fixture):
    from Genetic_Algorithm.data import load_stage

    stage = Stage("fixture", "2024-01-03", "2024-01-05")
    loaded = load_stage(stage_fixture, stage, warmup_days=2, fields=["close", "volume"])

    expected_history = pd.date_range("2024-01-01", "2024-01-05", freq="D")
    expected_stage = pd.date_range("2024-01-03", "2024-01-05", freq="D")
    assert loaded.stage == stage
    assert set(loaded.features) == {"close", "volume"}
    assert all(frame.index.equals(expected_history) for frame in loaded.features.values())
    assert loaded.eligible.index.equals(expected_history)
    assert loaded.quality_eligible.index.equals(expected_stage)
    assert loaded.opens.index.equals(expected_stage)
    assert all("future_ret" not in frame.columns for frame in loaded.features.values())
    assert loaded.audit["provider_cutoff"] == "2024-01-05"


def test_load_stage_quality_requires_current_membership_and_complete_non_placeholder_kline(
    stage_fixture,
):
    from Genetic_Algorithm.data import load_stage
    from data.crypto_quant.store import CryptoQuantStore

    panel = CryptoQuantStore(stage_fixture).read("research_panel_daily")
    panel = panel.drop(columns=["has_complete_kline"])
    with pd.HDFStore(stage_fixture, mode="a") as hdf:
        hdf.put(
            "research_panel_daily",
            panel,
            format="table",
            data_columns=["date", "binance_symbol"],
            index=False,
        )

    loaded = load_stage(
        stage_fixture,
        Stage("fixture", "2024-01-03", "2024-01-05"),
        warmup_days=0,
        fields=["close"],
    )

    assert loaded.eligible.loc[pd.Timestamp("2024-01-04"), "BUSDT"]
    assert not loaded.quality_eligible.loc[pd.Timestamp("2024-01-04"), "BUSDT"]
    assert loaded.audit["quality_counts"]["ineligible_placeholder"] >= 1
    assert loaded.audit["quality_counts"]["unknown"] >= 1


def test_load_stage_rejects_missing_membership_axis(stage_fixture, monkeypatch):
    from factor_common.data_provider import DataProvider
    from Genetic_Algorithm import data as data_module

    original = DataProvider.get_universe

    def missing_column(provider, *, start, end):
        return original(provider, start=start, end=end).drop(columns=["AUSDT"])

    monkeypatch.setattr(data_module.DataProvider, "get_universe", missing_column)
    with pytest.raises(ValueError, match="eligible membership axes do not match"):
        data_module.load_stage(
            stage_fixture,
            Stage("fixture", "2024-01-03", "2024-01-05"),
            warmup_days=0,
            fields=["close"],
        )


def test_load_stage_rejects_duplicate_membership_axis(stage_fixture, monkeypatch):
    from factor_common.data_provider import DataProvider
    from Genetic_Algorithm import data as data_module

    original = DataProvider.get_universe

    def duplicate_row(provider, *, start, end):
        frame = original(provider, start=start, end=end)
        return pd.concat([frame, frame.iloc[[0]]])

    monkeypatch.setattr(data_module.DataProvider, "get_universe", duplicate_row)
    with pytest.raises(ValueError, match="eligible membership axes contain duplicate"):
        data_module.load_stage(
            stage_fixture,
            Stage("fixture", "2024-01-03", "2024-01-05"),
            warmup_days=0,
            fields=["close"],
        )


def test_load_stage_full_and_cutoff_inputs_have_identical_training_rows(tmp_path):
    from Genetic_Algorithm.data import load_stage
    from tests.factor_common.conftest import write_h5_fixture

    full = write_h5_fixture(tmp_path / "full.h5", funding_schedule=pd.DataFrame())
    cutoff = write_h5_fixture(
        tmp_path / "cutoff.h5",
        calendar=pd.date_range("2024-01-01", "2024-01-05", freq="D"),
        funding_schedule=pd.DataFrame(),
    )
    stage = Stage("fixture", "2024-01-03", "2024-01-05")
    full_data = load_stage(full, stage, warmup_days=2, fields=["close", "volume"])
    cutoff_data = load_stage(cutoff, stage, warmup_days=2, fields=["close", "volume"])

    for field in full_data.features:
        diff = (full_data.features[field] - cutoff_data.features[field]).abs().to_numpy()
        assert float(pd.DataFrame(diff).max().max()) == 0.0
    pd.testing.assert_frame_equal(full_data.eligible, cutoff_data.eligible)
    assert full_data.audit["provider_cutoff"] == cutoff_data.audit["provider_cutoff"]


def test_warmup_availability_counts_rows_with_data(tmp_path):
    from Genetic_Algorithm.data import load_stage

    calendar = pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-04"])
    path = write_h5_fixture(
        tmp_path / "gapped.h5", calendar=calendar, funding_schedule=pd.DataFrame()
    )
    loaded = load_stage(
        path,
        Stage("fixture", "2024-01-01", "2024-01-04"),
        warmup_days=0,
        fields=["close"],
    )

    assert loaded.audit["warmup_availability"] == {
        "requested_days": 0,
        "available_days": 3,
        "history_start": "2024-01-01",
    }


def test_training_audit_is_retained_without_loading_another_stage(stage_fixture, tmp_path):
    from Genetic_Algorithm.data import load_stage, write_training_audit

    loaded = load_stage(
        stage_fixture,
        STAGES["train"],
        warmup_days=0,
        fields=["close"],
    )
    output = tmp_path / "run" / "audit_train.json"
    write_training_audit(output, loaded)

    import json

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stage"]["name"] == "train"
    assert payload["stage"]["end"] == "2024-12-31"
    assert payload["training_only"] is True
    assert payload["fingerprint"] == loaded.fingerprint


def test_run_training_audit_loads_and_retains_only_train(stage_fixture, tmp_path):
    from Genetic_Algorithm.data import run_training_audit

    output = tmp_path / "run" / "audit_train.json"
    run_training_audit(
        stage_fixture,
        output,
        stage=STAGES["train"],
        warmup_days=0,
        fields=["close"],
    )

    import json

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["training_only"] is True
    assert payload["stage"]["name"] == "train"


def test_search_entry_audits_before_callback_and_stops_on_audit_failure(
    stage_fixture, tmp_path, monkeypatch
):
    from Genetic_Algorithm import search as search_module

    output = tmp_path / "run" / "audit_train.json"
    callback_calls = []

    def search_stage(audit_path):
        callback_calls.append(audit_path)
        assert audit_path == output
        assert audit_path.exists()
        return "search-result"

    result = search_module.run_search(
        stage_fixture,
        output,
        stage=STAGES["train"],
        warmup_days=0,
        fields=["close"],
        search_stage=search_stage,
    )

    assert result == "search-result"
    assert callback_calls == [output]

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit failed")

    monkeypatch.setattr(search_module, "run_training_audit", fail_audit)
    callback_calls.clear()
    with pytest.raises(RuntimeError, match="audit failed"):
        search_module.run_search(
            stage_fixture,
            tmp_path / "failed" / "audit_train.json",
            stage=STAGES["train"],
            warmup_days=0,
            fields=["close"],
            search_stage=lambda audit_path: callback_calls.append(audit_path),
        )
    assert callback_calls == []


def test_load_stage_rejects_forged_train_bounds(stage_fixture):
    from Genetic_Algorithm import data as data_module

    with pytest.raises(ValueError, match="frozen train stage"):
        data_module.load_stage(
            stage_fixture,
            Stage("train", "2026-01-01", "2026-09-01"),
            warmup_days=0,
            fields=["close"],
        )


def test_run_training_audit_rejects_forged_train_bounds(stage_fixture, tmp_path):
    from Genetic_Algorithm import data as data_module

    with pytest.raises(ValueError, match="frozen train stage"):
        data_module.run_training_audit(
            stage_fixture,
            tmp_path / "audit.json",
            stage=Stage("train", "2026-01-01", "2026-09-01"),
            warmup_days=0,
            fields=["close"],
        )


def test_stage_fingerprint_changes_when_stage_identity_changes(stage_fixture):
    from Genetic_Algorithm import data as data_module

    fixture = data_module.load_stage(
        stage_fixture,
        STAGES["train"],
        warmup_days=0,
        fields=["close"],
    )
    other_stage = data_module._fingerprint(
        fixture.features,
        fixture.eligible,
        fixture.quality_eligible,
        fixture.opens,
        stage=Stage("fixture", "2024-01-04", "2024-01-05"),
        code_fingerprint=fixture.audit["provenance"]["code_fingerprint"],
    )

    assert other_stage != fixture.fingerprint


def test_stage_fingerprint_changes_when_code_identity_changes(stage_fixture, monkeypatch):
    from Genetic_Algorithm import data as data_module

    loaded = data_module.load_stage(
        stage_fixture,
        Stage("fixture", "2024-01-03", "2024-01-05"),
        warmup_days=0,
        fields=["close"],
    )
    monkeypatch.setattr(data_module, "_code_fingerprint", lambda: "changed-code")
    changed = data_module.load_stage(
        stage_fixture,
        Stage("fixture", "2024-01-03", "2024-01-05"),
        warmup_days=0,
        fields=["close"],
    )

    assert changed.fingerprint != loaded.fingerprint


def test_stage_fingerprint_declares_all_stage_loading_sources():
    from Genetic_Algorithm import data as data_module

    expected = {
        "Genetic_Algorithm/data.py",
        "factor_common/data_provider.py",
        "data/crypto_quant/reader.py",
        "data/crypto_quant/store.py",
        "data/crypto_quant/panel.py",
        "data/crypto_quant/schemas.py",
    }

    assert expected.issubset(set(data_module._CODE_FINGERPRINT_SOURCES))
    assert data_module._FINGERPRINT_VERSION >= 3
    assert data_module._OPERATOR_VERSION
