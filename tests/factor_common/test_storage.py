"""Tests for versioned factor value and evaluation artifact persistence."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factor_common.profiles import BacktestProfile
from factor_common.storage import (
    ConcurrentSourceChangeError,
    FactorStorage,
    snapshot_source_stats,
)


def _matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {"AUSDT": [0.2, 0.3], "BUSDT": [0.1, None]},
        index=pd.date_range("2024-01-01", periods=2, name="date"),
    )


def _metadata(**overrides) -> dict:
    metadata = {
        "source_sha256": "abc",
        "settings": {"window": 5},
        "requested_start": "2024-01-01",
        "requested_end": "2024-01-02",
        "loaded_start": "2023-12-31",
        "loaded_end": "2024-01-02",
        "input_hashes": {"close": "h-close", "membership": "h-member"},
    }
    metadata.update(overrides)
    return metadata


def _evaluation(**overrides) -> dict:
    result = {
        "status": "complete",
        "profile": BacktestProfile(),
        "evaluation_inputs": {
            "execution_tail_prices": "p1",
            "funding_events": "f1",
            "coverage_evidence": "c1",
        },
        "metrics": {"sharpe": 1.0},
    }
    result.update(overrides)
    return result


def test_factor_round_trip(tmp_path):
    import pandas as pd
    from factor_common.storage import FactorStorage
    matrix = pd.DataFrame({"AUSDT": [0.2, 0.3]},
                          index=pd.date_range("2024-01-01", periods=2, name="date"))
    store = FactorStorage(tmp_path)
    saved = store.save_value("momentum", matrix, {"source_sha256": "abc", "window": 1})
    pd.testing.assert_frame_equal(store.get_value("momentum", run_id=saved["run_id"]), matrix)


def test_flat_factor_cache_uses_one_stable_parquet_and_meta(tmp_path):
    matrix = _matrix()
    store = FactorStorage(tmp_path, flat=True)
    metadata = _metadata()
    saved = store.save_value("Volume_Stability_Factor", matrix, metadata)
    assert saved["factor_path"] == tmp_path / "volume_stability_factor.parquet"
    assert saved["metadata_path"] == tmp_path / "volume_stability_factor.meta.json"
    assert not (tmp_path / "Volume_Stability_Factor").exists()
    cached = store.load_cached_value(
        "Volume_Stability_Factor",
        source_sha256="abc",
        settings={"window": 5},
        requested_start="2024-01-01",
        requested_end="2024-01-02",
    )
    assert cached is not None
    pd.testing.assert_frame_equal(cached[0], matrix)


def test_flat_factor_cache_invalid_meta_is_a_miss(tmp_path):
    store = FactorStorage(tmp_path, flat=True)
    store.save_value("momentum", _matrix(), _metadata())
    assert store.load_cached_value(
        "momentum",
        source_sha256="changed",
        settings={"window": 5},
        requested_start="2024-01-01",
        requested_end="2024-01-02",
    ) is None


def test_flat_factor_cache_pipeline_fingerprint_mismatch_is_a_miss(tmp_path):
    store = FactorStorage(tmp_path, flat=True)
    store.save_value(
        "momentum", _matrix(), {**_metadata(), "pipeline_fingerprint": "old-code"}
    )
    query = {
        "source_sha256": "abc",
        "settings": {"window": 5},
        "requested_start": "2024-01-01",
        "requested_end": "2024-01-02",
    }
    assert store.load_cached_value(
        "momentum", pipeline_fingerprint="new-code", **query
    ) is None
    # Caches written before the fingerprint existed are also rejected.
    store.save_value("momentum", _matrix(), _metadata())
    assert store.load_cached_value(
        "momentum", pipeline_fingerprint="new-code", **query
    ) is None
    assert store.load_cached_value(
        "momentum", pipeline_fingerprint=None, **query
    ) is not None


def test_flat_evaluation_keeps_only_current_result_without_index_directory(tmp_path):
    ledger = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2), "equity": [1.0, 1.1]})
    store = FactorStorage(tmp_path, flat=True)
    saved = store.save_value("momentum", _matrix(), _metadata())
    first = store.save_evaluation(
        "momentum", saved["run_id"], _evaluation(ledger=ledger)
    )

    assert (tmp_path / "momentum.evaluation.json").is_file()
    assert (tmp_path / "momentum.evaluation.ledger.parquet").is_file()
    assert not (tmp_path / "momentum.evaluations").exists()
    assert not (tmp_path / "momentum.evaluations.json").exists()

    second = store.save_evaluation(
        "momentum",
        saved["run_id"],
        _evaluation(profile=replace(BacktestProfile(), fee_rate=0.0001)),
    )
    assert second["evaluation_id"] != first["evaluation_id"]
    assert not (tmp_path / "momentum.evaluation.ledger.parquet").exists()
    loaded = store.load_evaluation("momentum")
    assert loaded["evaluation_id"] == second["evaluation_id"]
    with pytest.raises(FileNotFoundError):
        store.load_evaluation("momentum", evaluation_id=first["evaluation_id"])


def test_round_trip_preserves_nan_and_empty_rows(tmp_path):
    matrix = pd.DataFrame(
        {"AUSDT": [0.2, None, 0.4], "BUSDT": [None, None, np.inf]},
        index=pd.date_range("2024-01-01", periods=3, name="date"),
    )
    store = FactorStorage(tmp_path)
    saved = store.save_value("momentum", matrix, _metadata())
    expected = matrix.replace([np.inf, -np.inf], np.nan)
    pd.testing.assert_frame_equal(
        store.get_value("momentum", run_id=saved["run_id"]), expected
    )


def test_round_trip_all_missing_matrix_keeps_axes(tmp_path):
    matrix = pd.DataFrame(
        {"AUSDT": [None, None], "BUSDT": [None, None]},
        index=pd.date_range("2024-01-01", periods=2, name="date"),
    ).astype("float64")
    store = FactorStorage(tmp_path)
    saved = store.save_value("momentum", matrix, _metadata())
    pd.testing.assert_frame_equal(
        store.get_value("momentum", run_id=saved["run_id"]), matrix
    )


def test_matrix_axes_are_normalized_before_saving(tmp_path):
    matrix = pd.DataFrame(
        {"BUSDT": [2.0, 4.0], "AUSDT": [1.0, 3.0]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-01"], tz="UTC"),
    )
    store = FactorStorage(tmp_path)
    saved = store.save_value("momentum", matrix, _metadata())
    expected = pd.DataFrame(
        {"AUSDT": [3.0, 1.0], "BUSDT": [4.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, name="date"),
    )
    pd.testing.assert_frame_equal(
        store.get_value("momentum", run_id=saved["run_id"]), expected
    )


def test_stored_table_has_unique_sorted_keys(tmp_path):
    store = FactorStorage(tmp_path)
    saved = store.save_value("momentum", _matrix(), _metadata())
    table = pd.read_parquet(saved["factor_path"])
    assert list(table.columns) == ["date", "instrument", "factor"]
    assert not table.duplicated(["date", "instrument"]).any()
    assert table.equals(
        table.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    )
    duplicated = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    duplicated.to_parquet(saved["factor_path"], index=False)
    with pytest.raises(ValueError, match="duplicate"):
        store.get_value("momentum", run_id=saved["run_id"])


def test_parameter_and_data_fingerprint_separation(tmp_path):
    store = FactorStorage(tmp_path)
    matrix = _matrix()
    base = store.save_value("momentum", matrix, _metadata())
    repeat = store.save_value("momentum", matrix, _metadata())
    assert repeat["run_id"] == base["run_id"]
    other_params = store.save_value("momentum", matrix, _metadata(settings={"window": 6}))
    assert other_params["run_id"] != base["run_id"]
    other_data = store.save_value(
        "momentum",
        matrix,
        _metadata(input_hashes={"close": "h-close-2", "membership": "h-member"}),
    )
    assert other_data["run_id"] != base["run_id"]
    assert other_data["run_id"] != other_params["run_id"]
    pd.testing.assert_frame_equal(
        store.get_value("momentum", run_id=other_params["run_id"]),
        store.get_value("momentum", run_id=base["run_id"]),
    )


def test_same_inputs_with_different_values_are_rejected(tmp_path):
    store = FactorStorage(tmp_path)
    store.save_value("momentum", _matrix(), _metadata())
    changed = _matrix().assign(AUSDT=[9.9, 9.9])
    with pytest.raises(ValueError, match="different factor values"):
        store.save_value("momentum", changed, _metadata())


def test_failed_write_preserves_success_pointer(tmp_path):
    store = FactorStorage(tmp_path)
    saved = store.save_value("momentum", _matrix(), _metadata())
    with pytest.raises(TypeError):
        store.save_value("momentum", _matrix(), _metadata(raw_output=object()))
    assert store.get_value("momentum") is not None
    pointer = json.loads((tmp_path / "momentum" / "latest_run.json").read_text())
    assert pointer["run_id"] == saved["run_id"]
    leftovers = [p for p in (tmp_path / "momentum").iterdir() if ".tmp-" in p.name]
    assert leftovers == []


def test_concurrent_source_change_aborts_save(tmp_path):
    source = tmp_path / "snapshot.h5"
    source.write_bytes(b"version-1")
    store = FactorStorage(tmp_path / "store")
    metadata = _metadata(source_stat_before=snapshot_source_stats([source]))
    source.write_bytes(b"version-2-with-different-size")
    with pytest.raises(ConcurrentSourceChangeError):
        store.save_value("momentum", _matrix(), metadata)
    with pytest.raises(FileNotFoundError):
        store.get_value("momentum")


def test_explicit_run_selection_and_latest_default(tmp_path):
    store = FactorStorage(tmp_path)
    first_matrix = _matrix()
    second_matrix = _matrix().assign(AUSDT=[0.5, 0.6])
    first = store.save_value("momentum", first_matrix, _metadata(settings={"window": 1}))
    second = store.save_value("momentum", second_matrix, _metadata(settings={"window": 2}))
    assert first["run_id"] != second["run_id"]
    pd.testing.assert_frame_equal(store.get_value("momentum"), second_matrix)
    pd.testing.assert_frame_equal(
        store.get_value("momentum", run_id=first["run_id"]), first_matrix
    )


def test_fee_only_change_creates_new_evaluation_under_same_run(tmp_path):
    store = FactorStorage(tmp_path)
    saved = store.save_value("momentum", _matrix(), _metadata())
    first = store.save_evaluation("momentum", saved["run_id"], _evaluation())
    cheaper = replace(BacktestProfile(), fee_rate=0.0001)
    second = store.save_evaluation(
        "momentum", saved["run_id"], _evaluation(profile=cheaper)
    )
    assert first["evaluation_id"] != second["evaluation_id"]
    loaded_first = store.load_evaluation(
        "momentum", run_id=saved["run_id"], evaluation_id=first["evaluation_id"]
    )
    assert loaded_first["profile"]["fee_rate"] == BacktestProfile().fee_rate
    loaded_default = store.load_evaluation("momentum")
    assert loaded_default["evaluation_id"] == second["evaluation_id"]
    assert loaded_default["profile"]["fee_rate"] == 0.0001
    pd.testing.assert_frame_equal(store.get_value("momentum"), _matrix())


def test_incomplete_evaluation_is_kept_but_not_promoted(tmp_path):
    store = FactorStorage(tmp_path)
    saved = store.save_value("momentum", _matrix(), _metadata())
    complete = store.save_evaluation("momentum", saved["run_id"], _evaluation())
    incomplete = store.save_evaluation(
        "momentum",
        saved["run_id"],
        _evaluation(
            status="incomplete",
            profile=replace(BacktestProfile(), fee_rate=0.0001),
            metrics={"sharpe": None},
        ),
    )
    assert incomplete["evaluation_id"] != complete["evaluation_id"]
    assert store.load_evaluation("momentum")["evaluation_id"] == complete["evaluation_id"]
    loaded = store.load_evaluation(
        "momentum", run_id=saved["run_id"], evaluation_id=incomplete["evaluation_id"]
    )
    assert loaded["status"] == "incomplete"
    pd.testing.assert_frame_equal(store.get_value("momentum"), _matrix())


def test_evaluation_tables_round_trip(tmp_path):
    ledger = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2),
            "equity": [1.0, 1.1],
        }
    )
    store = FactorStorage(tmp_path)
    saved = store.save_value("momentum", _matrix(), _metadata())
    evaluation = store.save_evaluation(
        "momentum", saved["run_id"], _evaluation(ledger=ledger)
    )
    loaded = store.load_evaluation(
        "momentum", run_id=saved["run_id"], evaluation_id=evaluation["evaluation_id"]
    )
    pd.testing.assert_frame_equal(loaded["ledger"], ledger)


def test_identical_evaluation_resave_is_idempotent(tmp_path):
    ledger = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2),
            "equity": [1.0, 1.1],
        }
    )
    store = FactorStorage(tmp_path)
    saved = store.save_value("momentum", _matrix(), _metadata())
    first = store.save_evaluation("momentum", saved["run_id"], _evaluation(ledger=ledger))
    second = store.save_evaluation(
        "momentum", saved["run_id"], _evaluation(ledger=ledger.copy())
    )
    assert second["evaluation_id"] == first["evaluation_id"]
    loaded = store.load_evaluation(
        "momentum", run_id=saved["run_id"], evaluation_id=first["evaluation_id"]
    )
    pd.testing.assert_frame_equal(loaded["ledger"], ledger)


def test_conflicting_evaluation_resave_is_rejected(tmp_path):
    store = FactorStorage(tmp_path)
    saved = store.save_value("momentum", _matrix(), _metadata())
    ledger = pd.DataFrame({"equity": [1.0, 1.1]})
    first = store.save_evaluation(
        "momentum", saved["run_id"], _evaluation(metrics={"sharpe": 1.0}, ledger=ledger)
    )
    with pytest.raises(ValueError, match="different content"):
        store.save_evaluation(
            "momentum",
            saved["run_id"],
            _evaluation(metrics={"sharpe": 2.0}, ledger=ledger),
        )
    changed_ledger = pd.DataFrame({"equity": [1.0, 9.9]})
    with pytest.raises(ValueError, match="different content"):
        store.save_evaluation(
            "momentum",
            saved["run_id"],
            _evaluation(metrics={"sharpe": 1.0}, ledger=changed_ledger),
        )
    loaded = store.load_evaluation(
        "momentum", run_id=saved["run_id"], evaluation_id=first["evaluation_id"]
    )
    assert loaded["metrics"]["sharpe"] == 1.0
    pd.testing.assert_frame_equal(loaded["ledger"], ledger)
    assert store.load_evaluation("momentum")["evaluation_id"] == first["evaluation_id"]


def test_nonfinite_scalars_become_json_null(tmp_path):
    store = FactorStorage(tmp_path)
    saved = store.save_value(
        "momentum",
        _matrix(),
        _metadata(diagnostics={"worst": float("nan"), "peak": float("inf")}),
    )
    raw = (tmp_path / "momentum" / saved["run_id"] / "metadata.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    payload = json.loads(raw)
    assert payload["metadata"]["diagnostics"] == {"worst": None, "peak": None}
    store.save_evaluation(
        "momentum", saved["run_id"], _evaluation(metrics={"sharpe": float("nan")})
    )
    assert store.load_evaluation("momentum")["metrics"]["sharpe"] is None


def test_missing_artifacts_raise_clear_errors(tmp_path):
    store = FactorStorage(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get_value("momentum")
    with pytest.raises(FileNotFoundError):
        store.load_evaluation("momentum")
    saved = store.save_value("momentum", _matrix(), _metadata())
    with pytest.raises(FileNotFoundError):
        store.load_evaluation("momentum", run_id=saved["run_id"])
    with pytest.raises(FileNotFoundError):
        store.get_value("momentum", run_id="0" * 16)
