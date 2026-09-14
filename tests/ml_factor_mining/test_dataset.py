from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ML_factor_mining.artifacts import discover_catalog, sha256, snapshot_inputs
from ML_factor_mining.dataset import FeatureMap, build_panel


def _write_factor(directory: Path, factor_id: str, frame: pd.DataFrame, **metadata) -> None:
    frame.to_parquet(directory / f"{factor_id}.parquet", index=False)
    payload = {
        "factor_id": factor_id,
        "setting": {"universe": "historical_top50"},
        "source_type": "module",
        **metadata,
    }
    (directory / f"{factor_id}.meta.json").write_text(json.dumps(payload), encoding="utf-8")


def _membership() -> pd.DataFrame:
    return pd.DataFrame(
        [[True, False], [True, True]],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        columns=["A", "B"],
    )


def test_discover_catalog_ignores_evaluation_sidecars(tmp_path):
    frame = pd.DataFrame(
        {"date": ["2024-01-01"], "instrument": ["A"], "factor": [1.0]}
    )
    _write_factor(tmp_path, "alpha", frame)
    _write_factor(tmp_path, "eval", frame, catalog="evaluation")

    records = discover_catalog(tmp_path, type("Config", (), {"include": (), "exclude": ()})())

    assert [record["factor_id"] for record in records] == ["alpha"]
    assert records[0]["values"].name == "alpha.parquet"
    assert records[0]["metadata"].name == "alpha.meta.json"


def test_build_panel_masks_membership_before_cross_sectional_ranking(tmp_path):
    factor = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "instrument": ["A", "B", "A", "B"],
            "factor": [1.0, 100.0, 10.0, 20.0],
        }
    )
    _write_factor(tmp_path, "alpha", factor)
    catalog = [{"factor_id": "alpha", "values": tmp_path / "alpha.parquet", "metadata": {}}]

    panel = build_panel(catalog, _membership())

    # B is ineligible on the first date, so A is a one-name cross-section and ranks to zero.
    assert panel.loc[(pd.Timestamp("2024-01-01"), "A"), "alpha"] == 0.0
    assert panel.loc[(pd.Timestamp("2024-01-02"), "A"), "alpha"] == -1.0
    assert panel.loc[(pd.Timestamp("2024-01-02"), "B"), "alpha"] == 1.0


def test_feature_map_admits_using_fit_only_and_never_predicts_all_missing(tmp_path):
    factor_a = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "instrument": ["A", "B", "A", "B", "A", "B"],
            "factor": [1.0, 2.0, 2.0, 4.0, 3.0, 6.0],
        }
    )
    factor_b = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "instrument": ["A", "B", "A", "B"],
            "factor": [3.0, 5.0, 4.0, 6.0],
        }
    )
    _write_factor(tmp_path, "a", factor_a)
    _write_factor(tmp_path, "b", factor_b)
    membership = pd.DataFrame(True, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), columns=["A", "B"])
    catalog = [
        {"factor_id": "a", "values": tmp_path / "a.parquet", "metadata": {}},
        {"factor_id": "b", "values": tmp_path / "b.parquet", "metadata": {}},
    ]
    panel = build_panel(catalog, membership)
    feature_map = FeatureMap.fit(panel.loc[:pd.Timestamp("2024-01-02")], 0.9)

    assert feature_map.factors == ("a", "b")
    prediction_panel = panel.copy()
    third_day = prediction_panel.index.get_level_values("date") == pd.Timestamp("2024-01-03")
    prediction_panel.loc[third_day, ["a", "b"]] = np.nan
    transformed, present = feature_map.transform(prediction_panel)
    assert transformed.columns.tolist() == ["a", "b", "__missing__a", "__missing__b"]
    assert transformed["a"].dtype == "float64"
    assert transformed.loc[(pd.Timestamp("2024-01-02"), "A"), "a"] == -1.0
    assert transformed.loc[(pd.Timestamp("2024-01-02"), "A"), "__missing__a"] == 0.0
    assert not present.loc[(pd.Timestamp("2024-01-03"), "A")]


def test_cleaning_rejects_duplicate_date_instrument(tmp_path):
    factor = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "instrument": ["A", "A"],
            "factor": [1.0, 2.0],
        }
    )
    _write_factor(tmp_path, "alpha", factor)
    catalog = [{"factor_id": "alpha", "values": tmp_path / "alpha.parquet", "metadata": {}}]

    with pytest.raises(ValueError, match="duplicate"):
        build_panel(catalog, _membership())


def test_feature_map_diagnostics_are_deeply_immutable():
    panel = pd.DataFrame({"alpha": [0.0, 1.0]}, index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01"), "A"), (pd.Timestamp("2024-01-01"), "B")],
        names=["date", "instrument"],
    ))
    feature_map = FeatureMap.fit(panel, 1.0)

    with pytest.raises(TypeError):
        feature_map.diagnostics["coverage"]["alpha"] = 0.0


def test_build_panel_rejects_duplicate_membership_axes(tmp_path):
    factor = pd.DataFrame(
        {"date": ["2024-01-01"], "instrument": ["A"], "factor": [1.0]}
    )
    _write_factor(tmp_path, "alpha", factor)
    membership = pd.DataFrame(
        [[True], [False]],
        index=pd.to_datetime(["2024-01-01", "2024-01-01"]),
        columns=["A"],
    )
    catalog = [{"factor_id": "alpha", "values": tmp_path / "alpha.parquet", "metadata": {}}]

    with pytest.raises(ValueError, match="duplicate"):
        build_panel(catalog, membership)


def test_snapshot_manifest_hashes_are_snapshot_relative(tmp_path):
    factor_dir = tmp_path / "factors"
    factor_dir.mkdir()
    frame = pd.DataFrame({"date": ["2024-01-01"], "instrument": ["A"], "factor": [1.0]})
    _write_factor(factor_dir, "alpha", frame)
    h5_path = tmp_path / "crypto_quant.h5"
    h5_path.write_bytes(b"h5")
    config = type("Config", (), {"include": (), "exclude": ()})()

    snapshot = snapshot_inputs(factor_dir, h5_path, tmp_path / "run", config)
    manifest = json.loads((snapshot.parent / "manifest.json").read_text())

    assert snapshot == tmp_path / "run" / "inputs"
    assert (snapshot / "crypto_quant.h5").is_file()
    assert (snapshot / "factors").is_dir()
    assert set(manifest["hashes"]) == {"crypto_quant.h5", "factors/alpha.parquet", "factors/alpha.meta.json"}
    assert manifest["hashes"]["crypto_quant.h5"] == sha256(h5_path)
