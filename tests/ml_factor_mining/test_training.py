import numpy as np
import pandas as pd
import json

from ML_factor_mining.config import Config, quarter_folds
from ML_factor_mining.dataset import FeatureMap
from ML_factor_mining.models import fit_model
from ML_factor_mining.training import predict_quarter, save_quarter


class FirstColumnModel:
    def predict(self, X):
        return X.iloc[:, 0].to_numpy()


def test_prediction_uses_only_membership_and_factor_values(tmp_path, monkeypatch):
    dates = pd.date_range("2025-01-01", periods=3, name="date")
    members = pd.DataFrame(True, index=dates, columns=["A", "B"])
    values = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "instrument": ["A", "B"] * 3,
            "factor": [-2, 2, np.nan, np.nan, -1, 1],
        }
    )
    path = tmp_path / "f.parquet"
    values.to_parquet(path)
    calls = []

    class Provider:
        def __init__(self, path, *, as_of):
            calls.append(as_of)

        def get_universe(self, *, start, end):
            return members.loc[start:end]

        def get_single_data(self, *args, **kwargs):
            raise AssertionError("prediction must not read label prices")

    monkeypatch.setattr("ML_factor_mining.training.DataProvider", Provider)
    fold = quarter_folds(Config(), dates[-1])[0]
    features = FeatureMap(("F",), {})
    result, coverage = predict_quarter(
        FirstColumnModel(),
        features,
        [{"factor_id": "F", "values": str(path)}],
        "unused.h5",
        fold,
        {"model_id": "m"},
    )
    assert calls == [dates[-1]]
    assert len(result) == 4
    assert dates[1] not in set(result.date)
    assert coverage.loc[coverage.date.eq(dates[1]), "coverage"].item() == 0
    assert set(result.columns) == {
        "date",
        "instrument",
        "factor",
        "model_id",
        "quarter",
        "trained_before",
    }


def test_save_quarter_serializes_immutable_feature_diagnostics(tmp_path):
    panel = pd.DataFrame({"F": [0.0, 1.0]}, index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01"), "A"), (pd.Timestamp("2024-01-01"), "B")],
        names=["date", "instrument"],
    ))
    feature_map = FeatureMap.fit(panel, 1.0)
    model = fit_model("linear", panel, pd.Series([0.0, 1.0]))
    destination = tmp_path / "2025Q1"

    save_quarter(
        destination,
        model,
        pd.DataFrame([{"candidate_id": "c000", "status": "valid"}]),
        {"model_id": "linear_2025Q1_x", "feature_admission": feature_map.diagnostics},
        pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "instrument": ["A"], "factor": [1.0]}),
        pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "coverage": [1.0]}),
    )

    payload = json.loads((destination / "audit.json").read_text(encoding="utf-8"))
    assert payload["feature_admission"]["selected"] == ["F"]
