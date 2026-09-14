from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ML_factor_mining.config import Config
from ML_factor_mining.models import (
    Fitted,
    candidate_bank,
    fit_model,
    load_model,
    require_backends,
    save_model,
)


def _data(n: int = 40):
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
    y = 2.0 * x["a"] - 0.5 * x["b"] + 0.25
    return x, y


def test_candidate_banks_are_deterministic_and_defensive():
    config = Config()
    lightgbm = candidate_bank("lightgbm", config)
    assert [item[0] for item in lightgbm] == [f"c{i:03d}" for i in range(6)]
    assert lightgbm[0][1] == {
        "num_leaves": 7,
        "min_data_in_leaf": 50,
        "lambda_l2": 1.0,
    }
    lightgbm[0][1]["num_leaves"] = 999
    assert candidate_bank("lightgbm", config)[0][1]["num_leaves"] == 7
    assert len(candidate_bank("xgboost", config)) == 6
    assert [item[1]["alpha"] for item in candidate_bank("ridge", config)] == [
        0.0001,
        0.001,
        0.01,
        0.1,
        1,
    ]
    assert candidate_bank("linear", config) == [("c000", {})]


def test_candidate_overrides_reject_duplicates_and_unknown_keys():
    with pytest.raises(ValueError, match="duplicate"):
        candidate_bank(
            "ridge",
            Config(candidate_overrides={"ridge": [{"alpha": 1}, {"alpha": 1}]}),
        )
    with pytest.raises(ValueError, match="unsupported"):
        candidate_bank(
            "ridge", Config(candidate_overrides={"ridge": [{"alpha": 1, "bad": 2}]}),
        )


def test_require_backends_is_lazy_mapping():
    backends = require_backends(("lightgbm", "xgboost", "ridge", "linear"))
    assert set(backends) == {"lightgbm", "xgboost", "ridge", "linear"}
    assert backends["ridge"] is backends["linear"]


@pytest.mark.parametrize("name", ["linear", "ridge", "lightgbm", "xgboost"])
def test_fit_save_load_round_trip(name, tmp_path):
    x, y = _data()
    fit = x.iloc[:28]
    val = x.iloc[28:]
    fit_y = y.iloc[:28]
    val_y = y.iloc[28:]
    kwargs = {"params": {"alpha": 0.1}} if name == "ridge" else {}
    fitted = fit_model(name, fit, fit_y, (val, val_y), Config(max_rounds=12, patience=3), **kwargs)
    before = fitted.predict(val)
    save_model(fitted, tmp_path)
    restored = load_model(tmp_path)
    np.testing.assert_allclose(restored.predict(val), before, rtol=1e-10, atol=1e-10)
    assert restored.columns == tuple(x.columns)
    assert restored.rounds == fitted.rounds


def test_fit_model_accepts_plan_positional_api():
    x, y = _data()
    fitted = fit_model(
        "ridge",
        {"alpha": 0.1},
        x,
        y,
        Config(max_rounds=12, patience=3),
    )
    assert fitted.columns == ("a", "b")
    assert np.isfinite(fitted.predict(x)).all()


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_tree_fixed_round_refit_does_not_require_validation(name):
    x, y = _data()
    params = {"num_leaves": 7, "min_data_in_leaf": 2, "lambda_l2": 1.0} if name == "lightgbm" else {
        "max_depth": 2, "min_child_weight": 1, "lambda": 1.0
    }
    fitted = fit_model(name, params, x, y, Config(max_rounds=4), rounds=2)
    assert fitted.rounds == 2
    assert np.isfinite(fitted.predict(x)).all()


def test_persistence_uses_single_json_metadata_file(tmp_path):
    x, y = _data()
    fitted = fit_model("linear", x, y)
    save_model(fitted, tmp_path)
    payload = __import__("json").loads((tmp_path / "model.json").read_text())
    assert payload["name"] == "linear"
    assert payload["columns"] == ["a", "b"]
    assert payload["weights"]["coef"]
    assert "intercept" in payload["weights"]
    assert not (tmp_path / "metadata.json").exists()


@pytest.mark.parametrize("coefficient_key", ["coef", "coefficients"])
def test_fitted_mapping_accepts_contract_and_legacy_coefficient_keys(coefficient_key):
    x, _ = _data()
    fitted = Fitted(
        "linear",
        {coefficient_key: [2.0, -0.5], "intercept": 0.25},
        ("a", "b"),
    )
    np.testing.assert_allclose(fitted.predict(x), 2.0 * x["a"] - 0.5 * x["b"] + 0.25)


def test_load_model_validates_rounds_and_columns_metadata(tmp_path):
    import json

    base = {"name": "linear", "columns": ["a", "b"], "rounds": None,
            "weights": {"coef": [2.0, -0.5], "intercept": 0.25}}
    for field, value, message in (("rounds", -1, "rounds"), ("columns", [], "columns"),
                                  ("columns", ["a", "a"], "columns")):
        payload = dict(base)
        payload[field] = value
        (tmp_path / "model.json").write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_model(tmp_path)


def test_predict_requires_exact_columns_and_is_one_dimensional():
    x, y = _data()
    fitted = fit_model("linear", x, y)
    with pytest.raises(ValueError, match="columns"):
        fitted.predict(x[["b", "a"]])
    with pytest.raises(ValueError, match="columns"):
        fitted.predict(x.assign(extra=1))


def test_ridge_repeated_rows_are_stable():
    x = pd.DataFrame({"a": [1.0, 1.0, 1.0, 2.0], "b": [0.0, 0.0, 0.0, 1.0]})
    y = pd.Series([1.0, 1.0, 1.0, 2.0])
    fitted = fit_model("ridge", x, y, params={"alpha": 0.1})
    assert np.isfinite(fitted.predict(x)).all()
