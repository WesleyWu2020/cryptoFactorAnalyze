"""Tests for quarterly ML research configuration and fold generation."""

from dataclasses import FrozenInstanceError, fields
from datetime import date, datetime, timezone

import pytest
import pandas as pd

from ML_factor_mining.config import Config, Fold, quarter_folds


def test_defaults_and_from_dict_normalize_sequence_fields():
    config = Config()

    assert config.train_start == date(2024, 1, 1)
    assert config.oos_start == date(2025, 1, 1)
    assert config.end == date(2026, 12, 31)
    assert config.mode == "expanding"
    assert config.models == ("lightgbm",)
    assert config.include == ()
    assert config.exclude == ()
    assert config.candidate_overrides == {}
    assert all(field.init for field in fields(Config))
    with pytest.raises(FrozenInstanceError):
        config.mode = "rolling"

    loaded = Config.from_dict({"models": ["ridge"], "include": ["btc"], "exclude": []})
    assert loaded.models == ("ridge",)
    assert loaded.include == ("btc",)
    assert loaded.exclude == ()


def test_expanding_quarter_folds_use_quarter_boundaries_and_evidence_stop():
    config = Config()
    folds = quarter_folds(config, date(2025, 12, 31))

    assert len(folds) == 4
    assert folds[0] == Fold(
        quarter="2025Q1",
        history_start=pd.Timestamp("2024-01-01"),
        validation_start=pd.Timestamp("2024-10-01"),
        retrain_at=pd.Timestamp("2025-01-01"),
        prediction_end=pd.Timestamp("2025-03-31"),
    )
    assert folds[-1].quarter == "2025Q4"
    assert folds[-1].prediction_end == pd.Timestamp("2025-12-31")


def test_rolling_quarter_folds_limit_history_to_rolling_months():
    config = Config(mode="rolling", rolling_months=12)
    folds = quarter_folds(config, date(2026, 6, 30))

    assert folds[0].history_start == pd.Timestamp("2024-01-01")
    assert folds[1].history_start == pd.Timestamp("2024-04-01")
    assert folds[4].quarter == "2026Q1"
    assert folds[4].history_start == pd.Timestamp("2025-01-01")
    assert folds[-1].prediction_end == pd.Timestamp("2026-06-30")


def test_incomplete_quarter_is_clipped_to_evidence_end():
    config = Config(end="2026-12-31")
    folds = quarter_folds(config, "2026-02-14")

    assert folds[-1].quarter == "2026Q1"
    assert folds[-1].prediction_end == pd.Timestamp("2026-02-14")


def test_no_folds_when_evidence_end_precedes_oos_start():
    assert quarter_folds(Config(), "2024-12-31") == []


def test_fold_without_fit_interval_raises():
    with pytest.raises(ValueError, match="^no fit interval before validation$"):
        quarter_folds(Config(train_start="2024-10-01"), "2025-03-31")


@pytest.mark.parametrize(
    "value",
    [None, [], ["not", "a", "mapping"], "not-a-mapping"],
)
def test_candidate_overrides_must_be_a_mapping(value):
    with pytest.raises((TypeError, ValueError)):
        Config(candidate_overrides=value)


def test_candidate_overrides_are_defensively_copied():
    source = {"lightgbm": {"n_estimators": 10}}
    config = Config(candidate_overrides=source)

    source["xgboost"] = {"max_depth": 3}
    assert config.candidate_overrides == {"lightgbm": {"n_estimators": 10}}
    assert config.candidate_overrides is not source


@pytest.mark.parametrize(
    "field",
    [
        "coverage",
        "validation_prediction_coverage",
        "ic_weight",
        "sharpe_weight",
        "fee_rate",
        "slippage",
    ],
)
def test_numeric_float_fields_reject_bool(field):
    kwargs = {field: True}
    if field == "ic_weight":
        kwargs["sharpe_weight"] = 0
    elif field == "sharpe_weight":
        kwargs["ic_weight"] = 0
    with pytest.raises((TypeError, ValueError)):
        Config(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("train_start", "2025-01-01"),
        ("end", "2024-12-31"),
        ("oos_start", "2025-02-01"),
        ("mode", "bad"),
        ("target_type", "bad"),
        ("rolling_months", 3),
        ("holding_days", 0),
        ("min_fit_dates", -1),
        ("min_val_dates", True),
        ("min_pairs", 4),
        ("max_rounds", 0),
        ("patience", 0),
        ("threads", 0),
        ("seed", 0),
        ("seed", True),
        ("models", []),
        ("models", ["lightgbm", "lightgbm"]),
        ("models", ["unknown"]),
        ("include", ["btc"]),
        ("exclude", ["btc"]),
        ("coverage", 0),
        ("validation_prediction_coverage", 1.1),
        ("ic_weight", float("nan")),
        ("ic_weight", 0.4),
        ("fee_rate", -0.1),
        ("slippage", 1),
    ],
)
def test_invalid_config_is_rejected(field, value):
    kwargs = {field: value}
    if field == "include":
        kwargs["exclude"] = ["btc"]
    if field == "exclude":
        kwargs["include"] = ["btc"]
    with pytest.raises((TypeError, ValueError)):
        Config(**kwargs)


@pytest.mark.parametrize(
    "value",
    [
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1),
    ],
)
def test_datetime_timezone_or_non_midnight_is_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        Config(train_start=value)
