"""Lazy, deterministic model backends used by quarterly factor mining."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import write_json


_BACKENDS = {
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "ridge": "sklearn.linear_model",
    "linear": "sklearn.linear_model",
}
_ALLOWED = {
    "lightgbm": {"num_leaves", "min_data_in_leaf", "lambda_l2"},
    "xgboost": {"max_depth", "min_child_weight", "lambda"},
    "ridge": {"alpha"},
    "linear": set(),
}


def require_backends(names: Sequence[str]) -> dict[str, Any]:
    """Import requested optional packages and fail loudly when unavailable."""
    if isinstance(names, str):
        names = (names,)
    result: dict[str, Any] = {}
    for requested in names:
        module_name = _BACKENDS.get(requested, requested)
        if requested not in _BACKENDS and module_name not in _BACKENDS.values():
            raise ValueError(f"unknown backend: {requested!r}")
        key = requested
        try:
            result[key] = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            raise RuntimeError(f"required backend {module_name!r} is unavailable") from exc
    return result


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _candidate_entries(
    values: Sequence[Mapping[str, Any]], name: str
) -> list[tuple[str, dict[str, Any]]]:
    allowed = _ALLOWED[name]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"candidate override for {name!r} must be a nonempty list")
    seen: set[str] = set()
    entries: list[tuple[str, dict[str, Any]]] = []
    for params in values:
        if not isinstance(params, Mapping):
            raise ValueError("candidate entries must be objects")
        params_copy = deepcopy(dict(params))
        unsupported = set(params_copy) - allowed
        if unsupported:
            raise ValueError(f"unsupported candidate keys for {name}: {sorted(unsupported)}")
        try:
            key = json.dumps(params_copy, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate parameters must be JSON-compatible") from exc
        if key in seen:
            raise ValueError(f"duplicate candidate for {name}: {params_copy}")
        seen.add(key)
        entries.append((f"c{len(entries):03d}", params_copy))
    return entries


def candidate_bank(name: str, config: Any = None) -> list[tuple[str, dict[str, Any]]]:
    """Return the deterministic candidate grid, or validated config overrides."""
    if name not in _ALLOWED:
        raise ValueError(f"unknown model: {name!r}")
    defaults: dict[str, list[dict[str, Any]]] = {
        "lightgbm": [
            {"num_leaves": leaves, "min_data_in_leaf": minimum, "lambda_l2": 1.0}
            for leaves in (7, 15, 31)
            for minimum in (50, 100)
        ],
        "xgboost": [
            {"max_depth": depth, "min_child_weight": 10, "lambda": penalty}
            for depth in (2, 3, 4)
            for penalty in (1, 10)
        ],
        "ridge": [{"alpha": alpha} for alpha in (0.0001, 0.001, 0.01, 0.1, 1)],
        "linear": [{}],
    }
    overrides = _config_value(config, "candidate_overrides", {}) or {}
    if not isinstance(overrides, Mapping):
        raise ValueError("candidate_overrides must be a mapping")
    values = overrides.get(name, defaults[name])
    return _candidate_entries(values, name)


def _validate_frame(frame: Any, label: str) -> tuple[pd.DataFrame, np.ndarray]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    if frame.empty or frame.shape[1] == 0:
        raise ValueError(f"{label} must be nonempty")
    if not frame.columns.is_unique:
        raise ValueError(f"{label} columns must be unique")
    try:
        values = frame.to_numpy(dtype="float64", copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values")
    return frame, values


def _validate_target(target: Any, n_rows: int, label: str = "y") -> np.ndarray:
    try:
        values = np.asarray(target, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric vector") from exc
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError(f"{label} must be a vector with length {n_rows}")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values")
    return values


def _validation_pair(validation: Any) -> tuple[Any, Any] | None:
    if validation is None:
        return None
    if isinstance(validation, Mapping):
        if "X" in validation and "y" in validation:
            return validation["X"], validation["y"]
        if "x" in validation and "y" in validation:
            return validation["x"], validation["y"]
    if isinstance(validation, (tuple, list)) and len(validation) == 2:
        return validation[0], validation[1]
    raise TypeError("validation must be an (X, y) pair")


@dataclass
class Fitted:
    name: str
    estimator: Any
    columns: tuple[str, ...]
    rounds: int | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        frame, values = _validate_frame(X, "X")
        if tuple(frame.columns) != self.columns:
            raise ValueError("X columns must exactly match fitted feature columns")

        if self.name == "lightgbm":
            estimator = self.estimator
            if hasattr(estimator, "booster_"):
                output = estimator.predict(frame, num_iteration=self.rounds)
            else:
                output = estimator.predict(frame, num_iteration=self.rounds)
        elif self.name == "xgboost":
            xgb = require_backends(("xgboost",))["xgboost"]
            estimator = self.estimator
            booster = estimator.get_booster() if hasattr(estimator, "get_booster") else estimator
            output = booster.predict(
                xgb.DMatrix(values, feature_names=list(self.columns)),
                iteration_range=(0, self.rounds),
            )
        elif isinstance(self.estimator, Mapping):
            weights = self.estimator.get("weights", self.estimator)
            if not isinstance(weights, Mapping):
                raise ValueError("serialized linear model weights must be an object")
            if "coef" in weights:
                coefficient = np.asarray(weights["coef"], dtype="float64")
            elif "coefficients" in weights:
                coefficient = np.asarray(weights["coefficients"], dtype="float64")
            else:
                raise ValueError("serialized linear model is missing coef weights")
            intercept = float(weights["intercept"])
            output = values @ coefficient + intercept
        else:
            output = self.estimator.predict(frame)

        result = np.asarray(output, dtype="float64")
        if result.ndim != 1:
            result = result.reshape(-1) if result.size == len(frame) else result
        if result.ndim != 1 or len(result) != len(frame) or not np.isfinite(result).all():
            raise ValueError("model prediction must be a finite one-dimensional vector")
        return result


def fit_model(name: str, *args: Any, **kwargs: Any) -> Fitted:
    """Fit one model using either the plan or legacy positional convention.

    The plan convention is ``fit_model(name, params, X, y, config, *,
    X_val=None, y_val=None, rounds=None)``.  Older callers in this repository
    pass ``X, y, validation, config`` and optionally ``params=...``; accepting
    both here keeps the adapter usable by the existing research scripts.
    """
    sentinel = object()
    params_kw = kwargs.pop("params", sentinel)
    X_kw = kwargs.pop("X", sentinel)
    y_kw = kwargs.pop("y", sentinel)
    config_kw = kwargs.pop("config", sentinel)
    validation_kw = kwargs.pop("validation", sentinel)
    X_val = kwargs.pop("X_val", None)
    y_val = kwargs.pop("y_val", None)
    rounds = kwargs.pop("rounds", None)
    if kwargs:
        raise TypeError(f"unexpected fit_model arguments: {sorted(kwargs)}")

    # A DataFrame in the first position unambiguously identifies the legacy
    # call.  Otherwise positional arguments follow the plan's parameter order.
    if args and isinstance(args[0], pd.DataFrame):
        X = args[0]
        y = args[1] if len(args) > 1 else (None if y_kw is sentinel else y_kw)
        validation = args[2] if len(args) > 2 else (
            None if validation_kw is sentinel else validation_kw
        )
        config = args[3] if len(args) > 3 else (
            None if config_kw is sentinel else config_kw
        )
        if len(args) > 4:
            raise TypeError("legacy fit_model accepts at most five positional arguments")
        params = None if params_kw is sentinel else params_kw
        if X_kw is not sentinel or y_kw is not sentinel and len(args) > 1:
            raise TypeError("X/y supplied both positionally and by keyword")
    else:
        if len(args) > 5:
            raise TypeError("fit_model accepts at most five positional arguments after name")
        positional = list(args) + [sentinel] * (5 - len(args))
        params = positional[0] if positional[0] is not sentinel else (
            None if params_kw is sentinel else params_kw
        )
        X = positional[1] if positional[1] is not sentinel else X_kw
        y = positional[2] if positional[2] is not sentinel else y_kw
        config = positional[3] if positional[3] is not sentinel else config_kw
        validation = None if validation_kw is sentinel else validation_kw
        if validation is not None:
            raise TypeError("validation is only supported by the legacy call convention")
    if X is sentinel or y is sentinel:
        raise TypeError("fit_model requires X and y")

    # In the short legacy form ``fit_model(name, X, y, config)``, the fourth
    # positional argument lands in ``validation`` by historical convention.
    if config is None and validation is not None and not isinstance(validation, (tuple, list, Mapping)):
        if all(hasattr(validation, attr) for attr in ("max_rounds", "patience", "seed", "threads")):
            config, validation = validation, None

    # The legacy validation tuple is equivalent to explicit X_val/y_val.
    if validation is not None:
        if X_val is not None or y_val is not None:
            raise TypeError("provide validation or X_val/y_val, not both")
        validation_pair = _validation_pair(validation)
        assert validation_pair is not None
        X_val, y_val = validation_pair

    if name not in _ALLOWED:
        raise ValueError(f"unknown model: {name!r}")
    frame, values = _validate_frame(X, "X")
    target = _validate_target(y, len(frame))
    if params is sentinel or params is None:
        candidate = {}
    elif isinstance(params, Mapping):
        candidate = dict(params)
    else:
        raise TypeError("params must be a mapping")
    unsupported = set(candidate) - _ALLOWED[name]
    if unsupported:
        raise ValueError(f"unsupported parameters for {name}: {sorted(unsupported)}")

    validation_pair = None if X_val is None and y_val is None else (X_val, y_val)
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val and y_val must be provided together")
    val_frame = val_values = val_target = None
    if validation_pair is not None:
        val_frame, val_values = _validate_frame(validation_pair[0], "validation X")
        if tuple(val_frame.columns) != tuple(frame.columns):
            raise ValueError("validation X columns must match training X columns")
        val_target = _validate_target(validation_pair[1], len(val_frame), "validation y")

    columns = tuple(frame.columns)
    if name in ("linear", "ridge"):
        sklearn = require_backends((name,))[name]
        if name == "linear":
            estimator = sklearn.LinearRegression()
        else:
            alpha = float(candidate.get("alpha", 1.0))
            if not np.isfinite(alpha) or alpha <= 0:
                raise ValueError("ridge alpha must be positive and finite")
            estimator = sklearn.Ridge(alpha=alpha * len(frame), solver="svd")
        estimator.fit(frame, target)
        return Fitted(name, estimator, columns, None)

    fixed_rounds = rounds is not None
    if validation_pair is None and not fixed_rounds:
        raise ValueError(f"{name} requires validation data for early-stopping tuning")
    max_rounds = rounds if rounds is not None else _config_value(config, "max_rounds", 1000)
    patience = _config_value(config, "patience", 50)
    seed = _config_value(config, "seed", 42)
    threads = _config_value(config, "threads", 1)
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, (int, np.integer)) or max_rounds <= 0:
        raise ValueError("max_rounds/rounds must be a positive integer")
    if isinstance(patience, bool) or not isinstance(patience, (int, np.integer)) or patience <= 0:
        raise ValueError("patience must be a positive integer")

    if name == "lightgbm":
        lgb = require_backends(("lightgbm",))["lightgbm"]
        fit_rounds = int(rounds) if fixed_rounds else int(max_rounds)
        estimator = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=fit_rounds,
            learning_rate=0.05,
            random_state=int(seed),
            n_jobs=int(threads),
            deterministic=True,
            feature_pre_filter=False,
            verbosity=-1,
            **candidate,
        )
        if fixed_rounds:
            estimator.fit(frame, target)
            selected = int(rounds)
        else:
            estimator.fit(
                frame,
                target,
                eval_X=val_frame,
                eval_y=val_target,
                callbacks=[lgb.early_stopping(int(patience), verbose=False)],
            )
            best = getattr(estimator, "best_iteration_", None)
            selected = int(best) if best is not None and int(best) > 0 else int(max_rounds)
    else:
        xgb = require_backends(("xgboost",))["xgboost"]
        fit_rounds = int(rounds) if fixed_rounds else int(max_rounds)
        xgb_kwargs = {
            "objective": "reg:squarederror",
            "n_estimators": fit_rounds,
            "tree_method": "hist",
            "random_state": int(seed),
            "n_jobs": int(threads),
            "eval_metric": "rmse",
            "reg_lambda": candidate.pop("lambda", 1.0),
            **candidate,
        }
        if not fixed_rounds:
            xgb_kwargs["early_stopping_rounds"] = int(patience)
        estimator = xgb.XGBRegressor(
            **xgb_kwargs,
        )
        if fixed_rounds:
            estimator.fit(frame, target, verbose=False)
            selected = int(rounds)
        else:
            estimator.fit(frame, target, eval_set=[(val_frame, val_target)], verbose=False)
            best = getattr(estimator, "best_iteration", None)
            selected = int(best) + 1 if best is not None and int(best) >= 0 else int(max_rounds)
    return Fitted(name, estimator, columns, selected)


def _metadata(model: Fitted) -> dict[str, Any]:
    return {"name": model.name, "columns": list(model.columns), "rounds": model.rounds}


def _linear_weights(estimator: Any, n_columns: int) -> dict[str, Any]:
    """Return canonical JSON-safe linear weights, accepting legacy mappings."""
    if isinstance(estimator, Mapping):
        weights = estimator.get("weights", estimator)
        if not isinstance(weights, Mapping):
            raise ValueError("serialized linear model weights must be an object")
        coefficient_key = "coef" if "coef" in weights else "coefficients"
        if coefficient_key not in weights or "intercept" not in weights:
            raise ValueError("serialized linear model must contain coef and intercept")
        coefficient = np.asarray(weights[coefficient_key], dtype="float64")
        intercept = float(weights["intercept"])
    else:
        coefficient = np.asarray(estimator.coef_, dtype="float64").reshape(-1)
        intercept = float(np.asarray(estimator.intercept_).reshape(-1)[0])
    if coefficient.ndim != 1 or len(coefficient) != n_columns or not np.isfinite(coefficient).all():
        raise ValueError("invalid serialized coefficients")
    if not np.isfinite(intercept):
        raise ValueError("invalid serialized intercept")
    return {"coef": coefficient.tolist(), "intercept": intercept}


def save_model(model: Fitted | str | Path, directory: str | Path | Fitted) -> Path:
    """Persist native tree models or JSON linear coefficients without pickle."""
    if not isinstance(model, Fitted):
        model, directory = directory, model
    if not isinstance(model, Fitted):
        raise TypeError("save_model expects a Fitted model")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    payload = _metadata(model)
    if model.name in ("linear", "ridge"):
        payload["weights"] = _linear_weights(model.estimator, len(model.columns))
    write_json(root / "model.json", payload)
    if model.name == "lightgbm":
        booster = model.estimator.booster_ if hasattr(model.estimator, "booster_") else model.estimator
        booster.save_model(str(root / "model.txt"))
    elif model.name == "xgboost":
        model.estimator.save_model(str(root / "model.ubj"))
    return root


def load_model(path_or_name: str | Path, directory: str | Path | None = None) -> Fitted:
    """Load a model saved by :func:`save_model` (optionally checking its name)."""
    expected_name = None if directory is None else str(path_or_name)
    root = Path(path_or_name if directory is None else directory)
    metadata = json.loads((root / "model.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("model metadata must be an object")
    name = metadata.get("name")
    if name not in _ALLOWED or (expected_name is not None and expected_name != name):
        raise ValueError("model metadata has an invalid or unexpected name")
    raw_columns = metadata.get("columns")
    if (
        not isinstance(raw_columns, list)
        or not raw_columns
        or any(not isinstance(column, str) or not column for column in raw_columns)
        or len(set(raw_columns)) != len(raw_columns)
    ):
        raise ValueError("model metadata has invalid columns")
    columns = tuple(raw_columns)
    rounds = metadata.get("rounds")
    if rounds is not None and (
        isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0
    ):
        raise ValueError("model metadata has invalid rounds")
    if name == "lightgbm":
        lgb = require_backends(("lightgbm",))["lightgbm"]
        estimator = lgb.Booster(model_file=str(root / "model.txt"))
    elif name == "xgboost":
        xgb = require_backends(("xgboost",))["xgboost"]
        estimator = xgb.Booster()
        estimator.load_model(str(root / "model.ubj"))
    else:
        weights = metadata.get("weights")
        if weights is None and "coefficients" in metadata:
            weights = {
                "coefficients": metadata["coefficients"],
                "intercept": metadata.get("intercept"),
            }
        if not isinstance(weights, Mapping):
            raise ValueError("model metadata is missing linear weights")
        estimator = _linear_weights(weights, len(columns))
    return Fitted(name, estimator, columns, rounds)
