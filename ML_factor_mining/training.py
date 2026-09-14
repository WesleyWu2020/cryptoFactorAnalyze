"""Quarterly fit, selection, refit, inference, and artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from factor_common.data_provider import DataProvider

from .artifacts import _json_safe, write_json, write_parquet
from .dataset import FeatureMap, build_panel
from .models import candidate_bank, fit_model, save_model
from .scoring import CandidateRejected, load_market, rank_candidates, score_validation
from .targets import build_targets, partition_masks


@dataclass
class PreparedFold:
    """Causal, fit-ready data for one quarterly selection/refit cycle."""

    X: pd.DataFrame
    targets: pd.DataFrame
    features: FeatureMap
    rows: dict[str, pd.MultiIndex]
    presence: pd.Series
    membership: pd.DataFrame
    market: Any


def prepare_fold(catalog, h5_path, fold, config) -> PreparedFold:
    """Prepare a fold using only evidence known before its retrain date."""
    cutoff = pd.Timestamp(fold.retrain_at).normalize() - pd.Timedelta(days=1)
    provider = DataProvider(h5_path, as_of=cutoff)
    membership = provider.get_universe(start=fold.history_start, end=cutoff)
    panel = build_panel(catalog, membership)

    masks = partition_masks(panel.index, fold, config.holding_days)
    features = FeatureMap.fit(panel.loc[masks["fit"]], config.coverage)
    X, presence = features.transform(panel)

    # Labels are deliberately built after causal features and remain separate
    # from FeatureMap admission.  Their forward prices are evaluation-only.
    opens = provider.get_single_data("open", start=fold.history_start, end=cutoff)
    opens = opens.reindex(index=membership.index, columns=membership.columns)
    targets = build_targets(opens, membership, config).reindex(panel.index)
    supervised = presence & targets["target"].notna()
    supervised_values = supervised.to_numpy(dtype=bool)
    rows = {
        name: X.index[mask & supervised_values]
        for name, mask in masks.items()
    }

    fit_dates = rows["fit"].get_level_values("date").nunique()
    validation_dates = rows["validation"].get_level_values("date").nunique()
    if fit_dates < config.min_fit_dates or validation_dates < config.min_val_dates:
        raise ValueError(
            "insufficient supervised dates: "
            f"fit={fit_dates}, validation={validation_dates}"
        )

    market = load_market(provider, fold.validation_start, cutoff, membership.columns)
    return PreparedFold(X, targets, features, rows, presence, membership, market)


def select_and_refit(prepared: PreparedFold, fold, config, name):
    """Validate each candidate, rank valid candidates, and refit the winner."""
    knowledge_end = pd.Timestamp(fold.retrain_at).normalize() - pd.Timedelta(days=1)
    X, targets, rows = prepared.X, prepared.targets, prepared.rows
    validation_rows = rows["validation"]
    inference_rows = X.index[
        partition_masks(X.index, fold, config.holding_days)["validation"]
        & prepared.presence.to_numpy(dtype=bool)
    ]
    records: list[dict[str, Any]] = []
    fitted: dict[str, tuple[dict[str, Any], int | None]] = {}

    for candidate_id, params in candidate_bank(name, config):
        model = fit_model(
            name,
            params,
            X.loc[rows["fit"]],
            targets.loc[rows["fit"], "target"],
            config,
            X_val=X.loc[validation_rows],
            y_val=targets.loc[validation_rows, "target"],
        )
        record = {
            "candidate_id": candidate_id,
            "params_json": json.dumps(params, sort_keys=True),
            "rounds": model.rounds,
        }
        predictions = pd.Series(
            model.predict(X.loc[inference_rows]), index=inference_rows, name="factor"
        )
        try:
            metrics = score_validation(
                predictions,
                targets,
                prepared.membership,
                prepared.market,
                fold,
                config,
            )
        except CandidateRejected as exc:
            record.update(status="rejected", reason=str(exc))
        else:
            # ``score_validation`` returns the accounting status (currently
            # ``complete``), while candidate tables use ``valid`` as their
            # selection status.  Merge first, then normalize the one shared
            # key to avoid passing duplicate ``status`` kwargs to ``dict.update``.
            record.update(metrics)
            record["status"] = "valid"
            fitted[candidate_id] = (dict(params), model.rounds)
        records.append(record)

    winner, table = rank_candidates(records, config)
    params, rounds = fitted[str(winner)]
    final = fit_model(
        name,
        params,
        X.loc[rows["refit"]],
        targets.loc[rows["refit"], "target"],
        config,
        rounds=rounds,
    )

    known = X.loc[rows["refit"]].copy()
    known["__target__"] = targets.loc[rows["refit"], "target"]
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(known, index=True).values.tobytes()
    )
    identity = {
        "model": name,
        "quarter": fold.quarter,
        "params": params,
        "rounds": rounds,
        "columns": list(X.columns),
        "holding_days": config.holding_days,
        "target_type": config.target_type,
    }
    digest.update(json.dumps(identity, sort_keys=True).encode())
    audit = {
        **identity,
        "model_id": f"{name}_{fold.quarter}_{digest.hexdigest()[:16]}",
        "winner": str(winner),
        "history_start": pd.Timestamp(fold.history_start).date().isoformat(),
        "validation_start": pd.Timestamp(fold.validation_start).date().isoformat(),
        "retrain_at": pd.Timestamp(fold.retrain_at).date().isoformat(),
        "knowledge_end": knowledge_end.date().isoformat(),
        "feature_admission": prepared.features.diagnostics,
        "selected_factors": list(prepared.features.factors),
        "rows": {key: len(index) for key, index in rows.items()},
        "max_exit": {
            key: pd.Timestamp(targets.loc[index, "exit_date"].max())
            .date()
            .isoformat()
            for key, index in rows.items()
        },
        "direction": 1,
        "scope": "model_level_oos",
    }
    return final, table, audit


def predict_quarter(model, features, catalog, h5_path, fold, audit):
    """Predict one quarter from point-in-time membership and factor values only."""
    provider = DataProvider(h5_path, as_of=fold.prediction_end)
    membership = provider.get_universe(
        start=fold.retrain_at, end=fold.prediction_end
    )
    panel = build_panel(catalog, membership)
    X, present = features.transform(panel)
    if not present.any():
        raise ValueError("no observable feature vectors in prediction quarter")

    selected = X.loc[present]
    predictions = pd.Series(
        model.predict(selected), index=selected.index, name="factor"
    ).reset_index()
    predictions["model_id"] = audit["model_id"]
    predictions["quarter"] = fold.quarter
    predictions["trained_before"] = pd.Timestamp(fold.retrain_at)
    if predictions.duplicated(["date", "instrument"]).any():
        raise AssertionError("duplicate predictions")

    coverage = present.groupby(level="date").agg(["sum", "count"])
    coverage = coverage.rename(columns={"sum": "predicted", "count": "eligible"})
    coverage["coverage"] = coverage["predicted"] / coverage["eligible"]
    return predictions, coverage.reset_index()


def save_quarter(directory, model, table, audit, predictions, coverage):
    """Atomically publish an immutable quarter directory and its native files."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"immutable quarter already exists: {directory}")
    pending = directory.with_name(directory.name + ".pending")
    pending.mkdir(parents=True, exist_ok=False)
    save_model(model, pending)
    # FeatureMap diagnostics are intentionally immutable MappingProxyType
    # objects; normalize them only at the persistence boundary.
    write_json(pending / "audit.json", _json_safe(audit))
    write_parquet(pending / "candidates.parquet", table)
    write_parquet(pending / "predictions.parquet", predictions)
    write_parquet(pending / "coverage.parquet", coverage)
    os.replace(pending, directory)


__all__ = [
    "PreparedFold",
    "prepare_fold",
    "select_and_refit",
    "predict_quarter",
    "save_quarter",
]
