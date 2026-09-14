"""Adversarial tests for quarterly ML point-in-time behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ML_factor_mining import training
from ML_factor_mining.config import Config, quarter_folds
from ML_factor_mining.models import Fitted
from ML_factor_mining.validation import compare_cutoff, replay_model


def test_full_retraining_is_invariant_to_future_perturbations(tmp_path, monkeypatch):
    """Quarterly fitting and prediction must ignore evidence after each cutoff."""
    dates = pd.date_range("2024-01-01", "2025-06-30", name="date")
    symbols = list("ABCDE")
    beta = np.linspace(-1.0, 1.0, len(symbols))
    time = np.arange(len(dates))
    changes = (0.002 + 0.0008 * np.sin(time / 11.0))[:, None] * beta[None, :]
    opens = pd.DataFrame(
        100.0 * np.exp(np.cumsum(changes, axis=0)), index=dates, columns=symbols
    )
    membership = pd.DataFrame(True, index=dates, columns=symbols)
    factors = pd.DataFrame(np.tile(beta, (len(dates), 1)), index=dates, columns=symbols)
    factor_path = tmp_path / "factor.parquet"

    def save_factors(frame):
        long = (
            frame.rename_axis(columns="instrument")
            .stack(future_stack=True)
            .rename("factor")
            .reset_index()
        )
        long.to_parquet(factor_path, index=False)

    save_factors(factors)
    world = {"opens": opens, "membership": membership}
    requests = []

    class BoundedProvider:
        def __init__(self, path, *, as_of):
            self.cutoff = pd.Timestamp(as_of).normalize()

        def bound(self, start, end):
            start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
            assert end <= self.cutoff
            requests.append((start, end, self.cutoff))
            return start, end

        def get_universe(self, *, start, end):
            start, end = self.bound(start, end)
            return world["membership"].loc[start:end].copy()

        def get_single_data(self, field, *, start, end):
            assert field == "open"
            start, end = self.bound(start, end)
            return world["opens"].loc[start:end].copy()

        def get_funding(self, *, start, end, symbols):
            self.bound(start, end)
            return pd.DataFrame(
                columns=["funding_time", "instrument", "funding_rate", "mark_price"]
            )

        def get_quality(self, *, start, end, symbols):
            start, end = self.bound(start, end)
            index = pd.MultiIndex.from_product(
                [pd.date_range(start, end), symbols], names=["date", "instrument"]
            )
            return pd.DataFrame(
                {
                    "has_placeholder_kline": False,
                    "funding_coverage_status": "not_applicable",
                },
                index=index,
            )

    def numpy_linear_fit(name, params, X, y, config, **kwargs):
        # This keeps the causal integration test independent of optional ML
        # backends; production adapter behavior is covered by test_models.py.
        design = np.column_stack([np.ones(len(X)), X.to_numpy()])
        coefficients = np.linalg.lstsq(design, np.asarray(y), rcond=None)[0]
        estimator = {
            "intercept": float(coefficients[0]),
            "coef": coefficients[1:].tolist(),
        }
        return Fitted("linear", estimator, tuple(X.columns), None)

    monkeypatch.setattr(training, "DataProvider", BoundedProvider)
    monkeypatch.setattr(training, "fit_model", numpy_linear_fit)

    config = Config(
        end="2025-06-30",
        models=("linear",),
        min_pairs=5,
    )
    catalog = [{"factor_id": "F", "values": factor_path}]

    def run_through(evidence_end):
        outputs = []
        for fold in quarter_folds(config, evidence_end):
            prepared = training.prepare_fold(catalog, "synthetic.h5", fold, config)
            model, _candidates, audit = training.select_and_refit(
                prepared, fold, config, "linear"
            )
            predictions, _coverage = training.predict_quarter(
                model, prepared.features, catalog, "synthetic.h5", fold, audit
            )
            outputs.append(predictions)
        return pd.concat(outputs, ignore_index=True)

    full = replay_model(run_through, dates[-1])
    cutoff = pd.Timestamp("2025-05-15")
    historical = replay_model(run_through, cutoff)
    comparison = compare_cutoff(full, historical, cutoff)
    assert comparison["status"] == "verified"
    assert comparison["max_abs_diff"] <= 1e-12

    # Change every evidence source strictly after the replay cutoff. The
    # replay still retrains at both 2025-01-01 and 2025-04-01 boundaries.
    world["opens"].loc[world["opens"].index > cutoff] *= 17.0
    world["membership"].loc[world["membership"].index > cutoff, "A"] = False
    changed = factors.copy()
    changed.loc[changed.index > cutoff] *= -100.0
    save_factors(changed)

    perturbed = replay_model(run_through, dates[-1])
    comparison = compare_cutoff(full, perturbed, cutoff)
    assert comparison["status"] == "verified"
    assert comparison["max_abs_diff"] <= 1e-12
    assert requests and all(end <= known for _start, end, known in requests)
