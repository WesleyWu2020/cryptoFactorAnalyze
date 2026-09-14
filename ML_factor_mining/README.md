# Quarterly ML factor mining

This package runs quarterly, model-level out-of-sample factor experiments. A
run first validates the requested optional model backends, copies the HDF5
market store and factor catalog into an immutable `inputs/` snapshot, and
records package versions, source hashes, and the Git revision in
`manifest.json`. Every quarter then performs fit-only feature admission,
candidate selection, refit, and daily inference.

## Run

From the repository root:

```bash
./.venv/bin/python -m ML_factor_mining.cli run \
  --config ML_factor_mining/configs/default.json \
  --factor-dir data/factor_data \
  --h5 data/crypto_quant.h5 \
  --output outputs/ml_runs/$(date +%Y%m%d_%H%M%S)
```

`--evidence-end YYYY-MM-DD` can be supplied to run only through an evidence
cutoff. Use `--no-evaluate` when only model and prediction artifacts are
needed. A successful run has `state.json` with `status: "complete"`; a failed
run retains the directory, writes `state.json` with the exception and
traceback, and writes `failed_candidates.json` when candidate diagnostics are
available.

## Verify

```bash
./.venv/bin/python -m ML_factor_mining.cli verify outputs/ml_runs/<run>
```

Verification checks immutable input and code-provenance hashes, scans the
provenance sources for prohibited future-data operations, and performs a
fresh cutoff replay. Cutoff replay calls the training pipeline again at each
cutoff; it never slices a full-run prediction table. A replay is accepted only
when its prefix keys and factor values match the full run within the reported
absolute tolerance. The complete result is saved as `verification.json`.

## Input catalog

Each factor requires a `<factor_id>.parquet` table and a matching
`<factor_id>.meta.json` sidecar. The sidecar must declare
`source_type: "module"` and `settings.universe: "historical_top50"`. Parquet
tables use the canonical `date`, `instrument`, and `factor` columns.

The model candidates are deterministic. Validation uses signed cross-sectional
RankIC and certified net Sharpe, then the winning candidate is refit on all
available pre-retrain observations. Future labels are evaluation-only and are
never admitted to feature construction, ranking, or inference.
