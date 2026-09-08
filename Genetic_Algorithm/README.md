# Daily genetic-algorithm workflow

This package searches causal daily cross-sectional formulas.  A formula at day
`t` may use only data available at `t` or earlier: rolling windows are trailing,
membership is point-in-time, and labels are never inputs to the formula.
Training is 2024, validation is 2025, and test is the frozen 2026 holdout.

Run each stage explicitly; `search` and `validate` never invoke `test`.

```bash
./.venv/bin/python -m Genetic_Algorithm audit --stage train --h5 data/crypto_quant.h5 --output Genetic_Algorithm/runs/audit_train.json
./.venv/bin/python -m Genetic_Algorithm search --config Genetic_Algorithm/configs/default.json --h5 data/crypto_quant.h5 --run-dir Genetic_Algorithm/runs/daily_001
./.venv/bin/python -m Genetic_Algorithm validate --run-dir Genetic_Algorithm/runs/daily_001 --h5 data/crypto_quant.h5
./.venv/bin/python -m Genetic_Algorithm freeze --run-dir Genetic_Algorithm/runs/daily_001
./.venv/bin/python -m Genetic_Algorithm test --manifest Genetic_Algorithm/runs/daily_001/frozen.json --h5 data/crypto_quant.h5
./.venv/bin/python -m Genetic_Algorithm export --manifest Genetic_Algorithm/runs/daily_001/frozen.json --output-dir factor_analyse/factor_mining
```

`configs/smoke.json` is a deliberately small smoke configuration; it confirms
the pipeline, not factor quality.  A successful pipeline is not evidence of a
useful factor.  `no_candidates` is a successful, informative outcome when all
candidates fail quality, novelty, or validation gates.

## Data, costs, and quality

Prices and volumes are daily Binance panels.  Quality eligibility excludes
unknown membership, placeholders, and incomplete klines.  Replay uses the
standard `perp_1d` profile: costs are fractions of gross notional (fee `0.0005`
and slippage `0.001` per side) and funding is enabled in strict mode.  Metrics
come from the certified `all_costs` ledger; an incomplete ledger is a failure,
not a zero return.

## Artifacts and reproduction

The run directory contains `config.json`, `audit_train.json`, `provenance.json`,
`training_candidates.json`, `generations.jsonl`, `deduplication.json`,
`validation.json`, and `frozen.json`.  Replay wrappers and their reports stay
under that run directory.  Exports alone are written to `factor_analyse/factor_mining`.
Test receipts are immutable evidence of holdout access; do not repeatedly use a
holdout as an iterative tuning target.

`training_candidates.json` holds training-only ASTs and diagnostics;
`deduplication.json` records novelty comparisons; `validation.json` holds stage
results and gate decisions; `frozen.json` binds selected ASTs, stage
fingerprints, configuration, profile, selection record, and runtime hashes.
To reproduce a frozen run, retain its H5 snapshot and run `test` or `export`
against that exact verified `frozen.json`; changed runtime hashes intentionally
block a test replay.

Exported factor modules depend on `Genetic_Algorithm.evaluator`,
`Genetic_Algorithm.expression`, and the normal `factor_common` point-in-time
context (including `__eligible__`).
