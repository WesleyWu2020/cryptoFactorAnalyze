# Daily portfolio signal pipeline

This package turns the fixed strategy from report
`20260920T080140Z_756193dec1b9` into a traceable daily target-weight signal.
It reuses the same factor computation, five-group construction, member target,
portfolio combination, and risk scaling code as the research workflow.

## Historical replay

Run this first. The selected date must exist in the source report:

```bash
./.venv/bin/python -m daily_pipeline.main replay \
  --signal-date 2026-08-24 --no-publish
```

The command writes daily artifacts plus `replay.json` under
`outputs/daily_pipeline/runs/<signal-date>/<run-id>/`. A successful replay has
zero (or <= 1e-12) difference from the report's factor values, member targets,
and combined portfolio target.

## Daily run

At or after 08:00 Asia/Shanghai:

```bash
./.venv/bin/python -m daily_pipeline.main run
```

The command updates `data/crypto_quant.h5`, validates the store and target-day
readiness, computes the fixed strategy, writes an immutable run directory, and
atomically publishes `outputs/daily_pipeline/published/latest.json`.

For a local calculation using an already-updated store:

```bash
./.venv/bin/python -m daily_pipeline.main run \
  --skip-update --signal-date YYYY-MM-DD --no-publish
```

`action=rebalance` is emitted only when the next execution day belongs to the
research schedule `(execution_date - 2024-01-01) % 5 == 0`. Other valid days
emit `action=hold`. A failed or missing signal is never interpreted as a flat
target.
