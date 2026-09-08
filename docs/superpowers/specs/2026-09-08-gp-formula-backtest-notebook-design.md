# GP Formula Export and Backtest Notebook

## Goal

Create `factor_analyse/gp_formula_backtest.ipynb`, following the operational
style of `factor_common_template_usage.ipynb`.  It lets a researcher export one
frozen GP candidate and backtest an exported GP factor over explicitly entered
dates.

## Inputs

All filesystem inputs are absolute paths:

- `MANIFEST_PATH`: a verified `frozen.json` manifest.
- `EXPRESSION_ID`: the full expression hash of the candidate to export.
- `EXPORT_DIR`: the absolute destination directory for the generated factor.
- `FACTOR_PATH`: the absolute path of an exported GP factor module to evaluate.
- `H5_PATH`: the absolute CryptoQuant H5 path.
- `START_DATE`, `END_DATE`, and `REBALANCE_DAYS`: explicit backtest bounds and
  frequency.

## Notebook flow

1. Resolve the repository root, put it first on `sys.path`, and enable
   autoreload for local development.
2. Load and verify the frozen manifest, display candidate IDs and ASTs, and
   export the selected candidate with its frozen direction.
3. Construct `FactorManager` from the absolute H5 and report paths, then run a
   bounded evaluation for the entered start/end dates.
4. Display the factor panel, IC summary, all-costs performance metrics, and
   save an HTML report under `reports/`.
5. Display static future-leak and cutoff diagnostics from the evaluation.

## Safety and errors

- The notebook never edits a frozen manifest or changes candidate direction.
- It rejects a requested end date that cannot be liquidated from available
  market data, rather than silently extending reads.
- The user must explicitly opt in before running an interval overlapping the
  2026 holdout, because that is a repeat holdout access and must not influence
  selection.
- Invalid paths, absent expression IDs, invalid date order, and incomplete
  backtests raise clear errors.

## Verification

Verify notebook JSON structure and compile every code cell.  Exercise the
input validation helpers without running a real backtest.
