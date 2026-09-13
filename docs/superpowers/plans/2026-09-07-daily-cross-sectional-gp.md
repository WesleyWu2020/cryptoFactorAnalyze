# Daily Cross-Sectional GP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible daily cross-sectional expression GP miner with 2024 training, 2025 validation, and an isolated 2026-01-01 through 2026-09-01 test period.

**Architecture:** Keep expression search and training evaluation inside `Genetic_Algorithm/`; reuse the existing H5 provider, labels, portfolio accounting, and report framework. Use bounded data reads, point-in-time masks, immutable candidate manifests, and separate commands for each research stage. Extend shared interfaces only where current behavior cannot satisfy these contracts, preserving existing callers.

**Tech Stack:** Python from `./.venv/bin/python`, NumPy, pandas, existing PyTables/HDF5 infrastructure, pytest, standard-library dataclasses/hashlib/json/argparse; no new heavy framework.

---

## Approved specification and execution rules

Read `docs/superpowers/specs/2026-09-07-daily-cross-sectional-gp-design.md` before implementation. The user approved that document and requested this plan in English. Numeric defaults below are research configuration, not empirically established performance guarantees.

Run commands from the repository root. Preserve the dirty working tree, including deleted legacy GA files. This planning pass only adds documentation; it does not create a worktree from HEAD because HEAD omits relevant uncommitted framework changes. Before implementation, record the actual working-tree baseline and use either this workspace or an isolated copy containing the same authorized dependency changes. Never silently implement against an older framework revision.

Follow tasks in order. Each numbered checklist item is a separate execution step. Run the named focused test after writing its assertions, observe an assertion/import failure, implement the listed contract, rerun it, then commit only task-owned changes. Do not stage whole shared files containing unrelated user changes; use reviewed hunks or defer their commit and document that limitation. No task authorizes pushing commits or sending external messages.

## File map

| Path | Responsibility |
|---|---|
| `Genetic_Algorithm/__init__.py` | Package marker, no import-time data access |
| `Genetic_Algorithm/config.py` | Frozen stage and search settings; strict JSON loading |
| `Genetic_Algorithm/data.py` | Stage-bounded panels, masks, quality audit, stage content fingerprints |
| `Genetic_Algorithm/expression.py` | Typed immutable AST, validation, canonical hash, dependencies, warmup |
| `Genetic_Algorithm/operators.py` | Causal numerical operators and explicit operator registry |
| `Genetic_Algorithm/features.py` | Named daily terminal definitions and their raw dependencies |
| `Genetic_Algorithm/evaluator.py` | AST interpretation and bounded intermediate cache |
| `Genetic_Algorithm/fitness.py` | Training IC diagnostics, direction, eligibility and objective vectors |
| `Genetic_Algorithm/evolution.py` | Tree generation, crossover, mutation, deterministic Pareto selection |
| `Genetic_Algorithm/selection.py` | Training correlation deduplication and validation ranking |
| `Genetic_Algorithm/artifacts.py` | Strict JSON, provenance, immutable manifests and training archive |
| `Genetic_Algorithm/replay.py` | Explicit-stage FactorManager adapter; no duplicate accounting |
| `Genetic_Algorithm/export.py` | Loader-compatible formula materialization |
| `Genetic_Algorithm/cli.py`, `Genetic_Algorithm/__main__.py` | Audit/search/validate/freeze/test/export command dispatch |
| `Genetic_Algorithm/configs/default.json`, `Genetic_Algorithm/configs/smoke.json` | Frozen production and low-budget configurations |
| `Genetic_Algorithm/README.md` | English usage, semantics, limitations and reproducibility instructions |
| `tests/genetic_algorithm/` | Focused tests plus synthetic integration fixtures |
| `scripts/verify_daily_gp.py` | Read-only verification of formula causality and run artifacts |
| `.gitignore` | Ignore `Genetic_Algorithm/runs/` only |

Shared changes, each guarded by compatibility tests: `factor_common/data_provider.py` for bounded reads; `factor_common/loader.py` and `factor_common/value_engine.py` for an opt-in eligibility context; `factor_common/manager.py` for an optional provider cutoff. Update `data/crypto_quant/reader.py` only if inspection identifies an unbounded stage path. Do not modify factor_miner contracts or replace existing utility implementations.

## Task 1: Freeze configuration and stage semantics

**Create:** package marker, `config.py`, both configuration JSON files, `tests/genetic_algorithm/test_config.py`.

- [ ] Write a parameterized boundary test using this exact expected calendar:

```python
import pandas as pd
import pytest
from Genetic_Algorithm.config import STAGES

@pytest.mark.parametrize("name,last", [
    ("train", "2024-12-29"),
    ("validation", "2025-12-29"),
    ("test", "2026-08-30"),
])
def test_signal_end(name, last):
    assert STAGES[name].signal_end == pd.Timestamp(last)
```

- [ ] Run `./.venv/bin/python -m pytest tests/genetic_algorithm/test_config.py -q`; expect missing-module failure initially.
- [ ] Define immutable `Stage(name, start, end)` with timestamps and `signal_end = end - pd.Timedelta(days=2)`. Define `STAGES` for the approved inclusive periods. Define frozen `SearchConfig` with strict unknown-key/type/range rejection and `load_config(path)` returning it. Use these defaults in both the dataclass and default JSON:

```json
{
  "seed": 42, "population": 200, "generations": 20,
  "max_depth": 4, "max_nodes": 15, "max_history": 180,
  "windows": [3, 5, 10, 20, 40, 60], "lags": [1, 3, 5, 10],
  "crossover_probability": 0.6, "mutation_probability": 0.3,
  "copy_probability": 0.1, "min_pairs": 20,
  "min_quarter_days": 45, "min_day_coverage": 0.8,
  "min_cell_coverage": 0.8, "correlation_limit": 0.9,
  "min_overlap_days": 120, "validation_limit": 20,
  "frozen_limit": 5, "cache_bytes": 268435456,
  "enable_funding_features": false
}
```

Smoke JSON changes only population to 12 and generations to 2. Keep statistical gates unchanged. Until funding terminals have a separately validated data contract, reject `enable_funding_features=true` with an explicit unsupported-feature error; do not silently ignore it.

- [ ] Add rejection assertions for booleans masquerading as integers, negative budgets, probabilities not summing to one, unknown keys, reversed dates and unsupported funding features. Run the focused test; expect PASS.
- [ ] Commit task-owned files with message `feat: define daily GP stages and configuration`.

## Task 2: Enforce bounded provider reads and audit real training data

**Modify:** `factor_common/data_provider.py`; conditionally `data/crypto_quant/reader.py`.
**Create:** `Genetic_Algorithm/data.py`, `tests/genetic_algorithm/test_data.py`.
**Extend:** `tests/factor_common/test_data_provider.py`.

- [ ] Inspect every provider path used by market, universe, funding and quality retrieval. `CryptoQuantStore.read(name, where=None)` already supports query pushdown. Its constructor call sites and provider range discovery currently read whole tables; replacing returned frames after a read does not satisfy isolation.
- [ ] Add a recording store spy that rejects reads without upper-bound predicates when `as_of` is set. Test market date, universe decision date and funding event timestamp boundaries, including an event exactly at the next midnight. Use a fixture with a future-only symbol and assert it cannot enter the truncated symbol axis. Run `./.venv/bin/python -m pytest tests/genetic_algorithm/test_data.py tests/factor_common/test_data_provider.py -q`; expect the new isolation assertions to fail.
- [ ] Push bounds into store queries. For daily tables use an inclusive UTC-naive day bound; for event tables use an exclusive UTC-aware next-midnight bound. Retain historical behavior when no cutoff is provided. Require queryable schemas and report an explicit data error instead of silently reading the whole table. Apply bounds before symbol discovery and stage content hashing.
- [ ] Implement `load_stage(h5_path, stage, warmup_days, fields)` returning `StageData(stage, features, eligible, quality_eligible, opens, fingerprint, audit)`. All panels preserve daily axes; the provider cutoff is `stage.end`. Labels are deliberately absent from this object. Feature history begins at `stage.start - warmup_days`; execution opens extend only to `stage.end`. Define quality eligibility as current membership AND complete non-placeholder kline; unknown quality is ineligible and counted, not relabeled as complete. Do not mask historical raw observations merely because a symbol was not yet a member: rolling features may use known pre-membership history.
- [ ] Hash only canonical sorted stage rows, required history, membership evidence, selected columns and schema versions. Record code/environment provenance separately. Use float64 values and explicit NaN masks; reject duplicate daily keys, invalid timestamps and inconsistent axes. Audit day counts, per-day eligible counts, missing fields, warmup availability and quality counts. Testing-period audit must not execute during search.
- [ ] Run the focused tests and the existing reader/store tests affected by changed paths. Before formal search, run a read-only training audit using `load_stage` and retain its result later in the run directory. Do not update H5, repair data, run a backtest or read test-period rows for this audit.
- [ ] Commit task-owned changes with message `feat: isolate GP stage data reads`.

## Task 3: Provide opt-in masks to exported formulas

**Modify:** `factor_common/loader.py`, `factor_common/value_engine.py`.
**Extend:** `tests/factor_common/test_loader.py`, `tests/factor_common/test_value_engine.py`.

The current engine calls `calc_factor(ctx)` before loading the universe. This is insufficient for nested cross-sectional operators: post-hoc masking cannot undo ranks computed using ineligible symbols.

- [ ] Add a test with three eligible values `[1, 2, 3]` and an ineligible extreme value. An opt-in factor reading `data_ctx["__eligible__"]` must produce the same eligible ranks whether that extreme column is present or absent. Add a legacy factor asserting `set(data_ctx) == {"close"}`. Run the focused tests and observe the opt-in test fail.
- [ ] Add optional boolean `SETTING.context_eligible`, default false, to loader validation. In `compute_factor`, load and validate the universe before the factor call only for opt-in modules, combine it with the same complete-kline/non-placeholder quality policy as Task 2, then add the reserved `__eligible__` matrix. Keep `data_needed` market-only and preserve the public `calc_factor(data_ctx)` signature. Reject a nonboolean setting and collisions with the reserved key.
- [ ] Apply the combined mask to the final opt-in output as well as to cross-sectional operators. Keep legacy behavior unchanged. Update pipeline fingerprint inputs so caches computed under the old context semantics cannot be reused for opt-in modules.
- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_loader.py tests/factor_common/test_value_engine.py tests/factor_common/test_compat.py -q`; expect PASS.
- [ ] Commit only reviewed new hunks with message `feat: support opt-in factor eligibility context`.

## Task 4: Build the causal expression runtime

**Create:** `expression.py`, `operators.py`, `features.py`, `evaluator.py` and `tests/genetic_algorithm/test_expression.py`, `test_operators.py`, `test_evaluator.py`.

- [ ] Write the first numerical test:

```python
import numpy as np
import pandas as pd
from Genetic_Algorithm.operators import safe_div

def test_safe_div_preserves_missing():
    a = pd.DataFrame([[2.0, 2.0, np.nan]])
    b = pd.DataFrame([[2.0, 0.0, 1.0]])
    result = safe_div(a, b)
    assert result.iloc[0, 0] == 1.0
    assert result.iloc[0, 1:].isna().all()
```

- [ ] Run the three test files, then implement `safe_div(a, b)` as `a.div(b.where(b.abs() > 1e-12)).replace([np.inf, -np.inf], np.nan)`. Register add/subtract/multiply/negate/abs, positive lag/delta, rolling mean/std/min/max/correlation, and masked average-percentile cross-sectional rank. Rolling operators require all observations and use `center=False`; standard deviation uses `ddof=1`. Rank singleton rows return NaN.
- [ ] Define frozen `Node(op, children=(), field=None, window=None)` and `validate_tree(node, config)`, `canonical_tree(node)`, `expression_hash(node)`, `required_fields(node)`, `history_days(node)`. Validate arity and distinguish terminal/operator attributes. Never use eval. History is maximum child history plus window minus one for rolling operators, plus lag for lag/delta. Depth zero means terminal. Canonicalize child order only for commutative add/multiply; preserve order elsewhere.
- [ ] Define deterministic terminal formulas: `return_1d = close / close.shift(1) - 1`, `range_relative = (high-low)/close`, `body_relative = (close-open)/open`, `volume_relative_20 = volume/volume.rolling(20).mean()`, `quote_volume_relative_20` analogously, `taker_base_ratio = taker_buy_base_volume/volume`, `taker_quote_ratio = taker_buy_quote_volume/quote_volume`, `quote_per_trade = quote_volume/trade_count`. All divisions use safe_div; raw OHLCV and flow fields remain available terminals. Terminal history is included in total warmup.
- [ ] Implement `evaluate_tree(node, data_ctx, eligible, cache=None)` returning a float64 matrix on unchanged axes. Mask every cross-sectional rank input using that day's eligible matrix; apply the final mask. Cache keys include expression, data fingerprint, operator-source hash and eligibility fingerprint. Evict least-recently-used entries until total accounted matrix bytes fit `cache_bytes`; do not cache an individually oversized entry.
- [ ] Add independent numerical expectations for ties, incomplete rolling windows, cumulative history, zero denominators, future-only symbols and nested rank expressions. Assert illegal negative lags, unknown fields and excessive trees are rejected. Run focused tests; expect PASS.
- [ ] Commit with message `feat: add constrained causal GP expressions`.

## Task 5: Implement training fitness and coverage accounting

**Create:** `fitness.py`, `tests/genetic_algorithm/test_fitness.py`.

- [ ] Add synthetic 30-symbol panels where known factor order exactly matches forward return order; use `factor_common.labels.make_labels(opens, 1)`, not a second label formula. Test a reversed factor obtains direction -1 using training only. Test fewer than 20 pairs and constant labels produce no valid daily IC. Run the focused file and observe failure.
- [ ] Define `TrainingScore(direction, mean_ic, worst_quarter_ic, icir, quarter_means, valid_days, day_coverage, cell_coverage, node_count, eligible, reasons)` and `score_training(values, labels, quality_eligible, config)`. Reject any signal outside the training range. Calculate pairwise daily average-rank Spearman, checking at least 20 finite pairs and nonconstant ranks. Reuse the existing IC implementation where possible without changing its legacy minimum; either filter its n_pairs output or add a backwards-compatible minimum argument.
- [ ] Derive direction from full training mean only. Apply it to all quarterly means and ICIR. Use four calendar quarters, at least 45 valid IC days each, day and cell coverage at least 0.8, and at least three strictly positive quarters. Cell coverage denominator is all quality-eligible signal-day cells, not observed factor cells or valid labels. Zero/invalid IC variance produces null ICIR, never Infinity. Objective vector is `(mean_ic, worst_quarter_ic, -node_count)` for eligible scores.
- [ ] Add a fixture that intentionally removes factors on losing days and assert the coverage denominator does not shrink. Assert purged December 30/31 rows cannot affect scores. Run `./.venv/bin/python -m pytest tests/genetic_algorithm/test_fitness.py -q`; expect PASS.
- [ ] Commit with message `feat: score GP training stability and coverage`.

## Task 6: Implement deterministic genetic search

**Create:** `evolution.py`, `tests/genetic_algorithm/test_evolution.py`.

- [ ] Test nondominated selection on hand-enumerated objective tuples, equal-score tie ordering, invalid offspring rejection and identical output for repeated seeds. Require an empty eligible result to remain empty. Run the test file; expect failure.
- [ ] Define `Candidate(tree, expression_id, score)` and `SearchResult(candidates, generation_log, evaluations)`. Implement `search(stage_data, config)` with a local `numpy.random.Generator`; labels are constructed only inside the training evaluation path. Generate valid trees with a maximum of `population * 50` attempts per requested population. Select parents by rank/crowding tournament, then crossover a random subtree, mutate a random subtree, or copy according to configured probabilities.
- [ ] Canonicalize and deduplicate expression hashes before evaluation. Merge parent/offspring pools, apply nondominated sorting and crowding-distance truncation, then deterministic complexity/hash ties. Eligibility failures may supply exploratory parents only when no valid candidates exist; they never enter the published training shortlist. Record attempted trees, unique evaluations, cache hits, invalid-reason counts and eligible population counts per generation. Stop with a structured error if the legal population cannot be formed within the attempt budget.
- [ ] Run `./.venv/bin/python -m pytest tests/genetic_algorithm/test_evolution.py -q`; expect PASS, including a fixed synthetic search with a known deterministic result.
- [ ] Commit with message `feat: evolve reproducible daily GP populations`.

## Task 7: Add training archives, deduplication and provenance

**Create:** `artifacts.py`, `selection.py`, `tests/genetic_algorithm/test_artifacts.py`, `test_selection.py`.
**Modify:** `.gitignore` by adding `Genetic_Algorithm/runs/`.

- [ ] Test identical and sign-inverted factor values are duplicates; 119 common valid days are insufficient evidence; 120 valid days with absolute mean daily correlation below 0.90 can establish novelty. The first candidate with no archive comparisons is admissible. Test corrupt hashes, nonfinite JSON metrics and attempts to overwrite an existing manifest.
- [ ] Implement `deduplicate_training(candidates, values_by_id, archive, config)`. Order by descending worst-quarter IC, descending mean IC, ascending node count, then hash. Compute mean of absolute daily Spearman correlations on at least 20 common symbols. Reject duplicate or insufficient-overlap comparisons against every accepted/archive candidate; keep at most 20. Store explicit rejection reasons.
- [ ] Define `write_artifact(path, payload)` using exclusive creation for immutable manifests and atomic temporary-file replacement for progress files. Serialize metrics with null instead of NaN/Infinity and `allow_nan=False`. Define `read_verified_manifest(path)` verifying a SHA-256 over canonical payload excluding its own digest.
- [ ] Record config, stage content hashes, selected code-content hashes, working-tree patch hash, package versions, seed, operator version and complete resolved backtest profile. Training archive contains only ASTs and training-derived diagnostics/fingerprints. Update it after training, independent of validation outcome; restrict comparisons to matching training fingerprints/operator versions. Record separate experiment IDs and validation attempt counts. No validation or test metrics may be used as archive search weights.
- [ ] Run both focused test files; expect PASS. Commit with message `feat: record GP provenance and training novelty`.

## Task 8: Export formulas and replay bounded stages

**Create:** `export.py`, `replay.py`, `tests/genetic_algorithm/test_export.py`, `test_replay.py`.
**Modify:** `factor_common/manager.py`; extend `tests/factor_common/test_manager.py`.

- [ ] Write a loader round-trip test for a nested rank expression and compare its result with `evaluate_tree` under changing membership. Test an existing different export cannot be overwritten. Add a manager cutoff regression: end-of-signal plus the two-day tail equals the stage end, and all underlying data reads honor that bound.
- [ ] Implement `export_factor(candidate, destination)` with identifier `GP_<expression_hash_prefix>`; detect collisions against full hashes. Render a literal AST dictionary, direction, dependencies and warmup into a regular factor module. Use `META` fields from the existing template, `preprocessing="none"`, `universe="historical_top50"`, and `context_eligible=True`. Its `calc_factor` reconstructs a validated Node and calls `evaluate_tree(..., data_ctx["__eligible__"])`; no runs-directory dependency or embedded training labels. Hash all transitive runtime modules in the artifact because the wrapper source alone is not a complete formula implementation fingerprint.
- [ ] Add keyword-only `as_of=None` to FactorManager construction, passed to its provider. Preserve default behavior. Reject an explicit requested execution tail beyond as_of. Ensure internal cutoff checks use the tighter of their cutoff and manager cutoff. Do not set a private `_dp` from GP code.
- [ ] Implement `replay(candidate, stage, h5_path, run_dir)` using exported script plus `FactorManager(as_of=stage.end, ...)`, params `start=stage.start`, `end=stage.signal_end`, `rebalance_days=1`, fixed training direction, `n_groups=10`, fees 0.0005, slippage 0.001, strict funding, gross exposure 1.0 from the resolved profile. Preserve the framework's 50/50 long-short weights and accounting. Verify final liquidation date is at most stage.end and final positions are zero. A non-complete all_costs result is a failed evaluation, not zero return.
- [ ] Extract stage-wide all_costs metrics from the framework's ledger and existing summary helpers; never use its automatic internal IS/OOS split as the GP stage selector. Store the resolved profile and complete framework result. Render each stage under `reports/` with stage/run identifiers. Disclose any internal framework split in report metadata rather than relabeling it as the three-way research split.
- [ ] Run `./.venv/bin/python -m pytest tests/genetic_algorithm/test_export.py tests/genetic_algorithm/test_replay.py tests/factor_common/test_manager.py tests/factor_common/test_reporting.py -q`; expect PASS, including entry/exit costs and boundary funding behavior.
- [ ] Commit reviewed changes with message `feat: export GP formulas and replay isolated stages`.

## Task 9: Implement validation, freezing and independent testing

**Extend:** `selection.py`, `artifacts.py`, `replay.py`.
**Create:** `tests/genetic_algorithm/test_stage_workflow.py`.

- [ ] Test negative validation net return fails, a validation direction reversal is not allowed, null Sharpe fails, and zero selected candidates is valid. Test any mutation of the frozen AST/config/direction or runtime source digest prevents test evaluation.
- [ ] Implement `select_validation(training_candidates, validation_results, config)` with unchanged training direction, 80% day/cell coverage, 45 valid days in each quarter, positive mean IC, three positive quarters, positive all_costs cumulative return and finite net Sharpe. Sort by descending net Sharpe, ascending turnover, ascending complexity, then hash; keep at most five. Record every rejection. Do not impose the four-quarter validation gate on the partial-year test period; test reports all candidates without eligibility-based deletion.
- [ ] Implement `freeze_candidates(...)` as a pure selection/provenance serialization step with no new market reads. Include the training and validation fingerprints, all candidate IDs, full ASTs, directions, complete config and profile, runtime source hashes and selection results. Test data fingerprint is absent until testing because freeze cannot read it.
- [ ] Implement `test_manifest(...)`: verify frozen inputs and historical training/validation fingerprints, then load bounded test data with permitted history; write test data fingerprint into a separate test receipt. Evaluate all frozen candidates, preserve failures, and never write the training archive. A repeated manifest invocation gets a new receipt marked as a repeat. Maintain a test-access log so later experiments cannot claim an unseen holdout after results were accessed.
- [ ] Run `./.venv/bin/python -m pytest tests/genetic_algorithm/test_stage_workflow.py -q`; expect PASS. Commit with message `feat: freeze validation candidates before holdout evaluation`.

## Task 10: Wire the command-line workflow and English documentation

**Create:** `cli.py`, `__main__.py`, `README.md`, `tests/genetic_algorithm/test_cli.py`.

- [ ] Test `--help`, missing manifest, invalid configuration, stage misuse and zero-candidate exit behavior using subprocesses against temporary paths. Run the focused file; expect failure.
- [ ] Expose these commands and arguments exactly; use argparse subcommands, explicit paths and nonzero exit codes for invalid inputs or infrastructure failures. Successful zero-candidate search/validation exits zero with `status=no_candidates`.

```bash
./.venv/bin/python -m Genetic_Algorithm audit --stage train --h5 data/crypto_quant.h5 --output Genetic_Algorithm/runs/audit_train.json
./.venv/bin/python -m Genetic_Algorithm search --config Genetic_Algorithm/configs/default.json --h5 data/crypto_quant.h5 --run-dir Genetic_Algorithm/runs/daily_001
./.venv/bin/python -m Genetic_Algorithm validate --run-dir Genetic_Algorithm/runs/daily_001 --h5 data/crypto_quant.h5
./.venv/bin/python -m Genetic_Algorithm freeze --run-dir Genetic_Algorithm/runs/daily_001
./.venv/bin/python -m Genetic_Algorithm test --manifest Genetic_Algorithm/runs/daily_001/frozen.json --h5 data/crypto_quant.h5
./.venv/bin/python -m Genetic_Algorithm export --manifest Genetic_Algorithm/runs/daily_001/frozen.json --output-dir factor_analyse/factor_mining
```

- [ ] Define output filenames as `config.json`, `audit_train.json`, `provenance.json`, `training_candidates.json`, `generations.jsonl`, `deduplication.json`, `validation.json`, `frozen.json`, and `test_receipts/<receipt_id>.json`. Intermediate replay wrappers stay under the run directory; final export goes to factor_mining. Never silently reuse an existing run directory for a different configuration. Test cannot be invoked automatically by search or validate.
- [ ] Document time semantics, cost units, quality handling, stage commands, smoke config, artifact schemas, formula runtime dependencies, no-candidate outcomes, and holdout reuse limitations in English. Explain that a successful pipeline is not proof of a useful factor. Include stage-specific report locations and how to reproduce a frozen run.
- [ ] Run `./.venv/bin/python -m pytest tests/genetic_algorithm/test_cli.py -q` and `./.venv/bin/python -m Genetic_Algorithm --help`; expect PASS and all six commands. Commit with message `feat: expose daily GP research CLI`.

## Task 11: Verify causality, stage isolation and the complete path

**Create:** `tests/genetic_algorithm/conftest.py`, `test_causality.py`, `test_isolation.py`, `test_end_to_end.py`, `scripts/verify_daily_gp.py`.

- [ ] Build a deterministic temporary H5 fixture using existing TABLE_SPECS/store helpers with at least 30 symbols, 180 warmup days, all three stages, membership changes, complete funding-price evidence and intentional gap scenarios. Block network use in tests. Reuse existing fixture construction patterns from `tests/factor_common/conftest.py` without importing a sibling conftest module as a library.
- [ ] Add prefix comparison for multiple cutoffs including a new listing and a year boundary. Reuse `factor_common.validation.check_cutoff` and its future-only-column policy; supplement relative tolerance only where needed. Check equal date axes and NaN masks, no finite values hidden by dropping columns, and `np.testing.assert_allclose(full_prefix, truncated, atol=1e-10, rtol=1e-8, equal_nan=True)`. Report maximum absolute difference and cutoff dates.
- [ ] Add mutation-isolation tests: changing validation/test rows cannot change the training expression IDs/scores/archive; changing test rows cannot change validation results or frozen digest. Separately use the recording read spy to prove upper-bound query isolation; value equality alone does not prove no future reads. Provenance must not include full-file content hashes that change when unrelated future rows change.
- [ ] Implement `scripts/verify_daily_gp.py --run-dir PATH --h5 PATH` to statically scan all new runtime modules and exported wrappers with the existing scanner, allowing future operations only in `factor_common/labels.py`, and dynamically recompute exported formulas at cutoffs. Save `verification.json` with file/line findings, allowance purposes, cutoff differences and failures. Exit nonzero on a prohibited feature operation or prefix mismatch.
- [ ] Run the complete relevant suite and syntax check:

```bash
./.venv/bin/python -m pytest tests/genetic_algorithm tests/factor_common tests/test_crypto_quant_store.py -q
./.venv/bin/python -m compileall -q Genetic_Algorithm scripts/verify_daily_gp.py
git diff --check
```

Expected: all tests pass, compileall succeeds, no whitespace errors. Broaden tests only if modified reader/provider paths affect additional consumers; run their existing tests before completion.

- [ ] Run a real-data smoke search with `configs/smoke.json` and a fresh run directory, then validation and freeze. If candidates survive, run the independent test once and export them; inspect generated stage HTML reports. If none survive, retain that result, run the full synthetic workflow, and verify a separately identified fixed momentum baseline through the report path. Never reduce gates to manufacture an accepted candidate. Data failures are recorded blockers, not successful smoke results.
- [ ] For each exported real candidate run the standard loader/report path with explicit stage bounds, for example the test period:

```bash
./.venv/bin/python factor_analyse/main.py /absolute/path/to/GP_IDENTIFIER.py 1 --start 2026-01-01 --end 2026-08-30
```

Replace the path with an actual generated filename, and check the command's existing argument spelling before execution. Use the bounded GP replay for holdout certification; the standard CLI call is compatibility verification and must be marked as repeat evaluation if it reopens the holdout. It must not influence selection.

- [ ] Run the verification script, record actual commands and output paths, and commit task-owned tests/docs with message `test: verify GP causality and staged workflow`.

## Completion checklist and handoff

- [ ] All specification sections map to Tasks 1–11: scope/config (1), data (2), interfaces (3), AST (4), fitness (5), search (6), novelty/provenance (7), export/accounting (8), freeze/test (9), CLI/artifacts (10), acceptance (11).
- [ ] Report changed files, executed commands, actual candidate counts, report/manifest paths, data limitations, static scan findings and dynamic max_abs_diff. Distinguish implemented, tested and statistically validated claims.
- [ ] Confirm the default budget remains 200 × 20; smoke completion does not claim a full-budget search was run.
- [ ] Preserve failed candidates and test receipts; do not claim an independent holdout if it was used for further design decisions.
- [ ] Perform an inline plan/spec review before execution. No implementation has been performed by writing this plan.
