# `factor_common` user guide

The `factor_common` package is the common factor-research framework over the
point-in-time `data/crypto_quant.h5` store. It computes daily factor values
from pure `.py` factor modules, persists versioned Parquet/JSON artifacts,
accounts a fixed-quantity daily long-short portfolio under explicit cost
scenarios, evaluates IC-family and portfolio metrics, and renders a
self-contained HTML report. This guide documents the exact execution order,
artifact schemas, cost and timing conventions, sample definitions, the
legacy compatibility mapping, unsupported features, and known data blockers.

## Quick start

```bash
# 0. Install the runtime, test, and notebook dependencies
./.venv/bin/python -m pip install -r requirements-dev.txt

# 1. Refresh and validate the data store (see docs/crypto_quant_data.md)
./.venv/bin/python data/update_crypto_quant.py update
./.venv/bin/python data/update_crypto_quant.py validate

# 2. Scaffold a new factor module (never overwrites an existing file)
./.venv/bin/python -c "from factor_common import FactorManager; print(FactorManager().create_template('my_factor'))"

# 3. Edit factor_analyse/factor_mining/my_factor.py, then evaluate it
./.venv/bin/python factor_analyse/main.py my_factor 1 --start 2024-02-01 --end 2024-03-15

# 4. Executable acceptance of the whole workflow (read-only against the H5)
./.venv/bin/python scripts/verify_factor_common.py --h5 data/crypto_quant.h5 \
    --start 2024-02-01 --end 2024-03-15 --output-dir <fresh-dir> --execute-notebook
```

The reference example is `factor_analyse/factor_mining/example_momentum.py`;
the interactive walkthrough is `factor_analyse/factor_common_usage.ipynb`.
Programmatic entry point:

```python
from factor_common import FactorManager

manager = FactorManager()  # defaults: data/crypto_quant.h5, data/factor_results, reports
result = manager.evaluate(
    "factor_analyse/factor_mining/example_momentum.py",
    params={"start": "2024-02-01", "end": "2024-03-15", "rebalance_days": 1},
)
print(result["status"])                       # complete | incomplete | insufficient_data
print(result["factor_performance"]["samples"]["full"]["ic"]["rank_ic_mean"])
print(result["factor_performance"]["scenarios"]["trading_net"]["full"])
print(result["paths"]["report_path"])
```

## Exact execution order

`FactorManager.evaluate(source, *, factor_name=None, profile_id="perp_1d",
params=None, plot=True)` executes, in order:

1. **Resolve parameters.** `params` may contain only `start`, `end`, and the
   profile fields `rebalance_days`, `anchor_date`, `n_groups`,
   `factor_direction`, `fee_rate`, `slippage`, `include_funding`,
   `funding_price_mode`, `split_date`, `out_of_sample_days`. Unknown keys
   raise `ValueError`;
   formula parameters belong in the module's `SETTING["params"]`. When
   `start`/`end` are omitted, `end` defaults to the last stored market date
   and `start` to 364 days earlier.
2. **Load and validate the source.** A `.py` path is loaded by
   `loader.load_factor` (contract: `TYPE="regular"`, `META`, `SETTING`,
   `calc_factor(data_ctx)`; the file stem must equal `META.factor_name`) and
   statically scanned by `validation.scan_future_leaks` — any banned pattern
   aborts before data is read. A DataFrame source is normalized into a daily
   date-by-instrument matrix (duplicate keys, intraday timestamps, and
   non-numeric values are errors) and requires an explicit `factor_name`.
3. **Snapshot the input store.** The H5 size/mtime are captured before the
   first value-phase read; `save_value` re-stats the file and raises
   `ConcurrentSourceChangeError` if it changed in between. All store reads
   are `mode="r"` — evaluation never mutates the H5.
4. **Compute values.** `value_engine.compute_factor` loads
   `SETTING["data_needed"]` fields plus point-in-time membership from
   `start - warmup_bars` through `end`, evaluates `calc_factor`, applies the
   configured preprocessing (`none` or `mad_rank`), and masks values for
   ineligible instruments.
5. **Validate causality.** `check_cutoff` re-runs the factor with provider
   knowledge (market history AND membership decisions) truncated at interior
   cutoffs and requires agreement with the full-history prefix
   (`max_abs_diff <= 1e-12`). A leak raises `ValueError` and nothing is
   persisted.
6. **Persist values.** The factor matrix is written before any
   execution-tail or funding data is touched, so a later funding failure can
   never discard a successfully computed value. The flat value cache is keyed
   on the factor source hash, settings, date range, input-store stat, and a
   `pipeline_fingerprint` of the value-pipeline sources (`value_engine`,
   `preprocessing`, `data_provider`, `data/crypto_quant/reader`); any edit to
   those modules — or a cache written before the fingerprint existed —
   invalidates the cache instead of silently reusing stale values.
7. **Load evaluation inputs.** Execution-tail opens (through
   `end + 1 + rebalance_days`), raw funding events, and per-day quality
   flags for the value columns; the content hashes of all three are recorded
   in the evaluation metadata.
8. **Account.** `backtest.run_backtest` simulates the three scenarios (see
   below) over the daily UTC grid.
9. **Evaluate metrics.** `metrics.evaluate_metrics` combines values,
   evaluation-only labels, and the single accounting into per-sample and
   per-scenario blocks; deterministic grouping produces saved group-return
   tables.
10. **Persist the evaluation** under its content-derived id and optionally
    **render** the HTML report (`reporting.render_result`), a pure display
    transform over the saved tables.

Result `status` is `"insufficient_data"` when no signal is usable,
`"incomplete"` when any scenario's accounting is not certified, and
`"complete"` only when every scenario completed with every cash flow
resolved. Data limitations surface through these statuses and null metrics —
never as silently zero-filled numbers. Invalid configuration, corrupt
schemas, and I/O failures raise exceptions instead.

## Factor module contract

```python
import numpy as np

TYPE = "regular"

META = {"factor_name": "example_momentum", "author": "local",
        "level": "daily", "category": "momentum", "description": "N-day log momentum"}

SETTING = {"data_needed": ["close"], "universe": "historical_top50",
           "warmup_bars": 20, "preprocessing": "mad_rank",
           "params": {"window": 20}, "factor_direction": 1}

def calc_factor(data_ctx):
    close = data_ctx["close"]
    return np.log(close / close.shift(SETTING["params"]["window"]))
```

- `data_needed` fields are the H5 daily kline columns (`open`, `high`,
  `low`, `close`, `volume`, ... as listed by `DataProvider.list_datas()`);
  `data_ctx[field]` is a date-by-instrument matrix covering the warmup
  prefix through `end`.
- `universe` is always `"historical_top50"`: the point-in-time membership
  reconstructed from decision dates, never future-informed.
- `factor_direction` is `1` (high factor value predicts outperformance) or
  `-1`; it selects which tail group is the long leg.
- Importing the module must be side-effect free; executable entry points
  live under `if __name__ == "__main__":`.
- The `factor` output may depend only on data at or before each timestamp.
  Banned in the value path (statically scanned): `shift(-k)` and any
  explicitly negated shift, `rolling(..., center=True)`, `bfill`/`backfill`
  (including `fillna(method=...)`), and `merge_asof(direction="forward")`.
  The only module permitted future data is `factor_common/labels.py`, whose
  forward returns are evaluation-only labels: they feed IC/group-return
  statistics and never enter factor construction, grouping, or trading.

## Artifact schemas

All artifacts live under `base_dir` (default `data/factor_results/`) as
Parquet tables and JSON scalars only — code and arbitrary objects are never
pickled. Run and evaluation ids are content-derived from canonical JSON over
the source digest, formula settings, requested/loaded ranges, input-content
hashes (including membership), the profile, and the evaluation-input hashes;
a parameter or data-fingerprint change can never silently reuse a previous
result, and re-saving identical fingerprints with different content raises.

### Factor run — `<factor_id>/<run_id>/`

- `factor.parquet`: long format, columns `date`, `instrument`, `factor`;
  unique, stably sorted keys; only finite values are stored as rows.
- `metadata.json`: full date/column axes (so empty rows/columns round-trip),
  value diagnostics, source digest/settings/ranges, input hashes, and the
  pre-read source snapshot. `latest_run.json` points at the latest
  successful run.

### Evaluation

Evaluation persistence is opt-in with `FactorManager(..., persist_evaluations=True)`.
The default `FactorManager` keeps the complete evaluation in the returned
`result` only, so running `evaluate()` does not create evaluation artifacts.
When persistence is enabled, the manager's flat storage mode retains only the
current evaluation:

- `<factor_id>.evaluation.json`: status, profile, evaluation-input hashes,
  metrics (`factor_performance`), diagnostics, and the scenario scalar blocks.
- `<factor_id>.evaluation.<key>.parquet`: one sidecar per saved DataFrame —
  `factor_value` is not duplicated because it loads from the factor cache; the
  flattened accounting tables are `<scenario>__<table>` plus `group_returns`
  and the optional `benchmark`.

Saving a new evaluation atomically replaces the current evaluation and removes
table sidecars that are no longer referenced. Historical evaluation archives
are available only in the non-flat storage mode.

### Scenario tables (per scenario `gross` / `trading_net` / `all_costs`)

- `ledger` (indexed by UTC date): `equity`, `return`, `price_pnl`,
  `funding_cashflow`, `fee`, `trade_notional`. The row for date `D` values
  holdings at `D`'s open and includes orders executed at that open plus
  every funding cash flow with a timestamp in `[D, D+1)` belonging to held
  quantities. The first row's `return` is relative to
  `profile.initial_equity`, so the inception fee is visible.
- `orders`: `date`, `instrument`, `side`, `target_weight`,
  `target_quantity`, `order_quantity`, `price`, `notional`, `fee`, `status`
  (`filled`/`failed`), `reason` (`invalid_price`, `prior_bar_ineligible`).
  Rejected weights are never redistributed.
- `positions`: date-by-instrument post-trade quantities (fixed between
  boundaries).
- `valuation_prices`: the last boundary open used to value each holding.
- `funding` (`all_costs` only): one row per settlement event on a held
  quantity — `funding_time`, `instrument`, `quantity`, `funding_rate`,
  `mark_price` (raw, preserved verbatim), `settlement_price`,
  `price_approximated`, `resolved`, `unresolved_reason`, `cashflow`.
  Unresolved events carry `cashflow=NaN` and are excluded from totals rather
  than treated as zero.
- `funding_coverage` (`all_costs` only): `date`, `instrument`,
  `observed_status`, `status`, `partial_day`, `accepted`.

## Cost, timing, and sign conventions

- **`fee_rate`** is charged per unit of traded notional on every fill
  (`fee = |order_quantity * price| * fee_rate`), sized after the day's
  already-due funding and deducted from equity after sizing. The default
  `0.0005` is 5 bps per side.
- **`slippage`** is charged the same way (`slippage = notional * slippage`)
  as a separate ledger/order column, modelling the one-sided market-impact
  cost of crossing the spread at the open. The default `0.001` is 0.1% per
  side. Both `fee` and `slippage` are zero in the `gross` scenario.
- **Scenarios** are independent accountings sharing signals and schedule:
  `gross` (no fees, no funding), `trading_net` (fees only), and `all_costs`
  (fees plus funding when `include_funding=True`). Each sizes orders from
  its own equity; no scenario's costs touch another's NAV.
- **Next-open execution**: a signal dated `t` trades at the open of
  `t + signal_delay_days` (`= 1` in the daily profile) on scheduled
  execution dates `(date - anchor_date).days % rebalance_days == 0`.
  Holdings never overlap: each entry is liquidated at the open exactly
  `rebalance_days` after entry.
- **Labels** are evaluation-only next-open forward returns built with
  `hold_days = rebalance_days`; entry at `t+1` open, exit at
  `t+1+rebalance_days` open.
- **Boundary event order** per UTC day `D`: (1) value pre-existing holdings
  at `D`'s open (an unpriceable holding halts certified valuation; the
  position stays in diagnostics and never disappears); (2) settle funding
  exactly at `D 00:00` on pre-trade quantities, so a first entry owns no
  boundary funding and the final boundary settles before liquidation;
  (3) execute scheduled orders; (4) settle intraday funding on the
  unchanged post-trade quantities; (5) check funding coverage of day `D` for
  every held instrument — an unresolved event or unaccepted coverage day
  halts certified continuation, retaining known cash flows through the first
  unresolved one and never computing later exact sizes from assumed-zero
  funding; (6) record the certified ledger row.
- **Cash-flow signs**: `funding_cashflow = -(quantity * settlement_price *
  funding_rate)` — longs pay positive rates, shorts receive them. `fee` is
  always nonnegative and reduces equity. `price_pnl` is signed by quantity.
- **Eligibility**: a zero-volume flat-OHLC placeholder bar at `D-1` blocks
  new entries at `D`'s open only; exits and reductions are always permitted.
  A placeholder flag known only at `D`'s close never excludes `D`'s opening
  trade; such fills are reported separately as retrospective nonexecution
  evidence.

## Funding price modes and coverage policy

- `funding_price_mode="strict"` (default): an invalid or missing mark price
  makes the cash flow unresolved — never approximated, never zero.
- `funding_price_mode="daily_open_approx"`: an invalid mark price is
  replaced by the same day's open (when the open exists and precedes the
  event timestamp), flagged `price_approximated=True`; the raw
  `mark_price` is preserved verbatim in the funding table.
- Coverage is proven per `(date, instrument)` by an **independent expected
  settlement schedule**: `complete` (all expected events observed),
  `not_applicable` (schedule proves no settlement), `missing`,
  `schedule_mismatch`, `invalid_rate`, `no_events` (observed none, no
  schedule), or `unknown` (no evidence). Only `complete` and
  `not_applicable` are accepted; a partially covered day cannot claim
  completeness from daily observations. **Observed events alone never prove
  completeness, and a settlement schedule must never be fabricated from
  observed timestamps.**

## Sample definitions

`factor_performance["samples"]` and the per-scenario blocks are sliced into
`full`, `in_sample`, and `out_of_sample`. The split date is
`profile.split_date` when given, otherwise `last_date - out_of_sample_days`
(180 natural days by default). In-sample keeps every signal whose concrete
exit date (`entry + rebalance_days`) is on or before the split;
out-of-sample admits signals whose entry is strictly after the split; a
signal whose holding period crosses the split is purged from both slices but
kept in the full sample. Purging compares concrete exit dates — never label
NaN-ness. Scenario slices are plain date slices of the certified ledger: the
portfolio is never rerun per slice, and out-of-sample continues the
in-sample equity path without any reset trade. When a scenario's accounting
is incomplete, its per-sample metric blocks are null and a separately
labeled `known_segment` summarizes only the certified ledger prefix (it is
not comparable to full-window metrics).

Missing metrics are always `None` (JSON `null`) — never NaN, inf, or
zero-filled artifacts. Return metrics compound (`prod(1+r)-1`); annualization
compounds over the actual daily count; Sharpe/ICIR use `ddof=1` with a
1e-12 std floor and annualize by `sqrt(365)`.

## Future-leak evidence and its limits

- The static scan is **evidence, not proof**: arbitrary Python can hide
  forward references from an AST scan, so a clean scan never certifies
  causality.
- `check_cutoff` "verified" means **replay-invariance of the compute
  callback**: the factor recomputed with all knowledge truncated at each
  cutoff agrees with the full-history prefix exactly (axes, NaN masks,
  `max_abs_diff <= 1e-12`). The replay roughly triples value-phase compute
  (full pass plus two truncated passes) — a known, accepted cost.
- External precomputed DataFrame inputs cannot be replayed; their cutoff
  block is recorded as `not_verified` and their static scan as
  `not_applicable`. Treat external factors as unverified by construction.

## Legacy `factor_miner` compatibility mapping

`factor_common/compat.py::LegacyFactorMiner` preserves the legacy
constructor's argument names/order and representative return shapes while
delegating all analytics to `FactorManager.evaluate`:

```python
LegacyFactorMiner(factor_data, factor_name, factor_direction, render_path=None,
                  api_key=None, api_secret=None, n_groups=5, rebalance_period=1,
                  out_of_sample_days=180, *, h5_path=None, base_dir=None,
                  reports_dir=None, profile_id="perp_1d", fee_rate=0.0003,
                  funding_price_mode="strict", include_funding=True,
                  start=None, end=None)
```

- `factor_data` is **copied, never mutated** (the legacy constructor sorted
  the caller's DataFrame in place); the `factor_name` column is renamed to
  the standard `factor` internally. `api_key`/`api_secret` are accepted for
  signature compatibility and ignored — there is no Binance client anywhere
  in the evaluation path.
- A provided legacy `future_ret` column is treated strictly as an external,
  evaluation-only label feeding the IC family; every PnL/turnover/drawdown
  number comes from the common backtest ledger over H5 next-open execution
  prices plus real funding events — never from `future_ret`.
- `fee_rate` remains an explicit keyword on
  `calculate_hedged_returns_with_fees*` and maps to the profile `fee_rate`
  (a distinct fee re-evaluates; the factor run artifact is reused).

| Legacy | New source | Convention changes |
| --- | --- | --- |
| `IC()[0]` `ic` | external-label Pearson per-date mean, else `samples.<s>.ic.ic_mean` | — |
| `IC()[1]` `acc_ic` | rebuilt from `ic.daily` cumsum, columns `[date, acc_ic]` | — |
| `IC()[2]` `ir` | `scenarios.gross.<s>.sharpe` (annualized long-short mean/std·√365) | zero-variance → `None` (was 0) |
| `IC()[3]` `ic_ir` | `ic.icir` (unannualized mean/std, ddof=1) | zero-variance → `None` (was 0) |
| `IC()[4]/[5]` `t_stat`, `p_value` | `ic.t_stat`, `ic.p_value` | need ≥2 dates + variance, else `None` (was 0/1; p no longer clamped to 0.0001) |
| `IC()[6]` `spearman_corr` | Spearman of group rank vs saved `group_returns` means | <2 groups → `None` (was 0) |
| `IC()[7]` `rank_ic` | `ic.rank_ic_mean` | — |
| `IC()[8]` `daily_ic` | DataFrame `[date, ic, rank_ic, acc_ic]` from `ic.daily` | dates with <3 pairs or constant cross-sections are omitted (legacy kept NaN rows) |
| `IC_with_sample_split(data, sample_type)` | `样本内→in_sample`, `样本外→out_of_sample`, else `full`; custom `data` slices are rejected (`ValueError`) | external-label sample slices use the same purge semantics as `evaluate_metrics`: in-sample keeps signals whose exit is on or before the split, out-of-sample admits by entry strictly after the split, and a crossing signal is purged from both slices (full sample only) |
| `performance(group_num)` columns | per-group daily series = saved `group_returns["group_{group_num+1}"]` summarized by `metrics.summarize_returns` | `return`←`total_return` (compounded, was summed); `annual_return` compounded (was linear ×365/n); `sharp`←`sharpe`; `max_drawback`←`max_drawdown` as **fraction** (was rounded percent); `win_percent`←`win_rate`; `profit-loss ratio`←`profit_loss_ratio` (mean gain/\|mean loss\|, was sum/sum); `turnover`→`None` per group (legacy measured membership churn; the common turnover is portfolio-level traded notional); `md_period_days`/`recovery_period_days`→`"-"`; `IC` column = pooled in-group Pearson vs external labels when provided, else `None` |
| `performance_with_sample_split` | same, slicing `group_returns` by the sample date masks | same |
| `show_plot()` / `show_plot_for_sample()` | **removed** — `NotImplementedError` pointing to `render()`; the common renderer emits one self-contained HTML report instead of pyecharts chart objects | no unstable pyecharts chart ids are snapshotted anywhere |
| `render()` | `FactorManager.plot_result(result, output_path=render_path or reports/<factor>_<run>.html)`; returns the renderer info dict (was `None`) | stable element ids; report states unavailable data explicitly |
| `get_latest_group_symbols(...)` | deterministic `grouping.assign_groups` (ties break by instrument; group count never silently reduced) on the evaluated matrix's latest date; values rounded to 4 | `include_incomplete_data` retained but grouping no longer depends on label completeness |
| `calculate_hedged_returns_with_fees(fee_rate)` | ledgers: `adjusted_hedged_return_no_fee` = gross daily return, `adjusted_hedged_return_with_fee` = trading_net daily return, `adjusted_fee` = gross−net, `cum_return_*` = ledger `equity − 1` (compounded, was simple-sum); legacy column layout preserved with `long_return`/`short_return`/`adjusted_turnover` carried as explicit `None` placeholders; stats dict keys preserved, values from `summarize_returns` | drawdown now on compounded NAV |
| `calculate_hedged_returns_with_fees_sample_split(data, ...)` | same, ledgers sliced by sample dates | custom `data` rejected |
| `calculate_rank_ic_decay(max_lag=10)` | `ic_decay` rows `{horizon, rank_ic}` → `[lag, rank_ic]` | horizons fixed 1..10 (larger max_lag capped); uncomputable → null (was 0-fill) |
| `calculate_rank_ic_autocorr(max_lag=20)` | `rank_ic_autocorr` rows over exact calendar lags 1..20 → `[lag, autocorr]` | uncomputable → null (was 0-fill); lags are calendar days (legacy correlated surviving rows) |
| `*_for_sample` variants | same source; external-label slices use the `evaluate_metrics` purge mask | custom `data` rejected |
| `calc_rankic_halflife(curve, threshold)` | first horizon whose \|RankIC\| ≤ half the \|horizon-1\| value | legacy exponential fit and `inf` return dropped; `threshold` kept for signature compatibility (0.5 rule) |
| in/out sample attributes | `split_date`, `in_sample_dates`, `out_of_sample_dates`, `data_in_sample`, `data_out_sample` exposed as lazy properties over the common split block | split respects hold-period purging |
| constant-symbol pruning in old `main.py` | **removed** — the value engine/universe layer owns eligibility | — |
| `future_ret` from deleted `data/kline_data/*.csv` / Binance API | **removed** — labels come from H5 next-open execution prices | — |

Dropped without replacement: the private helpers `_group_apply_preserve_columns`,
`_processing`, `_turnover*`, `_calculate_max_drawdown`,
`_calculate_future_returns*`, `_split_in_out_sample`.

### Migration differences from the old pipeline

- **Data source**: old factors read per-symbol CSV klines under
  `data/kline_data/` and could fetch labels from the Binance API; the common
  framework reads only the point-in-time H5 store. `future_ret` CSV/API
  fetching is gone.
- **Returns/labels**: legacy log-return next-bar labels are replaced by
  next-open execution labels with non-overlapping `rebalance_days` holdings.
- **Selection history**: the legacy pipeline evaluated over all history the
  CSVs happened to contain; the framework requires an explicit signal
  window (default: the last 365 stored days) over the point-in-time
  top-50 universe.
- **Numerics**: compounded (not summed) total/annual returns, fraction (not
  rounded-percent) drawdown on the compounded NAV, null (not 0/inf)
  degenerate statistics, calendar-lag (not surviving-row) RankIC
  autocorrelation.
- **Costs**: the legacy fee model adjusted a hedged return series after the
  fact; the framework books per-fill fees and signed funding cash flows in
  the ledger, with unresolved cash flows halting certification instead of
  being zero-filled.

## Unsupported features and known data blockers

Unsupported by design (rejected with errors, not silently approximated):

- Intraday profiles (`perp_4h`/`perp_8h`) and non-daily factor frequencies;
  only the daily `perp_1d` profile exists.
- `signal_delay_days != 1` and `price_field != "open"`.
- Custom per-call data slices on legacy `*_with_sample_split` methods.
- Binance API credentials/clients anywhere in the evaluation path.
- Funding coverage inferred from observed events or fetch watermarks.

Known data blockers on the production H5:

- **Funding coverage is `unknown` for every stored panel day**: no
  authoritative historical settlement schedules are available, so the
  `all_costs` scenario cannot be certified complete on real data. Real-H5
  evaluations truthfully report `status="incomplete"` with usable factor
  artifacts, null all-costs full metrics, and a labeled `known_segment`;
  `verified_complete_net_performance` stays false until verified schedules
  are supplied. This is an external-data limitation, not a framework gap —
  the synthetic fully-scheduled fixture certifies the complete accounting
  path.
- Holding-period placeholder klines remain in the store
  (`has_complete_kline=False`) and block new entries on the following day.
- The optional CMC100 benchmark is omitted from the report (with an explicit
  warning) when the store lacks `cmc100_daily`.
