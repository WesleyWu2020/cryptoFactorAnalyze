# Daily H5 Factor Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native daily factor framework that reads CryptoQuant H5 data, computes and persists one migrated factor, evaluates executable portfolios including funding cash flows, and exposes reusable Python, CLI, notebook, and HTML interfaces.

**Architecture:** Separate data access, factor computation, portfolio accounting, metrics, persistence, and rendering under `factor_common`. Reuse existing readers, preprocessing, operators, and applicable analytics while retaining `factor_miner` and CLI compatibility through adapters. Model daily execution and timestamped funding explicitly rather than accumulating overlapping forward-return labels.

**Tech Stack:** Python 3.10-compatible syntax, pandas, NumPy, SciPy, PyTables, pyecharts, pytest, and pyarrow for Parquet. Use standard-library dataclasses, importlib, pathlib, hashlib, json, tempfile, and string.Template; do not introduce a new application framework.

---

## Approved scope and execution constraints

- Design: `docs/superpowers/specs/2026-09-05-factor-common-h5-design.md` (approved by the user).
- Implement the common layer, one template, and one migrated N-day momentum example. Do not migrate other rule factors, Alpha101, ML, GA, or portfolio strategies in this pass.
- Work on `main`, as previously requested. Preserve existing workspace edits and deletions. Stage only files/hunks belonging to the task; never use `git add .`.
- If using subagents, respect the user's Luna medium model constraint. Verify model availability before dispatch; do not silently substitute another model.
- Run commands from the repository root with `./.venv/bin/python`.
- Do not mutate `data/crypto_quant.h5`, redownload market data, or run a derived-data rebuild as part of framework tests. Use temporary H5 fixtures.
- The external reference demonstrates interfaces, not implementation. Do not import `factor_system`, manufacture missing vendors, or claim exact compatibility with its undocumented numerical conventions.
- Daily bars cannot support intraday profiles. Reject `perp_4h`, `perp_8h`, minute templates, and formula strings with actionable errors.

### Evidence update affecting acceptance

Read `docs/superpowers/plans/2026-09-05-funding-quality.md` before implementation. The active workspace now has `has_placeholder_kline`, `funding_coverage_status`, and `funding_invalid_price_count`; reuse these instead of duplicating quality logic.

The funding-quality work records 47,250 panel rows with `unknown` funding coverage because authoritative historical settlement schedules are absent. Existing events do not prove complete coverage. Raw funding contains 547,193 events, including 25,617 missing and 3 nonpositive mark prices. Recheck these facts during implementation because another task may update the store.

The design's proposed complete real-data strict window may therefore not exist. The acceptance gate is a complete synthetic, independently specified schedule and accounting replay, plus a real-H5 end-to-end run that honestly returns `incomplete` with usable factor artifacts and diagnostics. A complete real-data net report remains contingent on independently verified coverage; never fabricate a schedule from observed timestamps or weaken strict mode to pass acceptance. This is a data limitation, not a reason to stop building the framework.

Neither pyarrow nor fastparquet is currently installed in `.venv`. Add pyarrow as the necessary Parquet dependency. Notebook execution tools are also absent; add them only to development requirements.

## File ownership and boundaries

| Path | Responsibility |
| --- | --- |
| `factor_common/__init__.py` | Public `FactorManager` export |
| `factor_common/definitions.py` | Factor specification and result validation |
| `factor_common/profiles.py` | Explicit daily execution and cost defaults |
| `factor_common/data_provider.py` | Existing H5 reader adaptation, matrices, events, coverage |
| `factor_common/loader.py` | Load trusted local factor modules and validate metadata |
| `factor_common/preprocessing.py` | Extract existing MAD clipping and same-day rank transforms |
| `factor_common/value_engine.py` | Warmup, eligible cross-sections, pure factor execution |
| `factor_common/storage.py` | Versioned factor/evaluation artifacts and atomic pointers |
| `factor_common/grouping.py` | Shared deterministic signal grouping and target weights |
| `factor_common/funding.py` | Event cash flow, settlement-price checks, coverage decisions |
| `factor_common/backtest.py` | Fixed-quantity holdings and event-ordered accounting |
| `factor_common/labels.py` | Calendar-aligned forward labels for evaluation only |
| `factor_common/metrics.py` | Shared full/in/out sample statistics |
| `factor_common/reporting.py` | Rendering from saved results without trading |
| `factor_common/templates/regular_daily.py.tmpl` | Minimal user factor template |
| `factor_common/templates/report.html` | Data-independent report layout |
| `factor_common/manager.py` | Public orchestration and artifact access |
| `factor_common/compat.py` | Legacy configuration and `factor_miner` adapters |
| `factor_common/validation.py` | Static and dynamic future-leak checks |
| `factor_analyse/factor_mining/example_momentum.py` | The single migrated formula |
| `factor_analyse/factor_common_usage.ipynb` | Local usage and result inspection |
| `scripts/verify_factor_common.py` | Read-only real-H5 acceptance and artifact manifest |
| `docs/factor_common.md` | User guide, accounting conventions, migration differences |
| `tests/factor_common/` | Isolated tests and temporary synthetic fixtures |

Modify `data/crypto_quant/reader.py` only where necessary for explicit symbols/as-of reads. Modify `factor_analyse/factor_mining/util_factor.py` only to delegate extracted preprocessing. Adapt `factor_analyse/factor_analyse_custom.py`, `factor_analyse/factor_config.py`, and `factor_analyse/main.py`; retain `factor_analyse/board.html` as a historical artifact. Do not carry embedded old chart observations into the new template.

## Shared interfaces and numerical conventions

Use these names consistently throughout the implementation:

```python
# definitions.py
from dataclasses import dataclass
from typing import Callable
import pandas as pd

@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    meta: dict
    setting: dict
    calc_factor: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]
    source_sha256: str

# profiles.py
@dataclass(frozen=True)
class BacktestProfile:
    profile_id: str = "perp_1d"
    rebalance_days: int = 1
    anchor_date: str = "2024-01-01"  # execution-calendar anchor
    signal_delay_days: int = 1
    price_field: str = "open"
    n_groups: int = 10
    factor_direction: int = 1
    initial_equity: float = 1.0
    gross_exposure: float = 1.0
    fee_rate: float = 0.0003
    include_funding: bool = True
    funding_price_mode: str = "strict"
    out_of_sample_days: int = 180
    split_date: str | None = None
    periods_per_year: int = 365
```

Validate positive integer periods, direction in {-1, 1}, positive equity, nonnegative fees, group count >= 2, exposure in (0, 1], supported funding modes, and daily/open/one-day-delay semantics. No unsupported setting may be silently ignored.

Public methods:

```python
# FactorManager signatures (implement in Task 12)
def evaluate(self, source, *, factor_name=None, profile_id="perp_1d",
             params=None, plot=True):
    """Return the standard result dictionary; source is a .py path or DataFrame."""

def create_template(self, factor_name="example_factor", *, output_dir=None):
    """Return a new daily factor file path; never overwrite an existing file."""

def get_value(self, factor_name, *, run_id=None):
    """Return the saved date-by-instrument matrix."""

def get_performance(self, factor_name, *, run_id=None, evaluation_id=None):
    """Return saved metrics for an explicitly selected or latest complete evaluation."""

def plot_result(self, result, *, output_path=None, start=None, end=None):
    """Render an existing result, optionally slicing its real timestamps."""
```

`params` accepts `start`, `end`, `rebalance_days`, `anchor_date`, `n_groups`, `factor_direction`, `fee_rate`, `include_funding`, `funding_price_mode`, `split_date`, and `out_of_sample_days`. Start/end specify inclusive signal dates; defaults are the last 365 calendar signal days ending at the last closed H5 bar. Factor formula parameters belong in SETTING, not params. Reject unrecognized keys; do not silently accept reference-only `cost` with ambiguous one-/two-sided semantics.

The standard result dictionary has `status`, `factor_value`, `factor_performance`, `factor_result`, `diagnostics`, `metadata`, and `paths`. Performance sample keys are `full`, `in_sample`, and `out_of_sample`. Accounting scenario keys are `gross`, `trading_net`, and `all_costs`. Keep missing metrics as JSON null rather than NaN literals. Record separate signal coverage, label coverage, and funding coverage.

Daily indexes are UTC-normalized timezone-naive dates by convention; funding timestamps stay timezone-aware UTC. Duplicate date/instrument keys and unrecognized matrix axes are errors. No implicit timestamp rounding from intraday inputs.

## Task 1: Dependency and configuration contracts

**Files:** Create `factor_common/definitions.py`, `factor_common/profiles.py`, `tests/factor_common/test_profiles.py`, `requirements-dev.txt`; modify `requirements.txt`.

- [ ] Write failing tests for unsupported profiles, illegal fees, mutable input protection, and strict funding defaults:

```python
import pytest
from factor_common.profiles import resolve_profile

def test_daily_defaults_include_strict_funding():
    p = resolve_profile("perp_1d", {})
    assert (p.signal_delay_days, p.price_field, p.include_funding) == (1, "open", True)
    assert p.funding_price_mode == "strict"

@pytest.mark.parametrize("name", ["perp_4h", "perp_8h"])
def test_intraday_is_rejected(name):
    with pytest.raises(ValueError, match="daily"):
        resolve_profile(name, {})
```

- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_profiles.py -q`; expect missing-module failure before implementation.
- [ ] Implement the dataclasses above and `resolve_profile(profile_id, overrides)` using `dataclasses.replace` plus explicit validation; reject unknown keys before replacement. Add `pyarrow>=14,<22` to runtime requirements (Python 3.10 compatible), and `nbformat>=5,<6`, `nbclient>=0.10,<0.11`, `ipykernel>=6,<7` with `-r requirements.txt` to development requirements. Use the existing interpreter; do not change pandas merely to add Parquet support.
- [ ] Install with `./.venv/bin/python -m pip install -r requirements-dev.txt`; if network approval is required, request it through the tool. Run the profile tests and `./.venv/bin/python -c 'import pyarrow, pandas, tables; print(pyarrow.__version__)'`; expect successful imports and passing tests.
- [ ] Commit only the task's files: `git commit -m "feat: define daily factor and execution contracts"` after explicit staging.

## Task 2: Temporary H5 fixtures and calendar-safe data provider

**Files:** Create `tests/factor_common/conftest.py`, `tests/factor_common/test_data_provider.py`, `factor_common/data_provider.py`; modify `data/crypto_quant/reader.py` only for necessary read extensions; extend `tests/test_crypto_quant_reader.py`.

- [ ] Build fixtures through `CryptoQuantStore.replace` using the current `TABLE_SPECS`, following `tests/test_crypto_quant_reader.py`. Include 12 instruments, 12 days, pre-start history, a membership transition, one missing day, flat zero-volume placeholder bars, positive/negative funding rates, and an explicitly supplied independent settlement schedule. Do not derive expected timestamps from observed events.
- [ ] Add tests for membership transition, no future entrant in earlier cross-section, missing-calendar-row retention, quote-volume naming, and independence of price eligibility from funding coverage. Core oracle:

```python
def test_unknown_funding_does_not_remove_price_inputs(h5_fixture):
    from factor_common.data_provider import DataProvider
    dp = DataProvider(h5_fixture)
    close = dp.get_single_data("close", start="2024-01-03", end="2024-01-08")
    mask = dp.get_universe(start="2024-01-03", end="2024-01-08")
    assert close.index.equals(mask.index)
    assert mask.loc["2024-01-03", "AUSDT"]
    assert len(close.index) == 6
```

- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_data_provider.py -q`; expect failure until DataProvider exists.
- [ ] Implement `list_datas()`, `get_time_range()`, `symbols`, `get_single_data(field, *, start, end)`, `get_universe(*, start, end)`, `get_funding(*, start, end, symbols)`, and `get_quality(*, start, end, symbols)`. Use `pd.date_range(start, end, freq="D")` and `pivot(...).reindex(...)`; do not fill prices. Read membership by effective intervals and known decision dates. Query raw klines for explicitly held symbols beyond their membership end; future eligibility must not erase valuation data. Reuse `placeholder_kline_mask` and `annotate_funding_prices`.
- [ ] Add explicit cutoff/as-of input to the provider's internal history path, so cutoff tests truncate market and membership knowledge together. Keep existing reader defaults unchanged; add optional keyword arguments rather than changing old positional contracts. Legacy schemas without new quality flags produce `unknown`, not inferred completeness.
- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_data_provider.py tests/test_crypto_quant_reader.py -q`; expect all cases to pass, including old reader expectations. Commit as `feat: adapt H5 history and quality for factor research`.

## Task 3: Standard factor loader and daily template

**Files:** Create `factor_common/loader.py`, `factor_common/templates/regular_daily.py.tmpl`, `tests/factor_common/test_loader.py`.

- [ ] Test missing META fields, minute frequency, unsafe factor identifiers/path traversal, unsupported TYPE, nonexistent fields, invalid warmup, and duplicate output axes. Verify module import does not execute any market reads or write files.
- [ ] Use this template as the concrete happy-path fixture:

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
    window = SETTING["params"]["window"]
    return np.log(close / close.shift(window))
```

- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_loader.py -q`; expect failure before implementation.
- [ ] Implement `load_factor(path) -> FactorSpec` with `importlib.util.spec_from_file_location`, source SHA-256, and validation. Accept only trusted local `.py` files; this is not a sandbox. Resolve fields against DataProvider's declared field set. Validate IDs with `^[A-Za-z][A-Za-z0-9_]*$`. Template generation substitutes only the validated identifier and creates files exclusively.
- [ ] Run loader tests; expect pass. Commit as `feat: add daily factor module contract and template`.

## Task 4: Pure factor values and reusable preprocessing

**Files:** Create `factor_common/preprocessing.py`, `factor_common/value_engine.py`, `tests/factor_common/test_value_engine.py`; modify only preprocessing definitions in `factor_analyse/factor_mining/util_factor.py`.

- [ ] Test window history before entry, missing dates, tail signal retention without labels, and same-day-only transforms. Use closes `[100, 110, 121]` with window 1: raw values on the last two days equal `log(1.1)`. Include a future entrant with an extreme value and assert earlier ranks do not change.
- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_value_engine.py -q`; expect failure.
- [ ] Move the existing `winsorize_by_date` and `rank_to_unit_by_date` implementations into `preprocessing.py` without changing their math; leave import/delegating wrappers in `util_factor.py`. Implement `compute_factor(spec, dp, *, start, end)` returning a matrix and diagnostics. Core order:

```python
history_start = pd.Timestamp(start) - pd.Timedelta(days=spec.setting["warmup_bars"])
ctx = {field: dp.get_single_data(field, start=history_start, end=end)
       for field in spec.setting["data_needed"]}
raw = spec.calc_factor(ctx).replace([np.inf, -np.inf], np.nan)
eligible = dp.get_universe(start=history_start, end=end)
masked = raw.where(eligible)
```

Validate returned axes before masking; perform configured same-day preprocessing only on eligible finite entries, then slice to start/end. Separate missing-value diagnostics from valid output. Support `none` and `mad_rank` only; reject ambiguous `pasteurization=True` instead of inventing its meaning.
- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_value_engine.py tests/test_util_factor_minimal.py tests/test_operator_utils_minimal.py -q`; expect pass. Commit as `refactor: centralize causal factor preprocessing`.

## Task 5: Versioned Parquet and metadata persistence

**Files:** Create `factor_common/storage.py`, `tests/factor_common/test_storage.py`.

- [ ] Test round trips, unique keys, parameter/data fingerprint separation, fee-only evaluation changes, failed-write pointer preservation, and explicit run selection. Concrete round-trip oracle:

```python
def test_factor_round_trip(tmp_path):
    import pandas as pd
    from factor_common.storage import FactorStorage
    matrix = pd.DataFrame({"AUSDT": [0.2, 0.3]},
                          index=pd.date_range("2024-01-01", periods=2, name="date"))
    store = FactorStorage(tmp_path)
    saved = store.save_value("momentum", matrix, {"source_sha256": "abc", "window": 1})
    pd.testing.assert_frame_equal(store.get_value("momentum", run_id=saved["run_id"]), matrix)
```

- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_storage.py -q`; expect failure.
- [ ] Implement `FactorStorage(base_dir)`, `save_value(factor_id, matrix, metadata)`, `get_value(factor_id, *, run_id=None)`, `save_evaluation(factor_id, run_id, result)`, and `load_evaluation(factor_id, *, run_id=None, evaluation_id=None)`. Normalize matrix columns/index before comparing or saving. Store non-null finite values as `date,instrument,factor` and record the full date/column axes in metadata so empty rows round-trip correctly.
- [ ] Compute run IDs from canonical JSON containing source, formula settings, requested/loaded ranges, normalized input-content hashes including membership, and framework version. Evaluation IDs additionally include Profile, execution-tail prices, funding events, and coverage evidence hashes. Verify source file stat before/after reading the H5 snapshot and abort on concurrent change. Write artifacts in a sibling temporary directory, rename when complete, then atomically replace a small success pointer. Preserve incomplete evaluations under their IDs without promoting latest-complete evaluation pointers; a successful factor value may still be promoted independently.
- [ ] Store tables as Parquet and scalar metadata/diagnostics as valid JSON with nonfinite scalars converted to null. Never pickle code or arbitrary objects. Run storage tests; expect pass. Commit as `feat: persist traceable factor and evaluation artifacts`.

## Task 6: Shared grouping and execution-calendar labels

**Files:** Create `factor_common/grouping.py`, `factor_common/labels.py`, `tests/factor_common/test_grouping.py`, `tests/factor_common/test_labels.py`.

- [ ] Test stable tie ordering, direction reversal, too-small universes, and daily calendar gaps. Group sorted `(factor, instrument)` rows using `floor(position * n_groups / count)`; require at least n_groups valid names for the configured grouping, otherwise return an insufficient-group diagnostic. Do not reduce group count silently.
- [ ] Test label alignment with this independent oracle:

```python
def test_label_uses_next_open_not_signal_close():
    import pandas as pd
    from factor_common.labels import make_labels
    opens = pd.DataFrame({"A": [90., 100., 110., 121.]},
                         index=pd.date_range("2024-01-01", periods=4))
    out = make_labels(opens, hold_days=1)
    assert abs(out.loc["2024-01-01", "A"] - 0.1) < 1e-12
    assert pd.isna(out.iloc[-1, 0])
```

- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_grouping.py tests/factor_common/test_labels.py -q`; expect failure.
- [ ] Implement `assign_groups(values, n_groups)` and `target_weights(values, profile)`. Produce separate targets for each long-only group, the directional long-only portfolio, and 50/50 long-short portfolio. Implement `make_labels(opens, hold_days)` on an already complete daily calendar using `opens.shift(-(hold_days + 1)) / opens.shift(-1) - 1`; annotate this as evaluation-only future data. Return label entry/exit dates alongside metric assembly, so boundary purge uses exit times.
- [ ] Run tests; expect pass. Commit as `feat: define deterministic groups and executable return labels`.

## Task 7: Funding event arithmetic and coverage policy

**Files:** Create `factor_common/funding.py`, `tests/factor_common/test_funding.py`.

- [ ] Add the arithmetic regression below plus negative-rate, invalid-price, and unheld-symbol cases:

```python
import pytest
from factor_common.funding import funding_cashflow

@pytest.mark.parametrize("quantity,rate,expected", [(2., .001, -.2),
                                                  (-2., .001, .2),
                                                  (2., -.001, .2)])
def test_funding_sign(quantity, rate, expected):
    assert funding_cashflow(quantity, 100., rate) == pytest.approx(expected)
```

- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_funding.py -q`; expect failure.
- [ ] Implement `funding_cashflow(quantity, mark_price, rate)` with finite/positive validation and the exact signed formula. Implement `settle_funding(quantities, events, *, profile, opens)` returning event rows and diagnostics, without editing raw records. Default strict invalid marks produce an unresolved cash flow, not zero. Explicit `daily_open_approx` uses the same UTC day's opening price only when timestamp <= event time; record original/replacement price and approximation flag. Missing rates are never approximated.
- [ ] Implement `check_funding_coverage(held_intervals, quality)` with complete/not_applicable accepted, missing/unknown/no_events unresolved unless independent schedule evidence establishes otherwise. Query holding intervals even after membership exit. An absent quality row is unknown. For partial-day intervals, conservative unknown is preferable to claiming completeness from daily observations. Aggregate invalid/unresolved costs only for symbols actually held at those times.
- [ ] Test zero holdings with unrelated bad events, independently scheduled zero-event days, unknown coverage with all observed marks valid, and approximation that does not upgrade unknown coverage. Run tests; expect pass. Commit as `feat: account for signed funding events and coverage uncertainty`.

## Task 8: Fixed-quantity daily portfolio accounting

**Files:** Create `factor_common/backtest.py`, `tests/factor_common/test_backtest.py`.

- [ ] Add hand-calculated fixtures: signal A long/B short, entry opens both 100, exit opens A=110/B=90, zero costs gives 10% return at 50/50 allocation. Repeating daily signals with rebalance_days=3 must not trade daily. Add a constant-price two-name fixture with initial equity 1 and fee_rate .001: initial entry turnover notional is 1, fee .001, remaining equity .999; immediate final liquidation of the same fixed quantities costs another .001, leaving .998.
- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_backtest.py -q`; expect failure.
- [ ] Implement `run_backtest(values, opens, events, quality, profile, *, signal_start, signal_end)` returning ledger, orders, positions, funding details, diagnostics, and status. Keep quantities, equity, and last valuation price explicitly. Size orders against pre-trade equity after already-due funding; fees reduce equity after sizing. No implicit continuous rebalancing.

```python
# Accounting identities to apply to each scenario independently.
price_pnl = (quantity * (new_price - last_price)).sum()
equity_before_trade = equity + price_pnl + funding_cashflows
target_quantity = target_weight * equity_before_trade / execution_price
trade_notional = ((target_quantity - quantity).abs() * execution_price).sum()
trade_fee = trade_notional * profile.fee_rate
equity_after_trade = equity_before_trade - trade_fee
```

- [ ] Process each UTC boundary in order: value old holdings at open; settle funding at that timestamp on old quantities; execute scheduled orders; settle intraday events using unchanged quantities; value at the next boundary. Track each cash flow exactly once. First entry owns no boundary funding; final boundary settles before liquidation. Scheduled execution dates satisfy `(date - anchor_date).days % rebalance_days == 0`; use the prior day's signal, not the latest arbitrary available row. Ending liquidation occurs one holding period after the last executed entry/rebalance, with actual evaluable tail recorded.
- [ ] Run three independent accounting scenarios sharing signals and schedule: gross (zero fees/funding), trading_net (fees only), all_costs (fees and funding). Each sizes from its own equity; do not subtract one scenario's costs from another's NAV. Persist actual daily return including inception fee relative to equity 1. For unknown future equity in all_costs, retain known events and quantities through the first unresolved cash flow and stop certified continuation; gross/trading-only paths may continue. Do not compute later exact sizes from assumed-zero funding.
- [ ] Treat failed entry prices as explicit failed orders and incomplete target execution. Do not silently redistribute rejected weights. Existing unpriceable/delisted positions stay in diagnostics and halt certified valuation, never disappear. Nonpositive equity halts accounting. Zero-volume placeholder knowledge from a day's close must not be used to exclude that same day's opening trade; use prior completed-bar eligibility, and report retrospective nonexecution evidence separately.
- [ ] Add tests for rebalance/funding same-time ordering, negative funding, missing tail, membership exit while held, empty signals, margin boundary, scenario-specific quantities, and plot-window-independent execution calendar. Run tests; expect pass. Commit as `feat: simulate daily positions with explicit event ordering`.

## Task 9: Shared metrics and sample boundaries

**Files:** Create `factor_common/metrics.py`, `tests/factor_common/test_metrics.py`; inspect `portfolio/backtester/metrics.py` and the IC/performance methods in `factor_analyse/factor_analyse_custom.py` for reusable formulas.

- [ ] Test compounded simple returns against `[.1, -.1] -> -.01`, drawdown against NAV `[1, 1.1, .99] -> .1`, zero-variance Sharpe -> missing, and perfect daily RankIC -> 1. Test sample purging when label exits cross split_date and no out-of-sample reset trades.
- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_metrics.py -q`; expect failure.
- [ ] Implement `summarize_returns(returns, *, periods_per_year=365)` and `evaluate_metrics(values, labels, accounting, profile)`. Use `prod(1+r)-1`, annualized compounded return over actual daily count, sample standard deviation (`ddof=1`), and annualized zero-risk-free Sharpe. Include initial NAV 1 in drawdown. Turnover is sum absolute traded notional divided by pretrade equity, without an undocumented factor 1/2. Define win rate on nonmissing returns; profit/loss ratio is mean positive return / absolute mean negative return, missing when either side is absent.
- [ ] Compute Pearson and Spearman IC per valid cross-section (minimum 3 nonconstant pairs), IC mean, unannualized mean/std ICIR, explicitly named annualized ICIR, and t/p only with sufficient dates. IC decay uses entry-aligned horizons 1..10; factor autocorrelation uses exact calendar lags 1..20; half-life is first lag <= half the absolute lag-1 magnitude, missing if no crossing. Share functions across sample slices, never rerun the portfolio on each slice. Retain applicable legacy metrics with their names documented in the compatibility map.
- [ ] Include coverage denominator as point-in-time eligible universe, not surviving labels. Incomplete all_costs metrics must be null, while separately labeled known-segment diagnostics may exist. Use 180 natural days for default out-of-sample split; exclude crossing labels, not historical observations required for computation. Run metrics tests; expect pass. Commit as `refactor: unify factor metrics and sample evaluation`.

## Task 10: Result-only HTML rendering

**Files:** Create `factor_common/reporting.py`, `factor_common/templates/report.html`, `tests/factor_common/test_reporting.py`.

- [ ] Test `plot=False` computation independence, HTML escaping, incomplete-state visibility, empty sample rendering, and no provider/backtest calls during plotting. Use a spy that raises if any data loader runs while rendering a saved result.
- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_reporting.py -q`; expect failure.
- [ ] Implement `render_result(result, output_path, *, start=None, end=None)` using the same result tables as notebook access. Build pyecharts options in Python and inject into `string.Template`; HTML-escape labels and encode JSON safely for script contexts. Render summary, full/in/out sample metrics, gross/trading/all-cost NAV, group NAV and demeaned group returns, cumulative RankIC, turnover, positions/coverage, decay/autocorrelation, latest groups, and funding/fee diagnostics. Unavailable curves say why and are not drawn as zeros.
- [ ] Extract layout concepts from existing render methods; do not copy `board.html`'s saved observations. Use real indexes and UTC labels. Optional CMC100 must be separately named and rebased to the selected display start; missing optional benchmark is a warning, not an evaluation failure. Do not rename it to the legacy equal-weight index.
- [ ] Render one synthetic complete and one incomplete report; inspect visible headings, cost labels, time range, and chart series in a browser during execution. Run tests; expect pass. Commit as `refactor: render reports from standard evaluation results`.

## Task 11: Static and cutoff future-leak validation

**Files:** Create `factor_common/validation.py`, `tests/factor_common/test_future_leak.py`.

- [ ] Test that the scanner flags `shift(-1)`, negative periods passed by keyword, centered rolling, bfill/backfill, and forward asof. Maintain an explicit evaluation-only allowance for `labels.py`, not a global ignore list. Add a deliberately leaking factor that passes shape checks but fails truncated replay.
- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_future_leak.py -q`; expect failure.
- [ ] Implement `scan_future_leaks(paths)` returning file/line/pattern findings, and `check_cutoff(compute, cutoffs, *, atol=1e-12)` returning index/mask equality and numeric differences. The compute callback receives a cutoff and must truncate provider knowledge as well as output; comparing only a sliced full result is not a valid test.

```python
pd.testing.assert_index_equal(full_prefix.index, truncated.index)
pd.testing.assert_index_equal(full_prefix.columns, truncated.columns)
assert full_prefix.isna().equals(truncated.isna())
delta = (full_prefix - truncated).abs().stack().dropna()
max_abs_diff = float(delta.max()) if len(delta) else 0.0
assert max_abs_diff <= 1e-12
```

Exclude all-NaN future-only columns only under a recorded common-axis policy and separately assert no finite historical output was removed. Include a future membership entrant and cutoff on a membership transition. Static scanning is evidence, not proof for arbitrary Python; external DataFrame inputs receive `not_verified`, not a passed replay.
- [ ] Run tests; expect causal fixture pass and intentionally leaking fixture detected. Commit as `test: enforce factor cutoff and membership causality`.

## Task 12: FactorManager orchestration and read APIs

**Files:** Create `factor_common/manager.py`, `factor_common/__init__.py`, `tests/factor_common/test_manager.py`.

- [ ] Test all standard result keys, `.py` and DataFrame sources, explicit identity for external persistence, factor-run reuse when only fees change, incomplete evaluations accessible by ID, and template no-overwrite.
- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_manager.py -q`; expect failure.
- [ ] Implement the public signatures specified above. Order the workflow: resolve date/Profile/source → snapshot input data and metadata → compute/validate values → persist values → load execution-tail and funding data → backtest → evaluate metrics → persist evaluation → optionally render. Use `compute_factor`, `run_backtest`, `make_labels`, `evaluate_metrics`, `FactorStorage`, and `render_result` rather than duplicating their internals. The manager receives project-root-resolved defaults but honors explicit absolute/relative paths.
- [ ] Return structured incomplete/insufficient_data outcomes for data limitations; use exceptions for invalid user configuration, corrupt schemas, and I/O failures. Ensure funding errors do not discard successfully saved factor values. `get_performance` defaults only to a complete evaluation and raises an informative error when none exists; explicit evaluation_id may retrieve an incomplete diagnostic result. Default factor direction comes from the factor definition unless overridden.
- [ ] Run manager tests with temp stores and monkeypatched networking that always fails if used. Expect all tests to pass without network access. Commit as `feat: expose factor research and artifact manager`.

## Task 13: Preserve legacy entry points and migrate one example

**Files:** Create `factor_common/compat.py`, `factor_analyse/factor_mining/example_momentum.py`, `tests/factor_common/test_compat.py`, `tests/factor_common/test_example.py`; modify `factor_analyse/factor_analyse_custom.py`, `factor_analyse/factor_config.py`, `factor_analyse/main.py`.

- [ ] Before editing, inventory every public `factor_miner` method and its current return shape, including `IC()` tuple order, `performance()` table columns, render, latest groups, sample methods, and plotting methods. Put the explicit old-to-new method/return mapping in `docs/factor_common.md`. Characterization tests must cover constructor positional arguments, custom factor-column rename, no mutation of input DataFrame, and representative public return shapes; do not snapshot unstable chart IDs.
- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_compat.py tests/factor_common/test_example.py -q`; expect new-path failures.
- [ ] Implement `LegacyFactorMiner` in `compat.py`, preserving constructor argument names/order and adding optional keyword-only H5/Profile context. Convert existing factor_name columns to the standard `factor` internally. Delegate analytics to common results; provided legacy future_ret is explicitly external-label IC input, never a source of claimed funding-adjusted PnL. Replace old class internals with the adapter export only after public contract tests pass. Preserve keyword names such as `fee_rate` through explicit mapping; remove obsolete Binance-client dependency from the active evaluation path.
- [ ] Register the example with `file_prefix`, `factor_name`, `factor_direction`, `factor_desc`, `rebalance_period`, plus a new `module_path`. Reuse the template formula from Task 3. Add an executable entry that resolves the project root and calls FactorManager under `if __name__ == "__main__"`; importing the factor module must remain side-effect-free. Keep existing `N_Day_Momentum.py` unchanged so the only migrated script is unambiguous.
- [ ] Make `main.py` a small argparse adapter: maintain positional factor type and rebalance days, `--list`, and add `--start`, `--end`, `--h5-path`, `--output-dir`, `--no-plot`, and `--funding-price-mode`. Explicit positional rebalance overrides registered defaults. For unmigrated entries without module_path or usable stored artifacts, report migration status and exit nonzero; do not search deleted CSV directories silently. Remove full-history constant-symbol pruning. Preserve report output under reports and optional existing web-copy behavior only when previously configured.
- [ ] Run new compatibility/example tests and `./.venv/bin/python factor_analyse/main.py --list`; expect existing names plus the migrated example and clearly indicated migration state. Run example cutoff tests before accepting it. Commit as `refactor: adapt legacy factor entry points to common framework`.

## Task 14: English user guide, notebook, and executable acceptance

**Files:** Create `docs/factor_common.md`, `factor_analyse/factor_common_usage.ipynb`, `scripts/verify_factor_common.py`, `tests/factor_common/test_e2e.py`; update `README.md` only for new usage links.

- [ ] Add an end-to-end synthetic test using the 12-name H5 fixture with independent settlement coverage: compute the example, write/reload Parquet, run all scenarios, inspect cost cash flows, and render. Require complete status. Add the same chain with unknown coverage and assert incomplete status, null all-cost full metrics, preserved factor output, and a visible report reason.
- [ ] Run `./.venv/bin/python -m pytest tests/factor_common/test_e2e.py -q`; expect to expose integration defects before documenting success.
- [ ] Implement `scripts/verify_factor_common.py` with `--h5`, `--start`, `--end`, `--output-dir`, and optional `--execute-notebook`. It runs the example and cutoff replays, records H5 size/mtime and SHA-256 before/after, writes `acceptance.json`, and returns nonzero for contract failures. A correctly diagnosed incomplete real-data evaluation is recorded as `real_data_status=incomplete` with an explicit `verified_complete_net_performance=false`, not falsely treated as a complete performance result. Save generated reports and runs under the supplied output directory for this verification command.
- [ ] Write notebook cells for repository-path setup, manager construction, fields/range inspection, template syntax display, .py evaluation, saved factor access, cost/quality diagnostics, and report rendering. Use pathlib root discovery (repo root or factor_analyse directory), not user-specific absolute paths. Show `factor_performance["full"]`, not reference-only `raw` keys. Notebook code must not execute copied `/root/...` imports or fabricate dates. Mark external matrices as unverified.
- [ ] Document exact execution order, schemas, fee rate meaning, next-open labels, event timing, cash-flow signs, sample definitions, strict/approx distinction, compatibility mapping, unsupported features, and known data blockers. Include migration differences from old CSV/log-return/all-history-selection behavior.
- [ ] Execute the notebook with nbclient using the current `.venv` interpreter's temporary kernel registration, store the executed notebook in the verification output directory, and inspect report HTML. The source notebook should remain free of huge embedded outputs and stale tracebacks.
- [ ] Run these final commands once after integration fixes:

```bash
./.venv/bin/python -m pytest tests/factor_common tests/test_crypto_quant_reader.py tests/test_util_factor_minimal.py tests/test_operator_utils_minimal.py -q
./.venv/bin/python -m compileall -q factor_common factor_analyse/factor_mining/example_momentum.py factor_analyse/main.py scripts/verify_factor_common.py
./.venv/bin/python scripts/verify_factor_common.py --h5 data/crypto_quant.h5 --start 2024-02-01 --end 2024-03-15 --output-dir /private/tmp/factor-common-acceptance --execute-notebook
git diff --check
```

If `/private/tmp/factor-common-acceptance` exists, create a new directory with `mktemp -d /private/tmp/factor-common-acceptance.XXXXXX` and pass the returned explicit path instead of overwriting. Expected: tests and imports pass; H5 unchanged; factor Parquet reloads; synthetic net accounting complete; real run truthfully reports its actual coverage status; cutoff max_abs_diff <= 1e-12; notebook executes without errors; reports label all costs and data limitations.
- [ ] Run affected crypto-quant regression files in addition if Task 2 altered shared readers beyond covered queries. Do not run an unrelated long full suite repeatedly without a concrete regression concern.
- [ ] Commit only completed implementation/docs/test files as `docs: document and verify H5 factor research workflow`. Hand off file changes, commands/results, artifact paths, static findings and dynamic max_abs_diff, numerical migration differences, and unresolved funding coverage explicitly.

## Review checkpoints and dependency order

Tasks 1–5 establish contracts and value artifacts; review before portfolio work. Tasks 6–9 establish accounting and metrics; require hand-calculated funding/sign/timing tests before reporting. Tasks 10–12 establish reusable presentation and manager APIs. Tasks 13–14 integrate legacy usage and prove the vertical slice.

After Task 1, loader and H5 fixture work may be assigned independently; after stable accounting result contracts, renderer and legacy characterization can proceed independently. Never assign two agents to edit the same legacy class, reader, or schema concurrently. Each task commit follows test verification and a review of only that task's diff.

## Specification coverage and self-review

| Design requirement | Tasks |
| --- | --- |
| Native common framework and reference-style APIs | 1, 3, 12 |
| H5 reuse, daily calendar, point-in-time pool, quality | 2, 4, 11 |
| Pure factor template, one example, existing operators | 3, 4, 13 |
| Parquet and reproducible run/evaluation metadata | 5, 12 |
| Next-open execution and non-overlapping holdings | 6, 8 |
| Actual funding events, strict missing-data policy | 7, 8 |
| Gross/fees/funding paths and auditable costs | 8, 9, 10 |
| Existing statistics, sample split, optional benchmark | 9, 10 |
| Legacy constructor/configuration/CLI/report availability | 10, 13 |
| Static and cutoff future-leak evidence | 11, 13, 14 |
| Notebook, real-H5 and synthetic end-to-end validation | 14 |

Plan review must verify consistent names (`fee_rate`, `rebalance_days`, `factor_performance["full"]`), no fabricated settlement schedules, no future-informed execution eligibility, no unresolved cash flows silently set to zero, and no implementation tests that require changing the production H5. Complete real-data net performance is an external-data-dependent outcome and must remain explicitly separate from framework implementation completion.
