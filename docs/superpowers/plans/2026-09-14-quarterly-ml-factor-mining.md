# Quarterly ML Factor Mining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Respect the active session's delegation permissions; this document does not itself dispatch agents.

**Goal:** Combine existing daily factor Parquets into quarterly retrained, point-in-time ML ranking factors, and compare their continuous out-of-sample, cost-adjusted performance from 2025 onward.

**Architecture:** Implement a standalone `ML_factor_mining` package that consumes immutable snapshots of factor artifacts and the existing CryptoQuant H5. Share feature preparation and quarter boundaries across LightGBM, XGBoost, Ridge, and OLS; reuse `factor_common` for historical membership, execution labels, portfolio accounting, and standard reports. Keep supervised labels, validation accounting, and prediction-time inputs separate.

**Tech Stack:** Python, pandas, NumPy, existing `factor_common`, scikit-learn, LightGBM, XGBoost, PyArrow, pytest; JSON configuration and Parquet/JSON/native model artifacts.

---

## 1. Status, scope, and repository facts

This is an English implementation document, including reference implementation code and executable tests. It is not a claim that the ML package has been implemented or trained. The user requested proceeding directly to this document; a separate design-document approval cycle is not required for this deliverable.

Repository root: `/Users/dmiwu/work/PythonProject/cryptoFactorAnalyze`.

Inspected on 2026-09-14:

- `ML_factor_mining/` is empty. Reference scripts are in `reference/ML_factor_mining/`.
- `data/factor_results/` contains 337 Parquets and 197 factor metadata sidecars. Evaluation sidecars account for additional Parquets; never glob every Parquet into the feature set.
- Among the 197 metadata files, 190 declare `mad_rank`, seven declare `none`. Date ranges differ. These counts are observations, not constants to hard-code.
- Factor values use `date`, `instrument`, `factor`; metadata contains `factor_id`, settings, dates, source fingerprints, and upstream validation evidence.
- `factor_common.data_provider.DataProvider(path, as_of=...)` supplies historical membership and raw market history, including prices after membership exit.
- `factor_common.labels.make_labels(opens, hold_days)` implements next-open labels. `label_dates` exposes concrete entry and exit dates.
- `factor_common.backtest.run_backtest` already executes the five-group top-minus-bottom portfolio with 50% long and 50% short exposure when gross exposure is one. It returns independent `gross`, `trading_net`, and `all_costs` ledgers.
- `FactorManager.evaluate` accepts a DataFrame and an explicit factor name. `factor_analyse/main.py` accepts a factor `.py`, not a Parquet path.
- External DataFrame evaluation is not automatically certified for cutoff invariance. ML must supply its own audit artifact; do not relabel the manager's `not_verified` result as verified.
- The current environment has pandas 3.0.5, NumPy 2.5.2, SciPy 1.18.1, PyArrow 22.0.0, and pytest 8.4.2. scikit-learn, LightGBM, and XGBoost are absent.
- The working tree contains unrelated user changes, including removal of the old ML location. Preserve them. `tests/test_ml_factor_mining_minimal.py` still references that removed location; do not claim a clean full-suite baseline or restore old files as part of this work.

This is one implementation scope: factor-combination research. Excluded: raw OHLCV feature generation, GP mining, neural networks, stacking models together, trading exchange integration, and automatically choosing a winning model after observing 2025–2026 test results.

## 2. Accepted research contract

| Setting | Contract |
|---|---|
| Initial historical interval | 2024-01-01 onward |
| First OOS quarter | 2025 Q1 |
| Retraining | At the beginning of each calendar quarter |
| Training mode | `expanding` default; `rolling` with 12 calendar months configurable |
| Validation | Last three calendar months before the prediction quarter |
| Refit | After selection, use all eligible, matured historical samples, including validation |
| Holding period | One `holding_days` parameter for labels and rebalancing; default 3 |
| Predictions | Every day, including non-rebalance days |
| Execution | Signal at day t close; earliest trade at t+1 open |
| Universe | Existing point-in-time `historical_top50`; do not freeze a quarter's constituents |
| Features | Per-factor, same-day eligible-universe rank in [-1, 1] |
| Feature admission | At least 90% finite coverage in the fit-only interval; remove constants |
| Missing inputs | Fill zero and add one binary missing indicator per admitted factor |
| All inputs absent | No prediction for that instrument/date |
| Labels | `rank_return` default; `raw_return` and `excess_return` supported |
| Models | `lightgbm` default; `xgboost`, `ridge`, `linear` selectable |
| Selection | 50% signed daily mean RankIC percentile + 50% all-costs long-short Sharpe percentile |
| Portfolio | Five equal-count groups; long group 5, short group 1; fixed positive direction |
| Outputs | Quarter predictions, concatenated OOS factor, model/audit files, quarterly metrics, HTML |

Operational defaults introduced to make implementation unambiguous: at least 120 fit dates, 45 valid validation-IC dates, 20 valid label pairs per IC date, and 90% prediction coverage on each validation date. These are configurable data-sufficiency checks, not return-performance acceptance thresholds. Undefined Sharpe or incomplete accounting rejects a candidate. Negative but finite results remain visible; a selected model is not automatically a successful strategy.

### 2.1 Exact time convention

All dates are UTC calendar dates. A factor row dated t is available after that day's close. A quarter-Q model uses only daily evidence dated strictly before Q. This conservative boundary excludes prices/events from Q itself, even if an opening print might operationally be available at midnight.

For holding period h:

`raw_return[t, i] = open[t + 1 + h, i] / open[t + 1, i] - 1`.

Use the repository implementation instead of introducing another negative-shift formula. For h=3, the exit is t+4. For the 2025-01-01 retrain, the latest eligible historical signal is 2024-12-27, whose exit is 2024-12-31. For the 2024-10-01 validation boundary, the latest fit signal is 2024-09-26. Always derive these dates from `label_dates`; never assume a three-day purge is enough.

Membership for a label's cross-section is membership on the signal date, even if a name exits before label realization. Missing execution prices produce missing labels, not zero returns or worst ranks. Rank only observed labels, record label coverage, and require the configured minimum cross-section. This does not eliminate potential missing-label selection bias; retain the coverage audit.

Expanding example: fit 2024 Q1–Q3, validate 2024 Q4, refit matured 2024 samples, predict 2025 Q1. At the next retrain, fit through 2024 Q4, validate 2025 Q1, refit matured history, predict 2025 Q2. Purge both boundaries. A past test quarter can become later historical training data; its saved predictions never change.

Feature selection is fitted only on the fit segment, before validation. Keep its selected columns fixed through validation, final refit, and that model's prediction quarter. No supervised factor-direction flips, future-period coverage filters, PCA, correlation pruning, or new StandardScaler in v1. Input ranks are already dimensionless. Missing indicators have fixed column order even if a training indicator is all zero.

### 2.2 Training loss versus selection objective

Tree training minimizes MSE against the configured label and uses validation MSE for early stopping. A candidate then runs one exact validation backtest. Its selected boosting-round count and parameters compete under the composite score. Ridge selects a penalty under the same score; OLS has a single candidate.

For each model and quarter, evaluate a predetermined candidate bank. Remove invalid candidates, then rank the two signed metrics within the remaining bank:

`P(x) = (average_rank(x) - 0.5) / number_of_valid_candidates`.

`selection_score = ic_weight * P(mean_daily_rank_ic) + sharpe_weight * P(net_long_short_sharpe)`.

Both weights default to 0.5 and sum to one. Percentiles are local to that bank; never compare these composite numbers across quarters or model families. Ties break by higher net Sharpe, higher signed RankIC, then candidate ID. A single valid candidate receives percentile 0.5 in both metrics. No absolute values. Freeze weights and candidate banks before examining the test quarter.

Ridge's configurable `alpha` is the penalty in a mean-squared-error objective. Pass `alpha * n_fit_rows` to sklearn's sum-of-squares Ridge so expanding sample counts and final refitting do not accidentally weaken the same configured penalty.

Validation portfolios start flat and liquidate before Q, using a common calendar and costs for every candidate. This is a selection experiment. Final OOS portfolios run continuously across quarters and liquidate only at the overall evaluation endpoint. Do not confuse these two accounting boundaries.

### 2.3 Evaluation boundaries and research provenance

Maintain separate `prediction_end` and `evaluation_signal_end`. Predictions need no labels and can extend to the latest available feature date. Certified evaluation requires the entire execution/funding tail. Conservatively use `evaluation_signal_end <= evidence_end - (holding_days + 1)` and cap it at the prediction endpoint.

Quarter returns are `prod(1 + daily_net_return) - 1` from date slices of one continuous ledger, including entry and exit costs. Never compound overlapping h-day label returns. Quarter maximum drawdown resets its reference wealth to one at that quarter's start; full-run drawdown remains a separate statistic. Funding retains the existing engine's daily attribution convention.

If a final liquidation falls into the next calendar quarter, retain that cash flow and identify an accounting-tail quarter explicitly. Extending a run can change the previous artificial terminal liquidation; only prediction-prefix invariance is required, not invariance of an earlier terminal liquidation.

The existing standard report has an IS/OOS split unrelated to quarterly ML. Set its split before the first prediction so the supplied factor is entirely model-level OOS; provide a separate quarterly HTML summary. Standard `group_returns` are forward-label diagnostics, not five independent net portfolio ledgers. V1 certifies the top-minus-bottom portfolio's quarterly PnL; do not present diagnostic group averages as cost-adjusted portfolio profits.

Snapshot all inputs for reproducibility. A catalog discovered in 2026 and backfilled to 2024 is retrospective research. Upstream cutoff verification proves causal value calculation, not that a factor formula or its selection was known in 2024. Mark outputs `model_level_oos` and retain upstream provenance; GP/ML factors selected on the evaluation years cannot be advertised as end-to-end unseen research.

## 3. File map

Create these files; existing `factor_common` and `Genetic_Algorithm` code needs no modification:

| File | Responsibility |
|---|---|
| `ML_factor_mining/__init__.py` | Package marker |
| `ML_factor_mining/config.py` | Configuration validation and quarter scheduling |
| `ML_factor_mining/artifacts.py` | Immutable input snapshots and atomic output utilities |
| `ML_factor_mining/dataset.py` | Catalog ingestion, membership alignment, causal feature preparation |
| `ML_factor_mining/targets.py` | Supervised labels and maturity masks |
| `ML_factor_mining/models.py` | Four model adapters, candidate banks, portable persistence |
| `ML_factor_mining/scoring.py` | Validation accounting and composite selection |
| `ML_factor_mining/training.py` | One-quarter fit/select/refit/predict orchestration |
| `ML_factor_mining/evaluation.py` | Continuous OOS replay and quarterly attribution |
| `ML_factor_mining/validation.py` | ML cutoff replay and separate upstream validation evidence |
| `ML_factor_mining/cli.py` | `run` command and artifact orchestration |
| `ML_factor_mining/configs/default.json` | Explicit default experiment |
| `ML_factor_mining/requirements-ml.txt` | Optional model dependencies |
| `ML_factor_mining/README.md` | Usage and interpretation |
| `tests/ml_factor_mining/` | New isolated regression suite |

Run artifacts live in `outputs/ml_factor_mining/<run_name>/`; HTML reports live in `reports/ml_factor_mining/<run_name>/`. Do not write generated ML factors back into the input catalog automatically, avoiding recursive self-ingestion.

Each `### File:` block below contains a complete file. Apply files in task order. Shell snippets are commands for implementation, not commands that were executed while writing this document.

## Task 1: Configuration and calendar contract

**Files:** Create `ML_factor_mining/__init__.py`, `config.py`, `requirements-ml.txt`, `configs/default.json`, and `tests/ml_factor_mining/test_config.py`.

- [ ] Create the configuration tests below first and run the task's test command; expect import failure before adding the package.
- [ ] Add the package marker, dependency file, configuration, and calendar implementation.
- [ ] Install model dependencies before model-specific tests; do not silently skip a selected production backend.
- [ ] Run the tests again; expect all tests in this task to pass.

### File: ML_factor_mining/__init__.py

```python
"""Quarterly supervised combinations of existing daily factor artifacts."""
```

### File: ML_factor_mining/requirements-ml.txt

```text
-r ../requirements.txt
pandas>=2.2,<4
scikit-learn>=1.6,<2
lightgbm>=4.6,<5
xgboost>=3.0,<4
```

Use supported wheels for the local Python version. If installation fails, report the exact package/interpreter error; do not replace a requested model. Record resolved versions in the run manifest. LightGBM may require an OpenMP runtime on macOS; do not install system software implicitly.

### File: ML_factor_mining/config.py

```python
from dataclasses import dataclass, field
import math
import pandas as pd


MODELS = ("lightgbm", "xgboost", "ridge", "linear")


@dataclass(frozen=True)
class Config:
    train_start: str = "2024-01-01"
    oos_start: str = "2025-01-01"
    end: str = "2026-12-31"
    mode: str = "expanding"
    rolling_months: int = 12
    holding_days: int = 3
    target_type: str = "rank_return"
    models: tuple[str, ...] = ("lightgbm",)
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    coverage: float = 0.9
    min_fit_dates: int = 120
    min_val_dates: int = 45
    min_pairs: int = 20
    validation_prediction_coverage: float = 0.9
    ic_weight: float = 0.5
    sharpe_weight: float = 0.5
    max_rounds: int = 1000
    patience: int = 50
    seed: int = 42
    threads: int = 1
    anchor_date: str = "2024-01-01"
    fee_rate: float = 0.0005
    slippage: float = 0.001
    candidate_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("train_start", "oos_start", "end", "anchor_date"):
            value = pd.Timestamp(getattr(self, name))
            if value.tzinfo is not None or value != value.normalize():
                raise ValueError(f"{name} must be a naive UTC calendar date")
        start, first, end = map(pd.Timestamp, (self.train_start, self.oos_start, self.end))
        if not start < first <= end:
            raise ValueError("require train_start < oos_start <= end")
        if first != first.to_period("Q").start_time:
            raise ValueError("oos_start must be a calendar-quarter start")
        if self.mode not in {"expanding", "rolling"}:
            raise ValueError("unknown training mode")
        if self.target_type not in {"rank_return", "raw_return", "excess_return"}:
            raise ValueError("unknown target_type")
        for name in ("rolling_months", "holding_days", "min_fit_dates", "min_val_dates",
                     "min_pairs", "max_rounds", "patience", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.rolling_months <= 3 or self.min_pairs < 5:
            raise ValueError("rolling window must exceed validation; require at least five pairs")
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be nonempty and unique")
        if set(self.models) - set(MODELS):
            raise ValueError("unsupported model")
        if set(self.include) & set(self.exclude):
            raise ValueError("include and exclude overlap")
        for name in ("coverage", "validation_prediction_coverage"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"invalid {name}")
        weights = (self.ic_weight, self.sharpe_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("invalid score weights")
        if not math.isclose(sum(weights), 1.0):
            raise ValueError("score weights must sum to one")
        if any(not math.isfinite(x) or not 0 <= x < 1 for x in (self.fee_rate, self.slippage)):
            raise ValueError("invalid trading costs")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for name in ("models", "include", "exclude"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)


@dataclass(frozen=True)
class Fold:
    quarter: str
    history_start: pd.Timestamp
    validation_start: pd.Timestamp
    retrain_at: pd.Timestamp
    prediction_end: pd.Timestamp


def quarter_folds(config, evidence_end):
    stop = min(pd.Timestamp(config.end), pd.Timestamp(evidence_end))
    if stop < pd.Timestamp(config.oos_start):
        return []
    folds = []
    for quarter in pd.period_range(config.oos_start, stop, freq="Q"):
        q = quarter.start_time
        start = pd.Timestamp(config.train_start)
        if config.mode == "rolling":
            start = max(start, q - pd.DateOffset(months=config.rolling_months))
        val_start = q - pd.DateOffset(months=3)
        if start >= val_start:
            raise ValueError("no fit interval before validation")
        folds.append(Fold(str(quarter), start, val_start, q,
                          min(stop, quarter.end_time.normalize())))
    return folds
```

### File: ML_factor_mining/configs/default.json

```json
{
  "train_start": "2024-01-01",
  "oos_start": "2025-01-01",
  "end": "2026-12-31",
  "mode": "expanding",
  "rolling_months": 12,
  "holding_days": 3,
  "target_type": "rank_return",
  "models": ["lightgbm"],
  "include": [],
  "exclude": [],
  "coverage": 0.9,
  "min_fit_dates": 120,
  "min_val_dates": 45,
  "min_pairs": 20,
  "validation_prediction_coverage": 0.9,
  "ic_weight": 0.5,
  "sharpe_weight": 0.5,
  "max_rounds": 1000,
  "patience": 50,
  "seed": 42,
  "threads": 1,
  "anchor_date": "2024-01-01",
  "fee_rate": 0.0005,
  "slippage": 0.001,
  "candidate_overrides": {}
}
```

### File: tests/ml_factor_mining/test_config.py

```python
import pandas as pd
import pytest
from ML_factor_mining.config import Config, quarter_folds


def test_expanding_and_calendar_rolling():
    expanding = quarter_folds(Config(), "2025-06-30")
    rolling = quarter_folds(Config(mode="rolling"), "2025-06-30")
    assert expanding[1].history_start == pd.Timestamp("2024-01-01")
    assert rolling[1].history_start == pd.Timestamp("2024-04-01")
    assert rolling[1].validation_start == pd.Timestamp("2025-01-01")
    assert expanding[0].prediction_end == pd.Timestamp("2025-03-31")


def test_incomplete_quarter_does_not_create_future_quarters():
    folds = quarter_folds(Config(), "2026-08-31")
    assert folds[-1].quarter == "2026Q3"
    assert folds[-1].prediction_end == pd.Timestamp("2026-08-31")


@pytest.mark.parametrize("kwargs", [
    {"holding_days": 0}, {"holding_days": True},
    {"oos_start": "2025-02-01"}, {"ic_weight": 0.9},
    {"models": ("unknown",)}, {"rolling_months": 3},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
```

Run:

```bash
./.venv/bin/python -m pip install -r ML_factor_mining/requirements-ml.txt
./.venv/bin/python -m pytest tests/ml_factor_mining/test_config.py -q
```

Checkpoint commit after passing: `feat: define quarterly ML research configuration`.

## Task 2: Immutable artifacts and causal feature panel

**Files:** Create `artifacts.py`, `dataset.py`, and `tests/ml_factor_mining/test_dataset.py`.

- [ ] Write the dataset tests below and confirm missing-module failure.
- [ ] Implement snapshot utilities and feature-panel preparation using the complete files below.
- [ ] Run the tests and confirm that out-of-universe values, validation-period data, and all-missing observations cannot affect fit-time admission.
- [ ] Commit only these task files after they pass.

### File: ML_factor_mining/artifacts.py

```python
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
                    encoding="utf-8")
    os.replace(temp, path)


def write_parquet(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def discover_catalog(directory, config):
    records = []
    for meta_path in sorted(Path(directory).glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        factor_id = meta["factor_id"]
        if config.include and factor_id not in config.include:
            continue
        if factor_id in config.exclude:
            continue
        values = meta_path.with_name(meta_path.name.removesuffix(".meta.json") + ".parquet")
        settings = meta.get("metadata", {}).get("settings", {})
        if not values.is_file():
            raise ValueError(f"missing factor values: {values}")
        if settings.get("universe") != "historical_top50":
            raise ValueError(f"nonconforming universe: {factor_id}")
        if meta.get("metadata", {}).get("source_type") != "module":
            raise ValueError(f"external/recursive input requires explicit provenance: {factor_id}")
        records.append({"factor_id": factor_id, "values": str(values),
                        "metadata": str(meta_path)})
    ids = [r["factor_id"] for r in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate factor catalog")
    missing = set(config.include) - set(ids)
    if missing:
        raise ValueError(f"requested factors not found: {sorted(missing)}")
    return sorted(records, key=lambda r: r["factor_id"])


def snapshot_inputs(factor_dir, h5_path, run_dir, config):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    records = discover_catalog(factor_dir, config)
    source_paths = [Path(h5_path)]
    for record in records:
        source_paths.extend([Path(record["values"]), Path(record["metadata"])])
    before = {str(p): sha256(p) for p in source_paths}
    snapshot = run_dir / "inputs"
    snapshot.mkdir()
    hashes = {}
    for source in source_paths:
        relative = Path("crypto_quant.h5") if source == Path(h5_path) else Path("factors") / source.name
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != before[str(source)]:
            raise RuntimeError(f"input changed during copy: {source}")
        hashes[str(relative)] = before[str(source)]
    if any(sha256(p) != before[str(p)] for p in source_paths):
        raise RuntimeError("input changed while constructing snapshot; keep run incomplete")
    versions = {}
    for package in ("pandas", "numpy", "scipy", "pyarrow", "scikit-learn", "lightgbm", "xgboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    manifest = {"config": asdict(config), "hashes": hashes, "versions": versions,
                "research_scope": "model_level_oos", "catalog_timing": "retrospective_snapshot",
                "source_paths": [str(p.resolve()) for p in source_paths]}
    write_json(run_dir / "manifest.json", manifest)
    return snapshot
```

Read source metadata during the final audit and carry forward its complete upstream validation object. Do not infer a historical formula-discovery date from a Parquet's first date. Snapshot failures leave an incomplete run directory for diagnosis; retry under a new run name. No automatic deletion or input-cache regeneration.

### File: ML_factor_mining/dataset.py

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd


def stack_matrix(matrix):
    return matrix.rename_axis(index="date", columns="instrument").stack(future_stack=True)


def clean_factor_table(table):
    if set(table.columns) != {"date", "instrument", "factor"}:
        raise ValueError("factor input must have exactly date/instrument/factor")
    table = table.copy()
    dates = pd.to_datetime(table["date"], utc=True, errors="raise").dt.tz_localize(None)
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("intraday factor timestamps are not supported")
    table["date"] = dates
    if table["instrument"].isna().any() or not table["instrument"].map(lambda x: isinstance(x, str)).all():
        raise ValueError("instrument must be a nonmissing string")
    if table.duplicated(["date", "instrument"]).any():
        raise ValueError("duplicate date/instrument")
    table["factor"] = pd.to_numeric(table["factor"], errors="raise").replace([np.inf, -np.inf], np.nan)
    return table.sort_values(["date", "instrument"])


def rank_panel(raw):
    groups = raw.groupby(level="date")
    rank = groups.rank(method="average")
    count = groups.transform("count")
    result = 2 * (rank - 1) / (count - 1).replace(0, np.nan) - 1
    # Preserve missing values even on dates with zero or one observed name.
    return result.mask(count.eq(1) & raw.notna(), 0.0).where(raw.notna())


def build_panel(catalog, membership):
    eligible = stack_matrix(membership.fillna(False).astype(bool))
    index = eligible.index[eligible.to_numpy()]
    if index.empty:
        raise ValueError("no historical eligible observations")
    raw = pd.DataFrame(index=index)
    for record in catalog:
        table = clean_factor_table(pd.read_parquet(record["values"]))
        series = table.set_index(["date", "instrument"])["factor"]
        raw[record["factor_id"]] = series.reindex(index)
    return rank_panel(raw.sort_index())


@dataclass(frozen=True)
class FeatureMap:
    factors: tuple[str, ...]
    diagnostics: dict

    @property
    def columns(self):
        # A reserved prefix prevents collisions with input factor identifiers.
        return list(self.factors) + ["__missing__" + f for f in self.factors]

    @classmethod
    def fit(cls, fit_panel, coverage):
        if fit_panel.empty:
            raise ValueError("empty feature fitting interval")
        if any(str(c).startswith("__missing__") for c in fit_panel.columns):
            raise ValueError("reserved feature prefix")
        rates = fit_panel.notna().mean()
        counts = fit_panel.nunique(dropna=True)
        selected = tuple(sorted(c for c in fit_panel if rates[c] >= coverage and counts[c] > 1))
        diagnostics = {c: {"coverage": float(rates[c]), "distinct_values": int(counts[c]),
                           "selected": c in selected} for c in fit_panel}
        if not selected:
            raise ValueError("no factors pass training-only admission")
        return cls(selected, diagnostics)

    def transform(self, panel):
        values = panel.reindex(columns=self.factors)
        present = values.notna().any(axis=1)
        missing = values.isna().astype(float)
        missing.columns = ["__missing__" + c for c in self.factors]
        result = pd.concat([values.fillna(0.0), missing], axis=1)
        return result.loc[:, self.columns].astype("float64"), present
```

### File: tests/ml_factor_mining/test_dataset.py

```python
import numpy as np
import pandas as pd
import pytest
from ML_factor_mining.artifacts import discover_catalog, write_json
from ML_factor_mining.config import Config
from ML_factor_mining.dataset import FeatureMap, build_panel, clean_factor_table


def test_catalog_excludes_evaluation_parquets(tmp_path):
    write_json(tmp_path / "f.meta.json", {"factor_id": "F", "metadata": {
        "source_type": "module", "settings": {"universe": "historical_top50"}}})
    (tmp_path / "f.parquet").touch()
    (tmp_path / "f.evaluation.positions.parquet").touch()
    assert len(discover_catalog(tmp_path, Config())) == 1


def test_membership_before_ranking_and_missing_preserved(tmp_path):
    day = pd.Timestamp("2024-01-02")
    table = pd.DataFrame({"date": [day] * 4, "instrument": ["A", "B", "C", "D"],
                          "factor": [2.0, 4.0, np.nan, -999.0]})
    path = tmp_path / "f.parquet"
    table.to_parquet(path)
    members = pd.DataFrame([[True, True, True, False]], index=[day], columns=list("ABCD"))
    panel = build_panel([{"factor_id": "F", "values": str(path)}], members)
    assert panel.loc[(day, "A"), "F"] == -1
    assert panel.loc[(day, "B"), "F"] == 1
    assert pd.isna(panel.loc[(day, "C"), "F"])
    assert (day, "D") not in panel.index


def test_admission_only_uses_fit_rows_and_all_missing_is_not_predicted():
    index = pd.MultiIndex.from_product([pd.date_range("2024-01-01", periods=2), list("AB")],
                                      names=["date", "instrument"])
    fit = pd.DataFrame({"good": [-1, 1, -1, 1], "late": [np.nan] * 4}, index=index)
    features = FeatureMap.fit(fit, 0.9)
    assert features.factors == ("good",)
    future = fit.copy()
    future["late"] = [-1, 1, -1, 1]
    future.loc[index[0], "good"] = np.nan
    transformed, present = features.transform(future)
    assert transformed.loc[index[0], "good"] == 0
    assert transformed.loc[index[0], "__missing__good"] == 1
    assert not present.loc[index[0]]
    assert "late" not in transformed


def test_duplicate_input_is_rejected():
    frame = pd.DataFrame({"date": ["2024-01-01"] * 2, "instrument": ["A"] * 2,
                          "factor": [1, 2]})
    with pytest.raises(ValueError, match="duplicate"):
        clean_factor_table(frame)
```

Run: `./.venv/bin/python -m pytest tests/ml_factor_mining/test_dataset.py -q`.

Checkpoint commit: `feat: ingest immutable factor catalogs with causal cross-sectional preprocessing`.

## Task 3: Targets and maturity-safe partitions

**Files:** Create `targets.py` and `tests/ml_factor_mining/test_targets.py`.

- [ ] Add the exact boundary and rank-equivalence tests below; confirm they fail before implementation.
- [ ] Implement target construction by calling the existing label functions.
- [ ] Keep missing-label filtering separate from feature admission; a feature's coverage denominator must not depend on whether its future return exists.
- [ ] Run the task tests, then commit the two files.

### File: ML_factor_mining/targets.py

```python
import numpy as np
import pandas as pd
from factor_common.labels import make_labels, label_dates
from .dataset import rank_panel, stack_matrix


def build_targets(opens, membership, config):
    # Supervised/evaluation-only data. Never pass this frame into FeatureMap.
    clean_opens = opens.where(np.isfinite(opens) & opens.gt(0))
    raw = make_labels(clean_opens, config.holding_days)
    eligible = membership.reindex(index=raw.index, columns=raw.columns).fillna(False)
    raw = raw.where(eligible)
    raw = raw.where(np.isfinite(raw))
    raw.loc[raw.count(axis=1) < config.min_pairs] = np.nan
    frame = stack_matrix(raw).rename("raw_return").to_frame()
    if config.target_type == "rank_return":
        frame["target"] = rank_panel(frame[["raw_return"]])["raw_return"]
    elif config.target_type == "excess_return":
        frame["target"] = frame["raw_return"] - frame.groupby(level="date")["raw_return"].transform("mean")
    else:
        frame["target"] = frame["raw_return"]
    entry, exit_ = label_dates(opens.index, config.holding_days)
    frame["entry_date"] = frame.index.get_level_values("date").map(pd.Series(entry, index=opens.index))
    frame["exit_date"] = frame.index.get_level_values("date").map(pd.Series(exit_, index=opens.index))
    return frame


def partition_masks(index, fold, holding_days):
    # Calendar maturity only: usable even before future labels are read.
    dates = index.get_level_values("date")
    exits = dates + pd.Timedelta(days=holding_days + 1)
    in_history = dates >= fold.history_start
    fit = in_history & (dates < fold.validation_start) & (exits < fold.validation_start)
    validation = (dates >= fold.validation_start) & (dates < fold.retrain_at) & (exits < fold.retrain_at)
    refit = in_history & (dates < fold.retrain_at) & (exits < fold.retrain_at)
    return {"fit": fit, "validation": validation, "refit": refit}
```

### File: tests/ml_factor_mining/test_targets.py

```python
import numpy as np
import pandas as pd
from ML_factor_mining.config import Config, quarter_folds
from ML_factor_mining.targets import build_targets, partition_masks


def test_three_day_horizon_purges_four_day_exit_boundary():
    config = Config()
    fold = quarter_folds(config, "2025-03-31")[0]
    dates = pd.to_datetime(["2024-09-26", "2024-09-27", "2024-12-27", "2024-12-28"])
    index = pd.MultiIndex.from_arrays([dates, ["A"] * 4], names=["date", "instrument"])
    masks = partition_masks(index, fold, 3)
    assert masks["fit"].tolist() == [True, False, False, False]
    assert masks["validation"].tolist() == [False, False, True, False]


def test_rank_return_equals_rank_excess_and_uses_next_open():
    dates = pd.date_range("2024-01-01", periods=10)
    symbols = list("ABCDE")
    opens = pd.DataFrame({s: 100 * (1 + 0.01 * (i + 1)) ** np.arange(10)
                          for i, s in enumerate(symbols)}, index=dates)
    members = pd.DataFrame(True, index=dates, columns=symbols)
    rank = build_targets(opens, members, Config(min_pairs=5))
    excess = build_targets(opens, members, Config(min_pairs=5, target_type="excess_return"))
    a = rank.loc[(dates[0], "A"), "raw_return"]
    assert np.isclose(a, opens.loc[dates[4], "A"] / opens.loc[dates[1], "A"] - 1)
    assert rank.loc[dates[0], "target"].rank().equals(excess.loc[dates[0], "target"].rank())
    assert rank.loc[dates[-1], "target"].isna().all()


def test_label_does_not_require_future_membership():
    dates = pd.date_range("2024-01-01", periods=8)
    opens = pd.DataFrame({s: np.arange(8) + 100 + i for i, s in enumerate("ABCDE")}, index=dates)
    members = pd.DataFrame(True, index=dates, columns=list("ABCDE"))
    members.loc[dates[1]:, "A"] = False
    targets = build_targets(opens, members, Config(min_pairs=5))
    assert pd.notna(targets.loc[(dates[0], "A"), "target"])
```

Run: `./.venv/bin/python -m pytest tests/ml_factor_mining/test_targets.py -q`.

Checkpoint commit: `feat: add maturity-purged next-open ML targets`.

## Task 4: Model adapters and reproducible candidate banks

**Files:** Create `models.py` and `tests/ml_factor_mining/test_models.py`.

- [ ] Write round-trip and refit tests first; they must fail before the adapter exists.
- [ ] Implement lazy imports, deterministic candidate banks, early stopping, and native model persistence.
- [ ] Validate all selected backends at run startup. Missing libraries are a setup error, not grounds for silently changing the model.
- [ ] Run tests for all four installed models and commit only this task's files.

### File: ML_factor_mining/models.py

```python
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import numpy as np
from .artifacts import write_json


def require_backends(names):
    packages = {"lightgbm": "lightgbm", "xgboost": "xgboost",
                "ridge": "sklearn.linear_model", "linear": "sklearn.linear_model"}
    for name in names:
        try:
            importlib.import_module(packages[name])
        except (ImportError, OSError) as exc:
            raise RuntimeError(f"backend {name} unavailable; install requirements-ml.txt") from exc


def candidate_bank(name, config):
    banks = {
        "lightgbm": [{"num_leaves": leaves, "min_data_in_leaf": minimum, "lambda_l2": 1.0}
                     for leaves in (7, 15, 31) for minimum in (50, 100)],
        "xgboost": [{"max_depth": depth, "min_child_weight": 10, "lambda": penalty}
                    for depth in (2, 3, 4) for penalty in (1.0, 10.0)],
        "ridge": [{"alpha": a} for a in (0.0001, 0.001, 0.01, 0.1, 1.0)],
        "linear": [{}],
    }
    allowed = {"lightgbm": {"num_leaves", "min_data_in_leaf", "lambda_l2"},
               "xgboost": {"max_depth", "min_child_weight", "lambda"},
               "ridge": {"alpha"}, "linear": set()}
    bank = config.candidate_overrides.get(name, banks[name])
    if not isinstance(bank, list) or not bank:
        raise ValueError("candidate bank must be a nonempty list")
    if any(not isinstance(p, dict) or set(p) - allowed[name] for p in bank):
        raise ValueError("unsupported candidate parameter")
    encoded = [json.dumps(p, sort_keys=True, allow_nan=False) for p in bank]
    if len(set(encoded)) != len(encoded):
        raise ValueError("duplicate candidates")
    return [(f"c{i:03d}", dict(params)) for i, params in enumerate(bank)]


@dataclass
class Fitted:
    name: str
    estimator: object
    columns: tuple[str, ...]
    rounds: int | None

    def predict(self, X):
        if tuple(X.columns) != self.columns:
            raise ValueError("prediction feature schema/order mismatch")
        if self.name == "xgboost":
            import xgboost as xgb
            result = self.estimator.predict(xgb.DMatrix(X.to_numpy(), feature_names=list(self.columns)),
                                            iteration_range=(0, self.rounds))
        elif self.name == "lightgbm":
            result = self.estimator.predict(X.to_numpy(), num_iteration=self.rounds)
        elif isinstance(self.estimator, dict):
            result = X.to_numpy() @ np.asarray(self.estimator["coef"]) + self.estimator["intercept"]
        else:
            result = self.estimator.predict(X.to_numpy())
        result = np.asarray(result, dtype=float)
        if result.shape != (len(X),) or not np.isfinite(result).all():
            raise ValueError("model produced nonfinite or malformed predictions")
        return result


def fit_model(name, params, X, y, config, *, X_val=None, y_val=None, rounds=None):
    if X.empty or len(y) != len(X) or not np.isfinite(X.to_numpy()).all() or not np.isfinite(y).all():
        raise ValueError("invalid fitting arrays")
    columns = tuple(X.columns)
    if name in {"ridge", "linear"}:
        from sklearn.linear_model import Ridge, LinearRegression
        if name == "ridge":
            alpha = float(params.get("alpha", 0.01))
            if not np.isfinite(alpha) or alpha <= 0:
                raise ValueError("Ridge alpha must be positive; use linear for OLS")
            model = Ridge(alpha=alpha * len(X), fit_intercept=True, solver="svd")
        else:
            model = LinearRegression(fit_intercept=True)
        model.fit(X.to_numpy(), np.asarray(y))
        return Fitted(name, model, columns, None)
    tuning = rounds is None
    if tuning and (X_val is None or y_val is None or X_val.empty):
        raise ValueError("tree selection requires a validation set")
    count = config.max_rounds if tuning else int(rounds)
    if count < 1:
        raise ValueError("invalid boosting round count")
    if name == "lightgbm":
        import lightgbm as lgb
        settings = {**params, "objective": "regression", "metric": "l2",
                    "learning_rate": 0.05, "verbosity": -1, "seed": config.seed,
                    "num_threads": config.threads, "deterministic": True,
                    "force_col_wise": True, "feature_pre_filter": False}
        training = lgb.Dataset(X.to_numpy(), label=np.asarray(y), feature_name=list(columns))
        kwargs = {}
        if tuning:
            validation = lgb.Dataset(X_val.to_numpy(), label=np.asarray(y_val), reference=training)
            kwargs = {"valid_sets": [validation],
                      "callbacks": [lgb.early_stopping(config.patience, verbose=False)]}
        model = lgb.train(settings, training, num_boost_round=count, **kwargs)
        selected_rounds = (model.best_iteration or count) if tuning else count
    elif name == "xgboost":
        import xgboost as xgb
        settings = {**params, "objective": "reg:squarederror", "eval_metric": "rmse",
                    "eta": 0.05, "tree_method": "hist", "seed": config.seed,
                    "nthread": config.threads, "verbosity": 0}
        training = xgb.DMatrix(X.to_numpy(), label=np.asarray(y), feature_names=list(columns))
        kwargs = {}
        if tuning:
            validation = xgb.DMatrix(X_val.to_numpy(), label=np.asarray(y_val), feature_names=list(columns))
            kwargs = {"evals": [(validation, "validation")],
                      "early_stopping_rounds": config.patience}
        model = xgb.train(settings, training, num_boost_round=count, verbose_eval=False, **kwargs)
        selected_rounds = model.best_iteration + 1 if tuning else count
    else:
        raise ValueError(f"unknown model: {name}")
    return Fitted(name, model, columns, int(selected_rounds))


def save_model(fitted, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    info = {"name": fitted.name, "columns": list(fitted.columns), "rounds": fitted.rounds}
    if fitted.name == "lightgbm":
        fitted.estimator.save_model(str(directory / "model.txt"), num_iteration=fitted.rounds)
    elif fitted.name == "xgboost":
        fitted.estimator.save_model(str(directory / "model.ubj"))
    else:
        estimator = fitted.estimator
        info["weights"] = estimator if isinstance(estimator, dict) else {
            "coef": estimator.coef_.tolist(), "intercept": float(estimator.intercept_)}
    write_json(directory / "model.json", info)


def load_model(directory):
    directory = Path(directory)
    info = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if info["name"] == "lightgbm":
        import lightgbm as lgb
        estimator = lgb.Booster(model_file=str(directory / "model.txt"))
    elif info["name"] == "xgboost":
        import xgboost as xgb
        estimator = xgb.Booster()
        estimator.load_model(directory / "model.ubj")
    else:
        estimator = info["weights"]
    return Fitted(info["name"], estimator, tuple(info["columns"]), info["rounds"])
```

Model files are trusted run outputs. Linear coefficients use JSON; tree models use their native formats. No pickle/joblib loading is needed. Store feature coefficients for both original factor columns and missing indicators; a coefficient is a conditional fitted association, not an individual factor's causal contribution.

### File: tests/ml_factor_mining/test_models.py

```python
import numpy as np
import pandas as pd
import pytest
from ML_factor_mining.config import Config
from ML_factor_mining.models import fit_model, load_model, save_model


@pytest.mark.parametrize("name,params", [
    ("linear", {}), ("ridge", {"alpha": 0.01}),
    ("lightgbm", {"num_leaves": 7, "min_data_in_leaf": 5}),
    ("xgboost", {"max_depth": 2, "min_child_weight": 1}),
])
def test_model_refit_and_native_round_trip(tmp_path, name, params):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(240, 3)), columns=["a", "b", "c"])
    y = 0.6 * X.a - 0.2 * X.b + 0.01 * rng.normal(size=len(X))
    config = Config(max_rounds=20, patience=3)
    trial = fit_model(name, params, X.iloc[:180], y.iloc[:180], config,
                      X_val=X.iloc[180:210], y_val=y.iloc[180:210])
    final = fit_model(name, params, X.iloc[:210], y.iloc[:210], config, rounds=trial.rounds)
    assert final.rounds == trial.rounds
    save_model(final, tmp_path)
    loaded = load_model(tmp_path)
    np.testing.assert_allclose(final.predict(X.iloc[210:]), loaded.predict(X.iloc[210:]),
                               rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError, match="schema"):
        loaded.predict(X.iloc[210:][["b", "a", "c"]])


def test_ridge_mean_loss_penalty_is_stable_when_rows_are_repeated():
    X = pd.DataFrame({"a": [-1.0, 0.0, 1.0], "b": [0.0, 1.0, -1.0]})
    y = pd.Series([-0.7, 0.2, 0.5])
    a = fit_model("ridge", {"alpha": 0.1}, X, y, Config())
    b = fit_model("ridge", {"alpha": 0.1}, pd.concat([X] * 4), pd.concat([y] * 4), Config())
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=1e-12)
```

Run: `./.venv/bin/python -m pytest tests/ml_factor_mining/test_models.py -q`.

Expected: all backend round trips pass; do not use `importorskip` to hide a backend installation failure in the required final run.

Checkpoint commit: `feat: support tree and linear quarterly factor models`.

## Task 5: Certified validation scoring

**Files:** Create `scoring.py` and `tests/ml_factor_mining/test_scoring.py`.

- [ ] Write the signed-score and accounting-boundary tests below first.
- [ ] Implement a validation scorer around the existing accounting engine, preserving its strict funding rules.
- [ ] Exclude invalid candidates before percentile normalization; save their explicit reasons.
- [ ] Verify direction, exposure, continuous execution anchor, and liquidation date; commit after passing.

### File: ML_factor_mining/scoring.py

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from factor_common.backtest import run_backtest
from factor_common.metrics import _daily_ic, summarize_returns
from factor_common.profiles import resolve_profile


class CandidateRejected(ValueError):
    """Expected data/accounting rejection; programming errors must still abort."""

    def __init__(self, message, *, records=None):
        super().__init__(message)
        self.records = records


@dataclass
class MarketWindow:
    opens: pd.DataFrame
    events: pd.DataFrame
    quality: pd.DataFrame


def load_market(provider, start, end, columns):
    calendar = pd.date_range(start, end, name="date")
    opens = provider.get_single_data("open", start=start, end=end).reindex(index=calendar, columns=columns)
    events = provider.get_funding(start=start, end=end, symbols=list(columns))
    quality = provider.get_quality(start=pd.Timestamp(start) - pd.Timedelta(days=1),
                                   end=end, symbols=list(columns))
    return MarketWindow(opens, events, quality)


def backtest_profile(config):
    return resolve_profile("perp_1d", {
        "rebalance_days": config.holding_days, "anchor_date": config.anchor_date,
        "signal_delay_days": 1, "n_groups": 5, "factor_direction": 1,
        "gross_exposure": 1.0, "fee_rate": config.fee_rate, "slippage": config.slippage,
        "include_funding": True, "funding_price_mode": "strict", "periods_per_year": 365,
    })


def score_validation(predictions, targets, membership, market, fold, config):
    start = fold.validation_start
    end = fold.retrain_at - pd.Timedelta(days=1)
    signal_end = end - pd.Timedelta(days=config.holding_days + 1)
    calendar = pd.date_range(start, end, name="date")
    columns = market.opens.columns
    values = predictions.unstack("instrument").reindex(index=calendar, columns=columns)
    eligible = membership.reindex(index=calendar, columns=columns).fillna(False)
    values = values.where(eligible)
    values.loc[values.index > signal_end] = np.nan
    signal_dates = calendar[calendar <= signal_end]
    denominator = eligible.loc[signal_dates].sum(axis=1)
    coverage = values.loc[signal_dates].notna().sum(axis=1) / denominator.replace(0, np.nan)
    if coverage.isna().any() or (coverage < config.validation_prediction_coverage).any():
        raise CandidateRejected("insufficient validation prediction coverage")
    labels = targets["raw_return"].unstack("instrument").reindex(index=calendar, columns=columns)
    daily = _daily_ic(values.loc[signal_dates], labels.loc[signal_dates])
    daily = daily.loc[daily["n_pairs"].ge(config.min_pairs) & np.isfinite(daily["rank_ic"])]
    if len(daily) < config.min_val_dates:
        raise CandidateRejected("insufficient valid validation RankIC dates")
    accounting = run_backtest(values, market.opens.reindex(index=calendar, columns=columns),
                              market.events, market.quality, backtest_profile(config),
                              signal_start=start, signal_end=signal_end)
    net = accounting["scenarios"]["all_costs"]
    diagnostics = net["diagnostics"]
    ledger = net["ledger"]
    if (net["status"] != "complete" or ledger.empty
            or not diagnostics.get("liquidation_reached")
            or pd.Timestamp(diagnostics["liquidation_date"]) >= fold.retrain_at
            or diagnostics.get("final_quantities")):
        raise CandidateRejected(f"uncertified validation ledger: {diagnostics.get('halt_reason')}")
    if not ledger.index.equals(pd.date_range(ledger.index[0], ledger.index[-1], name="date")):
        raise CandidateRejected("internal accounting gap")
    if not np.isfinite(ledger["return"]).all():
        raise CandidateRejected("nonfinite certified return")
    # Only outside the certified active interval is cash padding allowed.
    returns = ledger["return"].reindex(calendar, fill_value=0.0)
    metrics = summarize_returns(returns, periods_per_year=365)
    sharpe = metrics["sharpe"]
    if sharpe is None or not np.isfinite(sharpe):
        raise CandidateRejected("undefined validation net Sharpe")
    return {"mean_rank_ic": float(daily["rank_ic"].mean()), "net_sharpe": float(sharpe),
            "net_return": float(metrics["total_return"]),
            "max_drawdown": float(metrics["max_drawdown"]), "ic_dates": int(len(daily)),
            "min_prediction_coverage": float(coverage.min()),
            "liquidation_date": diagnostics["liquidation_date"]}


def rank_candidates(records, config):
    table = pd.DataFrame(records)
    if table.empty:
        raise CandidateRejected("no candidates evaluated")
    valid = table["status"].eq("valid")
    for name in ("mean_rank_ic", "net_sharpe"):
        if name not in table:
            table[name] = np.nan
        valid &= np.isfinite(pd.to_numeric(table[name], errors="coerce"))
    if not valid.any():
        raise CandidateRejected("all candidates rejected", records=records)
    count = int(valid.sum())
    for name in ("mean_rank_ic", "net_sharpe"):
        table.loc[valid, name + "_percentile"] = (table.loc[valid, name].rank(method="average") - 0.5) / count
    table.loc[valid, "selection_score"] = (
        config.ic_weight * table.loc[valid, "mean_rank_ic_percentile"]
        + config.sharpe_weight * table.loc[valid, "net_sharpe_percentile"])
    ranked = table.loc[valid].sort_values(
        ["selection_score", "net_sharpe", "mean_rank_ic", "candidate_id"],
        ascending=[False, False, False, True], kind="stable")
    return str(ranked.iloc[0]["candidate_id"]), table
```

Reuse of `_daily_ic` is deliberate: it keeps the existing tie and finite-pair behavior. It is a private helper, so the adapter's tests are the compatibility boundary. Do not modify that helper's behavior. The same applies to consuming the documented accounting dictionary.

### File: tests/ml_factor_mining/test_scoring.py

```python
import numpy as np
import pandas as pd
import pytest
from ML_factor_mining.config import Config, quarter_folds
from ML_factor_mining.scoring import CandidateRejected, MarketWindow, rank_candidates, score_validation


def test_composite_preserves_sign_and_excludes_invalid():
    rows = [
        {"candidate_id": "wrong", "status": "valid", "mean_rank_ic": -0.8, "net_sharpe": -3.0},
        {"candidate_id": "right", "status": "valid", "mean_rank_ic": 0.1, "net_sharpe": 1.0},
        {"candidate_id": "broken", "status": "rejected", "mean_rank_ic": 1.0, "net_sharpe": 99.0},
    ]
    winner, table = rank_candidates(rows, Config())
    assert winner == "right"
    assert pd.isna(table.set_index("candidate_id").loc["broken", "selection_score"])


def test_ties_and_single_candidate_are_deterministic():
    one = {"candidate_id": "a", "status": "valid", "mean_rank_ic": 0.1, "net_sharpe": 1.0}
    winner, table = rank_candidates([one], Config())
    assert winner == "a"
    assert table.iloc[0]["selection_score"] == 0.5
    assert rank_candidates([{**one, "candidate_id": "b"}, one], Config())[0] == "a"
    with pytest.raises(CandidateRejected) as rejected:
        rank_candidates([{**one, "net_sharpe": None}], Config())
    assert rejected.value.records[0]["candidate_id"] == "a"


def test_validation_tail_and_profile_are_bounded(monkeypatch):
    config = Config(min_pairs=5, min_val_dates=2)
    fold = quarter_folds(config, "2025-03-31")[0]
    dates = pd.date_range("2024-10-01", "2024-12-31", name="date")
    members = pd.DataFrame(True, index=dates, columns=list("ABCDE"))
    matrix = pd.DataFrame(np.tile(np.arange(5), (len(dates), 1)), index=dates, columns=list("ABCDE"))
    pred = matrix.rename_axis(columns="instrument").stack(future_stack=True)
    targets = pred.rename("raw_return").to_frame()
    market = MarketWindow(matrix + 100, pd.DataFrame(), pd.DataFrame())
    calls = {}

    def fake_backtest(values, opens, events, quality, profile, *, signal_start, signal_end):
        calls.update({"end": signal_end, "profile": profile, "values": values})
        ledger = pd.DataFrame({"return": np.resize([0.01, -0.004], len(dates))}, index=dates)
        return {"scenarios": {"all_costs": {"status": "complete", "ledger": ledger,
            "diagnostics": {"liquidation_reached": True, "liquidation_date": "2024-12-31",
                            "final_quantities": {}}}}}

    monkeypatch.setattr("ML_factor_mining.scoring.run_backtest", fake_backtest)
    result = score_validation(pred, targets, members, market, fold, config)
    assert calls["end"] == pd.Timestamp("2024-12-27")
    assert calls["values"].loc["2024-12-28":].isna().all().all()
    assert calls["profile"].rebalance_days == 3
    assert calls["profile"].n_groups == 5
    assert calls["profile"].include_funding
    assert result["mean_rank_ic"] > 0.99
```

The mocked test isolates the adapter's temporal contract; it does not replace the real engine integration test in Task 9.

Run: `./.venv/bin/python -m pytest tests/ml_factor_mining/test_scoring.py -q`.

Checkpoint commit: `feat: select ML candidates by signed RankIC and certified net Sharpe`.

## Task 6: Quarter training, refitting, and immutable predictions

**Files:** Create `training.py` and `tests/ml_factor_mining/test_training.py`.

- [ ] Write cutoff/prediction tests before implementing orchestration.
- [ ] Prepare each quarter's history once, with a provider capped before retraining. Reuse the prepared object across selected model families.
- [ ] Admit features from the fit-only panel; apply the frozen map to validation, refit, and inference.
- [ ] Select a candidate using history only, refit once, then load the prediction quarter's data.
- [ ] Save native models, scores, feature maps, boundaries, and predictions in a new quarter directory. Existing quarter directories must cause an error, not an overwrite.
- [ ] Run tests; commit after passing.

### File: ML_factor_mining/training.py

```python
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from factor_common.data_provider import DataProvider
from .artifacts import write_json, write_parquet
from .dataset import FeatureMap, build_panel
from .models import candidate_bank, fit_model, save_model
from .scoring import CandidateRejected, load_market, rank_candidates, score_validation
from .targets import build_targets, partition_masks


@dataclass
class PreparedFold:
    X: pd.DataFrame
    targets: pd.DataFrame
    features: FeatureMap
    rows: dict
    presence: pd.Series
    membership: pd.DataFrame
    market: object


def prepare_fold(catalog, h5_path, fold, config):
    cutoff = fold.retrain_at - pd.Timedelta(days=1)
    provider = DataProvider(h5_path, as_of=cutoff)
    membership = provider.get_universe(start=fold.history_start, end=cutoff)
    panel = build_panel(catalog, membership)
    masks = partition_masks(panel.index, fold, config.holding_days)
    features = FeatureMap.fit(panel.loc[masks["fit"]], config.coverage)
    X, presence = features.transform(panel)
    opens = provider.get_single_data("open", start=fold.history_start, end=cutoff)
    opens = opens.reindex(index=membership.index, columns=membership.columns)
    targets = build_targets(opens, membership, config).reindex(panel.index)
    supervised = presence & targets["target"].notna()
    rows = {name: X.index[mask & supervised.to_numpy()] for name, mask in masks.items()}
    fit_dates = rows["fit"].get_level_values("date").nunique()
    val_dates = rows["validation"].get_level_values("date").nunique()
    if fit_dates < config.min_fit_dates or val_dates < config.min_val_dates:
        raise ValueError(f"insufficient supervised dates: fit={fit_dates}, validation={val_dates}")
    market = load_market(provider, fold.validation_start, cutoff, membership.columns)
    return PreparedFold(X, targets, features, rows, presence, membership, market)


def select_and_refit(prepared, fold, config, name):
    X, targets, rows = prepared.X, prepared.targets, prepared.rows
    val_mask = partition_masks(X.index, fold, config.holding_days)["validation"]
    inference_rows = X.index[val_mask & prepared.presence.to_numpy()]
    records, fitted = [], {}
    for candidate_id, params in candidate_bank(name, config):
        model = fit_model(name, params, X.loc[rows["fit"]], targets.loc[rows["fit"], "target"],
                          config, X_val=X.loc[rows["validation"]],
                          y_val=targets.loc[rows["validation"], "target"])
        record = {"candidate_id": candidate_id, "params_json": json.dumps(params, sort_keys=True),
                  "rounds": model.rounds}
        predictions = pd.Series(model.predict(X.loc[inference_rows]), index=inference_rows, name="factor")
        try:
            metrics = score_validation(predictions, targets, prepared.membership,
                                       prepared.market, fold, config)
            record.update(status="valid", **metrics)
            fitted[candidate_id] = (params, model.rounds)
        except CandidateRejected as exc:
            record.update(status="rejected", reason=str(exc))
        records.append(record)
    # Unexpected fit/prediction errors propagate; they are not performance rejections.
    winner, table = rank_candidates(records, config)
    params, rounds = fitted[winner]
    final = fit_model(name, params, X.loc[rows["refit"]], targets.loc[rows["refit"], "target"],
                      config, rounds=rounds)
    known = X.loc[rows["refit"]].copy()
    known["__target__"] = targets.loc[rows["refit"], "target"]
    digest = hashlib.sha256(pd.util.hash_pandas_object(known, index=True).values.tobytes())
    identity = {"model": name, "quarter": fold.quarter, "params": params, "rounds": rounds,
                "columns": list(X.columns), "holding_days": config.holding_days,
                "target_type": config.target_type}
    digest.update(json.dumps(identity, sort_keys=True).encode())
    audit = {**identity, "model_id": f"{name}_{fold.quarter}_{digest.hexdigest()[:16]}",
             "winner": winner, "history_start": fold.history_start.date().isoformat(),
             "validation_start": fold.validation_start.date().isoformat(),
             "retrain_at": fold.retrain_at.date().isoformat(),
             "knowledge_end": (fold.retrain_at - pd.Timedelta(days=1)).date().isoformat(),
             "feature_admission": prepared.features.diagnostics,
             "selected_factors": list(prepared.features.factors),
             "rows": {key: len(index) for key, index in rows.items()},
             "max_exit": {key: targets.loc[index, "exit_date"].max().date().isoformat()
                          for key, index in rows.items()},
             "direction": 1, "scope": "model_level_oos"}
    return final, table, audit


def predict_quarter(model, features, catalog, h5_path, fold, audit):
    # No label construction or future price read is reachable from this function.
    provider = DataProvider(h5_path, as_of=fold.prediction_end)
    membership = provider.get_universe(start=fold.retrain_at, end=fold.prediction_end)
    panel = build_panel(catalog, membership)
    X, present = features.transform(panel)
    if not present.any():
        raise ValueError("no observable feature vectors in prediction quarter")
    selected = X.loc[present]
    predictions = pd.Series(model.predict(selected), index=selected.index, name="factor").reset_index()
    predictions["model_id"] = audit["model_id"]
    predictions["quarter"] = fold.quarter
    predictions["trained_before"] = fold.retrain_at
    if predictions.duplicated(["date", "instrument"]).any():
        raise AssertionError("duplicate predictions")
    coverage = present.groupby(level="date").agg(["sum", "count"])
    coverage = coverage.rename(columns={"sum": "predicted", "count": "eligible"})
    coverage["coverage"] = coverage["predicted"] / coverage["eligible"]
    return predictions, coverage.reset_index()


def save_quarter(directory, model, table, audit, predictions, coverage):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"immutable quarter already exists: {directory}")
    pending = directory.with_name(directory.name + ".pending")
    pending.mkdir(parents=True, exist_ok=False)
    save_model(model, pending)
    write_json(pending / "audit.json", audit)
    write_parquet(pending / "candidates.parquet", table)
    write_parquet(pending / "predictions.parquet", predictions)
    write_parquet(pending / "coverage.parquet", coverage)
    os.replace(pending, directory)
```

The selected feature list remains frozen during refit. Re-admitting factors using validation coverage would invalidate the candidate comparison. The final refit includes the fit/validation embargo gap once its labels are mature at Q, which is permissible because validation selection is already finished.

Prediction values are raw model scores, not clipped to [-1, 1]. A regressor can predict outside its target range; grouping depends on ordering. Preserve scores for diagnostics. Do not clip them into ties or reinterpret them as percentages when the target is `rank_return`.

### File: tests/ml_factor_mining/test_training.py

```python
import numpy as np
import pandas as pd
import pytest
from ML_factor_mining.config import Config, quarter_folds
from ML_factor_mining.dataset import FeatureMap
from ML_factor_mining.training import predict_quarter


class FirstColumnModel:
    def predict(self, X):
        return X.iloc[:, 0].to_numpy()


def test_prediction_uses_only_membership_and_factor_values(tmp_path, monkeypatch):
    dates = pd.date_range("2025-01-01", periods=3, name="date")
    members = pd.DataFrame(True, index=dates, columns=["A", "B"])
    values = pd.DataFrame({"date": dates.repeat(2), "instrument": ["A", "B"] * 3,
                           "factor": [-2, 2, np.nan, np.nan, -1, 1]})
    path = tmp_path / "f.parquet"
    values.to_parquet(path)
    calls = []

    class Provider:
        def __init__(self, path, *, as_of):
            calls.append(as_of)
        def get_universe(self, *, start, end):
            return members.loc[start:end]
        def get_single_data(self, *args, **kwargs):
            raise AssertionError("prediction must not read label prices")

    monkeypatch.setattr("ML_factor_mining.training.DataProvider", Provider)
    fold = quarter_folds(Config(), dates[-1])[0]
    features = FeatureMap(("F",), {})
    result, coverage = predict_quarter(FirstColumnModel(), features,
        [{"factor_id": "F", "values": str(path)}], "unused.h5", fold, {"model_id": "m"})
    assert calls == [dates[-1]]
    assert len(result) == 4
    assert dates[1] not in set(result.date)
    assert coverage.loc[coverage.date.eq(dates[1]), "coverage"].item() == 0
    assert set(result.columns) == {"date", "instrument", "factor", "model_id", "quarter", "trained_before"}
```

Run: `./.venv/bin/python -m pytest tests/ml_factor_mining/test_training.py -q`.

Checkpoint commit: `feat: orchestrate quarterly fit selection refit and daily inference`.

## Task 7: Continuous replay and quarterly reports

**Files:** Create `evaluation.py` and `tests/ml_factor_mining/test_evaluation.py`.

- [ ] Write a quarter-compounding test that would fail if quarter-start costs or returns were dropped.
- [ ] Implement continuous evaluation through `FactorManager`; never instantiate a new portfolio at every quarter boundary.
- [ ] Save the original daily ledger beside the quarterly table so returns can be reconciled independently.
- [ ] Label incomplete evidence, accounting tails, and label-based group diagnostics explicitly.
- [ ] Run tests and commit after passing.

### File: ML_factor_mining/evaluation.py

```python
from dataclasses import asdict
import html
from pathlib import Path
import numpy as np
import pandas as pd
from factor_common.manager import FactorManager
from factor_common.labels import make_labels
from factor_common.metrics import _daily_ic, _mean_turnover, summarize_returns
from .artifacts import write_json, write_parquet
from .dataset import clean_factor_table
from .scoring import backtest_profile


def quarterly_metrics(ledger, daily_ic, predictions, coverage, *, evidence_end, accounting_status):
    records = []
    first = pd.Timestamp(predictions.date.min()).to_period("Q")
    last_date = pd.Timestamp(predictions.date.max())
    if not ledger.empty:
        last_date = max(last_date, ledger.index[-1])
    for quarter in pd.period_range(first, last_date.to_period("Q"), freq="Q"):
        start, end = quarter.start_time, quarter.end_time.normalize()
        segment = ledger.loc[start:end]
        pred = predictions.loc[predictions.date.between(start, end)]
        ic = daily_ic.loc[daily_ic.date.between(start, end)]
        cover = coverage.loc[coverage.date.between(start, end)]
        summary = summarize_returns(segment["return"], periods_per_year=365)
        full_calendar = pd.date_range(start, end)
        actual_end = segment.index[-1] if len(segment) else None
        complete = actual_end is not None and actual_end >= end and pd.Timestamp(evidence_end) >= end
        row = {"quarter": str(quarter), **summary,
               "turnover": _mean_turnover(segment) if not segment.empty else None,
               "mean_rank_ic": float(ic.rank_ic.mean()) if len(ic) else None,
               "ic_dates": int(len(ic)), "prediction_days": int(pred.date.nunique()),
               "expected_calendar_days": len(full_calendar),
               "coverage_mean": float(cover.coverage.mean()) if len(cover) else None,
               "complete_calendar_quarter": bool(complete),
               "accounting_tail_only": bool(pred.empty and not segment.empty),
               "evaluated_through": actual_end.date().isoformat() if actual_end is not None else None,
               "accounting_status": accounting_status,
               "metrics_scope": "full_run_ledger_slice" if accounting_status == "complete" else "certified_prefix_slice"}
        records.append(row)
    return pd.DataFrame(records)


def manager_params(config, start, end):
    profile = backtest_profile(config)
    # FactorManager exposes a smaller override surface than resolve_profile.
    allowed = ("rebalance_days", "anchor_date", "n_groups", "factor_direction",
               "fee_rate", "slippage", "include_funding", "funding_price_mode")
    params = {key: getattr(profile, key) for key in allowed}
    params.update(start=pd.Timestamp(start).date().isoformat(),
                  end=pd.Timestamp(end).date().isoformat(),
                  split_date=(pd.Timestamp(start) - pd.Timedelta(days=1)).date().isoformat())
    return params


def evaluate_oos(predictions, coverage, *, config, h5_path, model_dir, report_dir, name, evidence_end):
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    factor = clean_factor_table(predictions[["date", "instrument", "factor"]])
    write_parquet(model_dir / "factor_oos.parquet", factor)
    end = min(factor.date.max(), pd.Timestamp(evidence_end) - pd.Timedelta(days=config.holding_days + 1))
    start = factor.date.min()
    if end < start:
        write_json(model_dir / "evaluation.json", {"status": "insufficient_execution_tail"})
        return "insufficient_execution_tail"
    evaluated = factor.loc[factor.date.between(start, end)]
    manager = FactorManager(h5_path=h5_path, base_dir=model_dir / "evaluation_cache",
                            reports_dir=report_dir, persist_evaluations=True, as_of=evidence_end)
    # External DataFrames are not automatically membership-masked by the manager.
    membership = manager.dp.get_universe(start=start, end=end)
    mask = membership.rename_axis(columns="instrument").stack(future_stack=True)
    keys = pd.MultiIndex.from_frame(evaluated[["date", "instrument"]])
    if not mask.reindex(keys).fillna(False).all():
        raise ValueError("OOS output contains ineligible historical instruments")
    params = manager_params(config, start, end)
    result = manager.evaluate(evaluated, factor_name=f"ML_{name}_{config.target_type}_h{config.holding_days}",
                              params=params, plot=True)
    net = result["factor_result"]["scenarios"]["all_costs"]
    ledger = net["ledger"]
    write_parquet(model_dir / "daily_ledger.parquet", ledger.reset_index())
    write_parquet(model_dir / "orders.parquet", net["orders"])
    write_parquet(model_dir / "positions.parquet", net["positions"].reset_index())
    write_parquet(model_dir / "funding.parquet", net["funding"])
    write_parquet(model_dir / "funding_coverage.parquet", net["funding_coverage"])
    prices = manager.dp.get_single_data("open", start=start,
        end=end + pd.Timedelta(days=config.holding_days + 1))
    labels = make_labels(prices, config.holding_days)
    matrix = evaluated.pivot(index="date", columns="instrument", values="factor")
    daily = _daily_ic(matrix, labels.reindex(index=matrix.index, columns=matrix.columns))
    daily = daily.loc[daily.n_pairs.ge(config.min_pairs) & np.isfinite(daily.rank_ic)]
    write_parquet(model_dir / "daily_ic.parquet", daily)
    table = quarterly_metrics(ledger, daily, predictions, coverage,
        evidence_end=evidence_end, accounting_status=net["status"])
    write_parquet(model_dir / "quarterly_metrics.parquet", table)
    write_parquet(model_dir / "group_forward_return_diagnostics.parquet",
                  result["group_returns"].reset_index())
    report_name = Path(result["paths"]["report_path"]).name
    explanation = (
        "All predictions are model-level out-of-sample. Quarterly portfolio returns are slices "
        "of one continuous all-costs ledger, not compounded forward labels. Latest-quarter "
        "and incomplete-accounting rows must be read with their coverage/status columns. "
        "Underlying factor discovery may be retrospective. Forward group averages are diagnostics."
    )
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>Quarterly ML factor evaluation</title>"
            "<style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse}"
            "td,th{padding:8px;border:1px solid #ddd}th{background:#eee}</style>"
            f"<h1>{html.escape(name)} — quarterly OOS evaluation</h1>"
            f"<p>{html.escape(explanation)}</p>"
            f"<p>Accounting status: {html.escape(net['status'])}</p>"
            f"<p><a href='{html.escape(report_name, quote=True)}'>Continuous NAV and standard factor report</a></p>"
            + table.to_html(index=False, escape=True, na_rep="unavailable") + "</html>")
    (report_dir / "quarterly.html").write_text(page, encoding="utf-8")
    write_json(model_dir / "evaluation.json", {
        "status": result["status"], "all_costs_status": net["status"],
        "prediction_end": factor.date.max().date().isoformat(),
        "evaluation_signal_end": end.date().isoformat(),
        "evidence_end": pd.Timestamp(evidence_end).date().isoformat(),
        "report": str(report_dir / "quarterly.html"),
        "manager_validation": result["diagnostics"]["validation"],
    })
    return result["status"]
```

The `FactorManager` report remains honest about external-factor validation. The ML audit created in Task 8 is separate. Do not manually edit the manager's returned validation status. Full synthetic and local-data cutoff tests are required for the ML computation chain; frozen upstream factors require their own source-generation evidence.

### File: tests/ml_factor_mining/test_evaluation.py

```python
import numpy as np
import pandas as pd
import pytest
from ML_factor_mining.evaluation import quarterly_metrics


def test_manager_parameters_match_existing_entrypoint():
    from factor_common.manager import FactorManager
    from factor_common.profiles import resolve_profile
    from ML_factor_mining.config import Config
    from ML_factor_mining.evaluation import manager_params
    from ML_factor_mining.scoring import backtest_profile
    config = Config()
    params = manager_params(config, "2025-01-01", "2025-03-31")
    start, end, overrides = FactorManager._split_params(params)
    actual = resolve_profile("perp_1d", overrides)
    expected = backtest_profile(config)
    for name in ("signal_delay_days", "price_field", "gross_exposure", "initial_equity",
                 "periods_per_year", "rebalance_days", "fee_rate", "slippage"):
        assert getattr(actual, name) == getattr(expected, name)
    assert start == pd.Timestamp("2025-01-01")
    assert actual.split_date == "2024-12-31"


def test_quarter_returns_reconcile_with_continuous_ledger():
    dates = pd.to_datetime(["2025-03-30", "2025-03-31", "2025-04-01", "2025-04-02"])
    ledger = pd.DataFrame({"return": [-0.01, 0.02, -0.03, 0.04],
        "equity": [0.99, 1.0098, 0.979506, 1.01868624],
        "fee": [0.01, 0, 0, 0], "slippage": [0, 0, 0, 0],
        "trade_notional": [1, 0, 0, 1]}, index=dates)
    pred = pd.DataFrame({"date": dates, "instrument": "A", "factor": 1.0})
    ic = pd.DataFrame({"date": dates, "rank_ic": [0.1] * 4})
    coverage = pd.DataFrame({"date": dates, "coverage": 1.0})
    table = quarterly_metrics(ledger, ic, pred, coverage,
        evidence_end="2025-04-02", accounting_status="complete").set_index("quarter")
    assert table.loc["2025Q1", "total_return"] == pytest.approx(0.99 * 1.02 - 1)
    assert table.loc["2025Q2", "total_return"] == pytest.approx(0.97 * 1.04 - 1)
    assert np.prod(1 + table.total_return) == pytest.approx(np.prod(1 + ledger["return"]))
    assert not table.loc["2025Q2", "complete_calendar_quarter"]
```

Run: `./.venv/bin/python -m pytest tests/ml_factor_mining/test_evaluation.py -q`.

Checkpoint commit: `feat: report quarterly returns from continuous ML OOS accounting`.

## Task 8: CLI, provenance, and cutoff verification

**Files:** Create `validation.py`, `cli.py`, `README.md`, and `tests/ml_factor_mining/test_cli.py`.

Add `ML_factor_mining/validation.py` to the file map: it owns ML-layer cutoff replay and explicitly separate upstream validation evidence. Its existence does not change `factor_common/validation.py`.

- [ ] Write CLI safety and artifact tests first; confirm import/parser failure before implementation.
- [ ] Implement `run` and `verify`; both resolve repository-relative paths consistently from the package's root.
- [ ] Preserve quarterly outputs on failure; record `state.json`. Reject a reused run name. Do not append predictions with a different input snapshot.
- [ ] Store source-code hashes for the new package and reused framework with dependency versions.
- [ ] Implement actual retraining at cutoff; simply slicing saved predictions does not verify causality.
- [ ] Add the README verbatim below, run the CLI tests, and commit the task files.

### File: ML_factor_mining/validation.py

```python
import json
from pathlib import Path
import pandas as pd
from factor_common.validation import check_cutoff, scan_future_leaks
from .artifacts import discover_catalog, sha256, write_json
from .config import Config, quarter_folds
from .training import prepare_fold, select_and_refit, predict_quarter


def assert_frozen_inputs(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["hashes"].items():
        if sha256(run_dir / "inputs" / relative) != expected:
            raise ValueError(f"snapshot hash mismatch: {relative}")
    return manifest


def replay_model(config, catalog, h5_path, name, cutoff):
    tables = []
    for fold in quarter_folds(config, cutoff):
        prepared = prepare_fold(catalog, h5_path, fold, config)
        final, scores, audit = select_and_refit(prepared, fold, config, name)
        predictions, coverage = predict_quarter(final, prepared.features, catalog, h5_path, fold, audit)
        tables.append(predictions[["date", "instrument", "factor"]])
    if not tables:
        raise ValueError("cutoff precedes first OOS quarter")
    table = pd.concat(tables, ignore_index=True)
    if table.duplicated(["date", "instrument"]).any():
        raise AssertionError("overlapping prediction quarters")
    matrix = table.pivot(index="date", columns="instrument", values="factor").sort_index(axis=1)
    return matrix.reindex(pd.date_range(config.oos_start, cutoff, name="date"))


def verify_run(run_dir, name, cutoffs):
    run_dir = Path(run_dir)
    manifest = assert_frozen_inputs(run_dir)
    config = Config.from_dict(manifest["config"])
    if name not in config.models:
        raise ValueError("model was not part of this run")
    root = Path(__file__).resolve().parents[1]
    for relative, expected in manifest.get("code_hashes", {}).items():
        if sha256(root / relative) != expected:
            raise ValueError(f"code differs from the trained run: {relative}")
    package = Path(__file__).resolve().parent
    findings = scan_future_leaks(package.glob("*.py"))
    if findings:
        raise AssertionError(f"ML static scan findings: {findings}")
    full_table = pd.read_parquet(run_dir / name / "factor_oos.parquet")
    full = full_table.pivot(index="date", columns="instrument", values="factor").sort_index(axis=1)
    evidence = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))["prediction_end"]
    full = full.reindex(pd.date_range(config.oos_start, evidence, name="date"))
    cutoffs = [pd.Timestamp(c) for c in cutoffs]
    if not cutoffs or any(c < full.index[0] or c >= full.index[-1] for c in cutoffs):
        raise ValueError("cutoffs must fall inside the OOS interval and before its end")
    catalog = discover_catalog(run_dir / "inputs" / "factors", config)

    def compute(cutoff):
        if cutoff is None:
            return full
        return replay_model(config, catalog, run_dir / "inputs" / "crypto_quant.h5", name, cutoff)

    checks = check_cutoff(compute, cutoffs, atol=1e-10)
    upstream = {}
    for record in catalog:
        meta = json.loads(Path(record["metadata"]).read_text(encoding="utf-8"))
        upstream[record["factor_id"]] = meta.get("metadata", {}).get("validation", {"status": "absent"})
    payload = {"scope": "ML transformations, training, selection, and inference on a frozen factor catalog",
               "static_findings": findings, "ml_cutoff": checks,
               "upstream_cached_evidence": upstream,
               "upstream_recomputed_here": False, "research_scope": "model_level_oos"}
    write_json(run_dir / name / "future_leak_audit.json", payload)
    return payload
```

`replay_model` rebuilds providers at historical boundaries and repeats selection/refit. It does not regenerate underlying factor formulas. Synthetic future-input perturbations in Task 9 prove the ML chain; upstream metadata is retained as separate evidence. A later end-to-end factor-generation replay can strengthen upstream certification without changing this interface.

### File: ML_factor_mining/cli.py

```python
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
import pandas as pd
from factor_common.data_provider import DataProvider
from .artifacts import discover_catalog, sha256, snapshot_inputs, write_json, write_parquet
from .config import Config, quarter_folds
from .evaluation import evaluate_oos
from .models import candidate_bank, require_backends
from .training import prepare_fold, select_and_refit, predict_quarter, save_quarter
from .validation import assert_frozen_inputs, verify_run


def parser():
    p = argparse.ArgumentParser(description="Quarterly ML combinations of stored factor values")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", default="ML_factor_mining/configs/default.json")
    run.add_argument("--run-name", required=True)
    run.add_argument("--factor-dir", default="data/factor_results")
    run.add_argument("--h5-path", default="data/crypto_quant.h5")
    run.add_argument("--models", nargs="+")
    run.add_argument("--mode", choices=["expanding", "rolling"])
    run.add_argument("--holding-days", type=int)
    run.add_argument("--end")
    run.add_argument("--as-of", help="Latest completed UTC data date allowed in this experiment")
    verify = sub.add_parser("verify")
    verify.add_argument("--run-name", required=True)
    verify.add_argument("--model", required=True)
    verify.add_argument("--cutoffs", nargs="+", required=True)
    return p


def safe_run_path(root, name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", name):
        raise ValueError("run name must contain only letters, digits, underscores, and hyphens")
    return Path(root) / "outputs" / "ml_factor_mining" / name


def execute_run(args, root):
    root = Path(root)
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    data = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    for name in ("models", "mode", "holding_days", "end"):
        value = getattr(args, name)
        if value is not None:
            data[name] = value
    config = Config.from_dict(data)
    require_backends(config.models)
    for name in config.models:
        candidate_bank(name, config)
    run_dir = safe_run_path(root, args.run_name)
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}; choose a new name")
    state = {"status": "preparing", "completed_quarters": []}
    try:
        snapshot = snapshot_inputs(resolve(args.factor_dir), resolve(args.h5_path), run_dir, config)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources = sorted((root / "ML_factor_mining").glob("*.py"))
        sources += sorted((root / "factor_common").glob("*.py"))
        sources += sorted((root / "data" / "crypto_quant").glob("*.py"))
        manifest["code_hashes"] = {str(p.relative_to(root)): sha256(p) for p in sources}
        manifest["requested_as_of"] = args.as_of
        write_json(manifest_path, manifest)
        dp = DataProvider(snapshot / "crypto_quant.h5", as_of=args.as_of)
        first, last = dp.get_time_range()
        if last is None:
            raise ValueError("empty market history")
        # Never treat the currently forming UTC bar as complete.
        completed_day = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
        evidence_end = min(last, completed_day)
        if args.as_of is not None:
            evidence_end = min(evidence_end, pd.Timestamp(args.as_of))
        prediction_end = min(pd.Timestamp(config.end), evidence_end)
        folds = quarter_folds(config, prediction_end)
        if not folds:
            raise ValueError("no OOS quarters available")
        state.update(status="training", prediction_end=prediction_end.date().isoformat(),
                     evidence_end=evidence_end.date().isoformat())
        write_json(run_dir / "state.json", state)
        catalog = discover_catalog(snapshot / "factors", config)
        accumulated = {name: [] for name in config.models}
        coverage_tables = {name: [] for name in config.models}
        for fold in folds:
            state.update(active_quarter=fold.quarter, active_model=None)
            write_json(run_dir / "state.json", state)
            prepared = prepare_fold(catalog, snapshot / "crypto_quant.h5", fold, config)
            for name in config.models:
                state["active_model"] = name
                model, scores, audit = select_and_refit(prepared, fold, config, name)
                predictions, coverage = predict_quarter(model, prepared.features, catalog,
                    snapshot / "crypto_quant.h5", fold, audit)
                save_quarter(run_dir / name / "quarters" / fold.quarter,
                             model, scores, audit, predictions, coverage)
                accumulated[name].append(predictions)
                coverage_tables[name].append(coverage)
                state["completed_quarters"].append(f"{name}:{fold.quarter}")
                write_json(run_dir / "state.json", state)
                print(f"saved {name} {fold.quarter}: {len(predictions)} predictions", flush=True)
        statuses = {}
        for name in config.models:
            predictions = pd.concat(accumulated[name], ignore_index=True).sort_values(["date", "instrument"])
            if predictions.duplicated(["date", "instrument"]).any():
                raise AssertionError("quarter overlap in OOS output")
            coverage = pd.concat(coverage_tables[name], ignore_index=True)
            write_parquet(run_dir / name / "predictions_oos.parquet", predictions)
            write_parquet(run_dir / name / "coverage.parquet", coverage)
            statuses[name] = evaluate_oos(predictions, coverage, config=config,
                h5_path=snapshot / "crypto_quant.h5", model_dir=run_dir / name,
                report_dir=root / "reports" / "ml_factor_mining" / args.run_name / name,
                name=name, evidence_end=evidence_end)
        assert_frozen_inputs(run_dir)
        state.update(status="complete" if all(s == "complete" for s in statuses.values()) else "incomplete",
                     evaluation_status=statuses, future_leak_verification="not_run",
                     active_quarter=None, active_model=None)
        write_json(run_dir / "state.json", state)
        print(json.dumps(state, indent=2))
        return 0 if state["status"] == "complete" else 2
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if run_dir.exists():
            rejected = getattr(exc, "records", None)
            if rejected is not None:
                write_parquet(run_dir / "failed_candidates.parquet", pd.DataFrame(rejected))
            write_json(run_dir / "state.json", state)
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.command == "run":
        return execute_run(args, root)
    run_dir = safe_run_path(root, args.run_name)
    require_backends([args.model])
    result = verify_run(run_dir, args.model, args.cutoffs)
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    state.setdefault("verified_models", {})[args.model] = result["ml_cutoff"]["status"]
    write_json(run_dir / "state.json", state)
    print(json.dumps({"model": args.model, "ml_cutoff": result["ml_cutoff"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

The run's `future_leak_verification` field initially says `not_run`; per-model completed checks live in `verified_models` and the audit JSON. A run can have completed training/accounting without having completed the research acceptance checklist. Keep these statuses distinct. V1 does not implement resume: use a new run name after a failure; existing files remain available for diagnosis.

### File: tests/ml_factor_mining/test_cli.py

```python
import pandas as pd
import pytest
from ML_factor_mining.artifacts import write_json, write_parquet
from ML_factor_mining.cli import parser, safe_run_path
from ML_factor_mining.validation import assert_frozen_inputs


def test_cli_model_and_horizon_options():
    args = parser().parse_args(["run", "--run-name", "check", "--models", "ridge", "linear",
                                "--holding-days", "3", "--mode", "rolling"])
    assert args.models == ["ridge", "linear"]
    assert args.holding_days == 3
    assert args.mode == "rolling"


def test_run_names_cannot_escape_output_root(tmp_path):
    for name in ("../outside", "/tmp/outside", "", "a/b"):
        with pytest.raises(ValueError):
            safe_run_path(tmp_path, name)


def test_atomic_output_and_snapshot_tamper_detection(tmp_path):
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]), "instrument": ["A"], "factor": [0.1]})
    write_parquet(tmp_path / "predictions.parquet", frame)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "predictions.parquet"), frame)
    assert not (tmp_path / "predictions.parquet.tmp").exists()
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "x").write_text("changed")
    write_json(tmp_path / "manifest.json", {"hashes": {"x": "not_the_real_hash"}})
    with pytest.raises(ValueError, match="hash mismatch"):
        assert_frozen_inputs(tmp_path)


def test_failed_selection_preserves_candidate_reasons(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from ML_factor_mining import cli
    from ML_factor_mining.config import Config
    from ML_factor_mining.scoring import CandidateRejected
    config = Config(models=("linear",), end="2025-03-31")
    write_json(tmp_path / "config.json", asdict(config))

    def fake_snapshot(factor_dir, h5_path, run_dir, config):
        run_dir.mkdir(parents=True)
        snapshot = run_dir / "inputs"
        snapshot.mkdir()
        write_json(run_dir / "manifest.json", {"config": asdict(config), "hashes": {}})
        return snapshot

    class Provider:
        def __init__(self, path, *, as_of):
            pass
        def get_time_range(self):
            return pd.Timestamp("2024-01-01"), pd.Timestamp("2025-03-31")

    def reject(*args, **kwargs):
        raise CandidateRejected("all candidates rejected", records=[{
            "candidate_id": "c000", "status": "rejected", "reason": "missing funding evidence"}])

    monkeypatch.setattr(cli, "require_backends", lambda names: None)
    monkeypatch.setattr(cli, "snapshot_inputs", fake_snapshot)
    monkeypatch.setattr(cli, "DataProvider", Provider)
    monkeypatch.setattr(cli, "discover_catalog", lambda *args: [])
    monkeypatch.setattr(cli, "prepare_fold", lambda *args: object())
    monkeypatch.setattr(cli, "select_and_refit", reject)
    args = cli.parser().parse_args(["run", "--run-name", "failure_case", "--config", "config.json"])
    with pytest.raises(CandidateRejected):
        cli.execute_run(args, tmp_path)
    directory = safe_run_path(tmp_path, "failure_case")
    saved = pd.read_parquet(directory / "failed_candidates.parquet")
    assert saved.iloc[0]["reason"] == "missing funding evidence"
    state = json.loads((directory / "state.json").read_text())
    assert state["active_quarter"] == "2025Q1"
    assert state["active_model"] == "linear"
    assert state["status"] == "failed"
```

### File: ML_factor_mining/README.md

```markdown
# Quarterly ML Factor Mining

Train quarterly combinations of stored factor values. Default: predict the rank of
the next executable three-day return, rebalance every three days, and infer daily.

Run all commands from the repository root using `./.venv/bin/python`.

Install: `./.venv/bin/python -m pip install -r ML_factor_mining/requirements-ml.txt`.

Default run:
`./.venv/bin/python -m ML_factor_mining.cli run --run-name ml_h3_expanding_v1`.

Four-model comparison:
`./.venv/bin/python -m ML_factor_mining.cli run --run-name ml_h3_four_models_v1 --models lightgbm xgboost ridge linear`.

Rolling twelve-month history:
`./.venv/bin/python -m ML_factor_mining.cli run --run-name ml_h3_rolling_v1 --mode rolling`.

Ridge smoke run:
`./.venv/bin/python -m ML_factor_mining.cli run --run-name ml_ridge_smoke_v1 --models ridge --end 2025-03-31`.

Cutoff replay:
`./.venv/bin/python -m ML_factor_mining.cli verify --run-name ml_ridge_smoke_v1 --model ridge --cutoffs 2025-02-15 2025-03-15`.

Outputs: `outputs/ml_factor_mining/<run_name>/<model>/factor_oos.parquet`
contains exactly `date`, `instrument`, and `factor`. Predictions represent scores,
not percent returns when `target_type=rank_return`.

Each model also saves rich predictions, quarterly model files, input coverage,
validation candidates, daily accounting, and quarterly metrics. Reports are under
`reports/ml_factor_mining/<run_name>/<model>/quarterly.html`.

The feature catalog uses metadata sidecars and excludes evaluation Parquets.
Include/exclude lists use metadata factor IDs. Missing values are not future-filled.
Each experiment snapshots factor inputs and H5 data; an existing run name is rejected.

Both training and validation labels must mature before their boundaries. Quarter
retraining does not reset the execution calendar or the continuous OOS portfolio.
The latest predictions can extend beyond the interval with fully realized labels;
evaluation dates and accounting-tail rows are explicitly recorded.

RankIC and all-costs long-short Sharpe select candidates using fixed equal weights.
Selection percentiles are local to one model/quarter, not comparable across models.
A negative-return winner is still a negative-return strategy. Inspect raw metrics.

Artifacts are model-level OOS research. Their existence does not establish that
underlying factor discovery was historically out of sample. The external-factor
manager report does not automatically certify the ML computation; inspect the
separate future-leak audit and upstream evidence.

Tests: `./.venv/bin/python -m pytest tests/ml_factor_mining -q`.
```

Run: `./.venv/bin/python -m pytest tests/ml_factor_mining/test_cli.py -q`.

Checkpoint commit: `feat: add reproducible ML experiment CLI and cutoff audits`.

## Task 9: Adversarial cutoff and real accounting integration tests

**Files:** Create `tests/ml_factor_mining/test_causality.py` and `tests/ml_factor_mining/test_accounting.py`.

- [ ] Add the tests below before calling the implementation complete.
- [ ] Run actual quarterly preparation, selection, refit, and prediction with a bounded synthetic provider. Recompute the historical result after perturbing future prices, membership, and feature values.
- [ ] Run a hand-calculated next-open/three-day portfolio through the real common engine, including nonzero commission, slippage, and a funding settlement.
- [ ] Run all ML tests and the relevant common-framework regression tests.
- [ ] Run at least one local H5 end-to-end experiment and two local-data cutoff replays. Archive evidence; if data or installation blocks completion, state the exact blocker rather than declaring success.

### File: tests/ml_factor_mining/test_causality.py

```python
import numpy as np
import pandas as pd
from ML_factor_mining.config import Config
from ML_factor_mining.models import Fitted
from ML_factor_mining.validation import replay_model


def test_full_retraining_is_invariant_to_future_perturbations(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", "2025-06-30", name="date")
    symbols = list("ABCDE")
    beta = np.linspace(-1, 1, len(symbols))
    time = np.arange(len(dates))
    changes = (0.002 + 0.0008 * np.sin(time / 11))[:, None] * beta[None, :]
    opens = pd.DataFrame(100 * np.exp(np.cumsum(changes, axis=0)), index=dates, columns=symbols)
    members = pd.DataFrame(True, index=dates, columns=symbols)
    factors = pd.DataFrame(np.tile(beta, (len(dates), 1)), index=dates, columns=symbols)
    path = tmp_path / "f.parquet"

    def save_factors(frame):
        frame.rename_axis(columns="instrument").stack(future_stack=True).rename("factor").reset_index().to_parquet(path, index=False)

    save_factors(factors)
    world = {"opens": opens, "membership": members}
    requests = []

    class BoundedProvider:
        def __init__(self, path, *, as_of):
            self.cutoff = pd.Timestamp(as_of)
        def bound(self, start, end):
            assert pd.Timestamp(end) <= self.cutoff
            requests.append((pd.Timestamp(start), pd.Timestamp(end), self.cutoff))
            return pd.Timestamp(start), pd.Timestamp(end)
        def get_universe(self, *, start, end):
            a, b = self.bound(start, end)
            return world["membership"].loc[a:b].copy()
        def get_single_data(self, field, *, start, end):
            assert field == "open"
            a, b = self.bound(start, end)
            return world["opens"].loc[a:b].copy()
        def get_funding(self, *, start, end, symbols):
            self.bound(start, end)
            return pd.DataFrame(columns=["funding_time", "instrument", "funding_rate", "mark_price"])
        def get_quality(self, *, start, end, symbols):
            a, b = self.bound(start, end)
            index = pd.MultiIndex.from_product([pd.date_range(a, b), symbols], names=["date", "instrument"])
            # This synthetic contract has no funding assessments; this is explicit fixture evidence.
            return pd.DataFrame({"has_placeholder_kline": False,
                                 "funding_coverage_status": "not_applicable"}, index=index)

    def numpy_linear_fit(name, params, X, y, config, **kwargs):
        # Exercise retraining without requiring a third-party backend in this causal test.
        # Production backend behavior is independently covered by test_models.py.
        design = np.column_stack([np.ones(len(X)), X.to_numpy()])
        coefficients = np.linalg.lstsq(design, np.asarray(y), rcond=None)[0]
        return Fitted("linear", {"intercept": float(coefficients[0]), "coef": coefficients[1:].tolist()},
                      tuple(X.columns), None)

    monkeypatch.setattr("ML_factor_mining.training.DataProvider", BoundedProvider)
    monkeypatch.setattr("ML_factor_mining.training.fit_model", numpy_linear_fit)
    catalog = [{"factor_id": "F", "values": str(path)}]
    config = Config(models=("linear",), min_pairs=5)
    full = replay_model(config, catalog, "synthetic.h5", "linear", dates[-1])
    cutoff = pd.Timestamp("2025-05-15")
    truncated = replay_model(config, catalog, "synthetic.h5", "linear", cutoff)
    pd.testing.assert_frame_equal(full.loc[:cutoff], truncated, atol=1e-12, rtol=0)
    world["opens"].loc[world["opens"].index > cutoff] *= 17
    world["membership"].loc[world["membership"].index > cutoff, "A"] = False
    changed = factors.copy()
    changed.loc[changed.index > cutoff] *= -100
    save_factors(changed)
    perturbed = replay_model(config, catalog, "synthetic.h5", "linear", dates[-1])
    pd.testing.assert_frame_equal(full.loc[:cutoff], perturbed.loc[:cutoff], atol=1e-12, rtol=0)
    assert requests and all(end <= known for start, end, known in requests)
```

This test perturbs information after a cutoff inside the second OOS quarter, so the replay must retrain at both January and April boundaries. It tests the full ML dependency chain while using the real validation accounting engine. Its synthetic NumPy fit is intentional test injection, not a fifth production backend.

### File: tests/ml_factor_mining/test_accounting.py

```python
import numpy as np
import pandas as pd
import pytest
from factor_common.backtest import run_backtest
from ML_factor_mining.config import Config
from ML_factor_mining.scoring import backtest_profile


def test_three_day_holding_fees_slippage_and_funding_reconcile():
    dates = pd.date_range("2025-01-01", periods=5, name="date")
    symbols = list("ABCDE")
    values = pd.DataFrame(np.nan, index=dates, columns=symbols)
    values.loc[dates[0]] = np.arange(5)
    opens = pd.DataFrame(100.0, index=dates, columns=symbols)
    opens.loc[dates[-1], "E"] = 110.0
    opens.loc[dates[-1], "A"] = 90.0
    events = pd.DataFrame({"funding_time": [pd.Timestamp("2025-01-03 08:00", tz="UTC")],
                           "instrument": ["E"], "funding_rate": [0.001], "mark_price": [100.0]})
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "instrument"])
    quality = pd.DataFrame({"has_placeholder_kline": False, "funding_coverage_status": "complete"}, index=index)
    config = Config(anchor_date="2025-01-02", fee_rate=0.0005, slippage=0.001)
    result = run_backtest(values, opens, events, quality, backtest_profile(config),
                          signal_start=dates[0], signal_end=dates[0])
    net = result["scenarios"]["all_costs"]
    assert net["status"] == "complete"
    assert net["diagnostics"]["first_order_date"] == "2025-01-02"
    assert net["diagnostics"]["liquidation_date"] == "2025-01-05"
    # Fixed quantities: long 0.005 E and short 0.005 A. Price PnL = 0.10.
    # Entry+exit traded notional = 2.0; fee+slippage = 0.003; funding paid = 0.0005.
    assert net["ledger"]["equity"].iloc[-1] == pytest.approx(1.0965)
    assert net["ledger"]["fee"].sum() == pytest.approx(0.001)
    assert net["ledger"]["slippage"].sum() == pytest.approx(0.002)
    assert net["positions"].iloc[-1].abs().sum() == 0
```

Run the focused suite:

```bash
./.venv/bin/python -m pytest tests/ml_factor_mining -q
./.venv/bin/python -m pytest tests/factor_common/test_labels.py tests/factor_common/test_grouping.py tests/factor_common/test_backtest.py tests/factor_common/test_metrics.py tests/factor_common/test_manager.py -q
./.venv/bin/python -m compileall -q ML_factor_mining
```

Expected: all new tests and selected shared-framework tests pass. A pre-existing failure must be reproduced without this implementation and reported separately, with its exact test name. Do not modify unrelated GP/portfolio tests to make this task green.

Local-data acceptance commands, after installing dependencies:

```bash
./.venv/bin/python -m ML_factor_mining.cli run --run-name ml_ridge_acceptance_v1 --models ridge --end 2025-03-31
./.venv/bin/python -m ML_factor_mining.cli verify --run-name ml_ridge_acceptance_v1 --model ridge --cutoffs 2025-02-15 2025-03-15
./.venv/bin/python -m ML_factor_mining.cli run --run-name ml_h3_four_models_acceptance_v1 --models lightgbm xgboost ridge linear --end 2026-12-31
```

The final command stops at available completed data. It must not fabricate the remainder of 2026. Add a cutoff verification run for every selected backend before calling that backend cutoff-verified. Keep default single-thread settings and the recorded dependency environment for replay.

Acceptance artifacts to inspect:

```text
outputs/ml_factor_mining/ml_h3_four_models_acceptance_v1/
  manifest.json
  state.json
  inputs/
    crypto_quant.h5
    factors/<factor>.parquet
    factors/<factor>.meta.json
  lightgbm/
    factor_oos.parquet
    predictions_oos.parquet
    coverage.parquet
    quarterly_metrics.parquet
    daily_ledger.parquet
    daily_ic.parquet
    orders.parquet
    positions.parquet
    funding.parquet
    funding_coverage.parquet
    group_forward_return_diagnostics.parquet
    evaluation.json
    quarters/2025Q1/
      model.json
      model.txt
      candidates.parquet
      audit.json
      predictions.parquet
      coverage.parquet
  xgboost/
  ridge/
  linear/
reports/ml_factor_mining/ml_h3_four_models_acceptance_v1/
  lightgbm/quarterly.html
  xgboost/quarterly.html
  ridge/quarterly.html
  linear/quarterly.html
```

Native model payloads differ: LightGBM has `model.txt`, XGBoost has `model.ubj`, Ridge/OLS coefficients live in `model.json`. `future_leak_audit.json` appears only after that model's explicit `verify` succeeds.

Checkpoint commit: `test: certify quarterly ML causality and accounting`.

## 4. Failure policy and implementation review checklist

| Condition | Required behavior |
|---|---|
| Wrong factor schema, duplicate keys, duplicate IDs | Abort with the artifact ID/path |
| Requested backend unavailable | Fail before training; installation instructions in error |
| Source changes during snapshot | Abort snapshot; preserve incomplete run directory |
| Input contains an external ML output with no source provenance | Reject admission; never recursively ingest it |
| Insufficient fit data or no eligible feature columns | Abort that run with quarter and reason; no silent fallback |
| Candidate has insufficient validation coverage or undefined net Sharpe | Save rejection; exclude from score percentiles |
| Every candidate rejected | Abort selection; do not switch to gross returns or MSE-only selection |
| Some prediction-time features missing | Frozen zero-plus-indicator rule |
| All admitted features missing for a row | Omit prediction; retain zero-coverage audit row |
| Fewer than five predictions on a trade date | Common engine marks signal insufficient; never silently reduce group count |
| Missing funding/price evidence | Preserve incomplete accounting status and certified prefix |
| Latest label unavailable | Keep prediction; exclude from label statistics/certified-tail evaluation |
| Existing run/quarter directory | Refuse overwrite |
| Snapshot/code hash mismatch during verification | Abort verification |
| Prediction prefix, missing mask, or axes differ in cutoff replay | Fail certification and report the difference |

Before merging implementation:

- [ ] Every output quarter uses a model trained strictly before its first signal date.
- [ ] All fit labels exit before validation; all refit/validation labels exit before retraining.
- [ ] Feature-admission denominators use eligible history, including rows with missing future labels.
- [ ] The same selected factor order and missing indicators reach fit, validation, refit, and inference.
- [ ] Changing `holding_days` changes both label exits and the execution profile.
- [ ] Five-group direction is positive and gross exposure is one: +0.5 top, -0.5 bottom.
- [ ] Validation Sharpe uses the certified all-costs daily ledger and signed metrics.
- [ ] Candidate banks and weights are frozen before each OOS test; test performance is never a search input.
- [ ] All four model adapters pass fitting/refit/persistence tests.
- [ ] Predictions are never regenerated by later quarters; concatenation rejects duplicate keys.
- [ ] Quarterly returns reconcile to the continuous ledger and full-period compound return.
- [ ] The latest-quarter status, execution tail, and input/label/prediction coverage are explicit.
- [ ] Static scan finds no banned patterns in new ML code. The existing `factor_common/labels.py` remains the sole negative-shift implementation, used only for matured supervision/evaluation.
- [ ] Synthetic full/truncated/perturbed replays pass; local-data replay reports `max_abs_diff <= 1e-10` with identical missing masks/common historical axes.
- [ ] Audit scope distinguishes ML-chain verification from upstream formula discovery and cached upstream validation.
- [ ] At least one real local-data end-to-end run completes, or the precise data/dependency blocker is reported.
- [ ] HTML files are opened and checked for dates, readable tables, working NAV-report links, and correct status text.
- [ ] Handoff lists changed files, executed commands, output paths, validation evidence, and remaining limitations.

## 5. Implementation sequencing and sources

Implement Tasks 1–9 in order. Use short test/implementation/verification steps within each task. Commit explicitly named task files only; do not stage the whole dirty worktree. The document itself does not authorize starting implementation, changing user factor caches, pushing commits, or publishing results.

Reference implementation code is intentionally localized to the new package. Reuse the current `factor_common` contracts; do not copy the reference scripts' feature engineering, global scaling, absolute-IC scoring, or future-quarter validation placement. No new factor module under `factor_analyse/factor_mining/` is needed for the first version because the manager's DataFrame path is already available. If an importable factor wrapper is later requested, separately implement the loader contract and its own replay semantics rather than loading a future-trained artifact inside `calc_factor`.

Primary API references used while designing this plan:

- [LightGBM training: custom evaluation, early stopping callbacks, and custom-loss derivatives](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html).
- [LightGBM objectives and model parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
- [XGBoost model parameters and regression objective](https://xgboost.readthedocs.io/en/stable/parameter.html).
- [scikit-learn Ridge objective and regularization strength](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).
- [scikit-learn ordinary least-squares regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

The research choices, coverage thresholds, candidate grids, and composite weights are this project's proposed implementation defaults, not performance guarantees or recommendations derived from those library documents.

## 6. Document-level verification performed on 2026-09-14

The author extracted the complete file blocks into an isolated temporary directory and checked them against the current repository without creating the production ML package.

- 24 complete file blocks were extracted; all 21 Python files passed AST syntax checking and the JSON configuration parsed successfully.
- 27 tests passed in the temporary copy, excluding `test_models.py` because scikit-learn, LightGBM, and XGBoost are not installed in the current environment.
- Passing tests include calendar/purge boundaries, membership-first ranks, missing-value admission, signed composite selection, failure-artifact preservation, FactorManager parameter compatibility, real fee/slippage/funding accounting, quarter compounding, and full retraining under future-data perturbation.
- The new ML source files produced zero findings under the repository's static future-leak scanner. This is separate from the dynamic synthetic tests and from unexecuted real-data certification.
- The review found and corrected a manager integration mismatch: `FactorManager.evaluate` accepts a narrower override set than `resolve_profile`. The explicit `manager_params` adapter and its regression test encode the correct contract.
- No placeholder markers or duplicate file-block paths remain.

Not executed while authoring this document: dependency installation, the five backend-specific model/refit/persistence test cases, full real-H5 training, real-data cutoff certification, or generated-report visual inspection. Those remain implementation acceptance steps, not claims of completed work. Only this implementation document was added to the repository; reference test execution occurred in the temporary directory.

For implementation handoff, the plan can be executed inline using `executing-plans`. Agent delegation is an alternative only when explicitly authorized in the implementation session. Keep this document's checkboxes as the progress record, and deliver the actual commands/results when implementation is complete.
