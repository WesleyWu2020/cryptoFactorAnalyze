# Crypto Market-Cap Top50 Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and maintain `data/crypto_quant.h5`, containing a point-in-time monthly CMC market-cap Top50 universe restricted to Binance USDT perpetuals, daily futures data, raw funding events, and factor-ready research views.

**Architecture:** Add a focused `data/crypto_quant/` package with injectable HTTP adapters, deterministic mapping/universe/panel builders, and a single-writer staged HDF5 store. The CLI supports initial backfill, idempotent daily updates, derived-table rebuilds, validation, and inspection; rolling factors load candidate history before filtering completed values to the effective Top50 universe.

**Tech Stack:** Python 3.11+ / pandas / requests / PyTables (`tables`) / pytest / standard-library `fcntl`, `pathlib`, `dataclasses`, and `argparse`.

**Spec:** `docs/superpowers/specs/2026-09-03-crypto-market-cap-top50-data-pipeline-design.md`

---

## Delivery boundaries

- Market-cap membership starts with the 2024-01-01 CMC snapshot and first becomes effective on 2024-01-02.
- Binance raw history starts up to 180 days before a candidate's first required universe date, so T-1 eligibility and rolling factors have support history.
- Existing CSV/liquidity scripts remain untouched and reproducible.
- No factor formula or `factor_miner` input/output contract changes in this plan.
- Only one durable data artifact is produced: `data/crypto_quant.h5`. The lock and staging HDF5 are runtime files and are removed or replaced after a successful run.
- Every network-dependent test uses fakes. Live API calls appear only in explicit acceptance commands.

## File structure

Create:

```text
data/
├── crypto_quant/
│   ├── __init__.py              # public package exports
│   ├── config.py                # paths, dates, endpoints, retry/update policy
│   ├── http.py                  # bounded JSON GET retry behavior
│   ├── cmc.py                   # CMC100 pagination and normalization
│   ├── binance.py               # exchange info, daily kline, funding adapters
│   ├── mapping.py               # stablecoin/wrapper rules and CMC-ID mapping
│   ├── universe.py              # monthly point-in-time Top50 construction
│   ├── panel.py                 # funding aggregation and daily panel construction
│   ├── schemas.py               # table columns, keys, and dtype normalization
│   ├── store.py                 # HDF5 upsert, metadata, lock, stage, publication
│   ├── validation.py            # source, universe, panel, and store invariants
│   ├── pipeline.py              # backfill/update/rebuild orchestration
│   └── reader.py                # factor-safe history and membership adapter
├── crypto_quant_rules.json      # versioned exclusions and mapping overrides
└── update_crypto_quant.py       # thin CLI entry point
docs/
└── crypto_quant_data.md         # operating and recovery runbook
tests/
├── fixtures/
│   ├── cmc100_page.json
│   ├── binance_exchange_info.json
│   ├── binance_klines.json
│   └── binance_funding.json
├── test_crypto_quant_http.py
├── test_crypto_quant_cmc.py
├── test_crypto_quant_binance.py
├── test_crypto_quant_mapping.py
├── test_crypto_quant_universe.py
├── test_crypto_quant_panel.py
├── test_crypto_quant_store.py
├── test_crypto_quant_validation.py
├── test_crypto_quant_pipeline.py
├── test_crypto_quant_reader.py
└── test_update_crypto_quant_cli.py
```

Modify:

- `requirements.txt`: declare direct `requests`, `tables`, and `pytest` dependencies.
- `README.md`: link the new data-pipeline runbook without replacing the legacy workflow.

## Stable cross-task interfaces

Later tasks must use these names and signatures exactly:

```python
PipelineConfig.default(repo_root: Path | None = None) -> PipelineConfig
JsonHttpClient.get_json(url: str, params: Mapping[str, object] | None = None) -> object
fetch_cmc_history(client, start: date, end: date, on_page=None) -> tuple[pd.DataFrame, pd.DataFrame]
fetch_exchange_info(client) -> pd.DataFrame
fetch_daily_klines(client, symbol: str, start: date, end: date) -> pd.DataFrame
fetch_funding_events(client, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame
build_contract_mappings(constituents, exchange_info, rules, historical_probe) -> tuple[pd.DataFrame, pd.DataFrame]
build_monthly_universe(constituents, mappings, klines, start: date, end: date, top_n: int = 50, current_trading_symbols=None, current_observed_date=None) -> pd.DataFrame
aggregate_funding_daily(events: pd.DataFrame) -> pd.DataFrame
build_research_panel(universe, klines, funding, panel_end: date, funding_complete_through: date) -> pd.DataFrame
validate_store(path: Path) -> ValidationReport
CryptoQuantPipeline.backfill(as_of: datetime) -> RunSummary
CryptoQuantPipeline.update(as_of: datetime) -> RunSummary
CryptoQuantPipeline.rebuild_derived(as_of: datetime) -> RunSummary
load_market_history(path, start, end, lookback_days=180) -> pd.DataFrame
load_daily_universe(path, start, end, require_complete=True) -> dict[pd.Timestamp, set[str]]
filter_factor_output(factors, universe_by_date) -> pd.DataFrame
```

All timestamps entering storage are UTC-naive pandas timestamps after conversion from timezone-aware source timestamps. All logical dates are normalized UTC dates.

## Preflight

- [ ] **Step 1: Create the repository-local environment if absent**

Run:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

Expected: `./.venv/bin/python` exists and dependency installation exits 0.

- [ ] **Step 2: Record the baseline test state**

Run: `./.venv/bin/python -m pytest tests/ -q`

Expected: existing tests pass. If an existing failure occurs, record its exact node ID and traceback before changing code, then ensure the final run has no additional failures.

---

### Task 1: Package scaffold, dependencies, and immutable configuration

**Files:**

- Create: `data/crypto_quant/__init__.py`
- Create: `data/crypto_quant/config.py`
- Modify: `requirements.txt`
- Test: `tests/test_crypto_quant_config.py`

- [ ] **Step 1: Write the failing configuration test**

```python
from datetime import date
from pathlib import Path

from data.crypto_quant.config import PipelineConfig


def test_default_config_uses_repo_data_directory(tmp_path: Path):
    cfg = PipelineConfig.default(tmp_path)
    assert cfg.store_path == tmp_path / "data" / "crypto_quant.h5"
    assert cfg.staging_path == tmp_path / "data" / ".crypto_quant.staging.h5"
    assert cfg.lock_path == tmp_path / "data" / ".crypto_quant.lock"
    assert cfg.universe_start == date(2024, 1, 1)
    assert cfg.warmup_days == 180
    assert cfg.top_n == 50
    assert cfg.cmc_overlap_days == 10
    assert cfg.binance_overlap_days == 7
```

- [ ] **Step 2: Run the test and confirm the missing package failure**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_config.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'data.crypto_quant'`.

- [ ] **Step 3: Declare dependencies**

Append to `requirements.txt`:

```text

# Crypto Top50 HDF5 pipeline
requests>=2.31.0,<3.0
tables>=3.9.0,<4.0
pytest>=8.0.0,<9.0
```

- [ ] **Step 4: Implement the package and configuration**

`data/crypto_quant/__init__.py`:

```python
"""Point-in-time crypto market-cap Top50 data pipeline."""

from .config import PipelineConfig

__all__ = ["PipelineConfig"]
```

`data/crypto_quant/config.py`:

```python
"""Immutable runtime configuration for the crypto-quant data pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    repo_root: Path
    store_path: Path
    staging_path: Path
    lock_path: Path
    rules_path: Path
    universe_start: date = date(2024, 1, 1)
    warmup_days: int = 180
    top_n: int = 50
    cmc_page_days: int = 10
    cmc_overlap_days: int = 10
    binance_overlap_days: int = 7
    request_timeout_seconds: float = 20.0
    max_attempts: int = 8
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    cmc_url: str = "https://pro-api.coinmarketcap.com/public-api/v3/index/cmc100-historical"
    exchange_info_url: str = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    klines_url: str = "https://fapi.binance.com/fapi/v1/klines"
    funding_url: str = "https://fapi.binance.com/fapi/v1/fundingRate"

    @classmethod
    def default(cls, repo_root: Path | None = None) -> "PipelineConfig":
        root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
        data_dir = root / "data"
        return cls(
            repo_root=root,
            store_path=data_dir / "crypto_quant.h5",
            staging_path=data_dir / ".crypto_quant.staging.h5",
            lock_path=data_dir / ".crypto_quant.lock",
            rules_path=data_dir / "crypto_quant_rules.json",
        )
```

- [ ] **Step 5: Run the focused test**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt data/crypto_quant/__init__.py data/crypto_quant/config.py tests/test_crypto_quant_config.py
git commit -m "feat(data): scaffold crypto quant pipeline"
```

---

### Task 2: Bounded HTTP retry client

**Files:**

- Create: `data/crypto_quant/http.py`
- Create: `tests/test_crypto_quant_http.py`

- [ ] **Step 1: Write tests for success, throttling, server errors, and hard failures**

Use small `FakeResponse` and `FakeSession` classes in the test. Cover these exact cases:

```python
def test_retries_429_using_retry_after():
    session = FakeSession([
        FakeResponse(429, {}, headers={"Retry-After": "2"}),
        FakeResponse(200, {"data": [1]}),
    ])
    sleeps = []
    client = JsonHttpClient(session, timeout=3, max_attempts=3, sleep=sleeps.append, random_fn=lambda: 0)
    assert client.get_json("https://example.test") == {"data": [1]}
    assert sleeps == [2.0]


def test_retries_500_then_succeeds():
    session = FakeSession([FakeResponse(500, {}), FakeResponse(200, {"ok": True})])
    client = JsonHttpClient(session, timeout=3, max_attempts=2, sleep=lambda _: None, random_fn=lambda: 0)
    assert client.get_json("https://example.test") == {"ok": True}


def test_does_not_retry_400():
    client = JsonHttpClient(FakeSession([FakeResponse(400, {"error": "bad"})]), max_attempts=3)
    with pytest.raises(HttpRequestError, match="HTTP 400"):
        client.get_json("https://example.test")


def test_rejects_non_json_success():
    response = FakeResponse(200, ValueError("invalid json"))
    client = JsonHttpClient(FakeSession([response]), max_attempts=1)
    with pytest.raises(HttpRequestError, match="invalid JSON"):
        client.get_json("https://example.test")
```

- [ ] **Step 2: Run the tests and confirm import failure**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_http.py -v`

Expected: FAIL because `data.crypto_quant.http` does not exist.

- [ ] **Step 3: Implement `JsonHttpClient`**

The implementation must:

- Accept an injected `requests.Session`, sleep function, and random function.
- Retry connection exceptions, 429, and 5xx only.
- Prefer numeric `Retry-After`; otherwise use `min(max_backoff, base_backoff * 2**attempt) + random_fn()`.
- raise `HttpRequestError` after the final attempt or immediately on non-retryable 4xx.
- call `response.json()` only after a 2xx response and wrap parse failures.

Public constructor: `JsonHttpClient(session: requests.Session | None = None, *, timeout: float = 20.0, max_attempts: int = 8, base_backoff: float = 1.0, max_backoff: float = 60.0, sleep: Callable[[float], None] = time.sleep, random_fn: Callable[[], float] = random.random)`.

Public request method: `get_json(self, url: str, params: Mapping[str, object] | None = None) -> object`.

The loop uses zero-based attempts and raises with URL, status, and attempt count in the message. Do not log query secrets; these endpoints require no credentials.

- [ ] **Step 4: Run the focused tests**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_http.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add data/crypto_quant/http.py tests/test_crypto_quant_http.py
git commit -m "feat(data): add bounded API retry client"
```

---

### Task 3: CMC100 historical adapter and restartable page callback

**Files:**

- Create: `data/crypto_quant/cmc.py`
- Create: `tests/fixtures/cmc100_page.json`
- Create: `tests/test_crypto_quant_cmc.py`

- [ ] **Step 1: Add a minimal official-shape fixture**

The fixture contains two dates, each with at least three constituent objects using the official fields `id`, `name`, `symbol`, `weight`, `update_time`, and `value`. Use 2024-01-01 and 2024-01-02 and distinct weights so ordering is testable.

- [ ] **Step 2: Write failing normalization and pagination tests**

Cover:

```python
def test_iter_cmc_windows_is_inclusive_and_ten_days():
    assert list(iter_cmc_windows(date(2024, 1, 1), date(2024, 1, 23))) == [
        (date(2024, 1, 1), date(2024, 1, 10)),
        (date(2024, 1, 11), date(2024, 1, 20)),
        (date(2024, 1, 21), date(2024, 1, 23)),
    ]


def test_normalize_cmc_payload_uses_update_date_and_cmc_id(fixture_payload):
    daily, members = normalize_cmc_payload(fixture_payload, fetched_at=pd.Timestamp("2026-09-03T00:20:00"))
    assert list(daily.columns) == ["date", "index_value", "source_update_time", "fetched_at_utc"]
    assert {"date", "cmc_id", "symbol", "name", "weight"} <= set(members.columns)
    assert not members.duplicated(["date", "cmc_id"]).any()


def test_fetch_history_calls_page_callback_after_each_valid_page(fake_client):
    calls = []
    daily, members = fetch_cmc_history(
        fake_client,
        date(2024, 1, 1),
        date(2024, 1, 2),
        on_page=lambda d, m, end: calls.append((len(d), len(m), end)),
    )
    assert calls == [(2, 6, date(2024, 1, 2))]
    assert len(daily) == 2
```

Also test malformed status, missing `constituents`, duplicate `date + cmc_id`, and the empty 2023 interval.

- [ ] **Step 3: Run tests and confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_cmc.py -v`

Expected: FAIL because the CMC adapter is missing.

- [ ] **Step 4: Implement the adapter**

Implementation rules:

- `iter_cmc_windows` emits inclusive windows of at most 10 calendar days.
- Every request sends `time_start`, `time_end`, `count=10`, and `interval=daily`.
- `status.error_code` values other than `0` or `None` raise `CmcSchemaError`.
- `update_time` is parsed as UTC and normalized into `date`; `source_update_time` keeps the full UTC timestamp.
- `id` is normalized to integer and `weight`/`value` to float.
- Rows are sorted and deduplicated only when duplicate values are identical; conflicting duplicates raise.
- `on_page` is called only after page validation, enabling the store task to checkpoint each successful page.

Public definitions are `CmcSchemaError(ValueError)`, `iter_cmc_windows(start: date, end: date, page_days: int = 10) -> Iterator[tuple[date, date]]`, `normalize_cmc_payload(payload: object, fetched_at: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]`, and `fetch_cmc_history(client: JsonHttpClient, start: date, end: date, on_page: Callable[[pd.DataFrame, pd.DataFrame, date], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]`.

- [ ] **Step 5: Run tests**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_cmc.py tests/test_crypto_quant_http.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add data/crypto_quant/cmc.py tests/fixtures/cmc100_page.json tests/test_crypto_quant_cmc.py
git commit -m "feat(data): fetch historical CMC100 constituents"
```

---

### Task 4: Binance USDT perpetual adapters

**Files:**

- Create: `data/crypto_quant/binance.py`
- Create: `tests/fixtures/binance_exchange_info.json`
- Create: `tests/fixtures/binance_klines.json`
- Create: `tests/fixtures/binance_funding.json`
- Create: `tests/test_crypto_quant_binance.py`

- [ ] **Step 1: Add fixtures in official response shapes**

- Exchange info includes `BTCUSDT` and `ETHUSDT` with `quoteAsset=USDT`, `contractType=PERPETUAL`, plus one quarterly and one USDC contract that must be excluded.
- Klines use 12-element arrays and include one completed daily candle plus one candle whose close time exceeds the requested completed-day boundary.
- Funding includes records with `fundingRate`, `fundingTime`, `markPrice`, one explicit `rateType`, and one omitted `rateType`.

- [ ] **Step 2: Write failing tests**

Required assertions:

```python
def test_exchange_info_keeps_only_usdt_perpetuals(payload):
    out = normalize_exchange_info(payload, fetched_at=pd.Timestamp("2026-09-03"))
    assert set(out["binance_symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert set(out["contract_type"]) == {"PERPETUAL"}


def test_kline_pagination_advances_by_last_open_time_plus_one_day(fake_client):
    out = fetch_daily_klines(fake_client, "BTCUSDT", date(2024, 1, 1), date(2024, 1, 3))
    assert not out.duplicated(["date", "symbol"]).any()
    assert out["date"].max() <= pd.Timestamp("2024-01-03")


def test_funding_pagination_advances_one_millisecond_and_normalizes_type(fake_client):
    out = fetch_funding_events(fake_client, "BTCUSDT", 1704067200000, 1704153600000)
    assert set(out["rate_type"]) >= {"Regular"}
    assert not out.duplicated(["funding_time", "symbol", "rate_type"]).any()


def test_historical_probe_requires_a_completed_t_minus_one_kline(fake_client):
    assert has_completed_daily_kline(fake_client, "BTCUSDT", date(2024, 1, 1)) is True
```

Also assert numeric dtypes, OHLC field positions, `trade_count`, taker-buy fields, `limit=1500` for klines, and `limit=1000` for funding.

- [ ] **Step 3: Run tests and confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_binance.py -v`

Expected: FAIL because the Binance adapter is missing.

- [ ] **Step 4: Implement exchange info normalization**

`fetch_exchange_info(client)` calls `/fapi/v1/exchangeInfo`, filters exact `quoteAsset == "USDT"` and `contractType == "PERPETUAL"`, and emits:

```text
binance_symbol, base_asset, quote_asset, contract_type,
onboard_date, status, fetched_at_utc
```

Convert Binance millisecond values with `pd.to_datetime(value, unit="ms", utc=True).tz_localize(None)`.

- [ ] **Step 5: Implement kline pagination**

`fetch_daily_klines` sends `interval=1d`, `limit=1500`, inclusive start/end milliseconds, advances to `last_open_time + 86_400_000`, and raises if the cursor does not advance. Normalize the exact 12 fields to the `/klines_daily` schema and discard any row whose `close_time` is after the requested completed-day end.

- [ ] **Step 6: Implement funding pagination and probe**

`fetch_funding_events` advances by `last_funding_time + 1`, stops on an empty or short page, normalizes a missing `rateType` to `Regular`, and preserves an absent `markPrice` as `NaN`. `has_completed_daily_kline(client, symbol, decision_date)` requests only `decision_date - 1 day` and returns true only for one completed candle on that date.

- [ ] **Step 7: Run focused tests**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_binance.py tests/test_crypto_quant_http.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add data/crypto_quant/binance.py tests/fixtures/binance_exchange_info.json tests/fixtures/binance_klines.json tests/fixtures/binance_funding.json tests/test_crypto_quant_binance.py
git commit -m "feat(data): fetch Binance futures market data"
```

---

### Task 5: Stablecoin rules and CMC-ID contract mapping

**Files:**

- Create: `data/crypto_quant_rules.json`
- Create: `data/crypto_quant/mapping.py`
- Create: `tests/test_crypto_quant_mapping.py`

- [ ] **Step 1: Add versioned rules**

Use this initial tracked JSON:

```json
{
  "version": "2026-09-03-v1",
  "stablecoin_symbols": [
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDP", "BUSD",
    "USDD", "USDE", "USD0", "USD1", "PYUSD", "RLUSD", "GUSD",
    "LUSD", "FRAX", "SUSD", "CRVUSD", "GHO", "EURC", "EURS"
  ],
  "wrapper_symbols": [
    "WBTC", "WETH", "STETH", "WSTETH", "CBETH", "CBBTC", "RETH"
  ],
  "overrides": [],
  "blocked_cmc_ids": []
}
```

Rules are defensive additions to CMC's methodology. Matching is case-insensitive and exact; no substring rule may exclude unrelated symbols.

- [ ] **Step 2: Write failing mapping tests**

Cover exact-symbol mapping keyed by CMC ID, exclusion, override precedence, same-symbol CMC-ID collision, historical probe success, and unresolved issue output:

```python
def test_mapping_is_keyed_by_cmc_id_and_excludes_stablecoins(rules):
    mappings, issues = build_contract_mappings(constituents, exchange_info, rules, lambda *_: False)
    assert 1 in set(mappings["cmc_id"])
    assert "USDT" not in set(mappings["cmc_symbol"])
    assert mappings.set_index("cmc_id").loc[1, "binance_symbol"] == "BTCUSDT"


def test_override_wins_and_records_mapping_source(rules_with_override):
    mappings, _ = build_contract_mappings(constituents, exchange_info, rules_with_override, lambda *_: False)
    row = mappings.loc[mappings["cmc_id"] == 999].iloc[0]
    assert row["binance_symbol"] == "1000SHIBUSDT"
    assert row["mapping_source"] == "explicit_override"


def test_historical_probe_confirms_contract_absent_from_current_exchange_info(rules):
    mappings, issues = build_contract_mappings(old_constituent, exchange_info, rules, lambda symbol, _: symbol == "OLDUSDT")
    assert mappings.iloc[0]["mapping_source"] == "historical_kline_probe"
    assert issues.empty
```

- [ ] **Step 3: Run tests and confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_mapping.py -v`

Expected: FAIL because mapping code is missing.

- [ ] **Step 4: Implement rules and mapping**

Define:

```python
@dataclass(frozen=True)
class MappingRules:
    version: str
    stablecoin_symbols: frozenset[str]
    wrapper_symbols: frozenset[str]
    overrides: Sequence[Mapping[str, object]]
    blocked_cmc_ids: frozenset[int]
```

Public functions are `load_mapping_rules(path: Path) -> MappingRules`, `is_excluded_asset(cmc_id: int, symbol: str, rules: MappingRules) -> bool`, and `build_contract_mappings(constituents: pd.DataFrame, exchange_info: pd.DataFrame, rules: MappingRules, historical_probe: Callable[[str, date], bool]) -> tuple[pd.DataFrame, pd.DataFrame]`.

Mapping order is exact and deterministic:

1. Exclude blocked IDs, stablecoin symbols, and wrapper symbols.
2. Apply a validity-matching explicit override by `cmc_id`.
3. Match exactly one current Binance `base_asset` to the CMC symbol.
4. If absent from current exchange info, probe exact `<CMC_SYMBOL>USDT` on the first-observed decision date.
5. Emit unresolved or ambiguous rows to `issues`; never fuzzy-match names or symbols.

Mappings emit all `/futures_contracts` columns. A current match uses Binance `onboard_date` as `valid_from`; a confirmed historical probe uses its first successful support date. `valid_to` remains `NaT` unless supplied by an override.

- [ ] **Step 5: Run tests**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_mapping.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add data/crypto_quant_rules.json data/crypto_quant/mapping.py tests/test_crypto_quant_mapping.py
git commit -m "feat(data): map CMC assets to futures contracts"
```

---

### Task 6: Monthly point-in-time Top50 universe

**Files:**

- Create: `data/crypto_quant/universe.py`
- Create: `tests/test_crypto_quant_universe.py`

- [ ] **Step 1: Write synthetic failing tests**

Generate 60 constituents with descending weights and 55 valid mappings. Make one of CMC ranks 1-50 ineligible and verify rank 51 enters, proving intersection occurs before selection.

Required assertions:

```python
def test_intersects_before_selecting_top50(synthetic_inputs):
    out = build_monthly_universe(*synthetic_inputs, date(2024, 1, 1), date(2024, 1, 31))
    january = out[out["decision_date"] == pd.Timestamp("2024-01-01")]
    assert len(january) == 50
    assert january["binance_symbol"].nunique() == 50
    assert january["market_cap_rank"].tolist() == list(range(1, 51))
    assert pd.Timestamp("2024-01-02") == january["effective_date"].iloc[0]


def test_requires_t_minus_one_kline(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    missing_symbol = mappings.iloc[0]["binance_symbol"]
    klines = klines[~((klines["symbol"] == missing_symbol) & (klines["date"] == pd.Timestamp("2023-12-31")))]
    out = build_monthly_universe(constituents, mappings, klines, date(2024, 1, 1), date(2024, 1, 31))
    assert missing_symbol not in set(out["binance_symbol"])


def test_fails_closed_below_50(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    with pytest.raises(UniverseBuildError, match="eligible contracts: 49"):
        build_monthly_universe(constituents, mappings.head(49), klines, date(2024, 1, 1), date(2024, 1, 31))
```

Also test deterministic CMC-ID tie-breaking, `effective_end_date = next effective_date - 1 day` for closed intervals, and a null `effective_end_date` for the latest accepted membership.

Add two status-timing tests: a contract currently marked `BREAK` remains eligible for a historical month when its T-1 kline exists, while a real current-day decision excludes a symbol absent from `current_trading_symbols` when `current_observed_date` equals that decision date.

- [ ] **Step 2: Run tests and confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_universe.py -v`

Expected: FAIL because `universe.py` is missing.

- [ ] **Step 3: Implement the universe builder**

Define `UniverseBuildError(ValueError)` and `build_monthly_universe(constituents: pd.DataFrame, mappings: pd.DataFrame, klines: pd.DataFrame, start: date, end: date, top_n: int = 50, current_trading_symbols: set[str] | None = None, current_observed_date: date | None = None) -> pd.DataFrame`.

Implementation order:

- Select snapshots whose normalized date is the first calendar day of each month.
- Join mappings by `cmc_id` and mapping validity range.
- Inner-join completed kline keys at `decision_date - 1 day`.
- Never filter historical decisions with the currently observed contract status. Apply `current_trading_symbols` only when `decision_date == current_observed_date`; replay dates receive both arguments as null.
- Sort by `weight DESC, cmc_id ASC`, take exactly 50, and assign `market_cap_rank` after eligibility filtering.
- Raise `UniverseBuildError` before emitting any rows for a month with fewer than 50.
- Set `effective_date = decision_date + 1 day`; derive each closed `effective_end_date` from the next accepted month and leave the latest membership open with `NaT`. Store a decision even when its T+1 effective date is after the latest completed kline date; the panel builder alone caps expansion at `panel_end`.
- Return only the `/universe_monthly` schema in deterministic order.

- [ ] **Step 4: Run tests**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_universe.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/crypto_quant/universe.py tests/test_crypto_quant_universe.py
git commit -m "feat(data): build point-in-time market-cap Top50"
```

---

### Task 7: Funding aggregation and daily research panel

**Files:**

- Create: `data/crypto_quant/panel.py`
- Create: `tests/test_crypto_quant_panel.py`

- [ ] **Step 1: Write failing funding aggregation tests**

```python
def test_aggregate_funding_preserves_zero_event_semantics():
    events = pd.DataFrame({
        "funding_time": pd.to_datetime(["2024-01-02 00:00", "2024-01-02 08:00"]),
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "funding_rate": [0.0, 0.0001],
        "mark_price": [42000.0, 42100.0],
        "rate_type": ["Regular", "Regular"],
    })
    out = aggregate_funding_daily(events)
    row = out.iloc[0]
    assert row["funding_rate_sum"] == pytest.approx(0.0001)
    assert row["funding_event_count"] == 2
    assert row["funding_rate_last"] == pytest.approx(0.0001)
```

- [ ] **Step 2: Write failing panel tests**

Use a two-day universe with 50 members. Assert:

- 50 membership rows remain even when one kline is missing.
- Missing kline fields are `NaN` and `has_complete_kline` is false; no zero fill occurs.
- A complete funding query with no event produces `funding_event_count=0`, aggregate rates `NaN`, and `has_complete_funding=true`.
- An incomplete funding coverage date has `has_complete_funding=false`.
- The panel starts at 2024-01-02, not the 2024-01-01 decision date.

- [ ] **Step 3: Run tests and confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_panel.py -v`

Expected: FAIL because `panel.py` is missing.

- [ ] **Step 4: Implement aggregation and panel expansion**

Define `aggregate_funding_daily(events: pd.DataFrame) -> pd.DataFrame`, `build_research_panel(universe: pd.DataFrame, klines: pd.DataFrame, funding: pd.DataFrame, panel_end: date, funding_complete_through: date) -> pd.DataFrame`, and `last_complete_panel_date(panel: pd.DataFrame) -> pd.Timestamp | None`.

Use a natural-day expansion of each membership interval, then left joins on `date + symbol`. Funding `last` is selected by the greatest `funding_time`, not input row order. `last_complete_panel_date` returns the end of the longest contiguous prefix where every date has exactly 50 unique members and every member has `has_complete_kline=true`.

- [ ] **Step 5: Run tests**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_panel.py tests/test_crypto_quant_universe.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add data/crypto_quant/panel.py tests/test_crypto_quant_panel.py
git commit -m "feat(data): build funding-aware daily research panel"
```

---

### Task 8: Schemas, HDF5 upsert, metadata, lock, and atomic publication

**Files:**

- Create: `data/crypto_quant/schemas.py`
- Create: `data/crypto_quant/store.py`
- Create: `tests/test_crypto_quant_store.py`

- [ ] **Step 1: Define table specifications in a failing test**

Assert that the seven required tables exist in `TABLE_SPECS`, with exact primary keys:

```python
EXPECTED_KEYS = {
    "cmc100_daily": ("date",),
    "cmc100_constituents": ("date", "cmc_id"),
    "futures_contracts": ("cmc_id", "valid_from"),
    "klines_daily": ("date", "symbol"),
    "funding_events": ("funding_time", "symbol", "rate_type"),
    "universe_monthly": ("effective_date", "binance_symbol"),
    "research_panel_daily": ("date", "binance_symbol"),
}
```

Declare columns in this exact order:

```text
cmc100_daily:
  date, index_value, source_update_time, fetched_at_utc
cmc100_constituents:
  date, cmc_id, symbol, name, weight
futures_contracts:
  cmc_id, cmc_symbol, binance_symbol, base_asset, quote_asset,
  contract_type, onboard_date, status, mapping_source, valid_from, valid_to
klines_daily:
  date, symbol, open_time, close_time, open, high, low, close, volume,
  quote_volume, trade_count, taker_buy_base_volume, taker_buy_quote_volume
funding_events:
  funding_time, symbol, funding_rate, mark_price, rate_type
universe_monthly:
  decision_date, effective_date, effective_end_date, cmc_id, cmc_symbol,
  binance_symbol, market_cap_rank, cmc_weight
research_panel_daily:
  date, binance_symbol, open_time, close_time, open, high, low, close,
  volume, quote_volume, trade_count, taker_buy_base_volume,
  taker_buy_quote_volume, decision_date, universe_effective_date,
  market_cap_rank, cmc_weight_at_decision, funding_rate_sum,
  funding_rate_mean, funding_rate_last, funding_event_count,
  has_complete_kline, has_complete_funding
```

Use schema version `1`. Date query columns are `date`, `decision_date`, `effective_date`, `funding_time`, and symbol query columns are `symbol` or `binance_symbol` as applicable. Set string capacities explicitly for names, symbols, statuses, mapping sources, metadata keys, and JSON metadata values so later valid rows do not fail because the first append inferred a shorter string.

Also assert that `normalize_table(name, df)` rejects missing columns and returns rows sorted by the primary key.

- [ ] **Step 2: Write failing store tests**

Required cases and assertions:

- `test_upsert_is_idempotent_and_new_rows_win`: upsert the same key with identical business values but a later `fetched_at_utc` and assert the original row is retained; then change one business value and assert one row remains with the changed value and newer provenance.
- `test_metadata_round_trips_json_values`: write a nested row-count dictionary; assert `read_metadata()` returns the same dictionary.
- `test_second_writer_fails_without_waiting`: hold the first lock and assert a nested second acquisition raises `StoreLockedError`.
- `test_failed_staging_context_keeps_active_bytes_unchanged`: raise `RuntimeError("forced failure")` in the context and compare pre/post SHA-256 hashes.
- `test_successful_publish_replaces_active_atomically`: write a new CMC row in staging, exit successfully, and assert the active store contains it while staging no longer exists.
- `test_resume_reuses_matching_staging_fingerprint`: leave staging after a forced failure, reopen with the same fingerprint, and assert its checkpoint metadata remains.
- `test_mismatched_staging_fingerprint_requires_reset`: reopen failed staging with a different fingerprint and assert `StagingMismatchError`; then pass `reset_staging=True` and assert a clean stage is created.

For rollback, hash the active file with SHA-256 before raising inside the staging context and assert the hash is unchanged afterward.

- [ ] **Step 3: Run tests and confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_store.py -v`

Expected: FAIL because schema/store modules are missing.

- [ ] **Step 4: Implement schema normalization**

Use a frozen `TableSpec(columns, key, data_columns, min_itemsize)` dataclass. `normalize_table` must:

- Require every declared column.
- Select columns in declared order.
- Normalize date/timestamp and numeric columns.
- Reject null primary-key values.
- Drop byte-identical duplicate keys and reject conflicting duplicate keys.
- Return stable key ordering.

Use HDF keys with leading slashes only when reading `HDFStore.keys()`; all write APIs receive names without a leading slash.

- [ ] **Step 5: Implement `CryptoQuantStore`**

Public methods are `CryptoQuantStore(path: Path)`, `read(name: str, where: str | None = None) -> pd.DataFrame`, `replace(name: str, frame: pd.DataFrame) -> None`, `upsert(name: str, frame: pd.DataFrame) -> None`, `read_metadata() -> dict[str, object]`, `write_metadata(updates: Mapping[str, object]) -> None`, and `keys() -> set[str]`.

Write `format="table"`, the declared `data_columns`, and declared `min_itemsize`. Metadata is a `/_metadata` table with `key`, JSON-encoded `value`, and `updated_at_utc`; duplicate keys are replaced.

- [ ] **Step 6: Implement lock and stage lifecycle**

Define `StoreLockedError(RuntimeError)`, `StagingMismatchError(RuntimeError)`, `single_writer_lock(lock_path: Path) -> Iterator[None]`, and `staged_store(active_path: Path, staging_path: Path, run_fingerprint: str, *, reset_staging: bool = False) -> Iterator[CryptoQuantStore]`.

Use `fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)`. On a new run, copy the active file to staging or create a new store. On resume, require matching `run_fingerprint`. Keep staging on exceptions for checkpoint recovery. After caller validation succeeds, flush/close all HDF handles and publish with `os.replace(staging_path, active_path)`.

- [ ] **Step 7: Run tests**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_store.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add data/crypto_quant/schemas.py data/crypto_quant/store.py tests/test_crypto_quant_store.py
git commit -m "feat(data): add atomic HDF5 research store"
```

---

### Task 9: Store and point-in-time validation, including future-leak反证

**Files:**

- Create: `data/crypto_quant/validation.py`
- Create: `tests/test_crypto_quant_validation.py`

- [ ] **Step 1: Write failing invariant tests**

Create one valid synthetic seven-table store, then parameterize corruptions:

- Duplicate source primary key.
- Negative volume.
- `high < max(open, close)` or `low > min(open, close)`.
- A monthly universe with 49 or 51 rows.
- A stablecoin member.
- A member absent from its CMC decision snapshot.
- Missing T-1 kline.
- Membership effective before `decision_date + 1 day`.
- Funding timestamps not monotonic after key sorting.
- Panel date with other than 50 memberships.

Each corruption must produce a `ValidationIssue(level="error", code="<specific invariant name>")` with the affected date or key in `detail`; use stable codes such as `duplicate_key`, `invalid_ohlc`, `universe_size`, `excluded_asset`, `missing_t_minus_one`, and `early_effective_date`.

- [ ] **Step 2: Write the dynamic cutoff test**

Build synthetic January and February source data, then append March source rows that radically alter symbols and weights. Run derived builders once on all data and once with every source truncated at `cutoff = 2024-02-15`. Compare rows at or before cutoff:

```python
pd.testing.assert_frame_equal(
    full_universe[full_universe["effective_date"] <= cutoff].reset_index(drop=True),
    cut_universe.reset_index(drop=True),
    check_exact=False,
    atol=1e-12,
    rtol=0,
)
pd.testing.assert_frame_equal(
    full_panel[full_panel["date"] <= cutoff].reset_index(drop=True),
    cut_panel.reset_index(drop=True),
    check_exact=False,
    atol=1e-12,
    rtol=0,
)
```

Calculate and print `max_abs_diff`; expected value is `0.0` for membership/ranks and at most `1e-12` for floats.

- [ ] **Step 3: Run tests and confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_validation.py -v`

Expected: FAIL because validation code is missing.

- [ ] **Step 4: Implement validation report types and checks**

```python
@dataclass(frozen=True)
class ValidationIssue:
    level: Literal["error", "warning"]
    code: str
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    issues: Sequence[ValidationIssue]

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def raise_for_errors(self) -> None:
        errors = [issue for issue in self.issues if issue.level == "error"]
        if errors:
            raise StoreValidationError("; ".join(f"{x.code}: {x.detail}" for x in errors))
```

Define `validate_frames(frames: Mapping[str, pd.DataFrame], metadata: Mapping[str, object]) -> ValidationReport` and `validate_store(path: Path) -> ValidationReport` after the report types.

Incomplete kline/funding flags are not themselves storage errors. They prevent `last_complete_panel_date` from advancing and produce warnings containing the first incomplete date.

- [ ] **Step 5: Run tests**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_validation.py tests/test_crypto_quant_universe.py tests/test_crypto_quant_panel.py -v`

Expected: PASS, including the dynamic cutoff comparison.

- [ ] **Step 6: Run the static future-pattern scan**

Run:

```bash
rg -n "shift\(-[1-9]|center\s*=\s*True|\.bfill\(|direction\s*=\s*['\"]forward" data/crypto_quant
```

Expected: no matches.

- [ ] **Step 7: Commit**

```bash
git add data/crypto_quant/validation.py tests/test_crypto_quant_validation.py
git commit -m "test(data): enforce Top50 point-in-time invariants"
```

---

### Task 10: Backfill and daily incremental orchestration

**Files:**

- Create: `data/crypto_quant/pipeline.py`
- Create: `tests/test_crypto_quant_pipeline.py`

- [ ] **Step 1: Write fake-source end-to-end tests**

Inject fake CMC and Binance adapters; no test may patch global `requests.get`. Implement these cases with explicit assertions:

- `test_backfill_writes_all_seven_tables_and_metadata`: assert the required key set and every required metadata key.
- `test_second_identical_update_is_idempotent`: snapshot all seven research tables after run one, run again at the same `as_of`, and use `pd.testing.assert_frame_equal` for every table; compare deterministic metadata while excluding run-attempt counters.
- `test_update_refetches_ten_cmc_days_and_seven_binance_days`: seed last dates 2024-02-20, update at 2024-02-25 00:20 UTC, and assert CMC starts 2024-02-11 while Binance overlap starts 2024-02-14.
- `test_new_candidate_receives_180_day_support_backfill`: add a candidate first observed 2024-07-01 and assert its requested start is 2024-01-03, capped by onboard date when later.
- `test_first_decision_uses_2023_12_31_kline_and_activates_2024_01_02`: assert the fake receives the support date and the stored universe effective date.
- `test_month_start_creates_t_plus_one_membership`: update on 2024-02-01 and assert decision/effective dates are 2024-02-01/2024-02-02.
- `test_api_failure_keeps_active_sha256_unchanged_and_staging_resumable`: force the third symbol fetch to fail, compare active hashes, and assert checkpoint metadata exists in staging.
- `test_validation_failure_never_publishes_staging`: make the fake return only 49 eligible mappings and assert `UniverseBuildError` plus an unchanged active hash.

The fake CMC adapter records requested ranges. The fake Binance adapter records symbol/date ranges and can raise on a selected request. Compare full DataFrames and metadata after duplicate updates, not only row counts.

- [ ] **Step 2: Run tests and confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_pipeline.py -v`

Expected: FAIL because pipeline orchestration is missing.

- [ ] **Step 3: Implement run summary and injectable adapters**

```python
@dataclass(frozen=True)
class RunSummary:
    mode: str
    as_of_utc: datetime
    published_path: Path
    last_complete_kline_date: date
    last_complete_panel_date: date | None
    row_counts: dict[str, int]
    warnings: Sequence[str]
```

Define `CryptoQuantPipeline(config: PipelineConfig, cmc_source: CmcSource, binance_source: BinanceSource)`, with methods `backfill(as_of: datetime, *, reset_staging: bool = False) -> RunSummary`, `update(as_of: datetime, *, reset_staging: bool = False) -> RunSummary`, and `rebuild_derived(as_of: datetime, *, reset_staging: bool = False) -> RunSummary`.

Use `Protocol` definitions for `CmcSource` and `BinanceSource` so tests supply fakes without network calls.

- [ ] **Step 4: Implement deterministic run windows**

Normalize `as_of` to UTC. For a run at time `R`:

```text
cmc_end = R UTC calendar date
kline_end = R UTC calendar date - 1 day
funding_end_ms = floor(R.timestamp() * 1000)
panel_end = kline_end
```

Backfill CMC from 2024-01-01. For every mapped candidate, backfill Binance data from `max(contract history start, first CMC observation - 180 days)`. Update starts CMC at `last_cmc_date - 9 days`, klines at `last_symbol_kline_date - 6 days`, and funding at `last_symbol_funding_time - 7 days`; deduplication makes overlap idempotent.

Pass current `status=TRADING` symbols to the universe builder only when the latest decision date equals the machine's real UTC date and `as_of` is not a historical replay. All older decisions and explicit past `--as-of` runs use T-1 klines without current-status filtering.

- [ ] **Step 5: Implement stage checkpoints and publication**

The run fingerprint contains mode, schema version, rule version, and resolved target-store path; it does not contain wall-clock `as_of`. Store `run_target_as_of` separately. A retry may keep or increase that target; a lower target raises `StagingMismatchError` unless `--reset-staging` is explicit. The CMC page callback immediately upserts validated page rows and writes `checkpoint.cmc_through`. Binance writes `checkpoint.klines.<symbol>` and `checkpoint.funding.<symbol>` only after each symbol range succeeds. Rebuild mappings and the full open-ended universe from CMC through `cmc_end`, then build the panel only through `panel_end`. Run `validate_store(staging_path).raise_for_errors()` before publication.

Metadata written before publication includes:

```text
schema_version, pipeline_version, rules_version, source_urls,
last_successful_cmc_date, last_successful_kline_date,
last_successful_funding_time, last_complete_panel_date,
table_row_counts, table_date_ranges, created_at_utc, updated_at_utc
```

- [ ] **Step 6: Run focused tests**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_pipeline.py tests/test_crypto_quant_store.py tests/test_crypto_quant_validation.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add data/crypto_quant/pipeline.py tests/test_crypto_quant_pipeline.py
git commit -m "feat(data): orchestrate backfill and incremental updates"
```

---

### Task 11: Factor-safe HDF5 reader adapter

**Files:**

- Create: `data/crypto_quant/reader.py`
- Create: `tests/test_crypto_quant_reader.py`

- [ ] **Step 1: Write failing adapter tests**

Create a store where `NEWUSDT` joins Top50 on 2024-03-02 but has raw klines from 2023-09-01. Assert:

```python
def test_market_history_includes_pre_membership_warmup(store_path):
    out = load_market_history(store_path, date(2024, 3, 2), date(2024, 3, 10), lookback_days=180)
    new = out[out["instrument"] == "NEWUSDT"]
    assert new["date"].min() == pd.Timestamp("2023-09-04")


def test_daily_universe_defaults_to_complete_panel_dates(store_path):
    universe = load_daily_universe(store_path, date(2024, 3, 2), date(2024, 3, 10))
    assert all(len(symbols) == 50 for symbols in universe.values())
    assert pd.Timestamp("2024-03-10") not in universe  # fixture marks this date incomplete


def test_filter_factor_output_keeps_only_point_in_time_members(factors, universe):
    out = filter_factor_output(factors, universe)
    assert list(out.columns) == ["date", "instrument", "factor"]
    assert out.groupby("date")["instrument"].nunique().eq(50).all()
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_reader.py -v`

Expected: FAIL because reader adapter is missing.

- [ ] **Step 3: Implement reader functions**

Define `load_market_history(path: str | Path, start: date, end: date, lookback_days: int = 180) -> pd.DataFrame`, `load_daily_universe(path: str | Path, start: date, end: date, require_complete: bool = True) -> dict[pd.Timestamp, set[str]]`, and `filter_factor_output(factors: pd.DataFrame, universe_by_date: Mapping[pd.Timestamp, set[str]]) -> pd.DataFrame`.

`load_market_history` first finds all symbols appearing in the requested membership interval, then queries `/klines_daily` from `start - lookback_days` through `end`, renames `symbol` to `instrument`, and returns deterministic `date, instrument, <market columns>` order. It does not filter pre-membership rows. `filter_factor_output` requires `date`, `instrument`, and `factor`, normalizes dates, applies exact same-day membership, and does not forward-fill future memberships.

- [ ] **Step 4: Run tests**

Run: `./.venv/bin/python -m pytest tests/test_crypto_quant_reader.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/crypto_quant/reader.py tests/test_crypto_quant_reader.py
git commit -m "feat(data): expose factor-safe HDF5 readers"
```

---

### Task 12: CLI, inspection output, and maintenance runbook

**Files:**

- Create: `data/update_crypto_quant.py`
- Create: `tests/test_update_crypto_quant_cli.py`
- Create: `docs/crypto_quant_data.md`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

Patch `build_pipeline`, not HTTP functions. Implement these cases:

- `test_update_command_uses_explicit_utc_as_of`: call `main(["update", "--as-of", "2026-09-03T00:20:00Z"])` and assert the fake receives an aware UTC datetime and return code 0.
- `test_backfill_command_forwards_reset_staging`: pass `--reset-staging` and assert the fake's keyword argument is true.
- `test_validate_command_exits_zero_for_valid_store`: return `ValidationReport(())` and assert code 0.
- `test_validate_command_exits_two_for_invalid_store`: return one error issue and assert code 2.
- `test_inspect_prints_row_counts_ranges_and_last_complete_date`: seed metadata, run inspect, and assert captured output contains all three values.

The expected parse format is `--as-of 2026-09-03T00:20:00Z`; reject timezone-free timestamps with an argparse error.

- [ ] **Step 2: Run tests and confirm failure**

Run: `./.venv/bin/python -m pytest tests/test_update_crypto_quant_cli.py -v`

Expected: FAIL because the CLI is missing.

- [ ] **Step 3: Implement the thin CLI**

Supported commands:

```text
backfill [--as-of ISO-UTC] [--store PATH] [--reset-staging]
update [--as-of ISO-UTC] [--store PATH] [--reset-staging]
rebuild-derived [--as-of ISO-UTC] [--store PATH] [--reset-staging]
validate [--store PATH]
inspect [--store PATH]
```

Default `as_of` is current UTC time. Instantiate one shared `requests.Session`, `JsonHttpClient`, concrete CMC/Binance sources, and `CryptoQuantPipeline`. Print a compact summary and return exit code 0 on success, 2 on validation failure, and 1 on source/store/runtime failure. Never catch `KeyboardInterrupt`.

- [ ] **Step 4: Write the operating runbook**

`docs/crypto_quant_data.md` must include exact commands for:

```bash
./.venv/bin/python data/update_crypto_quant.py backfill
./.venv/bin/python data/update_crypto_quant.py update
./.venv/bin/python data/update_crypto_quant.py validate
./.venv/bin/python data/update_crypto_quant.py inspect
./.venv/bin/python data/update_crypto_quant.py rebuild-derived
```

Document HDF keys and example reads:

```python
import pandas as pd

with pd.HDFStore("data/crypto_quant.h5", mode="r") as store:
    print(store.keys())
    panel = store.select(
        "research_panel_daily",
        where="date >= Timestamp('2026-08-01')",
    )
```

Include the daily 08:20 Asia/Shanghai cron entry:

```cron
20 8 * * * cd /Users/dmiwu/work/PythonProject/cryptoFactorAnalyze && ./.venv/bin/python data/update_crypto_quant.py update >> logs/crypto_quant_update.log 2>&1
```

Explain recovery: inspect the non-zero log, rerun the same command to resume matching staging, use `--reset-staging` only when configuration/rules/as-of changed, and confirm `validate` before consuming data. State that cron inherits a minimal environment and therefore uses absolute `cd` plus the repository venv.

- [ ] **Step 5: Link the runbook from README**

Add a `Crypto Market-Cap Top50 Data Pipeline` section that states this is the point-in-time CMC/Binance workflow and links `docs/crypto_quant_data.md`. Keep the existing legacy CSV instructions unchanged.

- [ ] **Step 6: Run CLI tests and syntax checks**

```bash
./.venv/bin/python -m pytest tests/test_update_crypto_quant_cli.py -v
./.venv/bin/python -m compileall -q data/crypto_quant data/update_crypto_quant.py
```

Expected: tests PASS and compileall exits 0.

- [ ] **Step 7: Commit**

```bash
git add data/update_crypto_quant.py tests/test_update_crypto_quant_cli.py docs/crypto_quant_data.md README.md
git commit -m "docs(data): add Top50 maintenance CLI and runbook"
```

---

### Task 13: Full offline regression and live acceptance

**Files:**

- Modify only if verification exposes a defect: files owned by the failing task and its focused test.

- [ ] **Step 1: Run all new offline tests**

Run:

```bash
./.venv/bin/python -m pytest \
  tests/test_crypto_quant_config.py \
  tests/test_crypto_quant_http.py \
  tests/test_crypto_quant_cmc.py \
  tests/test_crypto_quant_binance.py \
  tests/test_crypto_quant_mapping.py \
  tests/test_crypto_quant_universe.py \
  tests/test_crypto_quant_panel.py \
  tests/test_crypto_quant_store.py \
  tests/test_crypto_quant_validation.py \
  tests/test_crypto_quant_pipeline.py \
  tests/test_crypto_quant_reader.py \
  tests/test_update_crypto_quant_cli.py -v
```

Expected: all new tests PASS with no network access.

- [ ] **Step 2: Run repository regression tests**

Run: `./.venv/bin/python -m pytest tests/ -q`

Expected: no failures and no regression from the recorded baseline.

- [ ] **Step 3: Run static future-leak and syntax checks**

```bash
./.venv/bin/python -m compileall -q data/crypto_quant data/update_crypto_quant.py
rg -n "shift\(-[1-9]|center\s*=\s*True|\.bfill\(|direction\s*=\s*['\"]forward" data/crypto_quant
```

Expected: compileall exits 0 and ripgrep returns no matches. Record the dynamic cutoff test's `max_abs_diff`, expected `0.0` or at most `1e-12` for floating-point columns.

- [ ] **Step 4: Run a bounded live canary into `/tmp`**

Run:

```bash
./.venv/bin/python data/update_crypto_quant.py backfill \
  --as-of 2024-02-05T00:20:00Z \
  --store /tmp/crypto_quant_canary.h5 \
  --reset-staging
./.venv/bin/python data/update_crypto_quant.py validate --store /tmp/crypto_quant_canary.h5
./.venv/bin/python data/update_crypto_quant.py inspect --store /tmp/crypto_quant_canary.h5
```

Expected: validation exits 0; CMC starts 2024-01-01; the first universe is effective 2024-01-02; January and February accepted universes each contain 50 unique contracts; raw Binance support data may predate 2024-01-01; no project data file is changed.

- [ ] **Step 5: Run the full production backfill**

Run:

```bash
./.venv/bin/python data/update_crypto_quant.py backfill
./.venv/bin/python data/update_crypto_quant.py validate
./.venv/bin/python data/update_crypto_quant.py inspect
```

Expected: `data/crypto_quant.h5` is published only after validation, all seven logical tables and metadata are present, every accepted month has 50 unique contracts, and inspection reports the most recent complete panel date.

- [ ] **Step 6: Prove incremental idempotency on production data**

Run the same explicit as-of update twice:

```bash
./.venv/bin/python data/update_crypto_quant.py update --as-of 2026-09-03T00:20:00Z
./.venv/bin/python data/update_crypto_quant.py inspect
./.venv/bin/python data/update_crypto_quant.py update --as-of 2026-09-03T00:20:00Z
./.venv/bin/python data/update_crypto_quant.py inspect
```

Expected: both updates succeed; the second inspection has identical primary keys, row counts, date ranges, and values.

- [ ] **Step 7: Commit verification-only fixes if any were required**

Stage only the files changed to correct an observed failure and their focused tests, then commit with a message naming the actual defect. If no defect was found, do not create an empty commit.

---

## Final handoff checklist

- [ ] List every created/modified file.
- [ ] Report all executed commands and test counts.
- [ ] Report `data/crypto_quant.h5` table keys, row counts, date ranges, and size.
- [ ] Report CMC/Binance unresolved mappings and the exact rule overrides used.
- [ ] Report static future-pattern scan results with any matches and their purpose.
- [ ] Report dynamic full-versus-cutoff `max_abs_diff`.
- [ ] Report live canary and full backfill outcomes separately.
- [ ] State the daily maintenance command and cron schedule.
- [ ] State known operational risks: CMC throttling, Binance rate limits, mapping exceptions, incomplete panel dates, and single-writer HDF5 semantics.
