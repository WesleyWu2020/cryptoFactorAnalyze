# Crypto Market-Cap Top50 Data Pipeline Design

**Date:** 2026-09-03

**Status:** Approved design
**Scope:** `data/` market-universe, Binance USDⓈ-M futures data, funding data, and an HDF5 research panel

## 1. Objective

Build a point-in-time, reproducible data pipeline for personal cross-sectional crypto research. The pipeline will maintain a monthly market-cap Top50 universe, restricted to Binance USDT-margined perpetual contracts, and provide a daily panel suitable for five equal cross-sectional groups and 3-day or 5-day portfolio rebalancing.

The system starts on 2024-01-01, the first available CMC100 index date. It does not splice the existing Binance liquidity proxy into the pre-2024 period because that would change the universe definition and make results incomparable.

## 2. Confirmed Decisions

- Trading instruments: Binance USDⓈ-M USDT perpetual contracts only.
- Universe source: CoinMarketCap CMC100 historical index constituents and weights.
- Market-cap history starts on 2024-01-01.
- Universe size: fixed Top50 only; no Top100 mode.
- Stablecoins are always excluded. CMC's wrapper-token exclusions are retained as an additional rule.
- Universe is recomputed monthly.
- A monthly decision made from the CMC snapshot at `00:00 UTC` becomes effective on the next natural day (`T+1`).
- Market data frequency: daily futures OHLCV.
- Funding data frequency: preserve raw settlement events; derive daily aggregates later.
- Factor values are computed daily; portfolios normally rebalance every 3 or 5 days.
- Storage: one local HDF5 file containing multiple normalized logical tables.

## 3. Source Assessment

### 3.1 CoinMarketCap

Use the keyless endpoint:

`GET https://pro-api.coinmarketcap.com/public-api/v3/index/cmc100-historical`

Relevant parameters are `time_start`, `time_end`, `count`, and `interval=daily`. The public endpoint requires no account or API key. It returns at most 10 daily points per request, so callers must paginate and support throttling and retries.

The endpoint was validated end to end on 2026-09-03:

- 2023 dates returned no rows.
- 2024-01-01 returned the first row with index value 100.
- 2024-01-01 through 2026-09-03 contained 977 unique natural dates.
- No calendar dates were missing or duplicated in that interval.
- Each row included an index value and point-in-time constituent objects with weights.
- Individual responses contained at least 93 and at most 100 constituents, which is sufficient for Top50 selection but requires a minimum-count validation.
- A complete validation required 98 requests and encountered public-IP throttling, confirming the need for backoff and restartable checkpoints.

References:

- [CMC Index API](https://coinmarketcap.com/api/documentation/pro-api-reference/cmc-index)
- [CMC Keyless Public API](https://coinmarketcap.com/api/documentation/pro-api-reference/keyless-public-api)
- [Local CMC100 methodology](../../CoinMarketCap_100_Index_Methodology.pdf)

### 3.2 Binance USDⓈ-M Futures

Use Binance futures endpoints for:

- Current contract metadata and trading rules.
- Daily perpetual-contract klines.
- Historical funding-rate events.

The CMC100 feed supplies market-cap ordering, while Binance supplies tradability, futures prices, volumes, and funding. Binance data is not used as a market-cap proxy.

Historical eligibility must be established from observations available on or before the decision time. Current exchange status must not be projected backward. For historical months, a mapped contract is eligible only if it has a completed daily futures kline for `T-1`; this naturally excludes contracts not yet listed or already unavailable. The current exchange-information endpoint is used only for current metadata and the latest decision.

References:

- [Binance USDⓈ-M exchange information](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)
- [Binance funding-rate history](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)

## 4. Architecture

The pipeline has three independent source adapters and two deterministic derived stages:

1. CMC100 adapter fetches and normalizes index rows and constituent rows.
2. Binance contract adapter resolves point-in-time eligible USDT perpetuals.
3. Binance market-data adapter fetches daily klines and raw funding events.
4. Universe builder creates the monthly Top50 membership table.
5. Panel builder expands monthly membership to daily rows and joins market and funding data.

Raw normalized tables remain independent of derived rules. Changing a stablecoin rule, symbol mapping, or effective-date policy therefore requires rebuilding only the universe and panel, not downloading all source data again.

Existing CSV/liquidity scripts remain compatible and are not repurposed as true market-cap data. Shared retry, time-handling, and validation behavior should be reused or extended where practical, without changing the existing factor-miner input/output contract.

## 5. HDF5 Layout

The default store is:

`data/crypto_quant.h5`

It uses HDF5 table format with queryable date and symbol columns. Tables are normalized rather than storing nested JSON.

### 5.1 `/cmc100_daily`

One row per natural day:

- `date`
- `index_value`
- `source_update_time`
- `fetched_at_utc`

Primary key: `date`.

### 5.2 `/cmc100_constituents`

One row per CMC constituent per day:

- `date`
- `cmc_id`
- `symbol`
- `name`
- `weight`

Primary key: `date + cmc_id`.

### 5.3 `/futures_contracts`

Canonical CMC-to-Binance mapping and current contract metadata:

- `cmc_id`
- `cmc_symbol`
- `binance_symbol`
- `base_asset`
- `quote_asset`
- `contract_type`
- `onboard_date`
- `status`
- `mapping_source`
- `valid_from`
- `valid_to`

Mappings are keyed by CMC ID, not symbol alone. Renames and symbol collisions require explicit validity ranges or override records. Failed or ambiguous mappings are reported and never guessed silently.

### 5.4 `/klines_daily`

One row per completed UTC day and contract:

- `date`
- `symbol`
- `open_time`
- `close_time`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `quote_volume`
- `trade_count`
- `taker_buy_base_volume`
- `taker_buy_quote_volume`

Primary key: `date + symbol`.

### 5.5 `/funding_events`

One row per actual funding settlement event:

- `funding_time`
- `symbol`
- `funding_rate`
- `mark_price`
- `rate_type`, using Binance's value when supplied and normalizing a missing value to `Regular`

Primary key: `funding_time + symbol + rate_type`. Normalizing the missing type avoids nullable-key ambiguity. Raw events are retained because funding intervals may change and because a 3-day or 5-day holding period needs the actual accumulated funding cost.

### 5.6 `/universe_monthly`

One row per selected contract per monthly decision:

- `decision_date`
- `effective_date`
- `effective_end_date`
- `cmc_id`
- `cmc_symbol`
- `binance_symbol`
- `market_cap_rank`
- `cmc_weight`

Primary key: `effective_date + binance_symbol`. Every valid month must contain exactly 50 unique contracts.

### 5.7 `/research_panel_daily`

One row per effective Top50 member per day. It contains the daily kline columns plus:

- `decision_date`
- `universe_effective_date`
- `market_cap_rank`
- `cmc_weight_at_decision`
- `funding_rate_sum`
- `funding_rate_mean`
- `funding_rate_last`
- `funding_event_count`
- `has_complete_kline`
- `has_complete_funding`

Primary key: `date + binance_symbol`. This is the default table consumed by factor research. A factor script can load a date range, group by date, sort 50 factor values, and split them into five groups of 10.

### 5.8 Store metadata

Store metadata records or HDF5 attributes include:

- Schema version.
- Data-source URLs.
- Last successful update per source.
- Last complete UTC date.
- Row counts and date ranges per table.
- Stablecoin-rules version.
- Symbol-mapping version.
- Pipeline version and creation timestamp.

## 6. Monthly Top50 Methodology

For each first calendar day `T` at `00:00 UTC`:

1. Load that day's historical CMC100 constituent snapshot.
2. Remove stablecoins using both the upstream CMC exclusion and a local explicit denylist.
3. Retain CMC's asset-backed wrapper exclusions as a defensive rule.
4. Resolve each remaining CMC ID to a Binance USDT perpetual contract.
5. Require a completed Binance daily kline on `T-1`. Only information timestamped no later than `T` may be used for this check.
6. Sort eligible contracts by CMC weight descending, using CMC ID as a deterministic tie-breaker.
7. Select exactly 50 contracts and assign ranks 1 through 50.
8. Make the new membership effective on `T+1` and keep it unchanged until the next effective date.

The intersection is performed before selecting 50. Selecting CMC's first 50 and intersecting afterward could leave fewer than 50 Binance contracts and break five equal groups.

If fewer than 50 eligible contracts remain, the pipeline fails closed for that month. It must not silently add assets outside CMC100, reuse a stale contract that is no longer eligible, or produce uneven groups.

## 7. Initial Backfill

The first complete run performs these stages in order:

1. Fetch CMC100 history from 2024-01-01 through the current completed availability window in 10-day pages.
2. Persist each validated page as a restartable checkpoint.
3. Normalize all observed CMC constituents and build the candidate mapping set.
4. Resolve candidate contracts, recording explicit exceptions for collisions and renames.
5. Fetch daily futures klines from 2024-01-01 or the contract's first available date.
6. Fetch raw funding events over the same available interval.
7. Build all monthly Top50 memberships.
8. Build the daily research panel.
9. Validate the complete HDF5 store before publishing it as the active file.

The backfill fetches candidate data needed to prove eligibility, not only contracts that survive into today's Top50. This prevents current-membership and survivorship bias.

## 8. Incremental Update

Each daily update:

1. Acquires a single-writer lock.
2. Determines each source's last successful timestamp from store metadata.
3. Fetches only missing CMC pages, completed daily klines, and funding events after the last event timestamp.
4. Deduplicates source rows by their declared primary keys.
5. Rebuilds monthly Top50 only when a new CMC monthly decision is available or a mapping rule changes.
6. Rebuilds only affected daily-panel dates.
7. Runs all validations against a staging store.
8. Atomically replaces the active store only after validation succeeds.

The update is idempotent: running the same date range repeatedly produces identical keys, row counts, and values.

## 9. Funding Aggregation

Raw funding rows are assigned to the UTC date containing their actual `funding_time`. Daily panel aggregation produces:

- Sum of all funding rates that settled that day.
- Mean funding rate.
- Last observed funding rate.
- Number of funding events.

A missing event and a zero funding rate are different states. The pipeline does not fill missing raw events with zero. `funding_event_count` and `has_complete_funding` allow research code to decide whether a daily observation is usable.

The panel stores the signed raw funding-rate sum. Portfolio code applies position direction: a positive rate is a cost to a long and a benefit to a short.

## 10. Reliability and Failure Handling

### API behavior

- Honor Binance `Retry-After` responses.
- Use bounded exponential backoff with jitter for 429 and retryable 5xx responses.
- Keep concurrency conservative and configurable.
- Persist page-level progress so a failed backfill resumes rather than restarts.
- Treat malformed JSON, schema drift, and impossible timestamps as hard failures.

### Mapping behavior

- Never rely on symbol-only fuzzy matching.
- Record unresolved and ambiguous mappings in a structured validation result.
- Require explicit overrides for collisions, redenominations, or renamed contracts.

### HDF5 safety

- Permit one writer at a time.
- Write updated tables to a staging HDF5 file.
- Validate schemas, keys, date ranges, and row counts before publication.
- Replace the active file atomically.
- Leave the previous active file untouched when any stage fails.

The generated `.h5` file remains ignored by Git. Only code, tests, small metadata examples, and documentation are version controlled.

## 11. Validation and Tests

### Unit tests

- CMC 10-day pagination, empty pre-base interval, throttling, and retry behavior.
- Binance kline and funding pagination.
- Stablecoin and wrapper filtering.
- CMC ID to Binance contract mapping, including explicit overrides.
- Monthly intersection-before-ranking behavior.
- Exactly 50 unique members and deterministic ranks.
- `T+1` effective-date expansion.
- Funding-event daily aggregation and missing-event handling.
- HDF5 idempotent updates, lock behavior, and failed-update rollback.

### Data-quality validation

- CMC dates and `date + cmc_id` keys are unique.
- Each CMC decision snapshot has at least 50 valid constituents.
- Every monthly universe has exactly 50 unique, non-stablecoin USDT perpetuals.
- Every member belongs to that decision's CMC snapshot and has a completed `T-1` kline.
- Klines have unique `date + symbol` keys, positive prices, non-negative volumes, and valid OHLC relationships.
- Funding events have unique keys and monotonic timestamps per symbol.
- Every published panel day has 50 members and complete required kline fields.

### Future-leak validation

Static checks for all new or modified data and factor paths prohibit:

- Negative shifts used in feature construction.
- Centered rolling windows.
- Backward fill of feature data.
- Forward-looking as-of joins.

Dynamic cutoff tests run the pipeline twice: once with all available source data and once with every source truncated at a chosen cutoff. CMC constituents, monthly Top50 memberships, and panel values at or before the cutoff must be identical within floating-point tolerance. The test must also verify that no monthly membership becomes effective before `decision_date + 1 day`.

## 12. Dependencies and Compatibility

Pandas HDF5 support requires PyTables (`tables`). This is the only new storage dependency. No database server, distributed engine, or heavy data framework is introduced.

The existing factor-miner contract is unchanged. An adapter from `/research_panel_daily` will expose the repository's expected `date`, `instrument`, and market-data columns. Existing CSV-based scripts and reports remain reproducible while the new HDF5 pipeline is introduced alongside them.

## 13. Definition of Done

- A clean machine can build `data/crypto_quant.h5` from public endpoints starting at 2024-01-01.
- All seven logical tables and metadata are present with their declared schemas.
- Every published monthly universe contains exactly 50 eligible contracts.
- The daily research panel can be filtered by date and loaded directly for five-group factor analysis.
- Incremental reruns are idempotent and resume after interruption.
- API failures cannot replace a valid active HDF5 file with a partial file.
- Static future-leak checks pass.
- Dynamic full-versus-cutoff comparisons match within tolerance.
- Focused unit tests and one end-to-end backfill/update smoke test pass.

## 14. Known Limitations

- CMC100 history begins on 2024-01-01, so the strict market-cap Top50 backtest cannot cover earlier dates.
- The keyless CMC endpoint has a small page size and an unspecified shared-IP rate pool; first-time backfill is slower than the raw data volume suggests.
- Historical symbol mapping requires explicit handling of renames and collisions.
- An HDF5 store supports the intended single-machine, single-writer workflow but is not designed for multiple concurrent writers.
- If the CMC/Binance intersection contains fewer than 50 eligible contracts, the affected month is rejected rather than approximated.
