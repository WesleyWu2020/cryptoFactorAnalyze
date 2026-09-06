# Funding quality implementation plan

Goal: preserve raw rates/prices, reject known placeholder bars at decision time, and distinguish funding observation from proven coverage per symbol/day.

Approved design: raw data unchanged; derived quality flags only. No future backfill, no historical filtering by current contract status. Without authoritative historical settlement schedules, observed events do not prove completeness. An explicit per-day expected timestamp list may establish complete/missing/not_applicable; absent schedule yields unknown (or no_events). Existing global fetch watermarks are transport metadata only.

Implementation and validation:
- Add failing regression tests in tests/test_crypto_quant_panel.py and tests/test_crypto_quant_universe.py for independent coverage, price validity, placeholder eligibility and cutoff replay.
- Extend data/crypto_quant/panel.py with price validity annotation and optional expected settlement schedules; add funding_coverage_status, funding_invalid_price_count and has_placeholder_kline to derived output.
- Filter explicit zero-volume flat OHLC in data/crypto_quant/universe.py using only the prior completed day. Preserve existing memberships during holding periods.
- Extend schemas.py and pipeline.py to persist flags and migrate derived schema safely. Existing raw schemas remain unchanged.
- Update assertions tied to superseded global completeness behavior; run ./.venv/bin/python -m pytest tests/test_crypto_quant*.py -q.
- Rebuild through ./.venv/bin/python data/update_crypto_quant.py rebuild-derived; verify raw table hashes, RAY eligibility, panel quality counts and cutoff consistency. If cached data lacks replacement candidates, report the exact limitation rather than weakening eligibility.

Verified on the local store after rebuild:
- Five raw tables (CMC daily/constituents, mappings, klines, funding events) have identical pandas row-content SHA-256 hashes before and after rebuild.
- Funding events: 547,193; invalid prices: 25,620, preserved verbatim.
- Universe: 1,650 memberships; RAY: zero memberships; daily panel: 47,250 rows, exactly 50 symbols/day.
- Panel days without funding observations: zero. Every stored rate sum matches the raw daily event sum (max_abs_diff=0).
- Remaining holding-period placeholder rows: FTMUSDT 25, MATICUSDT 7; rows retained and has_complete_kline=False.
- Coverage statuses: 47,250 unknown because no authoritative historical schedule is available. This is an explicit data-source limitation, not proof that all days are missing; the strict funding-complete reader remains unavailable until verified schedules are supplied.
- Real-data cutoff replay at 2025-09-30: 30,900 panel rows exactly equal, max_abs_diff=0. Memberships equal except the intentionally future-closed effective_end_date metadata.
- Static scan of modified production files found no negative shifts, centered rolling, backfill or forward as-of joins. No factor scripts were modified in this task.

Commands: `./.venv/bin/python data/update_crypto_quant.py rebuild-derived`; `./.venv/bin/python -m pytest tests/test_crypto_quant*.py -q`; `git diff --check`; `./.venv/bin/python -m py_compile data/crypto_quant/panel.py data/crypto_quant/universe.py data/crypto_quant/schemas.py data/crypto_quant/pipeline.py`.

Files changed for this task (pre-existing workspace edits preserved):
- data/crypto_quant/panel.py
- data/crypto_quant/universe.py
- data/crypto_quant/schemas.py
- data/crypto_quant/pipeline.py
- tests/test_crypto_quant_panel.py
- tests/test_crypto_quant_universe.py
- tests/test_crypto_quant_store.py
- tests/test_crypto_quant_pipeline.py
- docs/crypto_quant_data.md
- docs/superpowers/plans/2026-09-05-funding-quality.md
- data/crypto_quant.h5 (derived tables and run metadata rebuilt; all five raw tables unchanged)

Final regression result: `./.venv/bin/python -m pytest tests/test_crypto_quant*.py -xq` — 234 passed in 221.64 seconds. Empty funding-table regression was reproduced and fixed before this final run. Syntax compilation and diff whitespace checks passed.
