# Volume Stability data repair — 2026-09-06

Implemented on main, preserving existing workspace changes.

## Findings and implemented repairs

The old saved report (69,414 eligible / 16,783 missing) does not describe the current H5. Before this repair, current data produced 38,900 eligible / 38,842 valid / 58 missing cells over 2024-01-01 through 2026-03-15. Do not attribute the entire difference from the old report to this patch.

Monthly membership ended the day before the next decision, although replacements take effect the day after that decision. This left 26 recurring empty days plus the legitimate initial 2024-01-01 warmup day. Keep old membership through the next decision day. Rebuilding derived tables restores 1,300 eligible cells in the report interval. The full panel grows from 47,250 to 48,850 rows.

Volume Stability now masks non-positive quote volume / trade count before rolling. A no-trade day has no measured average trade size; it must not create a valid stability value. Retain the 20-calendar-day requirement, without filling or shortening windows.

After repair: 40,200 eligible, 40,107 valid, 93 missing (0.23134%). Missing reasons: 33 no-trade current day, 32 missing current inputs, 28 insufficient observed history. MATIC 28, FTM 26, EOS 11; WLFI 8, ASTER 6, FORM 5, TRUMP 4, PENGU 3, S 2. The current-input gaps all belong to EOS/MATIC. Attribution CSV retains exact dates and symbols.

## Structural limits, not synthetic backfill candidates

- Binance EOSUSDT automatic settlement: 2025-05-21 09:00 UTC. Source: https://www.binance.com/id/support/announcement/detail/1e89a9ca957c4b0ca7502e60b993e201
- Binance MATICUSDT automatic settlement: 2024-09-04 09:00 UTC. Source: https://www.binance.com/de/support/announcement/detail/6a6de383727f4659a3050f7982e1620f

Post-delisting bars cannot be filled with fabricated prices, replacement-token bars, or forward-filled old prices. Full-window EOS holdings still need a verified terminal settlement price and accounting support. This remains unresolved, and the full-window report remains incomplete.

Funding schedule was not generated from observed events: doing so would certify an incomplete fetch against itself. All 48,850 panel rows retain unknown coverage. Existing build_research_panel supports a verified expected_times schedule, but no authoritative historical schedule was supplied or established in this repair. Current funding interval information is insufficient to reconstruct historical changes. Full-cost performance remains uncertified.

## Outputs and reproducibility

- Backed-up store: data/quality_repair_20260906/crypto_quant.before.h5
- Raw-table identity checks: data/quality_repair_20260906/raw_verification.json
- Missing detail: reports/volume_stability_quality_20260906/missing_rows.csv
- Daily coverage: reports/volume_stability_quality_20260906/daily_coverage.csv
- Summary and evaluation paths: reports/volume_stability_quality_20260906/summary.json
- Direction reports: reports/volume_stability_quality_20260906/positive/ and negative/

Commands:

```sh
./.venv/bin/python data/update_crypto_quant.py rebuild-derived --as-of 2026-09-05T15:36:53Z
./.venv/bin/python scripts/audit_volume_stability.py --evaluate
./.venv/bin/python data/update_crypto_quant.py validate
./.venv/bin/python -m pytest tests/test_crypto_quant_universe.py tests/test_crypto_quant_reader.py tests/test_crypto_quant_panel.py tests/test_volume_stability_quality.py -q
./.venv/bin/python -m pytest tests/test_crypto_quant_pipeline.py -q
```

Validation: 57 focused tests passed. Both newly added regression tests failed before fixes. Store validation passed with explicit incomplete-kline and incomplete-funding warnings. Five raw tables compare exactly equal before/after. Static factor scan is clean. Real-data full/cutoff comparisons at 2025-02-06 and 2026-03-14 have equal masks and max_abs_diff=0; synthetic zero-trade window replay also passes.

Changed source/test files: data/crypto_quant/universe.py, factor_analyse/factor_mining/Volume_Stability_Factor.py, tests/test_crypto_quant_universe.py, tests/test_volume_stability_quality.py, scripts/audit_volume_stability.py. H5 derived tables and metadata were rebuilt. No raw-table content changed.

## OOS experiment

`factor_direction` currently changes directional_long only; long_short is always high minus low, and IC uses the unmodified factor. Thus the positive/negative parameter-only reports intentionally have identical long-short metrics. Do not interpret them as opposite trades.

Additional isolated OOS runs from cash over 2025-09-16 through 2026-03-15, with funding explicitly excluded, finish complete:

| Signal | Trading-net total return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Original | 41.36% | 2.2680 | 16.02% |
| Negated saved factor matrix | -31.74% | -2.1324 | 39.66% |

These are experimental OOS-window restarts, not the uninterrupted full-window portfolio and not funding-inclusive performance. Original OOS RankIC is -0.0035719681; negating the same values reverses its sign. The original matrix was validated with real-data cutoffs; the negated external-matrix run is marked external (its own source scan is not applicable).

Reports: reports/volume_stability_quality_20260906/oos_1/ and oos_negated_factor/. The negated experiment.json records the exact source run_id and operation. Reproduction uses FactorManager.evaluate with start/end above, include_funding=False, rebalance_days=1, n_groups=10. For the reverse experiment, load original run 87f20a066c4ef0c5 with manager.get_value and pass its negative as the external DataFrame with factor_name='Volume_Stability_Negated_Experiment'.

Final verification: pipeline regression 48 passed in 239.94s; focused regression 57 passed (105 total). Updated audit rerun reproduces 40,200 / 40,107 / 93 and both cutoff checks. Syntax compilation and git diff --check passed.
