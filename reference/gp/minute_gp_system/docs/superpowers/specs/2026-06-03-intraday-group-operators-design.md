# Intraday Group Operators Design

## Goal

Add a new GP operator family that compresses intraday grouped minute information into daily factor values. The feature should let the search discover day-level signals from intraday shape, concentration, and A/B co-movement without replacing the existing mode1, mode2, and mode3 contracts.

## Existing Context

The unified v2 evaluator loads indicator tensors shaped `(P, N_IND, M, S)`, where `P` is period/day, `M` is minutes per period, and `S` is symbol. Existing base operators consume a selected window `(P, W, S)` and return `(P, S)`. Temporal and cross-sectional composition then operate on the daily `(P, S)` factor output.

Several fixed intraday path indicators already exist, such as `path_efficiency`, `high_time_frac`, `late_volume_share`, and `intraday_reversal`. Those are hand-coded features. The new work should expose a more general grouped intraday compression as GP-searchable operators.

## Design

Add a fourth raw mode, `mode == 3`, shown to users as `mode == 4`. This keeps old genomes compatible because existing mode ids and operator index registries remain unchanged.

Mode4 evaluates:

```text
minute indicator window -> intraday groups -> per-group summaries -> daily factor
```

The first release uses a compact operator set:

- `group_slope`: slope of group means from early to late groups.
- `group_dispersion`: standard deviation of group means.
- `group_early_late_diff`: late-half group mean minus early-half group mean.
- `group_top_share`: largest positive group contribution divided by total positive contribution.
- `group_corr`: correlation of A and B group means.
- `group_beta`: beta of A group means versus B group means.

Group counts are selected from `{4, 6, 12, 24}`. The implementation appends one new gene slot, `intraday_group_count`, instead of overloading existing slots. Archive and replay code already pads shorter genome arrays to `N_PARAMS`, so old results remain loadable.

## Files

- `engines/unified_v2/config.py`: append `MODE4_OPS`, `INTRADAY_GROUP_CHOICES`, `intraday_group_count`, bounds, decode, formula, search-space state, and repair projection.
- `engines/unified_v2/operators.py`: add grouped helper functions and `MODE4_DISPATCH`.
- `engines/unified_v2/evaluator.py`: group mode4 tasks, slice A/B data, call mode4 dispatch with the selected group count.
- `engines/unified_v2/evolution.py`: include mode4 in random generation, repair/mutation choices, and a small sampling lane or structural share.
- `tests/test_intraday_group_operators.py`: verify grouped operator numerics.
- `tests/test_intraday_group_evaluator.py`: verify a tiny population can evaluate a mode4 genome and render a formula.

## Search-Space Controls

Mode4 should start with a small sample share and field bias. Preferred A fields are returns, absolute return, price range, volume/log-volume, turnover, money flow, taker pressure, and xbinance taker pressure. Preferred B fields are liquidity, volatility, and order-flow context fields. This avoids flooding the main search with noisy intraday patterns before there is evidence the family transfers.

Mode4 should keep temporal and cross-sectional composition available. If mode4 output is noisy, `cs_zscore`, `ts_decay`, and `ts_zscore` are expected to be the most useful downstream normalizers.

## Edge Cases

- If a group has no valid observations, its group mean is NaN.
- Operators require at least two valid groups for one-variable shape operators and at least three paired valid groups for correlation or beta.
- `group_top_share` uses positive finite group sums and returns NaN when total positive contribution is too small.
- Group count is clipped to the available window length so small windows do not create empty-only groups.

## Validation

Unit tests should cover shape, NaN behavior, simple monotonic slopes, early-late differences, top-share concentration, and paired correlation/beta. A smoke test should construct a tiny `(P, N_IND, M, S)` tensor and a mode4 genome, then confirm `evaluate_population()` returns `(1, P, S)` finite values for a controlled case.
