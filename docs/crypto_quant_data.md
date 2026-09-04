# Crypto Market-Cap Top50 Data Pipeline

This is the point-in-time CoinMarketCap/Binance workflow. It writes the normalized research store to `data/crypto_quant.h5`; the existing CSV workflow remains separate.

## Daily commands

```bash
./.venv/bin/python data/update_crypto_quant.py backfill
./.venv/bin/python data/update_crypto_quant.py update
./.venv/bin/python data/update_crypto_quant.py validate
./.venv/bin/python data/update_crypto_quant.py inspect
./.venv/bin/python data/update_crypto_quant.py rebuild-derived
```

Mutation commands accept `--as-of 2026-09-03T00:20:00Z`, `--store PATH`, and `--reset-staging`. `--as-of` must include a timezone and is normalized to UTC; without it, the current UTC time is used.

## HDF5 layout

The store keys are `/cmc100_daily`, `/cmc100_constituents`, `/futures_contracts`, `/klines_daily`, `/funding_events`, `/universe_monthly`, `/research_panel_daily`, and `/_metadata`.

Example query:

```python
import pandas as pd

with pd.HDFStore("data/crypto_quant.h5", mode="r") as store:
    print(store.keys())
    panel = store.select(
        "research_panel_daily",
        where="date >= Timestamp('2026-08-01')",
    )
```

## Cron

The daily update runs at 08:20 Asia/Shanghai:

```cron
20 8 * * * cd /Users/dmiwu/work/PythonProject/cryptoFactorAnalyze && ./.venv/bin/python data/update_crypto_quant.py update >> logs/crypto_quant_update.log 2>&1
```

Cron inherits a minimal environment. The entry therefore uses an absolute `cd` and the repository's absolute virtual-environment path. Set `CMC_PRO_API_KEY` (or `CMC_API_KEY`) in the cron environment if the CMC endpoint requires authentication, and ensure `logs/` exists.

## Recovery and validation

On a non-zero run, inspect `logs/crypto_quant_update.log` and rerun the same command; the staged HDF5 is designed to resume from checkpoints. Use `--reset-staging` only when configuration, mapping rules, or the requested `--as-of` target changed and the existing staging state should be discarded. Run `validate` and require a zero exit code before consuming data. `validate` returns 2 for validation errors; source, store, and runtime failures return 1.
