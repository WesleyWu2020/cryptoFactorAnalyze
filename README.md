# cryptoFactorAnalyze

## CryptoQuant Data Pipeline

The point-in-time CoinMarketCap/Binance data workflow is maintained by
`data/update_crypto_quant.py`. Its commands operate on the preserved
`data/crypto_quant.h5` store:

```bash
./.venv/bin/python data/update_crypto_quant.py update
./.venv/bin/python data/update_crypto_quant.py validate
```

Use `backfill` or `rebuild-derived` when explicitly needed. After updating the
store, create factors in `factor_analyse/factor_mining/` and run
`factor_analyse/main.py` to generate reports in `reports/`.

For the English `factor_common` H5 workflow, artifact schemas, cost and
funding conventions, compatibility mapping, and executable acceptance command,
see [`docs/factor_common.md`](docs/factor_common.md).

## MaxDD-15 Hedged Strategy

Alt-long + BTC-perp-short portfolio targeting ≤15% OOS max drawdown.

### Architecture (4 layers)

| Layer | Component | Description |
|-------|-----------|-------------|
| A | IS/OOS Factor Filter | Spearman IC_IR ≥ 0.05 on IS (≤2023-12-31), Top-5 weekly |
| B | BTC Trend Score T | EMA(tanh(ratio)+tanh(zscore)) → T ∈ [-1,+1] |
| C | Exposure Controller | alt_exp = alt_min+(alt_max-alt_min)*(T+1)/2; hedge = min(beta*alt_exp*(1-T), alt_exp*cap) |
| D | Cost Model | Alt 10bps + BTC perp 5bps + funding 10.95%/yr |

### Run

```bash
# Default OOS backtest (v1 config)
./.venv/bin/python portfolio/main_hedged.py

# Pre-tuned best config (ema=20/50, alt=[0,0.3], hedge_cap=2.0)
./.venv/bin/python portfolio/main_hedged.py best

# Full parameter sweep (slow: ~90 min)
./.venv/bin/python portfolio/main_hedged.py sweep
```

### Key Config Params (`portfolio/config.py`)

| Param | Default | Best Config | Description |
|-------|---------|-------------|-------------|
| `is_end_date` | `2023-12-31` | same | IS/OOS cutoff |
| `min_ic_ir` | `0.05` | `0.02` | Factor keep threshold |
| `top_n` | `5` | same | Weekly Top-N coins |
| `ema_short` / `ema_long` | `50/200` | `20/50` | BTC trend EMA windows |
| `alt_min_exposure` | `0.5` | `0.0` | Min alt notional |
| `alt_max_exposure` | `1.0` | `0.3` | Max alt notional |
| `hedge_cap_multiplier` | `1.2` | `2.0` | BTC short max × alt_exp |
| `beta_prior` | `1.3` | `2.0` | Alt beta prior (bear market) |

### OOS Results (2024-01-01 to 2026-04-17)

| Config | MaxDD | Ann Return | Sharpe | Notes |
|--------|-------|-----------|--------|-------|
| v1 (default) | -78.16% | -25.67% | -0.42 | Baseline |
| best (tuned) | -22.54% | -5.49% | -0.66 | ema=20/50, alt=[0,0.3], hedge_cap=2 |
| **Target** | **≤-15%** | **≥+20%** | **≥1.2** | Not yet achieved |

### Why Target Not Met — Structural Findings

The OOS period (2024-2026) saw the altcoin universe collapse:
- **EW-Alt universe**: cum_ret = -45.2%, MaxDD = -81.5%
- The BTC hedge reduces but cannot fully offset the long-leg losses
- 72% of OOS days have T > 0 (bull regime), so the hedge is partially reduced

To achieve MaxDD ≤ 15%, the following changes are needed:
1. **Stop-loss / vol-targeting overlay** — hard cut when drawdown exceeds threshold
2. **Net-short capability** — allow negative alt_exp when T < -0.5
3. **Better universe filtering** — exclude coins with severe negative momentum IS

See spec: `docs/superpowers/specs/2026-04-20-crypto-portfolio-maxdd15-design.md`

---

## Daily 08:00 (BJT) Auto Runner + Feishu Push

Use `daily_feishu_scheduler.py` in repo root.

### 1) Put Feishu config in `.env`

```bash
cp .env.example .env
# edit .env
```

`.env` keys:

- `FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/your-webhook-token`
- `FEISHU_BOT_SECRET=your-feishu-bot-secret` (optional, only if signature is enabled)

### 2) Run once (for verification)

```bash
./.venv/bin/python daily_feishu_scheduler.py --once
```

### 3) Run as daily scheduler (08:00 Asia/Shanghai)

```bash
./.venv/bin/python daily_feishu_scheduler.py
```

Optional flags:

- `--run-now`: execute immediately once, then continue daily schedule
- `--dry-run`: do not send Feishu messages, print them to `logs/daily_feishu_scheduler.log`
- `--hour` / `--minute`: customize schedule time in BJT
