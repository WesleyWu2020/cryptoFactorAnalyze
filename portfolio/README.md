# Portfolio Layer

Multi-factor portfolio construction, backtesting, and reporting for Crypto assets.

## Overview

This layer sits on top of `factor_analyse/` and provides:
- **Factor loading & alignment** across multiple factor CSVs
- **Cross-sectional normalization** (winsorize → rank → z-score)
- **Symmetric orthogonalization** (optional, removes factor correlations)
- **IC/IC_IR weighting** (rolling, double-lag cutoff, no future leak)
- **Top-N equal-weight portfolio** construction
- **Drift-adjusted backtester** with transaction costs
- **HTML report** with equity curve and performance metrics

## Directory Structure

```
portfolio/
├── factor_pool/       # Load, align, label factors
│   ├── loader.py
│   ├── aligner.py
│   └── label_builder.py
├── combiner/          # Normalize, orthogonalize, weight, synthesize
│   ├── normalizer.py
│   ├── orthogonalizer.py
│   ├── ic_ir_weighter.py
│   └── synthesizer.py
├── portfolio_builder/ # Universe + Top-N selection
│   ├── constraints.py
│   ├── universe.py
│   └── topn_equal.py
├── backtester/        # Fees, metrics, engine
│   ├── fees.py
│   ├── metrics.py
│   └── engine.py
├── reporter/          # HTML report
│   └── html_renderer.py
├── config.py          # PortfolioConfig dataclass
├── pipeline.py        # End-to-end pipeline
└── main.py            # CLI entry point
```

## Quick Start

```bash
# Step 1: Generate factor data (run existing factor scripts)
./.venv/bin/python factor_analyse/factor_mining/momentum_factor.py

# Step 2: Run portfolio pipeline
./.venv/bin/python portfolio/main.py \
  --factors momentum=data/factor_data/momentum.csv \
             reversal=data/factor_data/reversal.csv \
  --kline data/kline_data/kline_data_1d.csv \
  --universe data/binance_coingecko_top100_marketcap_historical.csv \
  --top-n 10 \
  --ic-window 20 \
  --output portfolio/output/report.html
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_n` | 10 | Number of top instruments to hold |
| `ic_window` | 20 | Rolling IC window (days) |
| `label_period` | 1 | Forward return period (days) |
| `fee_rate` | 0.001 | One-way transaction fee (10bps) |
| `winsorize_pct` | (0.01, 0.99) | Winsorize bounds |
| `orthogonalize` | True | Enable symmetric orthogonalization |
| `output_dir` | portfolio/output | Default report output directory |

## Pipeline

```
Factor CSVs → load_many() → align_factors()
                                    ↓
kline CSV → build_future_ret()  normalize_cross_section()
                    ↓                   ↓
            [orthogonalize()]    compute_ic_ir_weights()
                                        ↓
universe CSV → build_universe() → synthesize() → build_topn_weights()
                                                        ↓
                                              run_backtest() → render_html()
```

## Future Leakage Policy

All components are verified no-leak by `tests/test_portfolio_cutoff_antileak.py`.
IC/IR weights use strict double-lag cutoff: window ends at `t-1`, never `t`.
