# Portfolio Research

日频固定组合的新入口是 `./.venv/bin/python -m portfolio.main_fixed`。
它按明确模块路径加载因子，合并各成员的方向多空目标，使用共享严格账本。
仅支持永续多空研究；不接实盘下单，也不模拟保证金和强平。
`freeze`、`run`、`audit` 均应在新 Python 进程执行；修改代码后不能沿用 notebook
已缓存的旧模块来宣称运行了新代码，必须重启解释器并重新冻结。

流程为 `freeze` → `run` → `audit`。`freeze` 记录代码版本和固定资本权重，
不执行因子选择；任何代码变化都需要生成新配置，旧配置不会被覆盖。
所有成员使用同一日频调仓日历，信号日 t 的目标在 t+1 open 基准执行。
缺失成员不沿用过期信号，也不把其预算重新分配给其他成员。

结果写入 `reports/portfolio/<run_id>/`。检查 `manifest.json` 的运行状态、
`metrics.json` 的各情景状态、订单/资金费/风险诊断，再查看 HTML。
`artifact_complete.json` 仅表示落盘完成；cutoff 审计以独立 audit JSON 为凭证，
必须核对其中 `configuration`、`code_hashes`、`source_stat` 与运行 manifest 一致。

固定组合结果中的 `artifact_complete` 表示文件集已经写完，`strategy_certified`
始终表示尚未通过实盘策略认证；两者不是同一个状态。研究账本不连接交易所 API，
也不模拟保证金、强平、最小订单、数量精度或盘口冲击。

最小流程示例（只生成配置，不自动运行大型 H5 回测）：

```bash
./.venv/bin/python -m portfolio.main_fixed freeze \
  --h5 data/crypto_quant.h5 --factor path/to/factor.py --allocation 1 \
  --start 2024-01-01 --end 2024-01-31 --as-of 2024-02-02 \
  --output reports/portfolio/frozen.json
./.venv/bin/python -m portfolio.main_fixed run \
  --config reports/portfolio/frozen.json --output-root reports/portfolio
./.venv/bin/python -m portfolio.main_fixed audit \
  --config reports/portfolio/frozen.json --cutoff 2024-01-15 \
  --output reports/portfolio/audit.json
```

旧 `main.py`、`main_hedged.py`、`main_pathB.py` 仅用于 legacy research 历史复现。
它们与新入口具有不同的持仓、计费、时间和异常数据规则，结果不可直接拼接。
旧测试通过不代表可用于实盘；旧入口不属于新固定组合的验收范围。

## Legacy research

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
