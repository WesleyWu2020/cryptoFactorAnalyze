#!/usr/bin/env python
"""CLI entry point for the portfolio pipeline."""
from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime

# Allow running as: python portfolio/main.py
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio.config import PortfolioConfig  # noqa: E402
from portfolio.pipeline import run_pipeline    # noqa: E402


def _parse_factor_pairs(pairs: list[str]) -> dict[str, str]:
    """Parse 'name=path' strings into dict."""
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                f"Factor argument must be 'name=path', got: {pair!r}"
            )
        name, path = pair.split("=", 1)
        result[name.strip()] = path.strip()
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="portfolio",
        description="Run the multi-factor portfolio pipeline and generate HTML report.",
    )
    p.add_argument(
        "--factors", nargs="+", required=True, metavar="NAME=PATH",
        help="Factor CSV files as name=path pairs (e.g. momentum=data/factor_data/mom.csv)",
    )
    p.add_argument("--kline", required=True, metavar="PATH",
                   help="Kline CSV with [symbol, date, close]")
    p.add_argument("--universe", required=True, metavar="PATH",
                   help="Universe CSV with Decision_Window columns")
    p.add_argument("--top-n", type=int, default=10, help="Number of top instruments (default: 10)")
    p.add_argument("--ic-window", type=int, default=20, help="Rolling IC window in days (default: 20)")
    p.add_argument("--label-period", type=int, default=1, help="Forward return period in days (default: 1)")
    p.add_argument("--fee-rate", type=float, default=0.001, help="One-way fee rate (default: 0.001)")
    p.add_argument("--cluster-threshold", type=float, default=0.0,
                   help="Merge factors with |corr| > threshold before orthogonalization (0=off, 0.6=recommended)")
    p.add_argument("--rebalance-period", type=int, default=1,
                   help="Rebalance every N trading days (1=daily, 5=weekly, default: 1)")
    p.add_argument("--no-orthogonalize", action="store_true", help="Disable factor orthogonalization")
    p.add_argument("--output", metavar="PATH",
                   help="Output HTML report path (default: portfolio/output/report_TIMESTAMP.html)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    factor_paths = _parse_factor_pairs(args.factors)

    config = PortfolioConfig(
        top_n=args.top_n,
        ic_window=args.ic_window,
        label_period=args.label_period,
        fee_rate=args.fee_rate,
        orthogonalize=not args.no_orthogonalize,
        rebalance_period=args.rebalance_period,
        cluster_threshold=args.cluster_threshold,
    )

    if args.output:
        output_path = pathlib.Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = pathlib.Path(config.output_dir) / f"report_{ts}.html"

    print(f"Running portfolio pipeline...")
    print(f"  Factors: {list(factor_paths.keys())}")
    print(f"  Top-N: {config.top_n}, Rebalance: every {config.rebalance_period}d, IC window: {config.ic_window}, Fee: {config.fee_rate:.4f}")

    returns = run_pipeline(
        factor_paths=factor_paths,
        kline_csv=args.kline,
        universe_csv=args.universe,
        config=config,
        output_path=output_path,
    )

    if returns.empty:
        print("Warning: No returns generated.")
        return 1

    # Print summary
    from portfolio.backtester.metrics import annual_return, sharpe, max_drawdown
    print(f"\n--- Backtest Summary ---")
    print(f"  Period:         {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"  Trading days:   {len(returns)}")
    print(f"  Annual return:  {annual_return(returns):.2%}")
    print(f"  Sharpe ratio:   {sharpe(returns):.3f}")
    print(f"  Max drawdown:   {max_drawdown(returns):.2%}")
    print(f"\nReport written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
