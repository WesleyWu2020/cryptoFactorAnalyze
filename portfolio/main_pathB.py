"""CLI for Path B: BTC Core + Alt Short Overlay OOS backtest."""
from __future__ import annotations
import argparse
import importlib.util
import pathlib
import sys
import pandas as pd

from portfolio.config import PortfolioConfig
from portfolio.pipeline import run_pathB_pipeline


def discover_factor_paths(factor_dir: str) -> dict:
    """Return {factor_name: latest CSV path} using factor_analyse/factor_config.py prefixes."""
    factor_dir = pathlib.Path(factor_dir)
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "factor_config", root / "factor_analyse" / "factor_config.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = getattr(mod, "FACTOR_CONFIG", {})
    paths = {}
    for ftype, meta in cfg.items():
        prefix = meta.get("file_prefix") or ftype
        matches = sorted(factor_dir.glob(f"{prefix}*.csv"))
        if matches:
            paths[ftype] = str(matches[-1])  # latest
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline", default="data/kline_data/binance_daily_klines_20260417.csv")
    ap.add_argument("--market-cap", default="data/binance_coingecko_top100_marketcap_historical.csv")
    ap.add_argument("--funding-dir", default="data/funding_rate_data")
    ap.add_argument("--factor-dir", default="data/factor_data")
    ap.add_argument("--is-end", default="2023-12-31")
    ap.add_argument("--oos-start", default="2024-01-01")
    ap.add_argument("--oos-end", default=None)
    ap.add_argument("--out", default="portfolio/output/pathB")
    ap.add_argument("--factors", default=None,
                    help="comma-separated subset of factor names; default = all discovered")
    args = ap.parse_args()

    factor_paths = discover_factor_paths(args.factor_dir)
    if args.factors:
        wanted = set(s.strip() for s in args.factors.split(","))
        factor_paths = {k: v for k, v in factor_paths.items() if k in wanted}
    if not factor_paths:
        print("ERROR: no factors discovered; check --factor-dir and factor_config.py", file=sys.stderr)
        sys.exit(2)
    print(f"📂 {len(factor_paths)} factors discovered")

    cfg = PortfolioConfig()
    result = run_pathB_pipeline(
        factor_paths=factor_paths,
        kline_csv=args.kline,
        universe_csv=args.market_cap,
        funding_dir=args.funding_dir,
        config=cfg,
        is_end_date=args.is_end,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
    )
    nav = result["nav"]
    outp = pathlib.Path(args.out)
    outp.mkdir(parents=True, exist_ok=True)
    nav.to_csv(outp / "pathB_nav.csv", header=["nav"])
    print(f"✅ NAV saved -> {outp/'pathB_nav.csv'} | len={len(nav)} | final={nav.iloc[-1]:.4f}" if len(nav) else "⚠️ empty NAV")
    print(f"   kept_factors: {result['kept_factors']}")
    print(f"   shorts days: {len(result['shorts'])}")


if __name__ == "__main__":
    main()
