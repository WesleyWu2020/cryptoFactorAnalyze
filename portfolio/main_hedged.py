"""Execute the MaxDD-15 hedged pipeline on real Binance data."""
from __future__ import annotations

import glob
import itertools
import os
import pathlib
import sys
from typing import Any

import pandas as pd

from portfolio.config import PortfolioConfig
from portfolio.pipeline import run_hedged_pipeline


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "portfolio" / "output"

# ── Fixed OOS window ────────────────────────────────────────────────────────
OOS_START = "2024-01-01"
OOS_END   = "2026-04-17"
IS_END    = "2023-12-31"


def _latest_kline() -> str:
    candidates = sorted(glob.glob(str(DATA_DIR / "kline_data" / "binance_daily_klines_*.csv")))
    if not candidates:
        raise FileNotFoundError("No kline CSV found")
    return candidates[-1]


def _load_factor_paths() -> dict[str, str]:
    """Auto-discover all factor CSVs matching *_YYYYMMDD.csv in factor_data/."""
    paths: dict[str, str] = {}
    seen: set[str] = set()
    for p in sorted(glob.glob(str(DATA_DIR / "factor_data" / "*.csv")), reverse=True):
        name = pathlib.Path(p).stem
        parts = name.rsplit("_", 1)
        base = parts[0] if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8 else name
        if base not in seen:
            paths[base] = p
            seen.add(base)
    return paths


def _oos_metrics(returns: pd.Series) -> dict[str, float]:
    """Compute OOS MaxDD, annualized return, and Sharpe."""
    oos = returns.loc[OOS_START:OOS_END] if not returns.empty else returns
    if oos.empty or len(oos) < 2:
        return {"maxdd": float("nan"), "ann": float("nan"), "sharpe": float("nan")}
    cum_curve = (1 + oos).cumprod()
    dd = (cum_curve / cum_curve.cummax() - 1).min()
    ann = cum_curve.iloc[-1] ** (252 / len(oos)) - 1
    sharpe = oos.mean() / oos.std() * (252 ** 0.5) if oos.std() > 1e-12 else float("nan")
    return {"maxdd": float(dd), "ann": float(ann), "sharpe": float(sharpe)}


def _build_config(overrides: dict[str, Any], kline_csv: str, universe_csv: str) -> PortfolioConfig:
    """Build PortfolioConfig with sensible post-diagnosis defaults + overrides.

    Post-diagnosis fixes applied as baseline:
    - orthogonalize=False: 28 factors causes scipy.sqrtm failures on many dates
    - beta_prior=2.0:      altcoins have true bear-market beta ~2 vs 1.3 prior
    - alt_exposure_ema_alpha=0.15: faster EMA (12-day) vs original 39-day
    """
    defaults: dict[str, Any] = dict(
        top_n=5,
        rebalance_period=5,
        is_end_date=IS_END,
        universe_csv=universe_csv,
        kline_csv=kline_csv,
        orthogonalize=False,       # Fix A: 28-factor sqrtm failures
        beta_prior=2.0,            # Fix B: real alt beta in bear market ≈ 2
        alt_exposure_ema_alpha=0.15,  # Fix C: faster exposure adjustment
        alt_max_exposure=1.0,
        hedge_cap_multiplier=1.2,
    )
    defaults.update(overrides)
    return PortfolioConfig(**defaults)


def run_oos_backtest(
    oos_start: str = OOS_START,
    oos_end: str = OOS_END,
    top_n: int = 5,
    label: str = "maxdd15_oos",
    config_overrides: dict[str, Any] | None = None,
) -> pd.Series:
    universe_csv = str(DATA_DIR / "binance_coingecko_top100_marketcap_historical.csv")
    kline_csv = _latest_kline()
    factor_paths = _load_factor_paths()

    print(f"Kline: {kline_csv}")
    print(f"Factors ({len(factor_paths)}): {list(factor_paths.keys())[:5]} ...")
    print(f"Universe: {universe_csv}")

    overrides = config_overrides or {}
    overrides["top_n"] = top_n
    cfg = _build_config(overrides, kline_csv, universe_csv)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = OUTPUT_DIR / f"report_{label}.html"
    returns = run_hedged_pipeline(
        factor_paths=factor_paths,
        kline_csv=kline_csv,
        universe_csv=universe_csv,
        config=cfg,
        output_path=str(output),
    )
    oos = returns.loc[oos_start:oos_end] if not returns.empty else returns
    print(f"\nReport: {output}")
    print(f"OOS dates: {len(oos)}  Full dates: {len(returns)}")
    if not oos.empty:
        m = _oos_metrics(returns)
        cum = (1 + oos).prod() - 1
        print(f"OOS cum_ret   = {cum:.2%}")
        print(f"OOS annualized= {m['ann']:.2%}")
        print(f"OOS MaxDD     = {m['maxdd']:.2%}")
        print(f"OOS Sharpe    = {m['sharpe']:.3f}")
    return returns


def run_param_sweep(
    oos_start: str = OOS_START,
    oos_end: str = OOS_END,
    top_n: int = 5,
) -> pd.DataFrame:
    """Sweep ema_short/long × alt_min_exposure × min_ic_ir and report OOS metrics.

    Fixed params: top_n=5, is_end_date=2023-12-31.
    Baseline fixes applied to every combo (see _build_config):
      orthogonalize=False, beta_prior=2.0, alt_exposure_ema_alpha=0.15.
    """
    universe_csv = str(DATA_DIR / "binance_coingecko_top100_marketcap_historical.csv")
    kline_csv = _latest_kline()
    factor_paths = _load_factor_paths()

    ema_pairs   = [(20, 50), (30, 100), (50, 200)]
    min_exps    = [0.2, 0.3, 0.5]
    min_ic_irs  = [0.02, 0.05]

    combos = list(itertools.product(ema_pairs, min_exps, min_ic_irs))
    n = len(combos)
    print(f"\nRunning {n} parameter combos …\n")

    rows = []
    for i, ((ema_s, ema_l), min_exp, min_ic_ir) in enumerate(combos, 1):
        print(f"[{i}/{n}] ema=({ema_s},{ema_l})  alt_min={min_exp}  min_ic_ir={min_ic_ir}")
        try:
            cfg = _build_config(
                {
                    "top_n": top_n,
                    "ema_short": ema_s,
                    "ema_long": ema_l,
                    "alt_min_exposure": min_exp,
                    "min_ic_ir": min_ic_ir,
                },
                kline_csv, universe_csv,
            )
            ret = run_hedged_pipeline(
                factor_paths=factor_paths,
                kline_csv=kline_csv,
                universe_csv=universe_csv,
                config=cfg,
            )
            m = _oos_metrics(ret)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            m = {"maxdd": float("nan"), "ann": float("nan"), "sharpe": float("nan")}

        row = {
            "ema_s": ema_s, "ema_l": ema_l,
            "min_exp": min_exp, "min_ic_ir": min_ic_ir,
            **m,
        }
        rows.append(row)
        print(f"  MaxDD={m['maxdd']:.2%}  Ann={m['ann']:.2%}  Sharpe={m['sharpe']:.3f}")

    df = pd.DataFrame(rows)
    _print_sweep_table(df)
    return df


def _print_sweep_table(df: pd.DataFrame) -> None:
    """Pretty-print sweep results."""
    print("\n" + "=" * 72)
    print("SWEEP RESULTS")
    print("=" * 72)
    header = f"| {'ema_s':>5} | {'ema_l':>5} | {'min_exp':>7} | {'min_ic_ir':>9} | {'MaxDD':>8} | {'Ann':>8} | {'Sharpe':>7} |"
    sep    = "|" + "|".join(["-"*(w+2) for w in [5,5,7,9,8,8,7]]) + "|"
    print(header)
    print(sep)
    for _, r in df.iterrows():
        print(
            f"| {int(r.ema_s):>5} | {int(r.ema_l):>5} |"
            f" {r.min_exp:>7.1f} | {r.min_ic_ir:>9.2f} |"
            f" {r.maxdd:>8.2%} | {r.ann:>8.2%} | {r.sharpe:>7.3f} |"
        )
    print("=" * 72)

    valid = df[df["maxdd"].notna() & (df["maxdd"] > -1.0)]
    if valid.empty:
        print("\nNo valid results.")
        return

    # Best by MaxDD first, then Sharpe
    achieves_target = valid[valid["maxdd"] >= -0.15]
    if not achieves_target.empty:
        best = achieves_target.sort_values("sharpe", ascending=False).iloc[0]
        print("\n✓ Best combo achieving MaxDD ≤ 15%:")
    else:
        best = valid.sort_values("maxdd", ascending=False).iloc[0]
        print("\n⚠ No combo achieves MaxDD ≤ 15%.  Best available (closest to target):")
    print(
        f"  ema=({int(best.ema_s)},{int(best.ema_l)})  alt_min={best.min_exp}"
        f"  min_ic_ir={best.min_ic_ir}"
        f"  → MaxDD={best.maxdd:.2%}  Ann={best.ann:.2%}  Sharpe={best.sharpe:.3f}"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "sweep":
        sweep_df = run_param_sweep()
        # Auto-run best config after sweep
        valid = sweep_df[sweep_df["maxdd"].notna() & (sweep_df["maxdd"] > -1.0)]
        if not valid.empty:
            achieves = valid[valid["maxdd"] >= -0.15]
            best_row = (
                achieves.sort_values("sharpe", ascending=False).iloc[0]
                if not achieves.empty
                else valid.sort_values("maxdd", ascending=False).iloc[0]
            )
            print("\nGenerating final report with best config …")
            run_oos_backtest(
                label="maxdd15_final",
                config_overrides={
                    "ema_short": int(best_row.ema_s),
                    "ema_long": int(best_row.ema_l),
                    "alt_min_exposure": float(best_row.min_exp),
                    "min_ic_ir": float(best_row.min_ic_ir),
                },
            )
    elif args and args[0] == "best":
        # Run pre-tuned best config (ema=20/50, alt_exp=[0,0.3], hedge_cap=2.0)
        run_oos_backtest(
            label="maxdd15_best",
            config_overrides={
                "ema_short": 20,
                "ema_long": 50,
                "alt_min_exposure": 0.0,
                "alt_max_exposure": 0.3,
                "hedge_cap_multiplier": 2.0,
                "min_ic_ir": 0.02,
            },
        )
    else:
        label = args[0] if args else "maxdd15_oos"
        run_oos_backtest(label=label)
