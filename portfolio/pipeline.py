"""End-to-end portfolio pipeline."""
from __future__ import annotations

import importlib.util
import pathlib
from typing import Mapping

import pandas as pd

from portfolio.config import PortfolioConfig
from portfolio.factor_pool.loader import load_many
from portfolio.factor_pool.aligner import align_factors
from portfolio.factor_pool.label_builder import build_future_ret
from portfolio.factor_pool.cluster_selector import cluster_and_select
from portfolio.portfolio_builder.universe import build_universe
from portfolio.combiner.normalizer import normalize_cross_section
from portfolio.combiner.orthogonalizer import orthogonalize
from portfolio.combiner.ic_ir_weighter import compute_ic_ir_weights
from portfolio.combiner.synthesizer import synthesize
from portfolio.portfolio_builder.topn_equal import build_topn_weights
from portfolio.backtester.engine import run_backtest
from portfolio.reporter.html_renderer import render_html


def _load_factor_directions(factor_names: list[str]) -> dict[str, int]:
    """Load factor directions from factor_analyse/factor_config.py."""
    try:
        spec = importlib.util.spec_from_file_location(
            "factor_config",
            pathlib.Path(__file__).resolve().parents[1] / "factor_analyse" / "factor_config.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        config = getattr(mod, "FACTOR_CONFIG", {})
        return {
            name: config.get(name, {}).get("factor_direction", 1)
            for name in factor_names
        }
    except Exception:
        return {name: 1 for name in factor_names}


def run_pipeline(
    factor_paths: Mapping[str, str | pathlib.Path],
    kline_csv: str | pathlib.Path,
    universe_csv: str | pathlib.Path,
    config: PortfolioConfig | None = None,
    output_path: str | pathlib.Path | None = None,
) -> pd.Series:
    """Run the full portfolio pipeline end-to-end.

    Args:
        factor_paths: {factor_name: csv_path} mapping
        kline_csv: Path to kline data CSV with [symbol, date, close]
        universe_csv: Path to universe CSV with Decision_Window columns
        config: Pipeline configuration (defaults to PortfolioConfig())
        output_path: If provided, write HTML report here

    Returns:
        pd.Series of daily portfolio returns after fees.
    """
    if config is None:
        config = PortfolioConfig()

    factor_names = list(factor_paths.keys())

    # 1. Load & align factors
    factors = load_many(factor_paths)
    universe_by_date = build_universe(universe_csv)
    panel = align_factors(factors, universe_by_date)

    if panel.empty or not factor_names:
        return pd.Series(dtype=float)

    # 2. Load kline and build labels
    kline = pd.read_csv(kline_csv)
    kline["date"] = pd.to_datetime(kline["date"]).dt.normalize()
    label_df = build_future_ret(kline, config.label_period)

    # 3. Normalize
    directions = _load_factor_directions(factor_names)
    panel = normalize_cross_section(panel, factor_names, directions, config.winsorize_pct)

    # 3.5. Cluster factors to remove highly-correlated redundancy
    if config.cluster_threshold > 0 and len(factor_names) > 1:
        factor_names = cluster_and_select(panel, factor_names, config.cluster_threshold)

    # 4. Optionally orthogonalize
    if config.orthogonalize and len(factor_names) > 1:
        panel = orthogonalize(panel, factor_names)

    # 5. Merge with labels for IC computation
    panel_with_labels = panel.merge(
        label_df[["date", "instrument", "future_ret"]],
        on=["date", "instrument"],
        how="left",
    )

    # 6. Compute IC_IR weights
    ic_weights = compute_ic_ir_weights(
        panel_with_labels, factor_names,
        label_col="future_ret",
        window=config.ic_window,
    )

    # 7. Synthesize composite scores
    scores = synthesize(panel, ic_weights, factor_names)

    # 8. Build top-N weights (only on rebalance days)
    portfolio_weights = build_topn_weights(
        scores, universe_by_date, config.top_n,
        rebalance_period=config.rebalance_period,
    )

    # 9. Run backtest
    returns = run_backtest(kline, portfolio_weights, config.fee_rate)

    # 10. Compute benchmarks (BTC buy-and-hold, Equal-Weight-All)
    benchmarks: dict[str, "pd.Series"] = {}
    btc_rows = kline[kline["symbol"].str.upper() == "BTCUSDT"].copy()
    if not btc_rows.empty:
        btc_rows = btc_rows.sort_values("date")
        btc_rows["ret"] = btc_rows["close"].pct_change()
        btc_ret = btc_rows.set_index("date")["ret"].dropna()
        if not returns.empty:
            btc_ret = btc_ret.reindex(returns.index).dropna()
        benchmarks["BTC Buy&Hold"] = btc_ret

    # 11. Optionally render HTML report
    if output_path is not None:
        render_html(returns, output_path, benchmarks=benchmarks)

    return returns


def run_hedged_pipeline(
    factor_paths: Mapping[str, str | pathlib.Path],
    kline_csv: str | pathlib.Path,
    universe_csv: str | pathlib.Path,
    config: "PortfolioConfig | None" = None,
    output_path: "str | pathlib.Path | None" = None,
) -> "pd.Series":
    """Run the hedged MaxDD-15 portfolio pipeline end-to-end.

    Flow:
      1. Load factors, kline, universe (same as long-only pipeline)
      2. Align + normalize + orthogonalize factors
      3. IS期预筛 factors by |IC_IR| >= min_ic_ir
      4. Compute rolling IC_IR weights on retained factors
      5. Synthesize composite score -> Top-N weights (weekly)
      6. Compute BTC trend T (daily)
      7. Compute rolling coin betas (daily)
      8. Per-date: portfolio_beta @ holdings, (alt_exp, hedge) = controller(T, beta)
      9. Smooth alt_exp with EMA
     10. Run hedged backtest -> daily returns
    """
    from portfolio.combiner.is_oos_filter import filter_factors_by_is_ic_ir
    from portfolio.regime.btc_trend import compute_btc_trend_score
    from portfolio.portfolio_builder.beta_estimator import (
        compute_coin_betas, compute_portfolio_beta
    )
    from portfolio.sizing.exposure_controller import (
        compute_exposure_and_hedge, smooth_exposure_series
    )
    from portfolio.backtester.engine import run_hedged_backtest

    if config is None:
        config = PortfolioConfig()
    factor_names = list(factor_paths.keys())

    # 1. Load & align
    factors = load_many(factor_paths)
    universe_by_date = build_universe(universe_csv)
    panel = align_factors(factors, universe_by_date)
    if panel.empty or not factor_names:
        return pd.Series(dtype=float)

    # 2. Load kline
    kline = pd.read_csv(kline_csv)
    kline["date"] = pd.to_datetime(kline["date"]).dt.normalize()
    label_df = build_future_ret(kline, config.label_period)

    # 3. Initial normalize (all directions +1 until IS filter)
    directions_init = {n: 1 for n in factor_names}
    panel = normalize_cross_section(panel, factor_names, directions_init, config.winsorize_pct)

    # 4. Merge labels for IS filter
    panel_labeled = panel.merge(
        label_df[["date", "instrument", "future_ret"]],
        on=["date", "instrument"], how="left",
    )

    # 5. IS/OOS filter
    selected_dirs = filter_factors_by_is_ic_ir(
        panel_labeled, factor_names,
        label_col="future_ret",
        is_end_date=config.is_end_date,
        min_ic_ir=config.min_ic_ir,
    )
    if not selected_dirs:
        return pd.Series(dtype=float)
    selected_factors = list(selected_dirs.keys())

    # 6. Re-normalize with corrected directions
    panel = normalize_cross_section(panel, selected_factors, selected_dirs, config.winsorize_pct)
    if config.orthogonalize and len(selected_factors) > 1:
        panel = orthogonalize(panel, selected_factors)
    panel_labeled = panel.merge(
        label_df[["date", "instrument", "future_ret"]],
        on=["date", "instrument"], how="left",
    )

    # 7. IC_IR weights
    ic_weights = compute_ic_ir_weights(
        panel_labeled, selected_factors,
        label_col="future_ret", window=config.ic_window,
    )

    # 8. Synthesize composite score
    scores = synthesize(panel, ic_weights, selected_factors)

    # 9. Top-N weekly weights
    portfolio_weights = build_topn_weights(
        scores, universe_by_date, config.top_n,
        rebalance_period=config.rebalance_period,
    )
    if not portfolio_weights:
        return pd.Series(dtype=float)

    # 10. BTC trend T
    T = compute_btc_trend_score(
        kline,
        ema_short=config.ema_short,
        ema_long=config.ema_long,
        zscore_window=config.zscore_window,
        ema_ratio_saturation=config.ema_ratio_saturation,
        zscore_saturation=config.zscore_saturation,
        smoothing_span=config.t_smoothing_span,
    )

    # 11. Coin betas
    betas_df = compute_coin_betas(kline, window=config.beta_window)

    # 12. Per-date exposure/hedge
    all_dates = sorted(kline["date"].unique())
    all_dates = [d for d in all_dates if d >= min(portfolio_weights.keys())]
    sorted_reb = sorted(portfolio_weights.keys())
    reb_idx = 0

    raw_alt: dict = {}
    raw_hedge: dict = {}
    current_holdings: dict[str, float] = {}
    for d in all_dates:
        while reb_idx < len(sorted_reb) and sorted_reb[reb_idx] <= d:
            current_holdings = portfolio_weights[sorted_reb[reb_idx]]
            reb_idx += 1
        if not current_holdings:
            continue
        # Get T for this date, default 0 if warmup period
        try:
            T_d = float(T.loc[d]) if d in T.index and not pd.isna(T.loc[d]) else 0.0
        except (KeyError, TypeError):
            T_d = 0.0
        beta_port = compute_portfolio_beta(
            betas_df, current_holdings, d, beta_prior=config.beta_prior
        )
        alt_e, hedge = compute_exposure_and_hedge(
            T=T_d, beta_port=beta_port,
            alt_min=config.alt_min_exposure,
            alt_max=config.alt_max_exposure,
            hedge_cap_mult=config.hedge_cap_multiplier,
        )
        raw_alt[d] = alt_e
        raw_hedge[d] = hedge

    if not raw_alt:
        return pd.Series(dtype=float)

    raw_alt_s = pd.Series(raw_alt).sort_index()
    raw_hedge_s = pd.Series(raw_hedge).sort_index()
    alt_exp_s = smooth_exposure_series(raw_alt_s, alpha=config.alt_exposure_ema_alpha)

    # 13. Hedged backtest
    returns = run_hedged_backtest(
        kline, portfolio_weights, alt_exp_s, raw_hedge_s,
        alt_fee=config.alt_fee_rate,
        perp_fee=config.perp_fee_rate,
        funding_annual=config.funding_rate_annual,
    )

    # 14. BTC benchmark + optional report
    benchmarks: dict[str, pd.Series] = {}
    btc_rows = kline[kline["symbol"].str.upper() == "BTCUSDT"].copy()
    if not btc_rows.empty:
        btc_rows = btc_rows.sort_values("date")
        btc_rows["ret"] = btc_rows["close"].pct_change()
        btc_ret = btc_rows.set_index("date")["ret"].dropna()
        if not returns.empty:
            btc_ret = btc_ret.reindex(returns.index).dropna()
        benchmarks["BTC Buy&Hold"] = btc_ret
    if output_path is not None:
        render_html(returns, output_path, benchmarks=benchmarks)
    return returns


def run_pathB_pipeline(
    factor_paths: Mapping[str, str | pathlib.Path],
    kline_csv: str | pathlib.Path,
    universe_csv: str | pathlib.Path,
    funding_dir: str | pathlib.Path,
    config: PortfolioConfig | None = None,
    output_path: str | pathlib.Path | None = None,
    is_end_date: str = "2023-12-31",
    oos_start: str = "2024-01-01",
    oos_end: str | None = None,
) -> dict:
    """Execute Path B: BTC long (dynamic 40-70%) + Alt short overlay.
    
    Returns dict with keys: nav (Series), shorts (dict), btc_weights (Series), kept_factors (list).
    """
    import glob
    from portfolio.combiner.is_oos_filter import filter_factors_by_is_ic_ir
    from portfolio.universe.rank_filter import filter_universe_by_rank
    from portfolio.regime.funding_filter import compute_funding_allowed_mask
    from portfolio.regime.btc_trend import compute_btc_trend_score
    from portfolio.sizing.btc_core_sizer import btc_core_weight
    from portfolio.risk.squeeze_stop import compute_squeeze_flags
    from portfolio.combiner.short_side_scorer import select_short_bottom_n
    from portfolio.backtester.engine import run_btc_core_short_backtest

    if config is None:
        config = PortfolioConfig()
    
    factor_names = list(factor_paths.keys())
    
    # 1. Load & align factors
    factors = load_many(factor_paths)
    universe_by_date = build_universe(universe_csv)
    panel = align_factors(factors, universe_by_date)
    
    if panel.empty or not factor_names:
        return {
            "nav": pd.Series(dtype=float),
            "shorts": {},
            "btc_weights": pd.Series(dtype=float),
            "kept_factors": []
        }
    
    # 2. Load kline and build labels
    kline = pd.read_csv(kline_csv)
    kline["date"] = pd.to_datetime(kline["date"]).dt.normalize()
    label_df = build_future_ret(kline, config.label_period)
    
    panel_labeled = panel.merge(
        label_df[["date", "instrument", "future_ret"]],
        on=["date", "instrument"],
        how="left"
    )
    
    # 3. IS-IC-IR filter
    directions = filter_factors_by_is_ic_ir(
        panel_labeled, factor_names,
        label_col="future_ret",
        is_end_date=is_end_date,
        min_ic_ir=getattr(config, "min_ic_ir", 0.05)
    )
    kept_factors = list(directions.keys())
    if not kept_factors:
        return {
            "nav": pd.Series(dtype=float),
            "shorts": {},
            "btc_weights": pd.Series(dtype=float),
            "kept_factors": []
        }
    
    # 4. Load and aggregate funding data
    funding_files = sorted(glob.glob(str(pathlib.Path(funding_dir) / "*_funding.csv")))
    funding_raw = (
        pd.concat([pd.read_csv(f) for f in funding_files], ignore_index=True)
        if funding_files else
        pd.DataFrame(columns=["symbol", "fundingTime", "fundingRate", "fundingTimeMs"])
    )
    
    # Daily aggregation: sum funding rates within each UTC day per symbol
    if not funding_raw.empty:
        fr = funding_raw.copy()
        fr["date"] = pd.to_datetime(fr["fundingTime"], utc=True).dt.tz_convert(None).dt.strftime("%Y-%m-%d")
        fr["fundingRate"] = pd.to_numeric(fr["fundingRate"], errors="coerce")
        funding_daily = (
            fr.groupby(["date", "symbol"], as_index=False)["fundingRate"]
            .sum()
            .rename(columns={"symbol": "instrument", "fundingRate": "funding_daily"})
        )
    else:
        funding_daily = pd.DataFrame(columns=["date", "instrument", "funding_daily"])
    
    # 5. Universe rank filter
    mc = pd.read_csv(universe_csv)
    mc_filt = filter_universe_by_rank(mc, config.universe_rank_min, config.universe_rank_max)
    mc_filt["date"] = pd.to_datetime(mc_filt["Date"]).dt.normalize()
    allowed_by_date = (
        mc_filt.groupby("date")["Trading_Pairs"]
        .apply(lambda s: set(str(x) for x in s.dropna() if str(x).endswith("USDT")))
        .to_dict()
    )
    
    # 6. Funding P80 filter (OOS only)
    oos_end_ts = pd.Timestamp(oos_end) if oos_end else pd.Timestamp(kline["date"].max())
    oos_dates = pd.DatetimeIndex([
        d for d in sorted(kline["date"].unique())
        if pd.Timestamp(oos_start) <= d <= oos_end_ts
    ])
    
    if not funding_raw.empty and len(oos_dates) > 0:
        fmask = compute_funding_allowed_mask(
            funding_raw, oos_dates,
            lookback_days=getattr(config, "funding_lookback_days", 30),
            percentile=config.funding_filter_percentile
        )
        fmask["date"] = pd.to_datetime(fmask["date"]).dt.normalize()
        fmask["instrument"] = fmask["instrument"].astype(str) + "USDT"
    else:
        fmask = pd.DataFrame(columns=["date", "instrument", "allowed_short"])
    
    # 7. Merge panel with masks (OOS only)
    panel_oos = panel[
        (panel["date"] >= pd.Timestamp(oos_start)) & (panel["date"] <= oos_end_ts)
    ].copy()
    
    if panel_oos.empty:
        return {
            "nav": pd.Series(dtype=float),
            "shorts": {},
            "btc_weights": pd.Series(dtype=float),
            "kept_factors": kept_factors
        }
    
    panel_oos = panel_oos.merge(
        fmask[["date", "instrument", "allowed_short"]],
        on=["date", "instrument"],
        how="left"
    )
    panel_oos["allowed_short"] = panel_oos["allowed_short"].fillna(False)
    
    # Universe rank filter
    panel_oos["in_universe"] = panel_oos.apply(
        lambda r: r["instrument"] in allowed_by_date.get(r["date"], set()),
        axis=1
    )
    panel_oos["allowed"] = panel_oos["allowed_short"] & panel_oos["in_universe"]
    
    # 8. Short selection (bottom-N)
    shorts = select_short_bottom_n(
        panel_oos, factors=kept_factors, directions=directions,
        n=config.short_top_n, allowed_col="allowed"
    )
    
    # 9. Apply squeeze stop
    squeeze = compute_squeeze_flags(
        kline.rename(columns={"symbol": "instrument"})[["date", "instrument", "close", "volume"]],
        ret_threshold=config.squeeze_stop_return,
        vol_multiple=3.0,
        vol_window=config.squeeze_stop_window
    )
    squeeze["date_str"] = pd.to_datetime(squeeze["date"]).dt.strftime("%Y-%m-%d")
    flagged_by_date = (
        squeeze[squeeze["squeeze_flag"]]
        .groupby("date_str")["instrument"]
        .apply(set)
        .to_dict()
    )
    for d_str, lst in list(shorts.items()):
        flagged = flagged_by_date.get(d_str, set())
        shorts[d_str] = [s for s in lst if s not in flagged]
    
    # 10. BTC trend + sizer
    T = compute_btc_trend_score(kline)
    btc_weights = btc_core_weight(T, w_min=config.btc_base_min, w_max=config.btc_base_max)
    
    # 11. Prices_wide for engine
    prices_wide = kline.pivot_table(index="date", columns="symbol", values="close").sort_index()
    prices_wide = prices_wide.loc[
        (prices_wide.index >= pd.Timestamp(oos_start)) & (prices_wide.index <= oos_end_ts)
    ]
    btc_weights_oos = btc_weights.reindex(prices_wide.index).ffill().fillna(config.btc_base_min)
    
    # 12. Run engine
    nav = run_btc_core_short_backtest(
        prices=prices_wide,
        btc_weights=btc_weights_oos,
        shorts_by_date=shorts,
        funding=funding_daily,
        alt_short_exposure=config.alt_short_exposure_max,
        fee_rate=getattr(config, "alt_fee_rate", 0.001),
        btc_symbol="BTCUSDT"
    )
    
    return {
        "nav": nav,
        "shorts": shorts,
        "btc_weights": btc_weights_oos,
        "kept_factors": kept_factors
    }
