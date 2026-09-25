#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H5 = PROJECT_ROOT / "data/crypto_quant.h5"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports/barra"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class BarraConfig:
    profile_id: str = "perp_1d"
    signal_freq: str = "1D"
    min_count: int = 20
    winsor_q: float = 0.01
    beta_window: int = 60
    vol_window: int = 20
    liq_window: int = 20
    size_window: int = 60
    funding_window: int = 3
    include_crypto_optional: bool = False

    def __post_init__(self):
        if self.profile_id != "perp_1d" or self.signal_freq not in {"D", "1D", "1d"}:
            raise ValueError("Only UTC daily perp_1d inputs are supported")
        for name in ("min_count", "beta_window", "vol_window", "liq_window", "size_window", "funding_window"):
            value = getattr(self, name)
            if type(value) is not int or value < (2 if name == "min_count" else 1):
                raise ValueError(f"{name} must be a positive integer (min_count >= 2)")
        if not np.isfinite(self.winsor_q) or not 0 <= self.winsor_q < 0.5:
            raise ValueError("winsor_q must be in [0, 0.5)")


def _data_provider(h5_path: str | Path, *, as_of=None):
    from factor_common.data_provider import DataProvider
    return DataProvider(h5_path, as_of=as_of)


def _profile_freq(profile_id: str) -> str:
    if profile_id != "perp_1d":
        raise ValueError(f"unknown profile_id: {profile_id}")
    return "1D"


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True, errors="raise")).tz_convert(None)
    return idx


def resample_alpha_to_profile(factor_value: pd.DataFrame, cfg: BarraConfig) -> pd.DataFrame:
    if not isinstance(factor_value, pd.DataFrame) or factor_value.empty:
        raise ValueError("factor_value must be a non-empty DataFrame")
    fv = factor_value.copy()
    if {"date", "instrument", "factor"}.issubset(fv.columns):
        fv["date"] = _normalize_index(fv["date"])
        fv["instrument"] = fv["instrument"].astype(str)
        if fv.duplicated(["date", "instrument"]).any():
            raise ValueError("duplicate date/instrument keys")
        fv = fv.pivot(index="date", columns="instrument", values="factor")
    elif not isinstance(fv.index, pd.DatetimeIndex):
        raise ValueError("wide factor input requires a DatetimeIndex")
    fv.index = _normalize_index(fv.index)
    fv.columns = fv.columns.map(str)
    if fv.index.hasnans or not fv.index.equals(fv.index.normalize()):
        raise ValueError("factor dates must be UTC midnight daily timestamps")
    if fv.index.has_duplicates or fv.columns.has_duplicates:
        raise ValueError("duplicate dates or instruments")
    fv = fv.apply(pd.to_numeric, errors="raise").astype(float).sort_index()
    fv.index.name = "date"
    fv.columns.name = "instrument"
    return fv.replace([np.inf, -np.inf], np.nan)


def winsor_zscore(s: pd.Series, q: float = 0.01) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    if q > 0:
        lo, hi = x.quantile([q, 1.0 - q])
        x = x.clip(lo, hi)
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def _rolling_beta(ret: pd.DataFrame, market: pd.Series, window: int) -> pd.DataFrame:
    min_periods = min(window, max(10, window // 2))
    paired_market = pd.DataFrame(
        np.broadcast_to(market.to_numpy()[:, None], ret.shape),
        index=ret.index, columns=ret.columns,
    ).where(ret.notna())
    cov = ret.rolling(window, min_periods=min_periods).cov(paired_market, ddof=0)
    var = paired_market.rolling(window, min_periods=min_periods).var(ddof=0)
    return (cov / var.where(var > 1e-15)).replace([np.inf, -np.inf], np.nan)


def _resample(dp: Any, field: str, freq: str, method: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = dp.get_single_data(field, start=start, end=end)
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame = frame.groupby(frame.index).last() if method == "last" else frame.groupby(frame.index).sum()
    frame.columns = [str(c) for c in frame.columns]
    return frame.replace([np.inf, -np.inf], np.nan)


def _resample_universe(dp: Any, freq: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    universe = dp.get_universe(start=start, end=end)
    universe.index = pd.DatetimeIndex(universe.index).normalize()
    universe = universe.groupby(universe.index).last()
    universe.columns = [str(c) for c in universe.columns]
    return universe.fillna(False).astype(bool)


def build_barra_exposures(
    alpha_daily: pd.DataFrame,
    *,
    h5_path: str | Path = DEFAULT_H5,
    cfg: BarraConfig,
    as_of=None,
    provider=None,
) -> dict[str, pd.DataFrame]:
    start = pd.Timestamp(alpha_daily.index.min())
    end = pd.Timestamp(alpha_daily.index.max())
    lookback = max(cfg.beta_window + cfg.vol_window, cfg.liq_window, cfg.size_window, cfg.funding_window, 20)
    load_start = start - pd.Timedelta(days=lookback + 2)
    load_end = end

    dp = provider if provider is not None else _data_provider(h5_path, as_of=as_of or end)
    close = _resample(dp, "close", cfg.signal_freq, "last", load_start, load_end)
    dollar_volume = _resample(dp, "quote_volume", cfg.signal_freq, "last", load_start, load_end)
    dollar_volume = dollar_volume.where(dollar_volume > 0)

    funding_raw = _resample(dp, "funding", cfg.signal_freq, "last", load_start, load_end)

    close = close.where(close > 0)
    ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    universe = dp.get_universe(start=load_start, end=load_end)
    market = ret.where(universe).mean(axis=1, skipna=True)
    beta = _rolling_beta(ret, market, cfg.beta_window)
    resid_ret = ret - beta.mul(market, axis=0)

    styles: dict[str, pd.DataFrame] = {
        "size": np.log(dollar_volume.rolling(cfg.size_window, min_periods=min(cfg.size_window, max(10, cfg.size_window // 2))).mean()),
        "liquidity": np.log(dollar_volume.rolling(cfg.liq_window, min_periods=min(cfg.liq_window, max(5, cfg.liq_window // 2))).mean()),
        "beta": beta,
        "momentum_5d": close / close.shift(5) - 1.0,
        "momentum_20d": close / close.shift(20) - 1.0,
        "reversal_1d": -ret,
        "volatility": ret.rolling(cfg.vol_window, min_periods=min(cfg.vol_window, max(5, cfg.vol_window // 2))).std(ddof=0) * math.sqrt(365),
        "residual_volatility": resid_ret.rolling(
            cfg.vol_window, min_periods=min(cfg.vol_window, max(5, cfg.vol_window // 2))
        ).std(ddof=0)
        * math.sqrt(365),
        "funding": funding_raw.rolling(cfg.funding_window, min_periods=1).mean(),
    }

    if cfg.include_crypto_optional:
        styles.update(_optional_crypto_exposures(dp, cfg, load_start, load_end))

    out = {}
    for name, frame in styles.items():
        trimmed = frame.where(universe).loc[(frame.index >= start) & (frame.index <= end)]
        out[name] = trimmed.replace([np.inf, -np.inf], np.nan).astype("float64")
    return out


def _optional_crypto_exposures(
    dp: Any,
    cfg: BarraConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    available = set(dp.list_datas())
    missing = sorted({"open_interest", "premium_close"} - available)
    if missing:
        warnings.warn(f"Optional Barra fields unavailable; skipped: {', '.join(missing)}", UserWarning)
    if "open_interest" in available:
        oi = _resample(dp, "open_interest", cfg.signal_freq, "last", start, end)
        log_oi = np.log(oi.where(oi > 0))
        out["open_interest"] = log_oi
        out["oi_change_5d"] = log_oi - log_oi.shift(5)
    if "premium_close" in available:
        out["premium"] = _resample(dp, "premium_close", cfg.signal_freq, "last", start, end)
    return out


def align_inputs(
    alpha_daily: pd.DataFrame,
    styles: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not styles:
        raise ValueError("at least one style is required")
    common_symbols = set(map(str, alpha_daily.columns))
    for frame in styles.values():
        common_symbols &= set(map(str, frame.columns))
    symbols = sorted(common_symbols)
    if not symbols:
        raise ValueError("alpha and Barra exposures have no common symbols")

    common_dates = set(alpha_daily.index)
    for frame in styles.values():
        common_dates &= set(frame.index)
    dates = pd.DatetimeIndex(sorted(common_dates))
    if len(dates) == 0:
        raise ValueError("alpha and Barra exposures have no common dates")

    alpha = alpha_daily.reindex(index=dates, columns=symbols)
    aligned_styles = {k: v.reindex(index=dates, columns=symbols) for k, v in styles.items()}
    return alpha, aligned_styles


def analyze_exposures(
    alpha_daily: pd.DataFrame,
    styles: dict[str, pd.DataFrame],
    *,
    cfg: BarraConfig,
) -> dict[str, pd.DataFrame]:
    alpha, styles = align_inputs(alpha_daily, styles)
    style_names = list(styles)

    corr_rows: list[dict[str, Any]] = []
    reg_rows: list[dict[str, Any]] = []
    residual = pd.DataFrame(index=alpha.index, columns=alpha.columns, dtype="float64")

    for dt in alpha.index:
        y = winsor_zscore(alpha.loc[dt], cfg.winsor_q)
        x_by_style = {name: winsor_zscore(styles[name].loc[dt].where(y.notna()), cfg.winsor_q) for name in style_names}

        for name, x in x_by_style.items():
            pair = pd.concat([y.rename("alpha"), x.rename("style")], axis=1).dropna()
            row = {"date": dt, "style": name, "n": int(len(pair)), "pearson": np.nan, "spearman": np.nan}
            if len(pair) >= cfg.min_count:
                row["pearson"] = float(pair["alpha"].corr(pair["style"], method="pearson"))
                row["spearman"] = float(pair["alpha"].corr(pair["style"], method="spearman"))
            corr_rows.append(row)

        xdf = pd.DataFrame(x_by_style)
        reg_df = pd.concat([y.rename("alpha"), xdf], axis=1).dropna()
        reg_row = {"date": dt, "n": int(len(reg_df)), "alpha_n": int(y.notna().sum()),
                   "r2": np.nan, "rank": np.nan, "condition_number": np.nan,
                   "status": "insufficient_data"}
        for name in style_names:
            reg_row[f"beta_{name}"] = np.nan

        if len(reg_df) >= max(cfg.min_count, len(style_names) + 3):
            yv = reg_df["alpha"].to_numpy(float)
            xv = np.column_stack([np.ones(len(reg_df)), reg_df[style_names].to_numpy(float)])
            coef, _, rank, singular = np.linalg.lstsq(xv, yv, rcond=None)
            reg_row["rank"] = int(rank)
            reg_row["condition_number"] = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
            if rank < xv.shape[1]:
                reg_row["status"] = "rank_deficient"
                reg_rows.append(reg_row)
                continue
            reg_row["status"] = "ok"
            fitted = xv @ coef
            ss_res = float(np.sum((yv - fitted) ** 2))
            ss_tot = float(np.sum((yv - yv.mean()) ** 2))
            reg_row["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
            for i, name in enumerate(style_names, start=1):
                reg_row[f"beta_{name}"] = float(coef[i])
            resid = pd.Series(yv - fitted, index=reg_df.index)
            residual.loc[dt, resid.index] = resid
        reg_rows.append(reg_row)

    corr = pd.DataFrame(corr_rows)
    reg = pd.DataFrame(reg_rows)
    summary = summarize(corr, reg, style_names)
    return {
        "alpha_daily": alpha,
        "daily_style_corr": corr,
        "daily_barra_regression": reg,
        "style_summary": summary,
        "alpha_barra_residual": residual,
    }


def _t_stat(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.nan
    return float(x.mean() / sd * math.sqrt(len(x)))


def summarize(corr: pd.DataFrame, reg: pd.DataFrame, style_names: list[str]) -> pd.DataFrame:
    rows = []
    for style in style_names:
        c = corr[corr["style"] == style]
        beta_col = f"beta_{style}"
        rows.append(
            {
                "style": style,
                "days": int(c["pearson"].notna().sum()),
                "pearson_mean": float(c["pearson"].mean()),
                "pearson_abs_mean": float(c["pearson"].abs().mean()),
                "pearson_t": _t_stat(c["pearson"]),
                "spearman_mean": float(c["spearman"].mean()),
                "spearman_abs_mean": float(c["spearman"].abs().mean()),
                "spearman_t": _t_stat(c["spearman"]),
                "beta_mean": float(reg[beta_col].mean()) if beta_col in reg else np.nan,
                "beta_t": _t_stat(reg[beta_col]) if beta_col in reg else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_abs_mean", ascending=False)


def load_universe_for_alpha(
    alpha_daily: pd.DataFrame,
    *,
    h5_path: str | Path = DEFAULT_H5,
    cfg: BarraConfig,
    provider=None,
) -> pd.DataFrame:
    start = pd.Timestamp(alpha_daily.index.min())
    end = pd.Timestamp(alpha_daily.index.max())
    dp = provider if provider is not None else _data_provider(h5_path, as_of=end)
    universe = _resample_universe(dp, cfg.signal_freq, start, end)
    return universe.reindex(index=alpha_daily.index, columns=alpha_daily.columns).fillna(False).astype(bool)


def analyze_evaluate_result(
    result: dict[str, Any],
    *,
    h5_path: str | Path = DEFAULT_H5,
    cfg: BarraConfig | None = None,
    as_of=None,
) -> dict[str, pd.DataFrame]:
    cfg = cfg or BarraConfig()
    if "factor_value" not in result:
        raise KeyError("result must contain result['factor_value']")
    alpha_daily = resample_alpha_to_profile(result["factor_value"], cfg)
    if as_of is not None:
        alpha_daily = alpha_daily.loc[:pd.Timestamp(as_of)]
    if alpha_daily.empty:
        raise ValueError("no alpha dates at or before cutoff")
    dp = _data_provider(h5_path, as_of=as_of or alpha_daily.index.max())
    universe = load_universe_for_alpha(alpha_daily, h5_path=h5_path, cfg=cfg, provider=dp)
    alpha_daily = alpha_daily.where(universe)
    styles = build_barra_exposures(alpha_daily, h5_path=h5_path, cfg=cfg, provider=dp)
    return analyze_exposures(alpha_daily, styles, cfg=cfg)


def write_outputs(results: dict[str, pd.DataFrame], out_dir: str | Path, *, metadata: dict[str, Any]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results["daily_style_corr"].to_csv(out / "daily_style_corr.csv", index=False)
    results["daily_barra_regression"].to_csv(out / "daily_barra_regression.csv", index=False)
    results["style_summary"].to_csv(out / "style_summary.csv", index=False)
    results["alpha_daily"].to_parquet(out / "alpha_daily.parquet")
    results["alpha_barra_residual"].to_parquet(out / "alpha_barra_residual.parquet")
    residual = results["alpha_barra_residual"].rename_axis(index="date", columns="instrument")
    residual.stack().dropna().rename("factor").reset_index().to_parquet(out / "alpha_barra_residual_long.parquet", index=False)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")


def analyze_and_write(
    result: dict[str, Any],
    *,
    out_dir: str | Path | None = None,
    h5_path: str | Path = DEFAULT_H5,
    cfg: BarraConfig | None = None,
    label: str = "alpha",
) -> dict[str, pd.DataFrame]:
    cfg = cfg or BarraConfig()
    before = Path(h5_path).stat()
    results = analyze_evaluate_result(result, h5_path=h5_path, cfg=cfg)
    after = Path(h5_path).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("H5 changed during analysis; rerun with a stable snapshot")
    reg = results["daily_barra_regression"]
    metadata = {
        "label": label,
        "h5_path": str(h5_path),
        "config": asdict(cfg),
        "dates": [
            str(results["alpha_daily"].index.min().date()),
            str(results["alpha_daily"].index.max().date()),
        ],
        "symbols": int(results["alpha_daily"].shape[1]),
        "valid_regression_days": int(reg["status"].eq("ok").sum()),
        "regression_status_counts": reg["status"].value_counts().to_dict(),
        "source_stat": {"size": before.st_size, "mtime_ns": before.st_mtime_ns},
        "funding_convention": "rolling mean of daily settlement-rate mean",
        "size_convention": "log rolling mean quote_volume; turnover proxy, not market capitalization",
        "universe": "historical_top50; same-day membership for alpha, styles and equal-weight market",
        "optional_fields_skipped": ["open_interest", "premium_close"] if cfg.include_crypto_optional else [],
        "causality": "not_verified_for_this_input; cutoff checks are separate; external alpha provenance is caller responsibility",
    }
    destination = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR / Path(label).name
    write_outputs(results, destination, metadata=metadata)
    return results


def _load_result_pickle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"pickle must contain result dict, got {type(obj)!r}")
    return obj


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Barra correlation/exposure analysis for fm.evaluate result dict.")
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--result-pkl", help="Trusted local pickle containing fm.evaluate(...) result dict.")
    inputs.add_argument("--factor-parquet", help="Daily factor Parquet (long or wide).")
    ap.add_argument("--out-dir")
    ap.add_argument("--h5", default=str(DEFAULT_H5))
    ap.add_argument("--profile-id", default="perp_1d")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--winsor-q", type=float, default=0.01)
    ap.add_argument("--include-crypto-optional", action="store_true")
    return ap.parse_args()


def main() -> int:
    ns = parse_args()
    cfg = BarraConfig(
        profile_id=ns.profile_id,
        signal_freq=_profile_freq(ns.profile_id),
        min_count=ns.min_count,
        winsor_q=ns.winsor_q,
        include_crypto_optional=ns.include_crypto_optional,
    )
    result = _load_result_pickle(ns.result_pkl) if ns.result_pkl else {"factor_value": pd.read_parquet(ns.factor_parquet)}
    result["factor_value"] = resample_alpha_to_profile(result["factor_value"], cfg).loc[ns.start:ns.end]
    label = Path(ns.result_pkl or ns.factor_parquet).stem
    destination = ns.out_dir or DEFAULT_OUT_DIR / label
    outputs = analyze_and_write(result, out_dir=destination, h5_path=ns.h5, cfg=cfg, label=label)
    reg = outputs["daily_barra_regression"]
    print(f"wrote outputs to {destination}")
    print(f"valid_regression_days={reg['status'].eq('ok').sum()}/{len(reg)}")
    print(f"dates={len(reg)} symbols={outputs['alpha_daily'].shape[1]} mean_r2={reg['r2'].mean():.4f}")
    print(outputs["style_summary"].to_string(index=False))
    return 0 if reg["status"].eq("ok").any() else 2


if __name__ == "__main__":
    raise SystemExit(main())
