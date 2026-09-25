"""Build an external behavior archive from the production factor base.

The archive contains standardized behavior signatures for every factor in
factor_base (precomputed factor-value parquets), aligned to the exact data
configuration (h5 / dates / tradable universe) of the GP mining run. Evolution
loads it through GP_EXTERNAL_BEHAVIOR_ARCHIVE and penalizes candidates whose
behavior is too similar to any production factor.

The npz format intentionally matches build_external_behavior_archive.py:
metadata_json (str), labels (str array), behavior_signatures (float16 2D),
phenotypes (float16 2D).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Prevent recursive archive replay while this utility instantiates the problem.
os.environ["GP_DISABLE_ARCHIVE_REPLAY"] = "1"
os.environ["GP_EXTERNAL_BEHAVIOR_ARCHIVE"] = ""

# The production GP run (run_gp_v3.sh, `python -m gp...` from the user root)
# resolves the `gp` package to /root/crypto-research/users/wesleywu/gp. Pin the
# same resolution here so the signature math matches the live engine byte for
# byte (the stale common/gp copy also requires numpy>=2, which this env lacks).
sys.path.insert(0, "/root/crypto-research/users/wesleywu")

from gp.minute_gp_system.engines.unified_v2.backend import set_backend
from gp.minute_gp_system.engines.unified_v2.behavior_signature import (
    BEHAVIOR_PATH_PERIODS,
    REGIME_FIELDS,
    build_behavior_context,
    build_behavior_signature_single,
)
from gp.minute_gp_system.engines.unified_v2.config import (
    CHUNK_PERIODS,
    INDICATOR_INDEX,
    TOP_QUANTILE,
)
from gp.minute_gp_system.engines.unified_v2.data_loader import CryptoDataLoader
from gp.minute_gp_system.engines.unified_v2.evolution import CryptoFactorProblemV2
from gp.minute_gp_system.engines.unified_v2.fitness import (
    _postprocess_single,
    _rank_ic_single,
    _resolve_ic_direction_single,
    sample_diversity_features,
    standardize_phenotypes,
)

DEFAULT_H5 = "/root/crypto-research/common/data/raw/h5/bybit_linear_1m_unified_gp_research.h5"
DEFAULT_MASK = "/root/crypto-research/common/data/raw/h5/bybit_tradable_mask_mining_gp_research.npy"
DEFAULT_FACTOR_DIR = "/root/crypto-research/common/factor_base/factor_values"
DEFAULT_REGISTRY = "/root/crypto-research/common/factor_base/factor_registry.json"
DEFAULT_OUTPUT = (
    "/root/crypto-research/users/wesleywu/gp/minute_gp_system/archives/"
    "factor_base_behavior_archive_20230101_20250630.npz"
)


def _regime_field_indices() -> np.ndarray:
    """Indicator fields the behavior context (regime matrix) needs."""
    return np.unique(
        np.array([INDICATOR_INDEX[name] for name in REGIME_FIELDS], dtype=np.intp)
    )


def _load_daily_matrix(
    parquet_path: Path,
    wanted_cols: list[str],
    active_days: np.ndarray,
    n_periods: int,
    n_coins: int,
    col_pos: dict[str, int],
    agg: str,
) -> tuple[np.ndarray | None, str | None]:
    """Read one factor parquet and align it to (n_periods, n_coins) float32.

    Rows are aligned by UTC day to the loader's active periods, columns by
    symbol to the loader's tradable-coin order. Missing entries stay NaN.
    Returns (matrix, skip_reason); exactly one of the two is not None.
    """
    try:
        df = pd.read_parquet(parquet_path, columns=wanted_cols)
    except Exception as exc:
        return None, f"read_error:{type(exc).__name__}"
    if df.empty:
        return None, "empty_frame"
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(idx))
        except Exception:
            return None, "bad_index"
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    if not idx.is_monotonic_increasing:
        order = np.argsort(idx.values, kind="stable")
        idx = idx[order]
        df = df.iloc[order]

    days = idx.values.astype("datetime64[D]")
    is_daily = bool(np.array_equal(idx.values.astype("datetime64[ns]"), days.astype("datetime64[ns]")))

    file_cols = list(df.columns)
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if is_daily:
        unique_days = days
        daily = arr
    else:
        if agg == "last":
            # Value at the last decision bar of each UTC day.
            change = np.flatnonzero(days[1:] != days[:-1]) + 1
            last_rows = np.concatenate([change - 1, [len(days) - 1]])
            unique_days = days[last_rows]
            daily = arr[last_rows]
        elif agg == "mean":
            daily_df = df.groupby(days).mean()
            unique_days = daily_df.index.values.astype("datetime64[D]")
            daily = daily_df.to_numpy(dtype=np.float32, copy=False)
        else:
            return None, f"bad_agg:{agg}"
    del df, arr

    rows = np.searchsorted(active_days, unique_days)
    rows = np.clip(rows, 0, n_periods - 1)
    matched = active_days[rows] == unique_days
    if not matched.any():
        return None, "no_period_overlap"

    out = np.full((n_periods, n_coins), np.nan, dtype=np.float32)
    src_rows = np.flatnonzero(matched)
    dst_rows = rows[matched]
    col_idx = np.array([col_pos[c] for c in file_cols], dtype=np.intp)
    out[np.ix_(dst_rows, col_idx)] = daily[src_rows][:, : len(file_cols)]
    return out, None


def _pairwise_value_corr(mats: list[np.ndarray], directions: list[float]) -> np.ndarray:
    n = len(mats)
    out = np.full((n, n), np.nan, dtype=np.float32)
    flats = []
    for mat, direction in zip(mats, directions):
        # float64: raw factor magnitudes (e.g. dollar-volume variances ~1e18)
        # overflow float32 when squared.
        flats.append((mat.astype(np.float64) * float(direction)).ravel())
    for i in range(n):
        for j in range(i, n):
            x, y = flats[i], flats[j]
            valid = np.isfinite(x) & np.isfinite(y)
            if int(valid.sum()) < 100:
                continue
            xv = x[valid] - float(x[valid].mean())
            yv = y[valid] - float(y[valid].mean())
            denom = float(np.sqrt(np.sum(xv * xv) * np.sum(yv * yv)))
            if denom > 1e-10:
                out[i, j] = out[j, i] = float(np.sum(xv * yv) / denom)
    return out


def build_archive(args: argparse.Namespace) -> None:
    t_start = time.time()
    from gp.minute_gp_system.engines.unified_v2 import evolution as _evo_mod

    print(f"[ENGINE] evolution module: {_evo_mod.__file__}", flush=True)
    factor_dir = Path(args.factor_values_dir)
    parquet_paths = sorted(factor_dir.glob("*.parquet"))
    if args.limit is not None:
        parquet_paths = parquet_paths[: int(args.limit)]
    if not parquet_paths:
        raise ValueError(f"no parquet files under {factor_dir}")

    set_backend(args.backend, args.gpu_id)
    loader = CryptoDataLoader(
        h5_path=args.h5_path,
        minutes_per_period=args.period_minutes,
        chunk_periods=args.chunk_periods,
        start_date=args.start_date,
        end_date=args.end_date,
        train_end_date=args.train_end_date or None,
        backend=args.backend,
        gpu_id=args.gpu_id,
        indicator_field_indices=_regime_field_indices(),
        tradable_mask_overlay_path=args.tradable_mask_overlay or None,
        fundamental_panel_path=args.fundamental_panel or None,
    )
    problem = CryptoFactorProblemV2(
        loader,
        backend=args.backend,
        gpu_id=args.gpu_id,
        eval_batch_size=args.batch_size,
        trading_cost=args.trading_cost,
    )
    behavior_context = build_behavior_context(
        problem.loader,
        problem.period_returns,
        problem.period_tradable_mask,
    )

    n_periods = int(loader.n_active_periods)
    n_coins = int(loader.n_tradable_coins)
    expected_dim = 2 * n_coins + BEHAVIOR_PATH_PERIODS + len(REGIME_FIELDS)
    print(
        f"[DIM] expected signature dim = 2*n_tradable_coins({n_coins}) "
        f"+ BEHAVIOR_PATH_PERIODS({BEHAVIOR_PATH_PERIODS}) "
        f"+ len(REGIME_FIELDS)({len(REGIME_FIELDS)}) = {expected_dim}",
        flush=True,
    )

    # dates_flat entries are 'YYYYMMDD' strings; np.datetime64 would parse a
    # bare 'YYYYMMDD' as a year, so insert dashes before casting.
    active_days = np.array(
        [
            f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:8]}"
            for d in loader.dates_flat[loader.active_period_indices]
        ],
        dtype="datetime64[D]",
    )
    tradable_symbols = [str(s) for s in loader.symbols[loader.coin_indices]]
    col_pos = {s: i for i, s in enumerate(tradable_symbols)}

    fitness_context = problem._fitness_context_is
    if fitness_context is None:
        raise ValueError("fitness context is None; cannot resolve ic_direction")
    idx_is, ret_slice, tradable_slice, ret_rank_cache = fitness_context

    labels: list[str] = []
    signatures: list[np.ndarray] = []
    phenotypes: list[np.ndarray] = []
    meta_rows: list[dict] = []
    skip_reasons: Counter = Counter()
    sanity_mats: list[np.ndarray] = []
    sanity_dirs: list[float] = []
    sanity_labels: list[str] = []

    for i, path in enumerate(parquet_paths, 1):
        factor_id = path.stem
        t0 = time.time()
        try:
            import pyarrow.parquet as pq

            file_cols = [
                c for c in pq.read_schema(path).names if c in col_pos
            ]
            if not file_cols:
                skip_reasons["no_symbol_overlap"] += 1
                print(f"[SKIP] {factor_id}: no_symbol_overlap", flush=True)
                continue
            mat, reason = _load_daily_matrix(
                path,
                file_cols,
                active_days,
                n_periods,
                n_coins,
                col_pos,
                args.agg,
            )
            if mat is None:
                skip_reasons[str(reason).split(":")[0]] += 1
                print(f"[SKIP] {factor_id}: {reason}", flush=True)
                continue
            period_has_data = np.isfinite(mat).any(axis=1)
            coverage = float(period_has_data.mean())
            if coverage < float(args.min_coverage):
                skip_reasons["low_coverage"] += 1
                print(f"[SKIP] {factor_id}: low_coverage ({coverage:.3f})", flush=True)
                continue

            factor_pp_is = _postprocess_single(mat[idx_is])
            ic_all = _rank_ic_single(
                factor_pp_is,
                ret_slice,
                tradable_slice=tradable_slice,
                ret_rank_cache=ret_rank_cache,
            )
            direction = float(_resolve_ic_direction_single(ic_all))

            sig, meta = build_behavior_signature_single(
                mat[behavior_context["idx"]],
                behavior_context["ret_sample"],
                behavior_context["tradable_sample"],
                behavior_context["regime_sample"],
                float(args.top_frac),
                direction,
            )
            pheno = sample_diversity_features(
                mat[None],
                tradable_mask=problem.period_tradable_mask,
            )[0]

            labels.append(factor_id)
            signatures.append(sig.astype(np.float32, copy=False))
            phenotypes.append(np.asarray(pheno, dtype=np.float32))
            meta_rows.append(
                {
                    "factor_id": factor_id,
                    "coverage": coverage,
                    "ic_direction": direction,
                    "behavior_key": meta.get("behavior_key"),
                    "effective_behavior_bars": meta.get("effective_behavior_bars"),
                    "dominant_regime": meta.get("dominant_regime"),
                }
            )
            if len(sanity_mats) < int(args.sanity_n):
                sanity_mats.append(mat)
                sanity_dirs.append(direction)
                sanity_labels.append(factor_id)
            del mat
        except Exception as exc:  # keep building; record and move on
            skip_reasons[f"error:{type(exc).__name__}"] += 1
            print(f"[WARN] {factor_id}: {exc}", file=sys.stderr, flush=True)
            continue
        finally:
            if i % 25 == 0 or i == len(parquet_paths):
                print(
                    f"[PROGRESS] {i}/{len(parquet_paths)} ok={len(labels)} "
                    f"skipped={sum(skip_reasons.values())} "
                    f"elapsed={time.time() - t_start:.0f}s",
                    flush=True,
                )
            _ = t0

    if not signatures:
        raise ValueError("no usable factor_base entries; archive would be empty")

    behavior = standardize_phenotypes(np.vstack(signatures))
    pheno_all = standardize_phenotypes(np.vstack(phenotypes))

    if behavior.shape[1] != expected_dim:
        raise ValueError(
            f"behavior dim mismatch: {behavior.shape[1]} != expected {expected_dim}"
        )

    registry_note = {}
    registry_path = Path(args.registry) if args.registry else None
    if registry_path is not None and registry_path.exists():
        try:
            registry = json.load(open(registry_path, "r", encoding="utf-8"))
            registry_note = {
                "registry_path": str(registry_path),
                "registry_entries": len(registry),
                "labels_in_registry": int(sum(1 for lb in labels if lb in registry)),
            }
        except Exception as exc:
            registry_note = {"registry_error": str(exc)}

    metadata = {
        "source": "factor_base",
        "factor_values_dir": str(factor_dir.resolve()),
        "h5_path": args.h5_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "train_end_date": args.train_end_date,
        "period_minutes": int(args.period_minutes),
        "chunk_periods": int(args.chunk_periods),
        "tradable_mask_overlay": loader.tradable_mask_overlay_path,
        "top_frac": float(args.top_frac),
        "agg": args.agg,
        "min_coverage": float(args.min_coverage),
        "n_tradable_coins": n_coins,
        "n_active_periods": n_periods,
        "n_entries": len(labels),
        "n_scanned": len(parquet_paths),
        "skip_reasons": dict(skip_reasons),
        "behavior_shape": list(behavior.shape),
        "labels": labels,
        "factor_meta": meta_rows,
        **registry_note,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
            labels=np.asarray(labels, dtype=np.str_),
            behavior_signatures=behavior.astype(np.float16),
            phenotypes=pheno_all.astype(np.float16),
        )
    os.replace(tmp, out_path)
    print(f"[OK] wrote {out_path} behavior_signatures={behavior.shape} entries={len(labels)}")
    print(f"[OK] skip_reasons={dict(skip_reasons)}")

    # --- verification report -------------------------------------------------
    print("[VERIFY] dim decomposition: "
          f"2*{n_coins} + {BEHAVIOR_PATH_PERIODS} + {len(REGIME_FIELDS)} = {expected_dim}; "
          f"archive dim = {behavior.shape[1]}")
    raw = np.vstack(signatures)
    for k in range(min(3, len(labels))):
        finite_ratio = float(np.isfinite(raw[k]).mean())
        raw_norm = float(np.linalg.norm(raw[k]))
        std_norm = float(np.linalg.norm(behavior[k]))
        print(
            f"[VERIFY] {labels[k]}: finite_ratio={finite_ratio:.4f} "
            f"raw_norm={raw_norm:.4f} standardized_norm={std_norm:.4f}"
        )

    if len(sanity_mats) >= 2:
        sig_corr = behavior[: len(sanity_mats)] @ behavior[: len(sanity_mats)].T
        val_corr = _pairwise_value_corr(sanity_mats, sanity_dirs)
        agree = 0
        total = 0
        for a in range(len(sanity_mats)):
            for b in range(a + 1, len(sanity_mats)):
                s, v = sig_corr[a, b], val_corr[a, b]
                if np.isfinite(s) and np.isfinite(v):
                    total += 1
                    if np.sign(s) == np.sign(v) or min(abs(s), abs(v)) < 0.05:
                        agree += 1
        print(f"[SANITY] labels={sanity_labels}")
        print(f"[SANITY] signature |corr| matrix:\n{np.round(np.abs(sig_corr), 3)}")
        print(f"[SANITY] direction-adjusted value corr matrix:\n{np.round(val_corr, 3)}")
        print(f"[SANITY] sign agreement: {agree}/{total} pairs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor_values_dir", default=DEFAULT_FACTOR_DIR)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--h5_path", default=DEFAULT_H5)
    parser.add_argument("--start_date", default="20230101")
    parser.add_argument("--end_date", default="20250630")
    parser.add_argument("--train_end_date", default="20241231",
                        help="IS cutoff for the fitness context used to resolve "
                             "ic_direction; must match the GP run")
    parser.add_argument("--period_minutes", type=int, default=1440)
    parser.add_argument("--chunk_periods", type=int, default=2)
    parser.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--trading_cost", type=float, default=0.0015)
    parser.add_argument("--tradable_mask_overlay", default=DEFAULT_MASK)
    parser.add_argument("--fundamental_panel", default="")
    parser.add_argument("--top_frac", type=float, default=TOP_QUANTILE)
    parser.add_argument("--agg", default="last", choices=["last", "mean"],
                        help="intraday-to-daily aggregation; 'last' takes the "
                             "last decision bar of each UTC day")
    parser.add_argument("--min_coverage", type=float, default=0.5,
                        help="skip factors with data on fewer than this "
                             "fraction of active periods")
    parser.add_argument("--sanity_n", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    build_archive(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
