"""Build a residual-IC basis archive from the production factor base.

Picks K representative factor_base factors (greedy farthest-point sampling on
the behavior signatures of an existing behavior archive) and stores their
full daily value panels — direction-adjusted and per-period z-scored — aligned
to the GP run's dates / tradable universe. Evolution loads the result through
GP_RESIDUAL_BASIS_ARCHIVE and penalizes candidates whose RankIC is fully
explained by the span of these basis factors (residual RankIC ≈ 0).
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

# Prevent recursive archive replay / penalties while this utility runs.
os.environ["GP_DISABLE_ARCHIVE_REPLAY"] = "1"
os.environ["GP_EXTERNAL_BEHAVIOR_ARCHIVE"] = ""
os.environ["GP_RESIDUAL_BASIS_ARCHIVE"] = ""

# Same package resolution pin as build_factor_base_behavior_archive.py.
sys.path.insert(0, "/root/crypto-research/users/wesleywu")

from gp.minute_gp_system.engines.unified_v2.backend import set_backend
from gp.minute_gp_system.engines.unified_v2.build_factor_base_behavior_archive import (
    DEFAULT_FACTOR_DIR,
    DEFAULT_H5,
    DEFAULT_MASK,
    _load_daily_matrix,
)
from gp.minute_gp_system.engines.unified_v2.data_loader import CryptoDataLoader
from gp.minute_gp_system.engines.unified_v2.fitness import _postprocess_single

DEFAULT_BEHAVIOR_ARCHIVE = (
    "/root/crypto-research/users/wesleywu/gp/minute_gp_system/archives/"
    "factor_base_behavior_archive_20230101_20250630.npz"
)
DEFAULT_OUTPUT = (
    "/root/crypto-research/users/wesleywu/gp/minute_gp_system/archives/"
    "factor_base_residual_basis_20230101_20250630.npz"
)


def _farthest_point_select(signatures: np.ndarray, k: int) -> np.ndarray:
    """Greedy k-center selection on unit-norm signature rows.

    Starts from the most central signature (max mean similarity) so dense
    regions get a representative early, then repeatedly adds the row with the
    largest minimum distance to the selected set.
    """
    sig = np.asarray(signatures, dtype=np.float32)
    n = sig.shape[0]
    k = max(1, min(int(k), n))
    sim = sig @ sig.T
    np.fill_diagonal(sim, 0.0)
    first = int(np.argmax(sim.mean(axis=1)))
    selected = [first]
    min_dist = 1.0 - sim[first]
    min_dist[first] = -np.inf
    for _ in range(k - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, 1.0 - sim[nxt])
        min_dist[nxt] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def build_basis(args: argparse.Namespace) -> None:
    t_start = time.time()
    with np.load(args.behavior_archive, allow_pickle=False) as data:
        labels_all = [str(x) for x in data["labels"].tolist()]
        signatures = data["behavior_signatures"].astype(np.float32)
        src_meta = {}
        if "metadata_json" in data.files:
            try:
                src_meta = json.loads(str(data["metadata_json"]))
            except Exception:
                src_meta = {}
    directions = {
        str(row.get("factor_id")): float(row.get("ic_direction", 1.0))
        for row in (src_meta.get("factor_meta") or [])
        if isinstance(row, dict) and row.get("factor_id")
    }

    sel = _farthest_point_select(signatures, args.n_basis)
    labels = [labels_all[i] for i in sel]
    print(f"[SELECT] {len(labels)} basis factors via farthest-point on "
          f"{signatures.shape[0]} signatures", flush=True)

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
        indicator_field_indices=None,
        tradable_mask_overlay_path=args.tradable_mask_overlay or None,
        fundamental_panel_path=args.fundamental_panel or None,
    )

    n_periods = int(loader.n_active_periods)
    n_coins = int(loader.n_tradable_coins)
    active_days = np.array(
        [
            f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:8]}"
            for d in loader.dates_flat[loader.active_period_indices]
        ],
        dtype="datetime64[D]",
    )
    tradable_symbols = [str(s) for s in loader.symbols[loader.coin_indices]]
    col_pos = {s: i for i, s in enumerate(tradable_symbols)}

    factor_dir = Path(args.factor_values_dir)
    panels: list[np.ndarray] = []
    kept_labels: list[str] = []
    kept_dirs: list[float] = []
    skip_reasons: Counter = Counter()

    for i, factor_id in enumerate(labels, 1):
        path = factor_dir / f"{factor_id}.parquet"
        if not path.exists():
            skip_reasons["missing_parquet"] += 1
            print(f"[SKIP] {factor_id}: missing_parquet", flush=True)
            continue
        try:
            import pyarrow.parquet as pq

            file_cols = [c for c in pq.read_schema(path).names if c in col_pos]
            if not file_cols:
                skip_reasons["no_symbol_overlap"] += 1
                continue
            mat, reason = _load_daily_matrix(
                path, file_cols, active_days, n_periods, n_coins, col_pos, args.agg,
            )
            if mat is None:
                skip_reasons[str(reason).split(":")[0]] += 1
                continue
            coverage = float(np.isfinite(mat).any(axis=1).mean())
            if coverage < float(args.min_coverage):
                skip_reasons["low_coverage"] += 1
                continue
            direction = directions.get(factor_id, 1.0)
            panel = _postprocess_single(mat * np.float32(direction))
            panels.append(panel.astype(np.float32, copy=False))
            kept_labels.append(factor_id)
            kept_dirs.append(direction)
            del mat
        except Exception as exc:
            skip_reasons[f"error:{type(exc).__name__}"] += 1
            print(f"[WARN] {factor_id}: {exc}", file=sys.stderr, flush=True)
            continue
        finally:
            if i % 8 == 0 or i == len(labels):
                print(f"[PROGRESS] {i}/{len(labels)} ok={len(panels)} "
                      f"elapsed={time.time() - t_start:.0f}s", flush=True)

    if not panels:
        raise ValueError("no usable basis panels; archive would be empty")

    basis = np.stack(panels).astype(np.float16)
    metadata = {
        "source": "factor_base_residual_basis",
        "behavior_archive": str(args.behavior_archive),
        "factor_values_dir": str(factor_dir.resolve()),
        "h5_path": args.h5_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "train_end_date": args.train_end_date,
        "period_minutes": int(args.period_minutes),
        "tradable_mask_overlay": loader.tradable_mask_overlay_path,
        "agg": args.agg,
        "min_coverage": float(args.min_coverage),
        "n_tradable_coins": n_coins,
        "n_active_periods": n_periods,
        "n_basis": len(kept_labels),
        "selection": "farthest_point_on_behavior_signatures",
        "labels": kept_labels,
        "ic_directions": kept_dirs,
        "skip_reasons": dict(skip_reasons),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
            labels=np.asarray(kept_labels, dtype=np.str_),
            symbols=np.asarray(tradable_symbols, dtype=np.str_),
            dates=np.asarray(active_days.astype(str), dtype=np.str_),
            basis_panels=basis,
        )
    os.replace(tmp, out_path)
    print(f"[OK] wrote {out_path} basis_panels={basis.shape}", flush=True)
    print(f"[OK] skip_reasons={dict(skip_reasons)}", flush=True)

    # Sanity: basis panels should be mutually low-correlated in value space.
    flats = [p.ravel().astype(np.float64) for p in panels]
    corrs = []
    for a in range(len(flats)):
        for b in range(a + 1, len(flats)):
            x, y = flats[a], flats[b]
            valid = np.isfinite(x) & np.isfinite(y)
            if int(valid.sum()) < 100:
                continue
            xv = x[valid] - float(x[valid].mean())
            yv = y[valid] - float(y[valid].mean())
            denom = float(np.sqrt(np.sum(xv * xv) * np.sum(yv * yv)))
            if denom > 1e-10:
                corrs.append(float(np.sum(xv * yv) / denom))
    if corrs:
        corrs = np.asarray(corrs)
        print(f"[SANITY] basis pairwise |corr|: median={np.median(np.abs(corrs)):.3f} "
              f"p90={np.quantile(np.abs(corrs), 0.9):.3f} max={np.max(np.abs(corrs)):.3f}",
              flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior_archive", default=DEFAULT_BEHAVIOR_ARCHIVE)
    parser.add_argument("--factor_values_dir", default=DEFAULT_FACTOR_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--n_basis", type=int, default=48)
    parser.add_argument("--h5_path", default=DEFAULT_H5)
    parser.add_argument("--start_date", default="20230101")
    parser.add_argument("--end_date", default="20250630")
    parser.add_argument("--train_end_date", default="20241231",
                        help="must match the GP run's IS cutoff")
    parser.add_argument("--period_minutes", type=int, default=1440)
    parser.add_argument("--chunk_periods", type=int, default=2)
    parser.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--tradable_mask_overlay", default=DEFAULT_MASK)
    parser.add_argument("--fundamental_panel", default="")
    parser.add_argument("--agg", default="last", choices=["last", "mean"])
    parser.add_argument("--min_coverage", type=float, default=0.5)
    args = parser.parse_args(argv)
    build_basis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
