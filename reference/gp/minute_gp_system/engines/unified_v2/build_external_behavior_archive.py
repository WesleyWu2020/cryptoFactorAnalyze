"""Build an external behavior archive for GP search-time correlation gating.

The archive contains standardized behavior signatures for historical accepted GP
factors. Evolution can load it through GP_EXTERNAL_BEHAVIOR_ARCHIVE and penalize
new candidates that behave too similarly to these factors.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Prevent recursive archive replay while this utility instantiates the problem.
os.environ["GP_DISABLE_ARCHIVE_REPLAY"] = "1"
os.environ["GP_EXTERNAL_BEHAVIOR_ARCHIVE"] = ""

from gp.minute_gp_system.engines.unified_v2.backend import set_backend
from gp.minute_gp_system.engines.unified_v2.behavior_signature import build_behavior_context
from gp.minute_gp_system.engines.unified_v2.config import CHUNK_PERIODS
from gp.minute_gp_system.engines.unified_v2.data_loader import CryptoDataLoader
from gp.minute_gp_system.engines.unified_v2.evolution import CryptoFactorProblemV2
from gp.minute_gp_system.engines.unified_v2.archive import reconstruct_archive_features
from gp.minute_gp_system.engines.unified_v2.fitness import standardize_phenotypes


def _key_from_label(label: str) -> tuple[str, int]:
    text = str(label)
    prefix, sep, rank_text = text.rpartition("_r")
    if not sep or not prefix:
        raise ValueError(f"cannot parse factor label: {label!r}")
    # gpbarra labels are often prefixed with a stable sort ordinal, e.g.
    # 045_gp_2022_2024_seed139_r14. The run name starts at gp_*.
    gp_pos = prefix.find("gp_2022_2024_seed")
    if gp_pos > 0:
        prefix = prefix[gp_pos:]
    return prefix, int(rank_text)


def _selected_keys_from_selection_log(selection_log: str | Path, selected_only: bool = True) -> set[tuple[str, int]]:
    df = pd.read_csv(selection_log)
    if "label" not in df.columns:
        raise ValueError(f"{selection_log} missing label column")
    if selected_only and "selected" not in df.columns:
        raise ValueError(f"{selection_log} missing selected column")
    if selected_only:
        selected = df["selected"]
        if selected.dtype != bool:
            selected = selected.astype(str).str.lower().isin({"1", "true", "yes", "on"})
        df = df[selected]
    return {_key_from_label(label) for label in df["label"]}


def _load_pareto_entry_map(runs_root: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for path in sorted(runs_root.glob("gp_2022_2024_seed*/raw/pareto_results.json")):
        run_name = path.parts[-3]
        try:
            rows = json.load(open(path, "r", encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] skip {path}: {exc}", file=sys.stderr)
            continue
        for row in rows:
            try:
                rank = int(row.get("rank"))
            except Exception:
                continue
            if "params" not in row:
                continue
            out[(run_name, rank)] = row
    return out


def _entries_from_passed_csv(
    csv_path: Path,
    runs_root: Path,
    limit: int | None = None,
    selection_log: str | Path | None = None,
    selected_only: bool = False,
) -> tuple[list[dict], list[str]]:
    df = pd.read_csv(csv_path)
    required = {"run_name", "rank"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    selected_keys = (
        _selected_keys_from_selection_log(selection_log, selected_only=selected_only)
        if selection_log
        else None
    )
    entry_map = _load_pareto_entry_map(runs_root)
    entries: list[dict] = []
    labels: list[str] = []
    seen_formula: set[str] = set()
    for _, row in df.iterrows():
        run_name = str(row["run_name"])
        rank = int(row["rank"])
        if selected_keys is not None and (run_name, rank) not in selected_keys:
            continue
        entry = entry_map.get((run_name, rank))
        if entry is None:
            print(f"[WARN] missing pareto entry for {run_name} rank={rank}", file=sys.stderr)
            continue
        formula = str(entry.get("formula") or row.get("formula") or "")
        if formula in seen_formula:
            continue
        seen_formula.add(formula)
        entries.append(entry)
        labels.append(f"{run_name}_r{rank}")
        if limit is not None and len(entries) >= int(limit):
            break
    if not entries:
        raise ValueError("no usable entries found from passed csv")
    return entries, labels


def merge_pareto_into_archive(
    archive_path: str,
    pareto_path: str,
    h5_path: str,
    period_minutes: int,
    start_date: str,
    end_date: str,
    run_label: str,
    keep_filter=None,
):
    archive_path = Path(archive_path)
    pareto_path = Path(pareto_path)
    if keep_filter is None:
        def keep_filter(row):
            gate = row.get('final_low_corr_gate')
            if gate is None:
                return True
            if isinstance(gate, dict):
                return bool(gate.get('passed', True))
            return True

    if not archive_path.exists():
        print(f'[AUTO_MERGE_ARCHIVE] missing archive: {archive_path}', flush=True)
        return 0
    if not pareto_path.exists():
        print(f'[AUTO_MERGE_ARCHIVE] missing pareto: {pareto_path}', flush=True)
        return 0

    with np.load(archive_path, allow_pickle=False) as data:
        behavior_key = 'behavior_signatures' if 'behavior_signatures' in data.files else 'archive'
        old_behavior = np.asarray(data[behavior_key], dtype=np.float32) if behavior_key in data.files else np.zeros((0, 0), dtype=np.float32)
        old_labels = [str(x) for x in data['labels'].tolist()] if 'labels' in data.files else []
        if 'metadata_json' in data.files:
            try:
                old_metadata = json.loads(str(data['metadata_json']))
            except Exception:
                old_metadata = {}
        else:
            old_metadata = {}
        old_phenotypes = np.asarray(data['phenotypes'], dtype=np.float32) if 'phenotypes' in data.files else None

    with open(pareto_path, 'r', encoding='utf-8') as f:
        pareto_rows = json.load(f)
    if not isinstance(pareto_rows, list):
        print(f'[AUTO_MERGE_ARCHIVE] invalid pareto payload: {pareto_path}', flush=True)
        return 0

    kept_rows = [row for row in pareto_rows if isinstance(row, dict) and 'params' in row and keep_filter(row)]
    if not kept_rows:
        print(f'[AUTO_MERGE_ARCHIVE] no usable pareto rows: {pareto_path}', flush=True)
        return 0

    params = [np.asarray(row['params'], dtype=np.int32) for row in kept_rows]
    field_indices = np.unique(np.vstack(params)[:, (0, 1, 4)].astype(np.intp, copy=False).ravel())

    set_backend('cpu', 0)
    loader = CryptoDataLoader(
        h5_path=h5_path,
        minutes_per_period=period_minutes,
        chunk_periods=CHUNK_PERIODS,
        start_date=start_date,
        end_date=end_date,
        backend='cpu',
        gpu_id=0,
        indicator_field_indices=field_indices,
    )
    problem = CryptoFactorProblemV2(loader, backend='cpu', gpu_id=0, eval_batch_size=8)
    behavior_context = build_behavior_context(problem.loader, problem.period_returns, problem.period_tradable_mask)
    new_phenotypes, new_behavior = reconstruct_archive_features(
        params,
        problem,
        behavior_context=behavior_context,
        source_meta={
            'source': 'auto_merge_pareto',
            'pareto_path': str(pareto_path),
            'run_label': run_label,
            'n_entries': len(kept_rows),
        },
    )
    if new_behavior is None or np.asarray(new_behavior).size == 0:
        print(f'[AUTO_MERGE_ARCHIVE] dim mismatch: archive={getattr(old_behavior, "shape", None)} new={None}; archive unchanged', flush=True)
        return 0
    new_behavior = standardize_phenotypes(new_behavior)
    if new_phenotypes is not None:
        new_phenotypes = standardize_phenotypes(new_phenotypes)

    if old_behavior.ndim != 2 or new_behavior.ndim != 2 or (old_behavior.size and old_behavior.shape[1] != new_behavior.shape[1]):
        print(f'[AUTO_MERGE_ARCHIVE] dim mismatch: archive={getattr(old_behavior, "shape", None)} new={new_behavior.shape}; archive unchanged', flush=True)
        return 0

    combined_behavior = new_behavior if old_behavior.size == 0 else np.vstack([old_behavior, new_behavior])
    kept_labels = [f'{run_label}_r{int(row["rank"])}' for row in kept_rows]
    combined_labels = old_labels + kept_labels

    if new_phenotypes is not None and np.asarray(new_phenotypes).size:
        if old_phenotypes is not None and old_phenotypes.size and old_phenotypes.shape[1] == np.asarray(new_phenotypes).shape[1]:
            combined_phenotypes = np.vstack([old_phenotypes, new_phenotypes])
        else:
            combined_phenotypes = new_phenotypes
    elif old_phenotypes is not None and old_phenotypes.size:
        combined_phenotypes = old_phenotypes
    else:
        combined_phenotypes = np.zeros((0, 0), dtype=np.float32)

    merged_metadata = dict(old_metadata)
    merged_metadata.update({
        'source': 'auto_merge_pareto',
        'last_pareto_path': str(pareto_path),
        'last_run_label': run_label,
        'h5_path': h5_path,
        'start_date': start_date,
        'end_date': end_date,
        'period_minutes': int(period_minutes),
        'n_entries': int(combined_behavior.shape[0]),
        'behavior_shape': list(combined_behavior.shape),
        'labels': combined_labels,
    })

    shutil.copy2(archive_path, Path(str(archive_path) + '.bak'))
    tmp_path = archive_path.with_suffix(archive_path.suffix + '.tmp')
    with open(tmp_path, 'wb') as f:
        np.savez_compressed(
            f,
            metadata_json=np.asarray(json.dumps(merged_metadata, sort_keys=True), dtype=np.str_),
            labels=np.asarray(combined_labels, dtype=np.str_),
            behavior_signatures=combined_behavior.astype(np.float16),
            phenotypes=np.asarray(combined_phenotypes, dtype=np.float16),
        )
    os.replace(tmp_path, archive_path)
    print(f'[AUTO_MERGE_ARCHIVE] appended={len(kept_rows)} archive_entries={combined_behavior.shape[0]} path={archive_path}', flush=True)
    return len(kept_rows)



def build_archive(args: argparse.Namespace) -> None:
    entries, labels = _entries_from_passed_csv(
        Path(args.passed_csv),
        Path(args.runs_root),
        limit=args.limit,
        selection_log=args.selection_log or None,
        selected_only=args.selected_only,
    )
    params = [np.asarray(entry["params"], dtype=np.int32) for entry in entries]
    field_indices = np.unique(np.vstack(params)[:, (0, 1, 4)].astype(np.intp, copy=False).ravel())

    set_backend(args.backend, args.gpu_id)
    loader = CryptoDataLoader(
        h5_path=args.h5_path,
        minutes_per_period=args.period_minutes,
        chunk_periods=args.chunk_periods,
        start_date=args.start_date,
        end_date=args.end_date,
        backend=args.backend,
        gpu_id=args.gpu_id,
        indicator_field_indices=field_indices,
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
    phenotypes, behavior = reconstruct_archive_features(
        params,
        problem,
        behavior_context=behavior_context,
        source_meta={
            "source": "passed_oos_full",
            "passed_csv": str(Path(args.passed_csv).resolve()),
            "n_entries": len(entries),
        },
    )
    if behavior is None or behavior.size == 0:
        raise ValueError("behavior signature reconstruction returned empty archive")
    behavior = standardize_phenotypes(behavior)
    phenotypes = standardize_phenotypes(phenotypes) if phenotypes is not None else np.zeros((0, 0), dtype=np.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "passed_oos_full",
        "passed_csv": str(Path(args.passed_csv).resolve()),
        "runs_root": str(Path(args.runs_root).resolve()),
        "h5_path": args.h5_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "period_minutes": int(args.period_minutes),
        "n_entries": len(entries),
        "behavior_shape": list(behavior.shape),
        "labels": labels,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
            labels=np.asarray(labels, dtype=np.str_),
            behavior_signatures=behavior.astype(np.float16),
            phenotypes=phenotypes.astype(np.float16),
        )
    os.replace(tmp, out_path)
    print(f"[OK] wrote {out_path} behavior_signatures={behavior.shape} entries={len(entries)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passed_csv", default="/home/wesleywu/gp/minute_gp_system/passed_oos_full.csv")
    parser.add_argument("--runs_root", default="/home/wesleywu/gp/minute_gp_system/runs_manual")
    parser.add_argument("--output", default="/home/wesleywu/gp/minute_gp_system/archives/passed_oos_behavior_archive.npz")
    parser.add_argument("--h5_path", default="/root/crypto-research/common/data/raw/h5/bybit_linear_1m_unified.h5")
    parser.add_argument("--start_date", default="20220102")
    parser.add_argument("--end_date", default="20241231")
    parser.add_argument("--period_minutes", type=int, default=1440)
    parser.add_argument("--chunk_periods", type=int, default=5)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--trading_cost", type=float, default=0.0015)
    parser.add_argument("--tradable_mask_overlay", default="")
    parser.add_argument("--fundamental_panel", default="/root/crypto-research/common/data/factor_platform/fundamentals/coinmetrics_cap_mrkt_est_daily.npz")
    parser.add_argument("--selection_log", default="")
    parser.add_argument("--selected_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    build_archive(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
