"""Screen 10 historical frozen GP candidates for the empty 8th portfolio seat.

Stage 1 (this script): export each frozen AST under current semantics to a
staging dir, compute its IS-window (2024-2025) matrix, and run the skill's
correlation gate (daily cross-sectional Spearman, mean over overlap) against
  a) the 7 live composite members (composite_cache wide matrices), and
  b) every library cache in data/factor_results (long format).

Writes portfolio/output/gp_archive_screen_20260924.json.

Usage: ./.venv/bin/python scripts/screen_gp_archive_candidates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from Genetic_Algorithm.export import export_factor  # noqa: E402
from factor_common.data_provider import DataProvider  # noqa: E402
from factor_common.loader import load_factor  # noqa: E402
from factor_common.value_engine import compute_factor  # noqa: E402

IS_START, IS_END = "2024-01-01", "2025-12-31"
STAGING = PROJECT_ROOT / "factor_analyse" / "unsubmit" / "gp_archive_screen"
RESULTS = PROJECT_ROOT / "data" / "factor_results"
MEMBER_CACHE = PROJECT_ROOT / "portfolio" / "output" / "composite_cache"
OUT = PROJECT_ROOT / "portfolio" / "output" / "gp_archive_screen_20260924.json"

LIVE_MEMBERS = [
    "GP_0006a47b612eaf9b", "GP_064185107a8f267e", "GP_3eaff470cdb222f5",
    "GP_77282e814dac81a8", "GP_a37c60cf4492e9f1", "GP_ed2c09f43439ff39",
    "Retail_Friction_Illiquidity_Factor",
]

RUNS = {
    "daily_dualref_seed43_20260913T104200": ["213d40636430a461"],
    "daily_fixed_2024_2025_2026_20260911T110141Z": [
        "260212412430b0ed", "6482dd08ea520376", "c69b1919d73bd9b3"],
    "daily_full_20260908T084421Z": [
        "82255f06cb27f55b", "9bcbe8613fe91402", "a970674cea838fdf",
        "e27478875f89de44", "febbae6c732d8b5f"],
    "daily_incremental_20260912T172900": ["7008d7895cac50bb", "852c52176e9860e1"],
}


def daily_rank_corr(a: pd.DataFrame, b: pd.DataFrame) -> float | None:
    dates = a.index.intersection(b.index)
    cols = a.columns.intersection(b.columns)
    if len(dates) < 30 or len(cols) < 5:
        return None
    corr = a.loc[dates, cols].corrwith(b.loc[dates, cols], axis=1, method="spearman")
    return float(corr.mean())


def main() -> int:
    STAGING.mkdir(parents=True, exist_ok=True)
    provider = DataProvider(PROJECT_ROOT / "data" / "crypto_quant.h5",
                            as_of=pd.Timestamp(IS_END))

    member_mats = {m: pd.read_parquet(MEMBER_CACHE / f"{m}.parquet") for m in LIVE_MEMBERS}
    lib_mats = {}
    for p in sorted(RESULTS.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
            lib_mats[p.stem] = df.pivot(index="date", columns="instrument", values="factor")
        except Exception as exc:
            print(f"skip lib cache {p.stem}: {exc}")

    report = {}
    for run, prefixes in RUNS.items():
        frozen = json.loads(
            (PROJECT_ROOT / "Genetic_Algorithm" / "runs" / run / "frozen.json").read_text())
        for cand in frozen["candidates"]:
            eid = cand.get("expression_id") or cand.get("identifier")
            if not any(eid.startswith(p) for p in prefixes):
                continue
            direction = int(cand.get("factor_direction") or cand.get("direction"))
            result = export_factor({"ast": cand["ast"]}, STAGING, direction=direction)
            spec = load_factor(result.path)
            matrix, _ = compute_factor(spec, provider, start=IS_START, end=IS_END)
            member_corr = {m: daily_rank_corr(matrix, mat) for m, mat in member_mats.items()}
            lib_corr = {fid: daily_rank_corr(matrix, mat) for fid, mat in lib_mats.items()}
            all_corr = {**member_corr, **lib_corr}
            valid = {k: v for k, v in all_corr.items() if v is not None}
            top = max(valid.items(), key=lambda kv: abs(kv[1])) if valid else (None, None)
            report[spec.factor_id] = {
                "source_run": run,
                "direction": direction,
                "member_corr": member_corr,
                "lib_max": {"factor": top[0], "rho": top[1]},
                "lib_corr_top5": dict(sorted(valid.items(), key=lambda kv: -abs(kv[1]))[:5]),
            }
            print(f"{spec.factor_id}: max|rho|={abs(top[1]):.3f} vs {top[0]}", flush=True)

    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
