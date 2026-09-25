"""Reproducible local-data acceptance: python -m barra.verify --help."""
import argparse
import json
from pathlib import Path

import pandas as pd

from barra.crypto_barra_exposure import (
    DEFAULT_H5, BarraConfig, analyze_and_write, analyze_evaluate_result,
    build_barra_exposures, resample_alpha_to_profile,
)
from factor_common import FactorManager
from factor_common.validation import check_cutoff, scan_future_leaks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-parquet", required=True)
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--factor-direction", type=int, choices=[-1, 1], default=1)
    args = parser.parse_args()
    out = Path(args.out_dir)
    cfg = BarraConfig()
    alpha = resample_alpha_to_profile(pd.read_parquet(args.factor_parquet), cfg)
    if len(alpha) < 3:
        raise ValueError("verification requires at least three dates")
    before = Path(args.h5).stat()
    findings = scan_future_leaks([Path(__file__).with_name("crypto_barra_exposure.py")])
    if findings:
        raise AssertionError(findings)
    full = analyze_and_write({"factor_value": alpha}, h5_path=args.h5, out_dir=out, cfg=cfg)
    cutoffs = [alpha.index[len(alpha)//3], alpha.index[2*len(alpha)//3]]
    analyses = {None: full}
    styles = {None: build_barra_exposures(alpha, h5_path=args.h5, cfg=cfg)}
    for cutoff in cutoffs:
        analyses[cutoff] = analyze_evaluate_result({"factor_value": alpha}, h5_path=args.h5, cfg=cfg, as_of=cutoff)
        styles[cutoff] = build_barra_exposures(alpha.loc[:cutoff], h5_path=args.h5, cfg=cfg, as_of=cutoff)
    checks = {name: check_cutoff(lambda cutoff: styles[cutoff][name], cutoffs)
              for name in styles[None]}
    checks["residual"] = check_cutoff(lambda cutoff: analyses[cutoff]["alpha_barra_residual"], cutoffs)
    residual = full["alpha_barra_residual"]
    if not residual.notna().any().any():
        raise AssertionError("no valid residual observations")
    manager = FactorManager(h5_path=args.h5, base_dir=out / "factor_results", reports_dir=out)
    params = {"start": str(alpha.index.min().date()), "end": str(alpha.index.max().date()),
              "n_groups": 5, "rebalance_days": 1, "factor_direction": args.factor_direction}
    reports = {}
    for name, values in {"original_matched": alpha.reindex_like(residual).where(residual.notna()),
                         "barra_residual": residual}.items():
        evaluated = manager.evaluate(values, factor_name=name, params=params)
        reports[name] = {"status": evaluated["status"], "paths": evaluated["paths"]}
    after = Path(args.h5).stat()
    unchanged = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    if not unchanged:
        raise AssertionError("source H5 changed during verification")
    evidence = {"static_findings": findings, "cutoff_checks": checks,
                "input_alpha_causality": "not_verified: external cached values",
                "h5_unchanged": unchanged, "evaluations": reports,
                "valid_regression_days": int(full["daily_barra_regression"]["status"].eq("ok").sum()),
                "mean_r2": float(full["daily_barra_regression"]["r2"].mean()),
                "residual_observations": int(residual.notna().sum().sum()),
                "alpha_observations": int(full["alpha_daily"].notna().sum().sum())}
    (out / "verification.json").write_text(json.dumps(evidence, indent=2, default=str) + "\n")
    print(json.dumps(evidence, indent=2, default=str))


if __name__ == "__main__":
    main()
