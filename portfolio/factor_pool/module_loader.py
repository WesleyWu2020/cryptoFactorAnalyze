"""Explicit loading and freeze verification for fixed-portfolio factors."""
from __future__ import annotations

from factor_common.loader import load_factor
from factor_common.validation import scan_future_leaks
from portfolio.fixed_provenance import code_snapshot


def load_members(config):
    paths = [member.path for member in config.factors]
    if code_snapshot(paths) != dict(config.code_hashes):
        raise ValueError("code snapshot differs from frozen configuration")
    findings = scan_future_leaks(paths)
    if findings:
        raise ValueError(f"factor source future-leak findings: {findings}")
    specs = [load_factor(path) for path in paths]
    names = [spec.factor_id for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate factor identifier")
    return specs


__all__ = ["load_members"]
