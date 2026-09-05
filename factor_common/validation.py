"""Future-leak validation: static pattern scan and cutoff replay comparison.

Static scanning is EVIDENCE, not proof: arbitrary Python can hide forward
references from an AST scan, so a clean scan never certifies causality. The
cutoff replay in ``check_cutoff`` is the stronger check: the compute callback
recomputes the factor with provider knowledge (market history AND membership
decisions) truncated at each cutoff, and the truncated result must agree with
the full-history prefix exactly (axis equality, NaN-mask equality, and numeric
differences within ``atol``).

The single evaluation-only allowance is ``factor_common/labels.py`` (forward
returns are permitted future data for evaluation only, per AGENTS.md). It is
an explicit, path-scoped allowance — not a global ignore list.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pandas as pd


# The only module permitted future data (evaluation-only labels); matched as a
# repo-relative path suffix so the same source at any other path still flags.
EVALUATION_ONLY_ALLOWANCE = frozenset({"factor_common/labels.py"})

# Recorded common-axis policy for check_cutoff: columns absent from the
# truncated replay are dropped from the full-history prefix only when they are
# entirely NaN there (future-only instruments such as membership entrants
# whose decision is not yet known at the cutoff), and a separate assertion
# guarantees no finite historical output was removed.
COMMON_AXIS_POLICY = (
    "common-axis: drop all-NaN future-only columns absent from the truncated "
    "replay; separately assert no finite historical output was removed"
)


def _negative_number(node: ast.expr) -> bool:
    """True for negative constants and any explicitly negated expression.

    ``shift(-horizon)`` is flagged even though the sign of ``horizon`` is only
    known at runtime: scanning is evidence, so negated periods are reported.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return True
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and node.value < 0
    )


def _constant(node: ast.expr):
    return node.value if isinstance(node, ast.Constant) else None


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _call_patterns(call: ast.Call) -> list[str]:
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    patterns = []
    if name == "shift":
        candidate = call.args[0] if call.args else _keyword(call, "periods")
        if candidate is not None and _negative_number(candidate):
            patterns.append("shift_negative_periods")
    elif name == "rolling":
        if _constant(_keyword(call, "center")) is True:
            patterns.append("rolling_center")
    elif name in ("bfill", "backfill"):
        patterns.append(name)
    elif name == "fillna":
        if _constant(_keyword(call, "method")) in ("bfill", "backfill"):
            patterns.append("bfill")
    elif name == "merge_asof":
        if _constant(_keyword(call, "direction")) == "forward":
            patterns.append("merge_asof_forward")
    return patterns


def scan_future_leaks(paths: Iterable[str | Path]) -> list[dict]:
    """Scan Python sources for banned forward-looking patterns.

    Returns one finding per banned call as ``{"path", "line", "pattern",
    "source"}`` sorted by path and line. Covers ``shift`` with negative or
    explicitly negated periods (positional or keyword), ``rolling(...,
    center=True)``, ``bfill``/``backfill`` (including
    ``fillna(method=...)``), and ``merge_asof(..., direction="forward")``.
    Files under ``EVALUATION_ONLY_ALLOWANCE`` are skipped; everything else is
    scanned.
    """
    findings = []
    for raw in sorted(Path(path) for path in paths):
        path = raw.as_posix()
        if any(path.endswith(allowed) for allowed in EVALUATION_ONLY_ALLOWANCE):
            continue
        source = raw.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=path)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for pattern in _call_patterns(node):
                findings.append({
                    "path": path,
                    "line": node.lineno,
                    "pattern": pattern,
                    "source": lines[node.lineno - 1].strip(),
                })
    return sorted(findings, key=lambda finding: (finding["path"], finding["line"]))


def check_cutoff(compute, cutoffs: Iterable, *, atol: float = 1e-12) -> dict:
    """Replay ``compute`` at each cutoff and compare with the full prefix.

    ``compute(cutoff)`` must return the daily factor matrix computed with ALL
    knowledge truncated at ``cutoff`` (a date-like, or ``None`` for the full
    history): rebuild the provider with ``DataProvider(path, as_of=cutoff)``
    or equivalent so market history and membership decisions are truncated
    together, and request output only through ``cutoff``. Slicing a
    full-history result is not a valid callback.

    External precomputed DataFrames cannot be replayed; they receive
    ``status="not_verified"`` rather than a passed replay.
    """
    if not callable(compute):
        return {
            "status": "not_verified",
            "reason": "external DataFrame inputs cannot be replayed; "
                      "cutoff invariance is unverified",
            "policy": COMMON_AXIS_POLICY,
            "cutoffs": [],
        }
    full = compute(None)
    if not isinstance(full, pd.DataFrame):
        raise TypeError("compute(None) must return a DataFrame")
    entries = []
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        truncated = compute(cutoff)
        full_prefix = full.loc[:cutoff]
        missing = [c for c in full_prefix.columns if c not in truncated.columns]
        excluded = [c for c in missing if full_prefix[c].isna().all()]
        removed = [c for c in missing if c not in excluded]
        assert not removed, (
            f"truncated replay at {cutoff.date()} removed finite historical "
            f"output columns: {removed}"
        )
        extra = [c for c in truncated.columns if c not in full_prefix.columns]
        assert not extra, (
            f"truncated replay at {cutoff.date()} introduced columns absent "
            f"from the full history: {extra}"
        )
        full_prefix = full_prefix.drop(columns=excluded)
        # Separate assertion under the recorded policy: no finite historical
        # output was removed by the all-NaN future-only column exclusion.
        assert not full.loc[:cutoff, excluded].notna().any().any()
        pd.testing.assert_index_equal(full_prefix.index, truncated.index)
        pd.testing.assert_index_equal(full_prefix.columns, truncated.columns)
        assert full_prefix.isna().equals(truncated.isna())
        delta = (full_prefix - truncated).abs().stack().dropna()
        max_abs_diff = float(delta.max()) if len(delta) else 0.0
        assert max_abs_diff <= atol
        entries.append({
            "cutoff": cutoff,
            "index_equal": True,
            "mask_equal": True,
            "max_abs_diff": max_abs_diff,
            "excluded_columns": excluded,
        })
    return {"status": "verified", "policy": COMMON_AXIS_POLICY, "cutoffs": entries}


__all__ = [
    "COMMON_AXIS_POLICY",
    "EVALUATION_ONLY_ALLOWANCE",
    "check_cutoff",
    "scan_future_leaks",
]
