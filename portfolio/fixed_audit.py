"""Cutoff replay certification for the fixed portfolio pipeline.

The audit deliberately recomputes from a ``DataProvider(as_of=...)`` for each
cutoff.  Comparing only cached prefixes would not detect a provider read that
quietly used future rows.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from portfolio.fixed_config import FixedConfig, daily_date
from portfolio.fixed_pipeline import run_fixed


def compare_matrix(
    full: pd.DataFrame,
    cut: pd.DataFrame,
    cutoff: Any,
    *,
    absent_zero: bool = False,
) -> dict[str, Any]:
    """Compare the historical prefix, preserving axes and NaN masks.

    A column may be absent from a cutoff replay only when it has no historical
    information.  Target/position matrices additionally permit an all-zero
    historical column, since a contract can be unknown until after cutoff.
    """
    cutoff = daily_date(cutoff)
    prefix = full.loc[:cutoff].copy()
    excluded_columns: list[Any] = []
    for column in prefix.columns:
        if column not in cut.columns:
            series = prefix[column]
            acceptable = series.isna().all()
            if absent_zero:
                acceptable = acceptable or (series.fillna(0.0) == 0.0).all()
            assert acceptable, f"removed historical column {column}"
            excluded_columns.append(column)

    unexpected = [column for column in cut.columns if column not in prefix.columns]
    assert not unexpected, f"cutoff introduced unexpected column(s): {unexpected}"
    prefix = prefix.drop(columns=excluded_columns)
    pd.testing.assert_index_equal(prefix.index, cut.index)
    pd.testing.assert_index_equal(prefix.columns, cut.columns)
    assert prefix.isna().equals(cut.isna()), "NaN mask changed"
    pd.testing.assert_frame_equal(prefix, cut, check_dtype=False, rtol=0.0, atol=1e-10)

    numeric = prefix.select_dtypes(include=[np.number]).columns
    if len(numeric):
        delta = (prefix[numeric] - cut[numeric]).abs().to_numpy(dtype=float)
        finite = delta[np.isfinite(delta)]
    else:
        finite = np.array([], dtype=float)
    return {
        "max_abs_diff": float(finite.max()) if finite.size else 0.0,
        "excluded_future_only_columns": [str(column) for column in excluded_columns],
        "rows": len(prefix),
    }


def _compare_orders(full: pd.DataFrame, cut: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    """Compare all order fields on or before cutoff, including status/costs."""
    def prefix(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy().reset_index(drop=True)
        result = frame.loc[pd.to_datetime(frame["date"]) <= cutoff].copy()
        keys = [key for key in ("date", "instrument", "side", "status") if key in result.columns]
        return result.sort_values(keys, kind="mergesort").reset_index(drop=True)

    before, after = prefix(full), prefix(cut)
    pd.testing.assert_frame_equal(before, after, check_dtype=False, rtol=0.0, atol=1e-10)
    numeric = before.select_dtypes(include=[np.number]).columns
    if len(numeric):
        delta = (before[numeric] - after[numeric]).abs().to_numpy(dtype=float)
        finite = delta[np.isfinite(delta)]
    else:
        finite = np.array([], dtype=float)
    return {"rows": len(before), "equal_within_tolerance": True,
            "max_abs_diff": float(finite.max()) if finite.size else 0.0}


def audit_fixed(config: FixedConfig, cutoffs: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
    """Certify that fixed-pipeline historical outputs are cutoff invariant."""
    if not isinstance(config, FixedConfig):
        raise TypeError("config must be a FixedConfig")
    if not cutoffs:
        raise ValueError("at least one cutoff required")
    days = sorted({daily_date(value) for value in cutoffs})
    start = daily_date(config.signal_start)
    end = daily_date(config.signal_end)
    for day in days:
        if not start + pd.Timedelta(days=1) <= day <= end:
            raise ValueError("cutoff must lie inside the active evaluation interval after first signal")

    full = run_fixed(config)
    if full["status"] != "complete":
        raise ValueError("full evaluation must be complete before cutoff certification")

    records: list[dict[str, Any]] = []
    for day in days:
        cut = run_fixed(config, as_of=day)
        if cut["source_stat"] != full["source_stat"]:
            raise RuntimeError("input source changed between cutoff replays")
        if cut["configuration"] != full["configuration"]:
            raise RuntimeError("configuration receipt changed between cutoff replays")

        checks: dict[str, dict[str, Any]] = {}
        for name in full["values"]:
            checks[f"values/{name}"] = compare_matrix(full["values"][name], cut["values"][name], day)
            checks[f"member_targets/{name}"] = compare_matrix(
                full["member_targets"][name], cut["member_targets"][name], day, absent_zero=True
            )
        checks["targets"] = compare_matrix(full["targets"], cut["targets"], day, absent_zero=True)

        for name in ("gross", "trading_net", "all_costs"):
            first = full["accounting"]["scenarios"][name]
            second = cut["accounting"]["scenarios"][name]
            checks[f"{name}/ledger"] = compare_matrix(first["ledger"], second["ledger"], day)
            checks[f"{name}/positions"] = compare_matrix(first["positions"], second["positions"], day, absent_zero=True)
            checks[f"{name}/orders"] = _compare_orders(first["orders"], second["orders"], day)

        scenario_status = {
            name: item["status"] for name, item in cut["accounting"]["scenarios"].items()
        }
        metrics_full_none = {
            name: cut["metrics"][name]["full"] is None
            for name in scenario_status
        }
        # A cutoff replay intentionally cannot observe the liquidation bar after
        # the final signal.  Preserve that incomplete state instead of forcing
        # an artificial close.  The known segment is the last ledger date that
        # was actually produced, not the requested cutoff label.
        ledger_dates = [
            pd.Timestamp(index).normalize()
            for scenario in cut["accounting"]["scenarios"].values()
            for index in scenario["ledger"].index
        ]
        known_segment_end = max(ledger_dates).isoformat() if ledger_dates else None
        incomplete_tail_expected = any(status == "incomplete" for status in scenario_status.values())
        if incomplete_tail_expected and not all(metrics_full_none.values()):
            raise AssertionError("incomplete cutoff scenarios must report metrics.full=None")

        records.append({
            "cutoff": day.isoformat(),
            "checks": checks,
            "source_stat": cut["source_stat"],
            "configuration": cut["configuration"],
            # Reads intentionally differ from the full run because every
            # cutoff replay uses a shorter requested date range.
            "input_reads": cut["input_reads"],
            "input_reads_comparison": "omitted: cutoff replay uses intentionally changed date ranges",
            "scenario_status": scenario_status,
            "metrics_full_none": metrics_full_none,
            "incomplete_tail_expected": incomplete_tail_expected,
            "known_segment_end": known_segment_end,
            "max_abs_diff": max(item["max_abs_diff"] for item in checks.values()),
        })

    return {
        "status": "verified",
        "cutoffs": records,
        "code_hashes": dict(config.code_hashes),
        "configuration": full["configuration"],
        "source_stat": full["source_stat"],
        "input_reads": full["input_reads"],
        "limitations": "验证历史计算一致性；不验证历史数据实际发布时间或未来收益。",
    }


__all__ = ["audit_fixed", "compare_matrix"]
