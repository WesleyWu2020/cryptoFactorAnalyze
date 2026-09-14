"""Persist and report the continuous out-of-sample ML evaluation.

The training package produces one prediction table per quarter.  This module
is the intentionally small boundary between those predictions and the
standard :class:`factor_common.manager.FactorManager` result: predictions are
concatenated once, the manager accounts the resulting portfolio once, and all
report slices are derived from that one accounting.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factor_common.metrics import _ledger_summary

from .dataset import clean_factor_table


_MANAGER_KEYS = frozenset(
    {
        "start",
        "end",
        "rebalance_days",
        "anchor_date",
        "n_groups",
        "factor_direction",
        "fee_rate",
        "slippage",
        "include_funding",
        "funding_price_mode",
        "split_date",
        "out_of_sample_days",
    }
)
_TABLES = ("ledger", "orders", "positions", "valuation_prices", "funding", "funding_coverage")


def _iso_date(value: Any, name: str) -> str:
    """Return a timezone-naive calendar date as ISO text."""
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError(f"{name} must be timezone-naive")
        if value.time() != datetime.min.time():
            raise ValueError(f"{name} must be normalized to midnight")
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        if value.tz is not None or value != value.normalize():
            raise ValueError(f"{name} must be a timezone-naive calendar date")
        return value.date().isoformat()
    if isinstance(value, str):
        parsed = pd.Timestamp(value)
        if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize():
            raise ValueError(f"{name} must be an ISO calendar date")
        return parsed.date().isoformat()
    raise TypeError(f"{name} must be a date or ISO calendar date string")


def manager_params(
    config: Any = None,
    *,
    start: Any = None,
    end: Any = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build the explicit, supported parameter map for ``FactorManager``.

    ML ``Config`` uses ``holding_days`` while the manager's daily profile uses
    ``rebalance_days``.  Only manager profile fields are emitted; model,
    training, and candidate settings never leak into the manager API.
    """
    unknown = sorted(set(overrides) - _MANAGER_KEYS)
    if unknown:
        raise ValueError(f"Unsupported FactorManager override: {unknown[0]}")

    if config is not None and isinstance(config, Mapping):
        source = dict(config)
    elif config is not None:
        source = {
            name: getattr(config, name)
            for name in (
                "oos_start",
                "end",
                "holding_days",
                "anchor_date",
                "fee_rate",
                "slippage",
                "factor_direction",
                "n_groups",
                "include_funding",
                "funding_price_mode",
                "split_date",
                "out_of_sample_days",
            )
            if hasattr(config, name)
        }
    else:
        source = {}

    params: dict[str, Any] = {}
    requested_start = start if start is not None else source.get("start", source.get("oos_start"))
    requested_end = end if end is not None else source.get("end")
    if requested_start is not None:
        params["start"] = _iso_date(requested_start, "start")
    if requested_end is not None:
        params["end"] = _iso_date(requested_end, "end")

    mapping = (
        ("holding_days", "rebalance_days"),
        ("anchor_date", "anchor_date"),
        ("n_groups", "n_groups"),
        ("factor_direction", "factor_direction"),
        ("fee_rate", "fee_rate"),
        ("slippage", "slippage"),
        ("include_funding", "include_funding"),
        ("funding_price_mode", "funding_price_mode"),
        ("split_date", "split_date"),
        ("out_of_sample_days", "out_of_sample_days"),
    )
    for source_name, target_name in mapping:
        value = overrides.get(target_name, source.get(source_name, source.get(target_name)))
        if value is None:
            continue
        if target_name in {"anchor_date", "split_date"}:
            params[target_name] = _iso_date(value, target_name)
        else:
            params[target_name] = value
    return params


def _validate_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(ledger, pd.DataFrame):
        raise TypeError("ledger must be a DataFrame")
    if ledger.empty:
        return ledger.copy()
    if not isinstance(ledger.index, pd.DatetimeIndex):
        raise ValueError("ledger index must be a DatetimeIndex")
    if ledger.index.tz is not None:
        raise ValueError("ledger index must be timezone-naive")
    if ledger.index.has_duplicates or ledger.index.hasnans:
        raise ValueError("ledger index must be unique and contain no NaT")
    if "return" not in ledger.columns or "equity" not in ledger.columns:
        raise ValueError("ledger must contain equity and return columns")
    result = ledger.sort_index().copy()
    result.index = result.index.normalize()
    return result


def quarterly_metrics(
    ledger: pd.DataFrame,
    *,
    periods_per_year: int = 365,
    status: str = "complete",
) -> pd.DataFrame:
    """Summarize calendar quarters by slicing one continuous ledger.

    ``total_return`` is compounded from the original daily return rows.  The
    ``reconciled`` flag checks both that compounding agrees with the equity
    path and that adjacent quarter equity slices join without a reset.
    """
    if status not in {"complete", "incomplete", "insufficient_data"}:
        raise ValueError("status must be complete, incomplete, or insufficient_data")
    frame = _validate_ledger(ledger)
    columns = [
        "quarter", "start_date", "end_date", "n_periods", "starting_equity",
        "ending_equity", "equity_return", "reconciled", "status",
        "missing_tail", "total_return", "annual_return", "volatility",
        "annual_volatility", "sharpe", "max_drawdown", "win_rate",
        "profit_loss_ratio", "turnover",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    periods = frame.index.to_period("Q")
    for quarter in periods.unique():
        piece = frame.loc[periods == quarter]
        summary = _ledger_summary(piece, periods_per_year=periods_per_year)
        compounded = float((1.0 + piece["return"].astype(float)).prod() - 1.0)
        first_position = int(frame.index.get_loc(piece.index[0]))
        starting = 1.0 if first_position == 0 else float(piece["equity"].iloc[0])
        ending = float(piece["equity"].iloc[-1])
        prior_equity = 1.0 if first_position == 0 else float(frame["equity"].iloc[first_position - 1])
        equity_return = ending / prior_equity - 1.0 if prior_equity != 0 else np.nan
        reconciled = bool(
            np.isfinite(compounded)
            and np.isfinite(equity_return)
            and np.isclose(compounded, equity_return, rtol=1e-9, atol=1e-12)
        )
        rows.append(
            {
                "quarter": str(quarter),
                "start_date": piece.index[0].date().isoformat(),
                "end_date": piece.index[-1].date().isoformat(),
                "n_periods": int(len(piece)),
                "starting_equity": starting,
                "ending_equity": ending,
                "equity_return": equity_return,
                "reconciled": reconciled,
                "status": status,
                "missing_tail": status != "complete" and quarter == periods[-1],
                **summary,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _factor_table(predictions: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(predictions, (str, Path)):
        predictions = pd.read_parquet(predictions)
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a DataFrame or parquet path")
    table = clean_factor_table(predictions.loc[:, ["date", "instrument", "factor"]])
    return table


def _write_table(path: Path, table: Any) -> None:
    if isinstance(table, pd.DataFrame):
        table.to_parquet(path, index=True)


def _daily_ic(result: Mapping[str, Any]) -> pd.DataFrame:
    rows = result.get("factor_performance", {}).get("samples", {}).get("full", {}).get("ic", {}).get("daily", [])
    if not rows:
        return pd.DataFrame(columns=["date", "ic", "rank_ic"])
    frame = pd.DataFrame(rows)
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def _render_html(result: Mapping[str, Any], path: Path) -> None:
    try:
        from factor_common.reporting import render_result

        render_result(dict(result), path)
    except Exception as exc:  # malformed/incomplete synthetic results still get a report
        path.write_text(
            "<html><body><h1>ML OOS evaluation</h1>"
            f"<p>Status: {_json_safe(result.get('status'))}</p>"
            f"<p>Report rendering unavailable: {_json_safe(str(exc))}</p>"
            "</body></html>",
            encoding="utf-8",
        )


def evaluate_oos(
    predictions: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    factor_name: str = "ml_oos",
    config: Any = None,
    manager: Any = None,
    manager_cls: Any = None,
    h5_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Evaluate and persist one continuous ML OOS prediction stream.

    All accounting tables are copied exactly from the manager result.  A
    halted all-costs scenario therefore remains halted in both the JSON and
    the report; no zero-filled tail or second per-quarter backtest is made.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table = _factor_table(predictions)
    factor_path = destination / "factor.parquet"
    table.to_parquet(factor_path, index=False)

    params = manager_params(config)
    if manager is None:
        if manager_cls is None:
            from factor_common.manager import FactorManager

            manager_cls = FactorManager
        kwargs: dict[str, Any] = {
            "reports_dir": destination,
            "persist_evaluations": False,
        }
        if h5_path is not None:
            kwargs["h5_path"] = h5_path
        if base_dir is not None:
            kwargs["base_dir"] = base_dir
        manager = manager_cls(**kwargs)
    result = manager.evaluate(
        table,
        factor_name=factor_name,
        profile_id="perp_1d",
        params=params,
        plot=False,
    )
    if not isinstance(result, Mapping):
        raise TypeError("FactorManager.evaluate must return a mapping")

    paths: dict[str, str] = {"factor_parquet": str(factor_path)}
    factor_result = result.get("factor_result", {})
    scenarios = factor_result.get("scenarios", {}) if isinstance(factor_result, Mapping) else {}
    scenario_payload: dict[str, Any] = {}
    for scenario_name, scenario in scenarios.items():
        if not isinstance(scenario, Mapping):
            continue
        scenario_payload[scenario_name] = {
            "status": scenario.get("status"),
            "diagnostics": _json_safe(scenario.get("diagnostics", {})),
            "tables": {},
        }
        for table_name in _TABLES:
            frame = scenario.get(table_name)
            if isinstance(frame, pd.DataFrame):
                table_path = destination / f"{scenario_name}__{table_name}.parquet"
                _write_table(table_path, frame)
                paths[f"{scenario_name}__{table_name}"] = str(table_path)
                scenario_payload[scenario_name]["tables"][table_name] = str(table_path)

    all_costs = scenarios.get("all_costs", {}) if isinstance(scenarios, Mapping) else {}
    all_ledger = all_costs.get("ledger", pd.DataFrame()) if isinstance(all_costs, Mapping) else pd.DataFrame()
    scenario_status = all_costs.get("status", result.get("status", "complete")) if isinstance(all_costs, Mapping) else result.get("status", "complete")
    quarterly = quarterly_metrics(all_ledger, status=scenario_status)
    quarterly_path = destination / "quarterly_metrics.parquet"
    quarterly.to_parquet(quarterly_path, index=False)
    paths["quarterly_metrics"] = str(quarterly_path)

    daily_ic = _daily_ic(result)
    daily_ic_path = destination / "daily_ic.parquet"
    daily_ic.to_parquet(daily_ic_path, index=False)
    paths["daily_ic"] = str(daily_ic_path)

    group_returns = result.get("group_returns")
    if isinstance(group_returns, pd.DataFrame):
        group_path = destination / "group_returns.parquet"
        _write_table(group_path, group_returns)
        paths["group_returns"] = str(group_path)
    group_diagnostics = result.get("diagnostics", {}).get("grouping", {})
    group_diag_path = destination / "group_diagnostics.json"
    group_diag_path.write_text(json.dumps(_json_safe(group_diagnostics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["group_diagnostics"] = str(group_diag_path)

    html_path = destination / "evaluation.html"
    if plot:
        _render_html(result, html_path)
    else:
        html_path.write_text("<html><body><h1>ML OOS evaluation</h1></body></html>", encoding="utf-8")
    paths["html"] = str(html_path)

    evaluation = {
        "status": result.get("status"),
        "factor_name": factor_name,
        "manager_params": params,
        "metadata": _json_safe(result.get("metadata", {})),
        "factor_performance": _json_safe(result.get("factor_performance", {})),
        "scenarios": scenario_payload,
        "quarterly_metrics": _json_safe(quarterly.to_dict(orient="records")),
        "group_diagnostics": _json_safe(group_diagnostics),
        "paths": paths,
    }
    evaluation_path = destination / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    paths["evaluation_json"] = str(evaluation_path)
    return {**evaluation, "paths": paths, "manager_result": result}


__all__ = ["evaluate_oos", "manager_params", "quarterly_metrics"]
