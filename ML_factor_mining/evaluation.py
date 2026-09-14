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
import html
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
    *legacy_bounds: Any,
    start: Any = None,
    end: Any = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build the explicit, supported parameter map for ``FactorManager``.

    ML ``Config`` uses ``holding_days`` while the manager's daily profile uses
    ``rebalance_days``.  Only manager profile fields are emitted; model,
    training, and candidate settings never leak into the manager API.

    ``start`` and ``end`` are keyword-only in the current API.  The positional
    form is retained for callers following the original Task 7 plan; that
    form also sets the split immediately before the OOS interval so the
    standard manager report has the same OOS boundary as the ML stream.
    """
    if len(legacy_bounds) > 2:
        raise TypeError("manager_params accepts at most positional start and end bounds")
    legacy_style = bool(legacy_bounds)
    if legacy_style:
        if start is not None or end is not None:
            raise TypeError("start/end cannot be supplied both positionally and by keyword")
        start = legacy_bounds[0]
        end = legacy_bounds[1] if len(legacy_bounds) == 2 else None

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

    # The ML experiment is intentionally a five-group portfolio, independent
    # of model/training configuration.  Always send this explicit profile
    # value so the manager cannot silently fall back to its ten-group default.
    params["n_groups"] = overrides.get("n_groups", 5)

    mapping = (
        ("holding_days", "rebalance_days"),
        ("anchor_date", "anchor_date"),
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
    if legacy_style and "split_date" not in params and start is not None:
        params["split_date"] = _iso_date(
            pd.Timestamp(start) - pd.Timedelta(days=1), "split_date"
        )
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
    if result.index.has_duplicates:
        raise ValueError("ledger index must contain unique calendar dates")
    if len(result.index) > 1:
        gaps = result.index.to_series().diff().dropna()
        if (gaps != pd.Timedelta(days=1)).any():
            raise ValueError("ledger index must contain a continuous daily calendar")
    return result


def quarterly_metrics(
    ledger: pd.DataFrame,
    *legacy_args: Any,
    predictions: pd.DataFrame | None = None,
    evidence_end: Any = None,
    periods_per_year: int = 365,
    status: str = "complete",
    accounting_status: str | None = None,
    daily_ic: pd.DataFrame | None = None,
    coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize calendar quarters by slicing one continuous ledger.

    ``total_return`` is compounded from the original daily return rows.  The
    ``reconciled`` flag checks both that compounding agrees with the equity
    path and that adjacent quarter equity slices join without a reset.
    """
    # The original plan passed daily IC, predictions, and coverage as the
    # first three positional arguments.  Keep that form readable while the
    # compact current API uses named ``predictions``/``evidence_end``.
    if legacy_args:
        if len(legacy_args) > 3:
            raise TypeError("quarterly_metrics accepts at most three legacy positional tables")
        if daily_ic is None:
            daily_ic = legacy_args[0]
        if len(legacy_args) >= 2 and predictions is None:
            predictions = legacy_args[1]
        if len(legacy_args) == 3 and coverage is None:
            coverage = legacy_args[2]
    if accounting_status is not None:
        status = accounting_status
    if status not in {"complete", "incomplete", "insufficient_data"}:
        raise ValueError("status must be complete, incomplete, or insufficient_data")
    frame = _validate_ledger(ledger)
    prediction_dates = pd.DatetimeIndex([], name="date")
    if predictions is not None:
        if not isinstance(predictions, pd.DataFrame) or "date" not in predictions.columns:
            raise ValueError("predictions must contain a date column")
        parsed = pd.to_datetime(predictions["date"], errors="raise", utc=True)
        prediction_dates = pd.DatetimeIndex(
            parsed.dt.tz_localize(None).dt.normalize(), name="date"
        )
    columns = [
        "quarter", "start_date", "end_date", "n_periods", "starting_equity",
        "ending_equity", "equity_return", "reconciled", "status",
        "missing_tail", "total_return", "annual_return", "volatility",
        "annual_volatility", "sharpe", "max_drawdown", "win_rate",
        "profit_loss_ratio", "turnover", "mean_rank_ic", "ic_dates",
        "prediction_days", "expected_calendar_days", "coverage_mean",
        "complete_calendar_quarter", "accounting_tail_only",
        "evaluated_through", "accounting_status", "metrics_scope",
    ]
    if frame.empty and prediction_dates.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    ledger_periods = frame.index.to_period("Q") if not frame.empty else pd.PeriodIndex([], freq="Q")
    prediction_periods = prediction_dates.to_period("Q") if not prediction_dates.empty else pd.PeriodIndex([], freq="Q")
    quarters = sorted(set(ledger_periods.tolist()) | set(prediction_periods.tolist()))
    evidence_timestamp = None
    if evidence_end is not None:
        evidence_timestamp = pd.Timestamp(_iso_date(evidence_end, "evidence_end"))
    for quarter in quarters:
        piece = frame.loc[ledger_periods == quarter] if not frame.empty else frame
        if piece.empty:
            summary = {
                "total_return": None, "annual_return": None, "volatility": None,
                "annual_volatility": None, "sharpe": None, "max_drawdown": None,
                "win_rate": None, "profit_loss_ratio": None, "turnover": None,
                "n_periods": 0,
            }
        else:
            summary = _ledger_summary(piece, periods_per_year=periods_per_year)
        compounded = float((1.0 + piece["return"].astype(float)).prod() - 1.0)
        if piece.empty:
            start_date = quarter.start_time.date().isoformat()
            end_date = None
            starting = ending = equity_return = np.nan
            reconciled = False
            actual_end = None
        else:
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
            start_date = piece.index[0].date().isoformat()
            end_date = piece.index[-1].date().isoformat()
            actual_end = piece.index[-1]
        quarter_prediction_dates = prediction_dates[prediction_dates.to_period("Q") == quarter]
        quarter_ic = daily_ic
        if isinstance(quarter_ic, pd.DataFrame) and "date" in quarter_ic.columns:
            ic_dates = pd.to_datetime(quarter_ic["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
            quarter_ic = quarter_ic.loc[ic_dates.dt.to_period("Q") == quarter]
        mean_rank_ic = (
            float(pd.to_numeric(quarter_ic["rank_ic"], errors="coerce").mean())
            if isinstance(quarter_ic, pd.DataFrame) and "rank_ic" in quarter_ic and quarter_ic["rank_ic"].notna().any()
            else None
        )
        quarter_coverage = coverage
        if isinstance(quarter_coverage, pd.DataFrame) and "date" in quarter_coverage.columns:
            coverage_dates = pd.to_datetime(quarter_coverage["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
            quarter_coverage = quarter_coverage.loc[coverage_dates.dt.to_period("Q") == quarter]
        coverage_mean = (
            float(pd.to_numeric(quarter_coverage["coverage"], errors="coerce").mean())
            if isinstance(quarter_coverage, pd.DataFrame) and "coverage" in quarter_coverage and quarter_coverage["coverage"].notna().any()
            else None
        )
        complete_calendar = bool(
            not piece.empty
            and actual_end is not None
            and actual_end >= quarter.end_time.normalize()
            and (evidence_timestamp is None or evidence_timestamp >= quarter.end_time.normalize())
            and status == "complete"
        )
        missing_tail = bool(
            status != "complete"
            or piece.empty
            or (evidence_timestamp is not None and evidence_timestamp < quarter.end_time.normalize())
        )
        rows.append(
            {
                "quarter": str(quarter),
                "start_date": start_date,
                "end_date": end_date,
                "n_periods": int(len(piece)),
                "starting_equity": starting,
                "ending_equity": ending,
                "equity_return": equity_return,
                "reconciled": reconciled,
                "status": status,
                "missing_tail": missing_tail,
                **summary,
                "mean_rank_ic": mean_rank_ic,
                "ic_dates": int(len(quarter_ic)) if isinstance(quarter_ic, pd.DataFrame) else 0,
                "prediction_days": int(quarter_prediction_dates.nunique()),
                "expected_calendar_days": int((quarter.end_time.normalize() - quarter.start_time).days + 1),
                "coverage_mean": coverage_mean,
                "complete_calendar_quarter": complete_calendar,
                "accounting_tail_only": bool(piece.empty and len(quarter_prediction_dates)),
                "evaluated_through": actual_end.date().isoformat() if actual_end is not None else None,
                "accounting_status": status,
                "metrics_scope": "full_run_ledger_slice" if status == "complete" else "certified_prefix_slice",
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


def _render_quarterly_html(
    quarterly: pd.DataFrame,
    *,
    path: Path,
    factor_name: str,
    accounting_status: str,
    standard_report_path: str | Path | None = None,
) -> None:
    """Write the ML-specific quarterly view and its interpretation notes."""
    disclaimer = (
        "Quarterly portfolio returns are compounded from slices of one "
        "continuous all-costs ledger. forward-label diagnostics (group "
        "returns) are not cost-adjusted portfolio profits. Incomplete "
        "accounting and the final execution tail are shown as unavailable or "
        "certified-prefix data rather than zero-filled returns."
    )
    table_html = quarterly.to_html(index=False, escape=True, na_rep="unavailable")
    report_link = ""
    if standard_report_path:
        report_name = Path(standard_report_path).name
        report_link = (
            f"<p><a href='{html.escape(report_name, quote=True)}'>"
            "Continuous NAV / standard evaluation report</a></p>"
        )
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        f"<title>{html.escape(factor_name)} — quarterly OOS evaluation</title>"
        "<style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse}"
        "td,th{padding:8px;border:1px solid #ddd}th{background:#eee}"
        ".note{max-width:1000px;line-height:1.5}</style>"
        f"<h1>{html.escape(factor_name)} — quarterly OOS evaluation</h1>"
        f"<p class='note'>{html.escape(disclaimer)}</p>"
        f"<p>All-costs accounting status: {html.escape(str(accounting_status))}</p>"
        f"{report_link}"
        f"{table_html}</html>"
    )
    path.write_text(page, encoding="utf-8")


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
    evidence_end: Any = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Evaluate and persist one continuous ML OOS prediction stream.

    All accounting tables are copied exactly from the manager result.  A
    halted all-costs scenario therefore remains halted in both the JSON and
    the report; no zero-filled tail or second per-quarter backtest is made.
    The canonical artifacts are ``factor_oos.parquet``, ``daily_ledger.parquet``,
    ``group_forward_return_diagnostics.parquet``, and ``quarterly.html``.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table = _factor_table(predictions)
    prediction_start = table["date"].min().normalize()
    prediction_end = table["date"].max().normalize()
    if isinstance(config, Mapping):
        holding_days = int(config.get("holding_days", 3))
    else:
        holding_days = int(getattr(config, "holding_days", 3)) if config is not None else 3
    if evidence_end is None and isinstance(config, Mapping):
        evidence_end = config.get("evidence_end")
    if evidence_end is None and config is not None and hasattr(config, "evidence_end"):
        evidence_end = getattr(config, "evidence_end")
    if evidence_end is None:
        evidence_day = prediction_end + pd.Timedelta(days=holding_days + 1)
    else:
        evidence_day = pd.Timestamp(_iso_date(evidence_end, "evidence_end"))
    evaluation_end = min(
        prediction_end,
        evidence_day - pd.Timedelta(days=holding_days + 1),
    )
    if evaluation_end < prediction_start:
        raise ValueError("evidence_end does not leave an executable OOS signal interval")
    evaluated = table.loc[table["date"].between(prediction_start, evaluation_end)].copy()

    # Keep the canonical name from the written plan and a compatibility alias
    # used by early callers of this standalone helper.
    factor_path = destination / "factor_oos.parquet"
    table.to_parquet(factor_path, index=False)
    factor_alias = destination / "factor.parquet"
    table.to_parquet(factor_alias, index=False)

    params = manager_params(
        config,
        start=prediction_start,
        end=evaluation_end,
        split_date=_iso_date(prediction_start - pd.Timedelta(days=1), "split_date"),
        n_groups=5,
    )
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
        kwargs["as_of"] = evidence_day
        manager = manager_cls(**kwargs)
    result = manager.evaluate(
        evaluated,
        factor_name=factor_name,
        profile_id="perp_1d",
        params=params,
        plot=plot,
    )
    if not isinstance(result, Mapping):
        raise TypeError("FactorManager.evaluate must return a mapping")

    paths: dict[str, str] = {
        "factor_parquet": str(factor_path),
        "factor_oos": str(factor_path),
        "factor_alias": str(factor_alias),
    }
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
    quarterly = quarterly_metrics(
        all_ledger,
        predictions=table,
        evidence_end=evidence_day,
        status=scenario_status,
    )
    quarterly_path = destination / "quarterly_metrics.parquet"
    quarterly.to_parquet(quarterly_path, index=False)
    paths["quarterly_metrics"] = str(quarterly_path)

    daily_ic = _daily_ic(result)
    daily_ic_path = destination / "daily_ic.parquet"
    daily_ic.to_parquet(daily_ic_path, index=False)
    paths["daily_ic"] = str(daily_ic_path)

    group_returns = result.get("group_returns")
    if isinstance(group_returns, pd.DataFrame):
        group_path = destination / "group_forward_return_diagnostics.parquet"
        _write_table(group_path, group_returns)
        group_alias = destination / "group_returns.parquet"
        _write_table(group_alias, group_returns)
        paths["group_returns"] = str(group_alias)
        paths["group_forward_return_diagnostics"] = str(group_path)
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

    quarterly_html_path = destination / "quarterly.html"
    standard_report_path = result.get("paths", {}).get("report_path") if isinstance(result.get("paths"), Mapping) else None
    _render_quarterly_html(
        quarterly,
        path=quarterly_html_path,
        factor_name=factor_name,
        accounting_status=str(scenario_status),
        standard_report_path=standard_report_path,
    )
    paths["quarterly_html"] = str(quarterly_html_path)

    if isinstance(all_ledger, pd.DataFrame):
        daily_ledger_path = destination / "daily_ledger.parquet"
        _write_table(daily_ledger_path, all_ledger)
        paths["daily_ledger"] = str(daily_ledger_path)

    evaluation = {
        "status": result.get("status"),
        "all_costs_status": scenario_status,
        "factor_name": factor_name,
        "prediction_start": prediction_start.date().isoformat(),
        "prediction_end": prediction_end.date().isoformat(),
        "evaluation_signal_start": prediction_start.date().isoformat(),
        "evaluation_signal_end": evaluation_end.date().isoformat(),
        "evidence_end": evidence_day.date().isoformat(),
        "manager_params": params,
        "metadata": _json_safe(result.get("metadata", {})),
        "factor_performance": _json_safe(result.get("factor_performance", {})),
        "scenarios": scenario_payload,
        "quarterly_metrics": _json_safe(quarterly.to_dict(orient="records")),
        "group_diagnostics": _json_safe(group_diagnostics),
        "group_diagnostics_scope": (
            "forward-label diagnostics (group returns) are not "
            "cost-adjusted portfolio profits."
        ),
        "paths": paths,
    }
    evaluation_path = destination / "evaluation.json"
    paths["evaluation_json"] = str(evaluation_path)
    evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {**evaluation, "paths": paths, "manager_result": result}


__all__ = ["evaluate_oos", "manager_params", "quarterly_metrics"]
