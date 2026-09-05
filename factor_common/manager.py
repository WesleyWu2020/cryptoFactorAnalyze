"""FactorManager: orchestrate factor evaluation, persistence, and rendering.

The manager wires together the focused ``factor_common`` modules without
duplicating their internals: loading/validation (``loader``,
``validation``), value computation (``value_engine``), versioned artifacts
(``storage``), accounting (``backtest``), metrics (``metrics``), grouping
(``grouping``), evaluation-only labels (``labels``), and HTML rendering
(``reporting``).

Workflow per ``evaluate`` call: resolve dates/Profile/source -> snapshot the
input store and build metadata -> compute and validate values -> persist
values -> load execution-tail prices, funding events, and quality evidence ->
backtest -> evaluate metrics -> persist the evaluation -> optionally render.
Values are persisted before any funding/execution data is touched, so a
funding failure can never discard a successfully computed factor value.

Data limitations surface as structured results (``status="incomplete"`` or
``"insufficient_data"``); invalid user configuration, corrupt schemas, and
I/O failures raise exceptions. Group return tables (computed from
evaluation-only labels at evaluation time) are saved inside the evaluation
artifact so rendering never recomputes them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data.crypto_quant.store import CryptoQuantStore

from .backtest import SCENARIOS, run_backtest
from .data_provider import DataProvider
from .grouping import assign_groups
from .labels import make_labels
from .loader import _validate_identifier
from .loader import create_template as _create_template
from .loader import load_factor
from .metrics import evaluate_metrics
from .profiles import resolve_profile
from .reporting import render_result
from .storage import FactorStorage, snapshot_source_stats
from .validation import check_cutoff, scan_future_leaks
from .value_engine import compute_factor

_SCENARIO_TABLES = (
    "ledger",
    "orders",
    "positions",
    "valuation_prices",
    "funding",
    "funding_coverage",
)
_DATE_PARAM_KEYS = frozenset({"start", "end"})
_PROFILE_PARAM_KEYS = frozenset({
    "rebalance_days",
    "anchor_date",
    "n_groups",
    "factor_direction",
    "fee_rate",
    "include_funding",
    "funding_price_mode",
    "split_date",
    "out_of_sample_days",
})
_ALLOWED_PARAM_KEYS = _DATE_PARAM_KEYS | _PROFILE_PARAM_KEYS
_DEFAULT_SIGNAL_DAYS = 365
_LONG_TABLE_COLUMNS = frozenset({"date", "instrument", "factor"})


def _hash_frame(frame: pd.DataFrame) -> str:
    """Content hash of a frame's values, index, and columns."""
    return hashlib.sha256(frame.to_csv().encode("utf-8")).hexdigest()


def _iso(day) -> str:
    return pd.Timestamp(day).date().isoformat()


def _signal_day(value, name: str) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if pd.isna(day):
        raise ValueError(f"{name} must be a valid date")
    if day.tzinfo is not None:
        day = day.tz_convert("UTC").tz_localize(None)
    if day != day.normalize():
        raise ValueError(f"{name} must be a daily date without intraday time")
    return day.normalize()


def _normalize_external_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize an external long table or date-by-instrument matrix.

    Duplicate date/instrument keys and unrecognized axes are errors; intraday
    timestamps are rejected rather than implicitly rounded. The result is a
    complete-calendar daily matrix with non-finite values masked to NaN.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("external factor source must be a DataFrame")
    if _LONG_TABLE_COLUMNS.issubset(frame.columns):
        table = frame.loc[:, ["date", "instrument", "factor"]].copy()
        dates = pd.to_datetime(table["date"], errors="raise", utc=True)
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
        if dates.hasnans:
            raise ValueError("factor date column contains invalid dates")
        if not dates.equals(dates.dt.normalize()):
            raise ValueError(
                "factor date column must hold daily midnights; "
                "intraday timestamps are never implicitly rounded"
            )
        table["date"] = dates
        table["instrument"] = table["instrument"].astype(str)
        if table.duplicated(["date", "instrument"]).any():
            raise ValueError("duplicate date/instrument key in factor data")
        matrix = table.pivot(index="date", columns="instrument", values="factor")
    elif "date" in frame.columns or "instrument" in frame.columns:
        raise ValueError(
            "unrecognized factor data axes: expected a date/instrument/factor "
            "long table or a date-by-instrument matrix"
        )
    else:
        index = frame.index
        if not (
            isinstance(index, pd.DatetimeIndex)
            or pd.api.types.is_object_dtype(index)
            or pd.api.types.is_string_dtype(index)
        ):
            raise ValueError(
                "unrecognized factor data axes: expected a date/instrument/"
                "factor long table or a date-by-instrument matrix"
            )
        try:
            index = pd.to_datetime(index, errors="raise", utc=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "unrecognized factor data axes: expected a date/instrument/"
                "factor long table or a date-by-instrument matrix"
            ) from exc
        index = pd.DatetimeIndex(index).tz_convert("UTC").tz_localize(None)
        if index.hasnans or not index.equals(index.normalize()):
            raise ValueError(
                "factor matrix index must be daily midnights; "
                "intraday timestamps are never implicitly rounded"
            )
        matrix = frame.copy()
        matrix.index = index

    if len(matrix.index) == 0 or len(matrix.columns) == 0:
        raise ValueError("factor data is empty")
    if not matrix.index.is_unique:
        raise ValueError("duplicate date axis in factor data")
    matrix = matrix.sort_index()
    matrix.columns = pd.Index([str(column) for column in matrix.columns])
    if not matrix.columns.is_unique:
        raise ValueError("duplicate instrument axis in factor data")
    try:
        matrix = matrix.astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("factor values must be numeric") from exc
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    calendar = pd.date_range(matrix.index[0], matrix.index[-1], freq="D", name="date")
    matrix = matrix.reindex(calendar)
    matrix.columns.name = None
    return matrix


def _flatten_accounting(accounting: dict) -> dict:
    """Lift scenario DataFrames to top-level ``<scenario>__<table>`` keys."""
    flattened = {}
    scenarios = {}
    for name, scenario in accounting["scenarios"].items():
        scalars = {}
        for key, value in scenario.items():
            if isinstance(value, pd.DataFrame):
                flattened[f"{name}__{key}"] = value
            else:
                scalars[key] = value
        scenarios[name] = scalars
    result = {key: value for key, value in accounting.items() if key != "scenarios"}
    result["scenarios"] = scenarios
    flattened["factor_result"] = result
    return flattened


class FactorManager:
    """Coordinate factor computation, storage, evaluation, and reporting.

    Defaults resolve against the project root (``data/crypto_quant.h5``,
    ``data/factor_results``, ``reports``); explicit absolute or relative
    paths are honored as given (relative paths resolve against the current
    working directory).
    """

    def __init__(self, h5_path=None, base_dir=None, *, reports_dir=None,
                 project_root=None):
        if project_root is not None:
            root = Path(project_root).resolve()
        else:
            root = Path(__file__).resolve().parent.parent
        self.project_root = root
        self.h5_path = (
            Path(h5_path) if h5_path is not None else root / "data" / "crypto_quant.h5"
        )
        self.base_dir = (
            Path(base_dir) if base_dir is not None else root / "data" / "factor_results"
        )
        self.reports_dir = (
            Path(reports_dir) if reports_dir is not None else root / "reports"
        )
        self.storage = FactorStorage(self.base_dir)
        self._dp: DataProvider | None = None

    @property
    def dp(self) -> DataProvider:
        """Point-in-time data provider over the configured H5 store."""
        if self._dp is None:
            self._dp = DataProvider(self.h5_path)
        return self._dp

    # ------------------------------------------------------------------
    # parameter and source resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _split_params(params):
        if params is None:
            return None, None, {}
        if not isinstance(params, Mapping):
            raise TypeError("params must be a mapping")
        unknown = sorted(set(params) - _ALLOWED_PARAM_KEYS)
        if unknown:
            allowed = ", ".join(sorted(_ALLOWED_PARAM_KEYS))
            raise ValueError(
                f"Unknown params key: {unknown[0]!r} (allowed: {allowed}); "
                "factor formula parameters belong in SETTING"
            )
        start = (
            _signal_day(params["start"], "start") if "start" in params else None
        )
        end = _signal_day(params["end"], "end") if "end" in params else None
        overrides = {
            key: value for key, value in params.items() if key in _PROFILE_PARAM_KEYS
        }
        return start, end, overrides

    def _default_end(self) -> pd.Timestamp:
        _, last = self.dp.get_time_range()
        if last is None:
            raise ValueError(
                "cannot default the signal end: the H5 store has no market "
                "data; pass explicit params start/end"
            )
        return last

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def _cutoff_check(self, spec, full_values, start, end) -> dict:
        span = (end - start).days
        candidates = [start + pd.Timedelta(days=span // 2), end - pd.Timedelta(days=1)]
        cutoffs = sorted({day for day in candidates if start <= day < end}) or [end]

        def compute(cutoff):
            if cutoff is None:
                return full_values
            cutoff = pd.Timestamp(cutoff)
            truncated = DataProvider(self.h5_path, as_of=cutoff)
            matrix, _ = compute_factor(
                spec, truncated, start=start, end=min(end, cutoff)
            )
            return matrix

        try:
            return check_cutoff(compute, cutoffs)
        except AssertionError as exc:
            raise ValueError(
                f"cutoff replay detected a future leak: {exc}"
            ) from None

    # ------------------------------------------------------------------
    # benchmark
    # ------------------------------------------------------------------

    def _load_benchmark(self):
        """Load optional CMC100 closes as a daily ``cmc100`` DataFrame."""
        table = CryptoQuantStore(self.h5_path).read("cmc100_daily")
        if table.empty or "index_value" not in table.columns:
            return None
        dates = pd.to_datetime(table["date"], errors="coerce", utc=True)
        dates = dates.dt.tz_localize(None).dt.normalize()
        frame = pd.DataFrame(
            {"cmc100": pd.to_numeric(table["index_value"], errors="coerce").to_numpy()},
            index=pd.DatetimeIndex(dates, name="date"),
        )
        frame = frame.dropna()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame if not frame.empty else None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def evaluate(self, source, *, factor_name=None, profile_id="perp_1d",
                 params=None, plot=True):
        """Return the standard result dictionary; source is a .py path or DataFrame."""
        start_param, end_param, overrides = self._split_params(params)

        spec = None
        external_matrix = None
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            spec = load_factor(source_path)
            if factor_name is not None and factor_name != spec.factor_id:
                raise ValueError(
                    f"factor_name {factor_name!r} does not match module factor "
                    f"{spec.factor_id!r}"
                )
            factor_id = spec.factor_id
            scan_findings = scan_future_leaks([source_path])
            if scan_findings:
                first = scan_findings[0]
                raise ValueError(
                    f"future-leak pattern {first['pattern']!r} at "
                    f"{first['path']}:{first['line']}: {first['source']}"
                )
        elif isinstance(source, pd.DataFrame):
            if factor_name is None:
                raise ValueError(
                    "external DataFrame factors require an explicit factor_name "
                    "for persistence"
                )
            factor_id = _validate_identifier(factor_name)
            external_matrix = _normalize_external_matrix(source)
        else:
            raise TypeError("source must be a .py path or a DataFrame")

        if spec is not None and "factor_direction" not in overrides:
            overrides["factor_direction"] = spec.setting["factor_direction"]
        profile = resolve_profile(profile_id, overrides)

        if spec is not None:
            end = end_param if end_param is not None else self._default_end()
            start = (
                start_param
                if start_param is not None
                else end - pd.Timedelta(days=_DEFAULT_SIGNAL_DAYS - 1)
            )
            if start > end:
                raise ValueError("start must be on or before end")
            warmup = spec.setting["warmup_bars"]
            history_start = start - pd.Timedelta(days=warmup)
            input_hashes = {
                field: _hash_frame(
                    self.dp.get_single_data(field, start=history_start, end=end)
                )
                for field in spec.setting["data_needed"]
            }
            input_hashes["membership"] = _hash_frame(
                self.dp.get_universe(start=history_start, end=end)
            )
            values, value_diag = compute_factor(spec, self.dp, start=start, end=end)
            validation = {
                "static_scan": {"status": "clean", "findings": []},
                "cutoff": self._cutoff_check(spec, values, start, end),
            }
            value_metadata = {
                "source_type": "module",
                "source_sha256": spec.source_sha256,
                "settings": dict(spec.setting),
                "requested_start": _iso(start),
                "requested_end": _iso(end),
                "loaded_start": _iso(history_start),
                "loaded_end": _iso(end),
                "input_hashes": input_hashes,
            }
            preprocessing = spec.setting["preprocessing"]
            source_type = "module"
        else:
            end = (
                end_param if end_param is not None else external_matrix.index[-1]
            )
            start = (
                start_param
                if start_param is not None
                else external_matrix.index[0]
            )
            if start > end:
                raise ValueError("start must be on or before end")
            calendar = pd.date_range(start, end, freq="D", name="date")
            values = external_matrix.reindex(calendar)
            valid = int(values.notna().to_numpy().sum())
            value_diag = {
                "eligible_count": None,
                "ineligible_count": None,
                "missing_count": None,
                "valid_count": valid,
            }
            source_sha = _hash_frame(external_matrix)
            validation = {
                "static_scan": {
                    "status": "not_applicable",
                    "reason": "external DataFrame inputs carry no source code",
                },
                "cutoff": check_cutoff(None, []),
            }
            value_metadata = {
                "source_type": "dataframe",
                "source_sha256": source_sha,
                "settings": {"external": True},
                "requested_start": _iso(start),
                "requested_end": _iso(end),
                "loaded_start": _iso(external_matrix.index[0]),
                "loaded_end": _iso(external_matrix.index[-1]),
                "input_hashes": {"external_matrix": source_sha},
            }
            preprocessing = "external"
            source_type = "dataframe"

        # Persist values before touching execution-tail or funding data so a
        # later funding failure can never discard the computed factor value.
        # The H5 store's read path bumps the file mtime on every open, so the
        # change-detection snapshot is taken only after the final value-phase
        # read; save_value re-stats the file and aborts if it changed between
        # this snapshot and the write.
        value_metadata["source_stat_before"] = snapshot_source_stats([self.h5_path])
        saved = self.storage.save_value(factor_id, values, value_metadata)
        run_id = saved["run_id"]

        tail_end = end + pd.Timedelta(days=1 + profile.rebalance_days)
        tail_index = pd.date_range(start, tail_end, freq="D", name="date")
        bt_values = values.reindex(tail_index)
        opens = self.dp.get_single_data("open", start=start, end=tail_end)
        opens = opens.reindex(columns=values.columns)
        events = self.dp.get_funding(
            start=start, end=tail_end, symbols=list(values.columns)
        )
        quality = self.dp.get_quality(
            start=start - pd.Timedelta(days=1),
            end=tail_end,
            symbols=list(values.columns),
        )

        accounting = run_backtest(
            bt_values, opens, events, quality, profile,
            signal_start=start, signal_end=end,
        )
        # Labels are evaluation-only future data: they feed metrics and group
        # returns here and never enter factor construction or grouping.
        labels = make_labels(opens, profile.rebalance_days).loc[start:end]
        performance = evaluate_metrics(values, labels, accounting, profile)

        groups, grouping_diag = assign_groups(values, profile.n_groups)
        group_returns = pd.DataFrame(
            {
                f"group_{group_id}": labels.where(groups == group_id).mean(axis=1)
                for group_id in range(1, profile.n_groups + 1)
            },
            dtype="float64",
        )
        group_returns.index.name = "date"

        benchmark = self._load_benchmark()

        if value_diag["valid_count"] == 0 or accounting["diagnostics"]["no_usable_signals"]:
            status = "insufficient_data"
        elif accounting["status"] != "complete":
            status = "incomplete"
        else:
            status = "complete"

        coverage_table = accounting["scenarios"]["all_costs"].get("funding_coverage")
        funding_coverage = {
            "include_funding": bool(profile.include_funding),
            "status_counts": {},
        }
        if isinstance(coverage_table, pd.DataFrame) and not coverage_table.empty:
            funding_coverage["status_counts"] = {
                str(key): int(count)
                for key, count in coverage_table["status"].value_counts().items()
            }
        if spec is not None:
            signal_coverage = {
                "eligible_count": value_diag["eligible_count"],
                "ineligible_count": value_diag["ineligible_count"],
                "missing_count": value_diag["missing_count"],
                "valid_count": value_diag["valid_count"],
            }
        else:
            signal_coverage = {
                "valid_count": value_diag["valid_count"],
                "cell_count": int(values.size),
            }
        diagnostics = {
            "value": value_diag,
            "grouping": {
                "insufficient_dates": [
                    _iso(day) for day in grouping_diag["insufficient_dates"]
                ]
            },
            "validation": validation,
            "coverage": {
                "signal": signal_coverage,
                "label": performance["samples"]["full"]["coverage"],
                "funding": funding_coverage,
            },
        }
        result_metadata = {
            "factor_name": factor_id,
            "profile_id": profile.profile_id,
            "factor_direction": profile.factor_direction,
            "n_groups": profile.n_groups,
            "signal_start": _iso(start),
            "signal_end": _iso(end),
            "source_type": source_type,
            "preprocessing": preprocessing,
        }

        stored = {
            "status": status,
            "profile": profile,
            "evaluation_inputs": {
                "execution_tail_prices": _hash_frame(opens),
                "funding_events": _hash_frame(events),
                "coverage_evidence": _hash_frame(quality),
                "benchmark": _hash_frame(benchmark) if benchmark is not None else None,
            },
            "metadata": result_metadata,
            "factor_performance": performance,
            "group_returns": group_returns,
            "diagnostics": diagnostics,
        }
        stored.update(_flatten_accounting(accounting))
        if benchmark is not None:
            stored["benchmark"] = benchmark
        saved_evaluation = self.storage.save_evaluation(factor_id, run_id, stored)
        evaluation_id = saved_evaluation["evaluation_id"]

        paths = {
            "run_id": run_id,
            "evaluation_id": evaluation_id,
            "factor_path": str(saved["factor_path"]),
            "metadata_path": str(saved["metadata_path"]),
            "evaluation_dir": str(saved_evaluation["dir"]),
            "report_path": None,
        }
        result = {
            "status": status,
            "factor_value": values,
            "factor_performance": performance,
            "factor_result": accounting,
            "group_returns": group_returns,
            "diagnostics": diagnostics,
            "metadata": {
                **result_metadata,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            "paths": paths,
            "run_id": run_id,
            "evaluation_id": evaluation_id,
        }
        if benchmark is not None:
            result["benchmark"] = benchmark

        if plot:
            report_path = (
                self.reports_dir
                / f"{factor_id}_{profile.rebalance_days}d_{run_id}.html"
            )
            info = render_result(result, report_path, start=start, end=end)
            paths["report_path"] = info["output_path"]
        return result

    def create_template(self, factor_name="example_factor", *, output_dir=None):
        """Return a new daily factor file path; never overwrite an existing file."""
        directory = (
            Path(output_dir)
            if output_dir is not None
            else self.project_root / "factor_analyse" / "factor_mining"
        )
        return _create_template(factor_name, directory)

    def get_value(self, factor_name, *, run_id=None):
        """Return the saved date-by-instrument matrix."""
        return self.storage.get_value(factor_name, run_id=run_id)

    def get_performance(self, factor_name, *, run_id=None, evaluation_id=None):
        """Return saved metrics for an explicitly selected or latest complete evaluation."""
        try:
            loaded = self.storage.load_evaluation(
                factor_name, run_id=run_id, evaluation_id=evaluation_id
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"evaluation lookup failed for factor {factor_name!r}: {exc}"
            ) from exc
        tables = {
            key: value for key, value in loaded.items()
            if isinstance(value, pd.DataFrame)
        }
        flat_result = loaded["factor_result"]
        scenarios = {}
        for name in SCENARIOS:
            scenario = dict(flat_result["scenarios"][name])
            prefix = f"{name}__"
            for table_key, frame in tables.items():
                if table_key.startswith(prefix):
                    scenario[table_key[len(prefix):]] = frame
            scenarios[name] = scenario
        accounting = {
            key: value for key, value in flat_result.items() if key != "scenarios"
        }
        accounting["scenarios"] = scenarios
        result = {
            "status": loaded["status"],
            "factor_performance": loaded["factor_performance"],
            "factor_result": accounting,
            "diagnostics": loaded.get("diagnostics", {}),
            "metadata": dict(loaded.get("metadata", {})),
            "run_id": loaded["run_id"],
            "evaluation_id": loaded["evaluation_id"],
            "paths": {
                "run_id": loaded["run_id"],
                "evaluation_id": loaded["evaluation_id"],
            },
        }
        if "group_returns" in tables:
            result["group_returns"] = tables["group_returns"]
        if "benchmark" in tables:
            result["benchmark"] = tables["benchmark"]
        result["factor_value"] = self.storage.get_value(
            factor_name, run_id=loaded["run_id"]
        )
        return result

    def plot_result(self, result, *, output_path=None, start=None, end=None):
        """Render an existing result, optionally slicing its real timestamps."""
        if output_path is None:
            factor_name = result.get("metadata", {}).get("factor_name", "factor")
            run_id = result.get("run_id") or result.get("paths", {}).get("run_id") or "result"
            output_path = self.reports_dir / f"{factor_name}_{run_id}.html"
        return render_result(result, output_path, start=start, end=end)


__all__ = ["FactorManager"]
