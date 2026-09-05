"""Versioned Parquet/JSON persistence for factor values and evaluations.

Factor matrices are stored per ``<factor_id>/<run_id>/`` as a long-format
``factor.parquet`` (columns ``date``, ``instrument``, ``factor``; unique,
stably sorted keys) plus a ``metadata.json`` recording the full date/column
axes so rows or columns without any finite value round-trip exactly. Run IDs
are content-derived from canonical JSON over the caller-supplied metadata
(source digest, formula settings, requested/loaded ranges, input-content
hashes including membership) and the framework version, so a parameter or
data-fingerprint change can never silently reuse a previous result.

Evaluations live under ``<run_id>/evaluations/<evaluation_id>/``; the
evaluation ID additionally covers the profile and the evaluation input hashes
(execution-tail prices, funding events, coverage evidence), so a fee-only
profile change produces a new evaluation without touching the factor value.
Artifacts are written in a sibling temporary directory, renamed when complete,
and only then is a small success pointer atomically replaced. Incomplete
evaluations remain loadable by ID without promoting the latest-complete
pointer; a successful factor value is promoted independently.

Only Parquet tables and JSON scalars are written — code and arbitrary objects
are never pickled. Non-finite JSON scalars are serialized as ``null``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

FRAMEWORK_VERSION = "1"
VALUE_TABLE_COLUMNS = ("date", "instrument", "factor")
VALID_EVALUATION_STATUS = frozenset({"complete", "incomplete", "insufficient_data"})
_VOLATILE_METADATA_KEYS = frozenset({"created_at", "source_stat_before"})


class ConcurrentSourceChangeError(RuntimeError):
    """Raised when a source snapshot file changed after it was read."""


def snapshot_source_stats(paths: Iterable[str | Path]) -> dict[str, dict[str, int]]:
    """Capture mtime/size per source file for later change detection.

    Callers snapshot the H5 inputs before reading them and pass the result as
    ``metadata["source_stat_before"]``; ``save_value`` re-stats the same files
    and aborts when any of them changed in between.
    """
    stats = {}
    for path in paths:
        stat = os.stat(path)
        stats[str(path)] = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    return stats


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    raise TypeError(
        f"Unsupported value in storage metadata: {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _artifact_id(payload: Mapping) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:16]


def _write_json(path: Path, payload: Mapping) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _validate_factor_id(factor_id: Any) -> None:
    if (
        not isinstance(factor_id, str)
        or not factor_id
        or factor_id in {".", ".."}
        or "/" in factor_id
        or "\\" in factor_id
    ):
        raise ValueError(f"invalid factor_id: {factor_id!r}")


def _normalize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix must be a DataFrame with daily axes")
    index = pd.DatetimeIndex(matrix.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    if index.hasnans or not index.equals(index.normalize()):
        raise ValueError("matrix date axis must be daily midnight without NaT")
    if not index.is_unique:
        raise ValueError("matrix date axis must be unique")
    columns = pd.Index([str(column) for column in matrix.columns])
    if not columns.is_unique:
        raise ValueError("matrix instrument axis must be unique")
    normalized = matrix.copy()
    normalized.index = index.rename("date")
    normalized.columns = columns
    normalized = normalized.sort_index(axis=0).sort_index(axis=1)
    normalized.columns.name = None
    try:
        normalized = normalized.astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix values must be numeric") from exc
    return normalized


def _to_long_table(matrix: pd.DataFrame) -> pd.DataFrame:
    values = matrix.to_numpy(dtype="float64")
    rows, cols = np.nonzero(np.isfinite(values))
    table = pd.DataFrame(
        {
            "date": matrix.index.take(rows),
            "instrument": matrix.columns.take(cols),
            "factor": values[rows, cols],
        }
    )
    return table.sort_values(list(VALUE_TABLE_COLUMNS[:2]), kind="mergesort").reset_index(
        drop=True
    )


def _axes_payload(matrix: pd.DataFrame) -> dict[str, list]:
    return {
        "dates": [day.isoformat() for day in matrix.index],
        "columns": list(matrix.columns),
    }


def _dates_index(dates: Iterable[str]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex([pd.Timestamp(day) for day in dates], name="date")
    if len(index) and index.equals(pd.date_range(index[0], index[-1], freq="D")):
        index = pd.date_range(index[0], index[-1], freq="D", name="date")
    return index


def _verify_source_stats(before: Any) -> None:
    if before is None:
        return
    if not isinstance(before, Mapping):
        raise TypeError("source_stat_before must be a mapping of path to stat")
    for path, previous in before.items():
        try:
            current = snapshot_source_stats([path])[str(path)]
        except OSError as exc:
            raise ConcurrentSourceChangeError(
                f"source file vanished after the snapshot was read: {path}"
            ) from exc
        if current != _json_safe(previous):
            raise ConcurrentSourceChangeError(
                f"source file changed after the snapshot was read: {path}"
            )


def _temp_sibling(final_dir: Path) -> Path:
    return final_dir.with_name(f"{final_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")


class FactorStorage:
    """Atomic versioned storage rooted at ``base_dir``."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def _factor_dir(self, factor_id: str) -> Path:
        _validate_factor_id(factor_id)
        return self.base_dir / factor_id

    def _run_dir(self, factor_id: str, run_id: str | None) -> Path:
        factor_dir = self._factor_dir(factor_id)
        if run_id is None:
            pointer = factor_dir / "latest_run.json"
            if not pointer.is_file():
                raise FileNotFoundError(
                    f"no completed factor run recorded for {factor_id!r}"
                )
            run_id = json.loads(pointer.read_text())["run_id"]
        run_dir = factor_dir / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"unknown run {run_id!r} for {factor_id!r}")
        return run_dir

    def _promote_pointer(self, path: Path, payload: Mapping) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _temp_sibling(path)
        _write_json(tmp, payload)
        os.replace(tmp, path)

    def save_value(
        self, factor_id: str, matrix: pd.DataFrame, metadata: Mapping
    ) -> dict[str, Any]:
        """Persist a factor matrix and promote it as the latest successful run.

        ``metadata`` should carry the source digest, formula settings,
        requested/loaded ranges, and normalized input-content hashes including
        membership; together with the framework version these define the run
        ID. Re-saving identical inputs with different values is rejected.
        """
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        normalized = _normalize_matrix(matrix)
        table = _to_long_table(normalized)
        axes = _axes_payload(normalized)
        run_id = _artifact_id(
            {
                "framework_version": FRAMEWORK_VERSION,
                "metadata": {
                    key: value
                    for key, value in metadata.items()
                    if key not in _VOLATILE_METADATA_KEYS
                },
            }
        )
        factor_dir = self._factor_dir(factor_id)
        run_dir = factor_dir / run_id
        if run_dir.is_dir():
            existing_meta = json.loads((run_dir / "metadata.json").read_text())
            existing = pd.read_parquet(run_dir / "factor.parquet")
            if existing_meta.get("axes") != axes or not existing.equals(table):
                raise ValueError(
                    "run already exists with different factor values; "
                    "input fingerprints must change when outputs change"
                )
        else:
            _verify_source_stats(metadata.get("source_stat_before"))
            payload = {
                "run_id": run_id,
                "factor_id": factor_id,
                "framework_version": FRAMEWORK_VERSION,
                "axes": axes,
                "diagnostics": {
                    "valid_count": int(len(table)),
                    "missing_count": int(normalized.size - len(table)),
                },
                "metadata": dict(metadata),
            }
            tmp_dir = _temp_sibling(run_dir)
            tmp_dir.mkdir(parents=True)
            try:
                table.to_parquet(tmp_dir / "factor.parquet", index=False)
                _write_json(tmp_dir / "metadata.json", payload)
                os.rename(tmp_dir, run_dir)
            except BaseException:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise
        self._promote_pointer(factor_dir / "latest_run.json", {"run_id": run_id})
        return {
            "run_id": run_id,
            "factor_path": run_dir / "factor.parquet",
            "metadata_path": run_dir / "metadata.json",
        }

    @staticmethod
    def _assert_same_evaluation(
        eval_dir: Path, payload: Mapping, tables: Mapping[str, pd.DataFrame]
    ) -> None:
        """Reject a re-save whose content differs from the stored evaluation."""
        conflict = (
            "evaluation already exists with different content; "
            "input fingerprints must change when outputs change"
        )
        stored = json.loads((eval_dir / "result.json").read_text())
        if _canonical_json(stored) != _canonical_json(payload):
            raise ValueError(conflict)
        for key, frame in tables.items():
            existing = pd.read_parquet(eval_dir / f"table-{key}.parquet")
            if not existing.equals(frame):
                raise ValueError(conflict)

    def get_value(self, factor_id: str, *, run_id: str | None = None) -> pd.DataFrame:
        """Rebuild the stored matrix for a run, defaulting to the latest success."""
        run_dir = self._run_dir(factor_id, run_id)
        metadata_path = run_dir / "metadata.json"
        table_path = run_dir / "factor.parquet"
        if not metadata_path.is_file() or not table_path.is_file():
            raise FileNotFoundError(f"incomplete factor artifact under {run_dir}")
        axes = json.loads(metadata_path.read_text())["axes"]
        table = pd.read_parquet(table_path)
        if table.duplicated(["date", "instrument"]).any():
            raise ValueError(f"duplicate date/instrument key in {table_path}")
        matrix = table.pivot(index="date", columns="instrument", values="factor")
        matrix = matrix.reindex(
            index=_dates_index(axes["dates"]), columns=pd.Index(axes["columns"])
        )
        matrix = matrix.astype("float64")
        matrix.index.name = "date"
        matrix.columns.name = None
        return matrix

    def save_evaluation(
        self, factor_id: str, run_id: str, result: Mapping
    ) -> dict[str, Any]:
        """Persist an evaluation result under its content-derived ID.

        The evaluation ID covers the run ID, the profile, and the evaluation
        input hashes (execution-tail prices, funding events, coverage
        evidence), so cost-only profile changes never overwrite prior results.
        Re-saving identical content is idempotent; re-saving the same ID with
        different content raises ``ValueError``, mirroring the value path.
        Only ``status="complete"`` results promote the latest-complete pointer;
        incomplete evaluations stay loadable by explicit ID.
        """
        if not isinstance(result, Mapping):
            raise TypeError("result must be a mapping")
        status = result.get("status")
        if status not in VALID_EVALUATION_STATUS:
            raise ValueError(
                f"evaluation status must be one of {sorted(VALID_EVALUATION_STATUS)}"
            )
        run_dir = self._run_dir(factor_id, run_id)
        evaluation_id = _artifact_id(
            {
                "framework_version": FRAMEWORK_VERSION,
                "run_id": run_id,
                "profile": result.get("profile"),
                "evaluation_inputs": result.get("evaluation_inputs"),
            }
        )
        eval_dir = run_dir / "evaluations" / evaluation_id
        tables = {
            key: value
            for key, value in result.items()
            if isinstance(value, pd.DataFrame)
        }
        scalars = {key: value for key, value in result.items() if key not in tables}
        payload = {
            "evaluation_id": evaluation_id,
            "run_id": run_dir.name,
            "framework_version": FRAMEWORK_VERSION,
            "tables": sorted(tables),
            "result": scalars,
        }
        if eval_dir.is_dir():
            self._assert_same_evaluation(eval_dir, payload, tables)
        else:
            tmp_dir = _temp_sibling(eval_dir)
            tmp_dir.mkdir(parents=True)
            try:
                for key, frame in tables.items():
                    frame.to_parquet(tmp_dir / f"table-{key}.parquet")
                _write_json(tmp_dir / "result.json", payload)
                os.rename(tmp_dir, eval_dir)
            except BaseException:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise
        if status == "complete":
            self._promote_pointer(
                run_dir / "evaluations" / "latest_complete.json",
                {"evaluation_id": evaluation_id},
            )
        return {"evaluation_id": evaluation_id, "dir": eval_dir}

    def load_evaluation(
        self,
        factor_id: str,
        *,
        run_id: str | None = None,
        evaluation_id: str | None = None,
    ) -> dict[str, Any]:
        """Load an evaluation, defaulting to the latest run's latest complete one."""
        run_dir = self._run_dir(factor_id, run_id)
        evaluations_dir = run_dir / "evaluations"
        if evaluation_id is None:
            pointer = evaluations_dir / "latest_complete.json"
            if not pointer.is_file():
                raise FileNotFoundError(
                    f"no complete evaluation recorded for run {run_dir.name!r}"
                )
            evaluation_id = json.loads(pointer.read_text())["evaluation_id"]
        result_path = evaluations_dir / evaluation_id / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(
                f"unknown evaluation {evaluation_id!r} for run {run_dir.name!r}"
            )
        payload = json.loads(result_path.read_text())
        result = dict(payload["result"])
        for key in payload["tables"]:
            result[key] = pd.read_parquet(result_path.parent / f"table-{key}.parquet")
        result["run_id"] = payload["run_id"]
        result["evaluation_id"] = payload["evaluation_id"]
        return result


__all__ = [
    "ConcurrentSourceChangeError",
    "FRAMEWORK_VERSION",
    "FactorStorage",
    "VALID_EVALUATION_STATUS",
    "VALUE_TABLE_COLUMNS",
    "snapshot_source_stats",
]
