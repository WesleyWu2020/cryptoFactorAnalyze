"""Parquet/JSON persistence for factor values and evaluations.

The legacy versioned mode stores factor matrices per ``<factor_id>/<run_id>/`` as a long-format
``factor.parquet`` (columns ``date``, ``instrument``, ``factor``; unique,
stably sorted keys) plus a ``metadata.json`` recording the full date/column
axes so rows or columns without any finite value round-trip exactly. Run IDs
are content-derived from canonical JSON over the caller-supplied metadata
(source digest, formula settings, requested/loaded ranges, input-content
hashes including membership) and the framework version, so a parameter or
data-fingerprint change can never silently reuse a previous result.

In versioned storage, evaluations live under
``<run_id>/evaluations/<evaluation_id>/``. In flat storage, only the current
evaluation is kept in ``<factor_id>.evaluation.json`` and table sidecars. The
evaluation ID additionally covers the profile and the evaluation input hashes
(execution-tail prices, funding events, coverage evidence), so a fee-only
profile change produces a new evaluation without touching the factor value.
Artifacts are written through temporary files and promoted only after they are
complete. Incomplete evaluations remain loadable by an explicit current ID,
while the default lookup accepts only a current complete evaluation.

The manager uses flat mode: one ``<factor_id>.parquet`` and one
``<factor_id>.meta.json`` for factor values. Evaluation JSON and table
sidecars are written only when the manager is constructed with
``persist_evaluations=True``. A cache is accepted only when source code,
settings, requested start, requested end, and source-file statistics still
match. If the date range is insufficient or metadata differs, the caller
recomputes the factor and atomically replaces the cache files.

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
    """Atomic factor storage rooted at ``base_dir``.

    ``flat=True`` stores one stable value file and metadata file per factor.
    Evaluation artifacts are also flat and keep only the current evaluation;
    a new evaluation replaces the previous evaluation and its table sidecars.
    """

    def __init__(self, base_dir: str | Path, *, flat: bool = False):
        self.base_dir = Path(base_dir)
        self.flat = flat

    @staticmethod
    def _flat_name(factor_id: str) -> str:
        _validate_factor_id(factor_id)
        return factor_id.lower()

    def _flat_paths(self, factor_id: str) -> tuple[Path, Path]:
        name = self._flat_name(factor_id)
        return (
            self.base_dir / f"{name}.parquet",
            self.base_dir / f"{name}.meta.json",
        )

    def _flat_evaluation_path(self, factor_id: str) -> Path:
        return self.base_dir / f"{self._flat_name(factor_id)}.evaluation.json"

    def _flat_evaluation_table_path(self, factor_id: str, key: str) -> Path:
        if (
            not isinstance(key, str)
            or not key
            or "/" in key
            or "\\" in key
        ):
            raise ValueError(f"invalid evaluation table key: {key!r}")
        return self.base_dir / f"{self._flat_name(factor_id)}.evaluation.{key}.parquet"

    def _evaluation_root(self, factor_id: str, run_id: str) -> Path:
        _validate_factor_id(factor_id)
        if self.flat:
            return self.base_dir / f"{self._flat_name(factor_id)}.evaluations" / run_id
        return self.base_dir / factor_id / "evaluations" / run_id

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
        if self.flat:
            return self._save_flat_value(factor_id, matrix, metadata)
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

    def _save_flat_value(
        self, factor_id: str, matrix: pd.DataFrame, metadata: Mapping
    ) -> dict[str, Any]:
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
        factor_path, metadata_path = self._flat_paths(factor_id)
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
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        temp_factor = factor_path.with_name(f".{factor_path.name}.tmp-{os.getpid()}")
        temp_meta = metadata_path.with_name(f".{metadata_path.name}.tmp-{os.getpid()}")
        try:
            table.to_parquet(temp_factor, index=False)
            _write_json(temp_meta, payload)
            os.replace(temp_factor, factor_path)
            os.replace(temp_meta, metadata_path)
        except BaseException:
            for path in (temp_factor, temp_meta):
                path.unlink(missing_ok=True)
            raise
        return {
            "run_id": run_id,
            "factor_path": factor_path,
            "metadata_path": metadata_path,
        }

    def load_cached_value(
        self,
        factor_id: str,
        *,
        source_sha256: str,
        settings: Mapping,
        requested_start: str,
        requested_end: str,
        source_stat: Mapping | None = None,
        pipeline_fingerprint: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        """Load a flat factor only when source/settings/date metadata still fit.

        ``pipeline_fingerprint`` identifies the value-pipeline code that
        produced the matrix; caches written before the fingerprint existed
        (or by older code) are rejected rather than silently reused.
        """
        if not self.flat:
            return None
        factor_path, metadata_path = self._flat_paths(factor_id)
        if not factor_path.is_file() or not metadata_path.is_file():
            return None
        try:
            payload = json.loads(metadata_path.read_text())
            metadata = payload["metadata"]
            if metadata.get("source_sha256") != source_sha256:
                return None
            if _canonical_json(metadata.get("settings")) != _canonical_json(settings):
                return None
            if metadata.get("requested_start") != requested_start:
                return None
            if metadata.get("requested_end", "") < requested_end:
                return None
            saved_stats = metadata.get("source_stat_before")
            if source_stat is not None and saved_stats != _json_safe(source_stat):
                return None
            if pipeline_fingerprint is not None and metadata.get(
                "pipeline_fingerprint"
            ) != pipeline_fingerprint:
                return None
            table = pd.read_parquet(factor_path)
            if table.duplicated(["date", "instrument"]).any():
                raise ValueError(f"duplicate date/instrument key in {factor_path}")
            axes = payload["axes"]
            matrix = table.pivot(index="date", columns="instrument", values="factor")
            matrix = matrix.reindex(
                index=_dates_index(axes["dates"]), columns=pd.Index(axes["columns"])
            ).astype("float64")
            matrix.index.name = "date"
            matrix.columns.name = None
            return matrix.loc[requested_start:requested_end], payload
        except (KeyError, OSError, ValueError, TypeError, pd.errors.ParserError):
            return None

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

    def _save_flat_evaluation(
        self, factor_id: str, run_id: str, result: Mapping
    ) -> dict[str, Any]:
        """Persist only the current evaluation in flat sidecar files."""
        evaluation_id = _artifact_id(
            {
                "framework_version": FRAMEWORK_VERSION,
                "run_id": run_id,
                "profile": result.get("profile"),
                "evaluation_inputs": result.get("evaluation_inputs"),
            }
        )
        evaluation_path = self._flat_evaluation_path(factor_id)
        tables = {
            key: value
            for key, value in result.items()
            if isinstance(value, pd.DataFrame)
        }
        scalars = {key: value for key, value in result.items() if key not in tables}
        payload = {
            "evaluation_id": evaluation_id,
            "run_id": run_id,
            "framework_version": FRAMEWORK_VERSION,
            "tables": sorted(tables),
            "result": scalars,
        }
        conflict = (
            "evaluation already exists with different content; "
            "input fingerprints must change when outputs change"
        )

        previous = None
        if evaluation_path.is_file():
            previous = json.loads(evaluation_path.read_text())
            if _canonical_json(previous) == _canonical_json(payload):
                for key in tables:
                    table_path = self._flat_evaluation_table_path(factor_id, key)
                    if not table_path.is_file() or not pd.read_parquet(table_path).equals(tables[key]):
                        raise ValueError(conflict)
                return {
                    "evaluation_id": evaluation_id,
                    "dir": evaluation_path.parent,
                    "path": evaluation_path,
                }

        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        temp_files: list[tuple[Path, Path]] = []
        try:
            for key, frame in tables.items():
                final_path = self._flat_evaluation_table_path(factor_id, key)
                temp_path = final_path.with_name(
                    f".{final_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
                )
                frame.to_parquet(temp_path)
                temp_files.append((temp_path, final_path))
            temp_json = evaluation_path.with_name(
                f".{evaluation_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
            )
            _write_json(temp_json, payload)
            for temp_path, final_path in temp_files:
                os.replace(temp_path, final_path)
            os.replace(temp_json, evaluation_path)
        except BaseException:
            for temp_path, _ in temp_files:
                temp_path.unlink(missing_ok=True)
            if "temp_json" in locals():
                temp_json.unlink(missing_ok=True)
            raise

        if previous is not None:
            previous_tables = set(previous.get("tables", []))
            for key in previous_tables - set(tables):
                self._flat_evaluation_table_path(factor_id, key).unlink(missing_ok=True)
        return {
            "evaluation_id": evaluation_id,
            "dir": evaluation_path.parent,
            "path": evaluation_path,
        }

    def get_value(self, factor_id: str, *, run_id: str | None = None) -> pd.DataFrame:
        """Rebuild the stored matrix for a run, defaulting to the latest success."""
        if self.flat:
            factor_path, metadata_path = self._flat_paths(factor_id)
            if not factor_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"no flat factor artifact for {factor_id!r}")
            payload = json.loads(metadata_path.read_text())
            table = pd.read_parquet(factor_path)
            if table.duplicated(["date", "instrument"]).any():
                raise ValueError(f"duplicate date/instrument key in {factor_path}")
            matrix = table.pivot(index="date", columns="instrument", values="factor")
            matrix = matrix.reindex(
                index=_dates_index(payload["axes"]["dates"]),
                columns=pd.Index(payload["axes"]["columns"]),
            ).astype("float64")
            matrix.index.name = "date"
            matrix.columns.name = None
            return matrix
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
        if self.flat:
            return self._save_flat_evaluation(factor_id, run_id, result)
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
        if self.flat:
            result_path = self._flat_evaluation_path(factor_id)
            if not result_path.is_file():
                raise FileNotFoundError(
                    f"no evaluation recorded for {factor_id!r}"
                )
            payload = json.loads(result_path.read_text())
            saved_run_id = payload.get("run_id")
            saved_evaluation_id = payload.get("evaluation_id")
            if evaluation_id is None and payload.get("result", {}).get("status") != "complete":
                raise FileNotFoundError(
                    f"no complete evaluation recorded for {factor_id!r}"
                )
            if run_id is not None and run_id != saved_run_id:
                raise FileNotFoundError(
                    f"unknown run {run_id!r} for {factor_id!r}"
                )
            if evaluation_id is not None and evaluation_id != saved_evaluation_id:
                raise FileNotFoundError(
                    f"unknown evaluation {evaluation_id!r} for {factor_id!r}"
                )
            result = dict(payload["result"])
            for key in payload["tables"]:
                table_path = self._flat_evaluation_table_path(factor_id, key)
                if not table_path.is_file():
                    raise FileNotFoundError(
                        f"missing evaluation table {key!r} for {factor_id!r}"
                    )
                result[key] = pd.read_parquet(table_path)
            result["run_id"] = saved_run_id
            result["evaluation_id"] = saved_evaluation_id
            return result
        else:
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
