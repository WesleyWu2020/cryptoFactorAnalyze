"""Bounded, label-free daily data bundles for the GP stages."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from data.crypto_quant.schemas import TABLE_SPECS
from factor_common.data_provider import DataProvider
from .config import STAGES, Stage


_FINGERPRINT_VERSION = 2
_OPERATOR_VERSION = "task2-stage-data-v1"


@dataclass(frozen=True)
class StageData:
    stage: Stage
    features: dict[str, pd.DataFrame]
    eligible: pd.DataFrame
    quality_eligible: pd.DataFrame
    opens: pd.DataFrame
    fingerprint: str
    audit: dict[str, Any]


def _day(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    timestamp = timestamp.normalize()
    return timestamp


def _validate_panel(frame: pd.DataFrame, name: str, dates: pd.DatetimeIndex, columns: pd.Index) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None:
        raise ValueError(f"{name} must use a UTC-naive daily index")
    if not frame.index.equals(dates) or not frame.columns.equals(columns):
        raise ValueError(f"{name} axes do not match the stage axes")
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"{name} contains duplicate axes")


def _validate_membership_axes(
    frame: pd.DataFrame, dates: pd.DatetimeIndex, columns: pd.Index
) -> None:
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("eligible membership axes contain duplicate labels")
    if not frame.index.equals(dates) or not frame.columns.equals(columns):
        raise ValueError("eligible membership axes do not match the feature axes")


def _canonical_frame(name: str, frame: pd.DataFrame) -> bytes:
    frame = frame.sort_index().sort_index(axis=1)
    payload = {
        "name": name,
        "index": [pd.Timestamp(value).isoformat() for value in frame.index],
        "columns": [str(value) for value in frame.columns],
        "schema": [str(frame[column].dtype) for column in frame.columns],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for column in frame.columns:
        series = frame[column]
        mask = series.isna().to_numpy(dtype=np.uint8)
        digest.update(mask.tobytes())
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="raise").astype("float64").to_numpy().copy()
            values[mask.astype(bool)] = 0.0
            digest.update(np.ascontiguousarray(values).tobytes())
        else:
            digest.update(json.dumps(series.fillna("").astype(str).tolist()).encode())
    return digest.digest()


def _fingerprint(
    features: dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
    quality_eligible: pd.DataFrame,
    opens: pd.DataFrame,
    *,
    stage: Stage,
    code_fingerprint: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "fingerprint_version": _FINGERPRINT_VERSION,
        "operator_version": _OPERATOR_VERSION,
        "stage": {
            "name": stage.name,
            "start": _day(stage.start).isoformat(),
            "end": _day(stage.end).isoformat(),
        },
        "code_fingerprint": code_fingerprint,
    }, sort_keys=True).encode())
    for name, frame in [
        *sorted(features.items()),
        ("eligible", eligible),
        ("quality_eligible", quality_eligible),
        ("opens", opens),
    ]:
        digest.update(_canonical_frame(name, frame))
    digest.update(json.dumps({
        name: {
            "schema_version": spec.schema_version,
            "columns": list(spec.columns),
        }
        for name, spec in sorted(TABLE_SPECS.items())
        if name in {"klines_daily", "universe_monthly", "research_panel_daily"}
    }, sort_keys=True).encode())
    return digest.hexdigest()


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    provider_path = Path(inspect.getsourcefile(DataProvider) or "")
    for path in (Path(__file__), provider_path):
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_training_stage(stage: Stage) -> None:
    frozen = STAGES["train"]
    if (
        stage.name != frozen.name
        or _day(stage.start) != _day(frozen.start)
        or _day(stage.end) != _day(frozen.end)
    ):
        raise ValueError("training audit requires the frozen train stage")


def _matrix_from_quality(
    quality: pd.DataFrame, field: str, dates: pd.DatetimeIndex, columns: pd.Index
) -> pd.DataFrame:
    if field not in quality:
        return pd.DataFrame(index=dates, columns=columns, dtype="float64")
    matrix = quality[field].unstack("instrument")
    return matrix.reindex(index=dates, columns=columns)


def load_stage(
    h5_path: str | Path,
    stage: Stage,
    warmup_days: int,
    fields: Iterable[str],
) -> StageData:
    """Load one point-in-time stage and its required feature history.

    The provider cutoff is the stage end. This function deliberately never
    requests labels or any rows after that cutoff.
    """
    if stage.name == "train":
        _validate_training_stage(stage)
    if isinstance(warmup_days, bool) or not isinstance(warmup_days, int) or warmup_days < 0:
        raise ValueError("warmup_days must be a non-negative integer")
    fields = tuple(dict.fromkeys(fields))
    if not fields:
        raise ValueError("fields must not be empty")
    start, end = _day(stage.start), _day(stage.end)
    history_start = start - pd.Timedelta(days=warmup_days)
    provider = DataProvider(h5_path, as_of=end)
    features = {
        field: provider.get_single_data(field, start=history_start, end=end).astype("float64")
        for field in fields
    }
    history_dates = pd.date_range(history_start, end, freq="D", name="date")
    stage_dates = pd.date_range(start, end, freq="D", name="date")
    columns = features[fields[0]].columns
    for field, frame in features.items():
        _validate_panel(frame, f"feature {field}", history_dates, columns)
    eligible = provider.get_universe(start=history_start, end=end)
    _validate_membership_axes(eligible, history_dates, columns)
    _validate_panel(eligible, "eligible", history_dates, columns)
    if eligible.isna().any().any():
        raise ValueError("eligible contains unknown membership")
    eligible = eligible.astype(bool)

    opens = provider.get_single_data("open", start=start, end=end).reindex(
        index=stage_dates, columns=columns
    ).astype("float64")
    _validate_panel(opens, "opens", stage_dates, columns)
    quality = provider.get_quality(start=start, end=end, symbols=columns.tolist())
    quality = quality.reindex(pd.MultiIndex.from_product(
        [stage_dates, columns], names=["date", "instrument"]
    ))
    complete = _matrix_from_quality(quality, "has_complete_kline", stage_dates, columns)
    placeholder = _matrix_from_quality(quality, "has_placeholder_kline", stage_dates, columns)
    current_members = eligible.loc[stage_dates]
    quality_eligible = (
        current_members
        & complete.fillna(False).astype(bool)
        & ~placeholder.fillna(True).astype(bool)
    )
    _validate_panel(quality_eligible, "quality_eligible", stage_dates, columns)

    active = current_members
    unknown = active & (complete.isna() | placeholder.isna())
    placeholder_count = int((active & placeholder.fillna(False).astype(bool)).to_numpy().sum())
    incomplete_count = int((active & ~complete.fillna(False).astype(bool)).to_numpy().sum())
    code_fingerprint = _code_fingerprint()
    audit = {
        "provider_cutoff": end.strftime("%Y-%m-%d"),
        "stage_days": int(len(stage_dates)),
        "history_days": int(len(history_dates)),
        "per_day_eligible_counts": {
            day.strftime("%Y-%m-%d"): int(value)
            for day, value in current_members.sum(axis=1).items()
        },
        "missing_fields": {
            field: int(frame.loc[stage_dates].isna().sum().sum())
            for field, frame in features.items()
        },
        "warmup_availability": {
            "requested_days": warmup_days,
            "available_days": int(
                pd.concat(
                    [frame.loc[history_dates].notna().any(axis=1) for frame in features.values()],
                    axis=1,
                ).any(axis=1).sum()
            ),
            "history_start": history_start.strftime("%Y-%m-%d"),
        },
        "quality_counts": {
            "eligible": int(active.to_numpy().sum()),
            "quality_eligible": int(quality_eligible.to_numpy().sum()),
            "unknown": int(unknown.to_numpy().sum()),
            "ineligible_placeholder": placeholder_count,
            "ineligible_incomplete_kline": incomplete_count,
        },
        "provenance": {
            "code_fingerprint": code_fingerprint,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "pid": os.getpid(),
            },
        },
    }
    return StageData(
        stage=stage,
        features=features,
        eligible=eligible,
        quality_eligible=quality_eligible,
        opens=opens,
        fingerprint=_fingerprint(
            features,
            eligible,
            quality_eligible,
            opens,
            stage=stage,
            code_fingerprint=code_fingerprint,
        ),
        audit=audit,
    )


def write_training_audit(output_path: str | Path, stage_data: StageData) -> Path:
    """Retain the already-loaded training audit without reading another stage.

    The helper accepts only ``StageData`` so an audit retention step cannot
    accidentally load validation or test rows. The temporary file is placed
    beside the destination and atomically renamed into place.
    """
    _validate_training_stage(stage_data.stage)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "training_only": True,
        "stage": {
            "name": stage_data.stage.name,
            "start": _day(stage_data.stage.start).strftime("%Y-%m-%d"),
            "end": _day(stage_data.stage.end).strftime("%Y-%m-%d"),
        },
        "fingerprint": stage_data.fingerprint,
        "audit": stage_data.audit,
    }
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def run_training_audit(
    h5_path: str | Path,
    output_path: str | Path,
    *,
    stage: Stage,
    warmup_days: int,
    fields: Iterable[str],
) -> Path:
    """Load and retain the training audit before a search entry point.

    This workflow loads exactly one train stage and never loads validation or
    test data. The search boundary calls it before handing off to its search
    stage callback.
    """
    _validate_training_stage(stage)
    stage_data = load_stage(h5_path, stage, warmup_days=warmup_days, fields=fields)
    return write_training_audit(output_path, stage_data)


__all__ = ["StageData", "load_stage", "run_training_audit", "write_training_audit"]
