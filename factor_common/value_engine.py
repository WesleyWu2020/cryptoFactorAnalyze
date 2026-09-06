"""Calculate label-independent daily factor values using historical context."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from data.crypto_quant import reader as _cq_reader

from . import data_provider as _data_provider
from . import preprocessing as _preprocessing
from .definitions import FactorSpec
from .preprocessing import rank_to_unit_by_date, winsorize_by_date


def value_pipeline_fingerprint() -> str:
    """Hash of the sources that define cached factor-value semantics.

    Masking, preprocessing, and universe/quality access all shape the saved
    value matrix, so any edit to those modules must invalidate cached factor
    values instead of silently reusing a matrix built by older code.
    """

    digest = hashlib.sha256()
    for path in (_cq_reader.__file__, _data_provider.__file__, _preprocessing.__file__, __file__):
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _validate_axes(matrix, *, name, expected=None):
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame with daily axes")
    index = matrix.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{name} date axis must be a DatetimeIndex")
    if index.tz is not None or index.hasnans or not index.equals(index.normalize()):
        raise ValueError(f"{name} date axis must be daily UTC-naive midnight without NaT")
    if not index.is_unique or not matrix.columns.is_unique:
        raise ValueError(f"{name} axes must be unique")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} date axis must be increasing")
    if len(index) and not index.equals(pd.date_range(index[0], index[-1], freq="D")):
        raise ValueError(f"{name} date axis must preserve every calendar day")
    if expected is not None:
        dates, instruments = expected
        if not index.equals(dates) or not matrix.columns.equals(instruments):
            raise ValueError(f"{name} axes must exactly match context axes")


def _day(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError("start/end must be daily UTC-naive midnight dates")
    return day


def compute_factor(spec: FactorSpec, dp, *, start, end) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a daily matrix and counts over the inclusive requested interval.

    Warmup uses calendar days and includes pre-entry market history. Only after
    calculation are values masked by same-day membership and preprocessed.
    Missing/non-finite eligible values remain NaN; no labels are requested.
    Diagnostics count eligible, ineligible, missing eligible, and valid cells
    after slicing (warmup is excluded). Formula causality is the caller's contract.
    """
    setting = spec.setting
    if setting.get("pasteurization", False):
        raise ValueError("pasteurization=True has no defined meaning; use preprocessing")
    mode = setting["preprocessing"]
    if mode not in ("none", "mad_rank"):
        raise ValueError("preprocessing must be 'none' or 'mad_rank'")
    warmup = setting["warmup_bars"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup_bars must be a non-negative integer")
    start, end = _day(start), _day(end)
    if start > end:
        raise ValueError("start must be on or before end")
    history_start = start - pd.Timedelta(days=warmup)
    ctx = {field: dp.get_single_data(field, start=history_start, end=end)
           for field in setting["data_needed"]}
    if not ctx:
        raise ValueError("data_needed must not be empty")
    axes = None
    for field, matrix in ctx.items():
        _validate_axes(matrix, name=f"context {field}", expected=axes)
        if axes is None:
            axes = (matrix.index.copy(), matrix.columns.copy())
    raw = spec.calc_factor(ctx)
    _validate_axes(raw, name="factor output", expected=axes)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    eligible = dp.get_universe(start=history_start, end=end)
    _validate_axes(eligible, name="universe", expected=axes)
    if not all(is_bool_dtype(dtype) for dtype in eligible.dtypes) or eligible.isna().any().any():
        raise ValueError("universe must contain only non-missing boolean eligibility")
    masked = raw.where(eligible)
    if mode == "mad_rank":
        # Omit missing entries before ranking: legacy singleton math returns 0
        # even for NaN rows when a date has at most one finite observation.
        rows, cols = np.nonzero(masked.notna().to_numpy())
        long = pd.DataFrame({"date": masked.index.take(rows),
                             "factor": masked.to_numpy()[rows, cols]})
        ranked = rank_to_unit_by_date(winsorize_by_date(long, "factor"), "factor")
        values = np.full(masked.shape, np.nan)
        values[rows, cols] = ranked["factor"].to_numpy()
        masked = pd.DataFrame(values, index=masked.index, columns=masked.columns)
    result = masked.loc[start:end].copy()
    selected = eligible.loc[start:end]
    eligible_count = int(selected.to_numpy().sum())
    valid_count = int(result.notna().to_numpy().sum())
    diagnostics = {
        "eligible_count": eligible_count,
        "ineligible_count": int(selected.size - eligible_count),
        "missing_count": eligible_count - valid_count,
        "valid_count": valid_count,
    }
    return result, diagnostics
