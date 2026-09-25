"""Data readiness checks specific to the frozen daily strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.crypto_quant.validation import validate_store
from factor_common.data_provider import DataProvider


def check_readiness(h5_path, signal_date, *, min_valid_instruments: int) -> dict:
    day = pd.Timestamp(signal_date).normalize()
    report = validate_store(h5_path)
    errors = [issue.__dict__ for issue in report.issues if issue.level == "error"]
    warnings = [issue.__dict__ for issue in report.issues if issue.level == "warning"]
    if errors:
        return {"ready": False, "signal_date": day.date().isoformat(),
                "errors": errors, "warnings": warnings}

    provider = DataProvider(h5_path, as_of=day)
    _, market_end = provider.get_time_range()
    universe = provider.get_universe(start=day, end=day)
    active = list(universe.columns[universe.iloc[0]]) if len(universe.index) else []
    quality = provider.get_quality(start=day, end=day, symbols=active)
    if quality.empty:
        complete = placeholder = 0
    else:
        complete = int(quality["has_complete_kline"].map(
            lambda value: type(value) in (bool, np.bool_) and bool(value)
        ).sum())
        placeholder = int(quality["has_placeholder_kline"].map(
            lambda value: type(value) in (bool, np.bool_) and bool(value)
        ).sum())
    reasons = []
    if market_end is None or market_end < day:
        reasons.append("target_daily_kline_not_available")
    if len(active) < min_valid_instruments:
        reasons.append("insufficient_active_universe")
    if complete - placeholder < min_valid_instruments:
        reasons.append("insufficient_complete_non_placeholder_klines")
    return {
        "ready": not reasons,
        "signal_date": day.date().isoformat(),
        "market_end": None if market_end is None else market_end.date().isoformat(),
        "active_instruments": len(active),
        "complete_klines": complete,
        "placeholder_klines": placeholder,
        "reasons": reasons,
        "errors": errors,
        "warnings": warnings,
    }
