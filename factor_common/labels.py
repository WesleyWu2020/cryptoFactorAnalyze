"""Executable forward-return labels aligned to the daily execution calendar.

EVALUATION-ONLY FUTURE DATA: labels are the single framework location where
future information is permitted. ``label[t]`` buys at the open of ``t+1``
(the signal_delay_days=1 execution bar) and sells at the open of
``t+1+hold_days``. Labels feed metric assembly and sample-boundary purging
only; they must never enter factor construction, preprocessing, grouping,
or instrument selection.

``label_dates`` returns the concrete entry/exit execution dates per signal
date so downstream purging compares exit times against sample boundaries.
Exit dates remain concrete past the data tail (they describe the calendar,
not observed prices), while the label values themselves are NaN there.
"""

import pandas as pd

from .value_engine import _validate_axes


def _hold_days(hold_days) -> int:
    if isinstance(hold_days, bool) or not isinstance(hold_days, int) or hold_days < 1:
        raise ValueError("hold_days must be a positive integer")
    return hold_days


def make_labels(opens: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    """Return next-open forward returns; EVALUATION-ONLY future data.

    ``label[t] = open[t+1+hold_days] / open[t+1] - 1`` on an already complete
    daily calendar. The last ``hold_days + 1`` rows are NaN because their
    execution prices are unobserved.
    """
    _validate_axes(opens, name="opens")
    hold_days = _hold_days(hold_days)
    return opens.shift(-(hold_days + 1)) / opens.shift(-1) - 1


def label_dates(dates: pd.DatetimeIndex, hold_days: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Return (entry_dates, exit_dates) per signal date on the daily calendar.

    Entry is the next day's open (``t+1``); exit is ``hold_days`` after entry
    (``t+1+hold_days``). Dates stay concrete beyond the observed tail so
    boundary purge compares exit times against split dates.
    """
    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError("dates must be a DatetimeIndex")
    if len(dates) and not dates.equals(pd.date_range(dates[0], dates[-1], freq="D")):
        raise ValueError("dates must preserve every calendar day")
    hold_days = _hold_days(hold_days)
    entry = dates + pd.Timedelta(days=1)
    exit_ = dates + pd.Timedelta(days=1 + hold_days)
    return entry, exit_
