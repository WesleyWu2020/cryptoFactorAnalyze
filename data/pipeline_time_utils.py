"""Time utilities for data pipeline scheduling windows."""

from __future__ import annotations

from datetime import date, timedelta


def add_months(d: date, months: int) -> date:
    """Return first-day-safe month offset."""
    if months == 0:
        return d
    month_idx = d.month - 1 + months
    year = d.year + month_idx // 12
    month = month_idx % 12 + 1
    return d.replace(year=year, month=month)


def previous_month_window(decision_date: date) -> tuple[date, date]:
    """
    Build non-leaky monthly window:
    decision day is month-start T, and window uses T-1 calendar month.
    """
    month_start = decision_date.replace(day=1)
    prev_month_end = month_start - timedelta(days=1)
    prev_month_start = prev_month_end.replace(day=1)
    return prev_month_start, prev_month_end

