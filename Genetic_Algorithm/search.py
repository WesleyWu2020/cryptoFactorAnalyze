"""Audit-first boundary for the future daily GP search stage."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, TypeVar

from .config import Stage
from .data import run_training_audit


Result = TypeVar("Result")


def run_search(
    h5_path: str | Path,
    audit_path: str | Path,
    *,
    stage: Stage,
    warmup_days: int,
    fields: Iterable[str],
    search_stage: Callable[[Path], Result],
) -> Result:
    """Persist the train audit before handing off to the future GP search.

    This is the Task 2 boundary for Task 6/10 integration. The callback is
    intentionally injected: this module does not implement GP search and it
    does not load validation or test rows.
    """
    persisted_audit = run_training_audit(
        h5_path,
        audit_path,
        stage=stage,
        warmup_days=warmup_days,
        fields=fields,
    )
    return search_stage(persisted_audit)


__all__ = ["run_search"]
