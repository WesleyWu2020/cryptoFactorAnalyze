"""Static-scan and cutoff-replay future-leak enforcement.

Static scanning is evidence, not proof; the cutoff replay recomputes the
factor with market AND membership knowledge truncated at each cutoff and
requires exact agreement with the full-history prefix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_common.data_provider import DataProvider
from factor_common.definitions import FactorSpec
from factor_common.validation import (
    EVALUATION_ONLY_ALLOWANCE,
    check_cutoff,
    scan_future_leaks,
)
from factor_common.value_engine import compute_factor


REPO_ROOT = Path(__file__).resolve().parents[2]

LEAKING_SOURCE = '''
import pandas as pd


def leaking(df, other):
    positional = df.shift(-1)
    keyword = df.shift(periods=-2)
    centered = df.rolling(3, center=True).mean()
    filled = df.bfill()
    legacy = df.backfill()
    merged = pd.merge_asof(df, other, on="date", direction="forward")
    return positional, keyword, centered, filled, legacy, merged
'''

CLEAN_SOURCE = '''
import pandas as pd


def causal(df, other):
    past = df.shift(1)
    rolled = df.rolling(3).mean()
    filled = df.ffill()
    merged = pd.merge_asof(df, other, on="date", direction="backward")
    return past, rolled, filled, merged
'''


def test_scanner_flags_each_banned_pattern_with_location(tmp_path):
    target = tmp_path / "leaking_factor.py"
    target.write_text(LEAKING_SOURCE)
    findings = scan_future_leaks([target])
    by_line = {finding["line"]: finding["pattern"] for finding in findings}
    assert by_line == {
        6: "shift_negative_periods",
        7: "shift_negative_periods",
        8: "rolling_center",
        9: "bfill",
        10: "backfill",
        11: "merge_asof_forward",
    }
    assert all(finding["path"].endswith("leaking_factor.py") for finding in findings)


def test_scanner_clean_source_has_no_findings(tmp_path):
    target = tmp_path / "causal_factor.py"
    target.write_text(CLEAN_SOURCE)
    assert scan_future_leaks([target]) == []


def test_labels_allowance_is_explicit_and_path_scoped(tmp_path):
    # The allowance names exactly one evaluation-only module; it is not a
    # global ignore list for the banned patterns.
    assert EVALUATION_ONLY_ALLOWANCE == frozenset({"factor_common/labels.py"})
    labels_path = REPO_ROOT / "factor_common" / "labels.py"
    assert scan_future_leaks([labels_path]) == []
    copy = tmp_path / "labels.py"
    copy.write_text(labels_path.read_text())
    findings = scan_future_leaks([copy])
    assert any(f["pattern"] == "shift_negative_periods" for f in findings)


def test_factor_construction_modules_scan_clean():
    modules = sorted((REPO_ROOT / "factor_common").glob("*.py"))
    scanned = [path for path in modules if path.name not in {"labels.py", "metrics.py"}]
    assert {path.name for path in scanned} >= {
        "value_engine.py", "data_provider.py", "validation.py",
    }
    assert scan_future_leaks(scanned) == []
    # Recorded evidence, not a violation: metrics.py:298 negatively shifts the
    # evaluation-only labels matrix (allowed future data), never factor values.
    findings = scan_future_leaks([REPO_ROOT / "factor_common" / "metrics.py"])
    assert [(f["line"], f["pattern"]) for f in findings] == [(298, "shift_negative_periods")]
    assert "labels.shift" in findings[0]["source"]


START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2024-01-12")
# Membership transition: the entrant cohort (GUSDT..LUSDT) is decided on
# 2024-01-06 and effective 2024-01-07; LUSDT never has market history.
DECISION_DAY = pd.Timestamp("2024-01-06")
EFFECTIVE_DAY = pd.Timestamp("2024-01-07")
CUTOFFS = [pd.Timestamp("2024-01-05"), DECISION_DAY, pd.Timestamp("2024-01-08")]


def _causal(ctx):
    return ctx["close"] / ctx["close"].shift(3) - 1


def _leaking(ctx):
    return ctx["close"].shift(-1) / ctx["close"] - 1


def _compute(path, calc, *, warmup):
    spec = FactorSpec("task11", {}, {
        "data_needed": ["close"],
        "warmup_bars": warmup,
        "preprocessing": "none",
    }, calc, "task11-test")

    def compute(cutoff):
        # The callback truncates provider knowledge (market history AND
        # membership decisions via DataProvider's as_of) as well as output;
        # slicing a full-history result is not a valid callback.
        end = END if cutoff is None else min(pd.Timestamp(cutoff), END)
        dp = DataProvider(path, as_of=None if cutoff is None else cutoff)
        values, _ = compute_factor(spec, dp, start=START, end=end)
        return values

    return compute


def test_causal_factor_passes_cutoff_replay_with_future_entrant(h5_fixture):
    report = check_cutoff(_compute(h5_fixture, _causal, warmup=3), CUTOFFS)
    assert report["status"] == "verified"
    assert "all-NaN" in report["policy"]
    entries = report["cutoffs"]
    assert [entry["cutoff"] for entry in entries] == CUTOFFS
    assert [entry["max_abs_diff"] for entry in entries] == [0.0, 0.0, 0.0]
    assert all(entry["index_equal"] and entry["mask_equal"] for entry in entries)
    # Recorded common-axis policy: at the 2024-01-05 cutoff LUSDT's membership
    # decision (2024-01-06) is still unknown and it has no market history, so
    # its column is an all-NaN future-only column excluded under the policy.
    assert entries[0]["excluded_columns"] == ["LUSDT"]
    assert entries[1]["excluded_columns"] == []
    assert entries[2]["excluded_columns"] == []


def test_leaking_factor_passes_shape_checks_but_fails_truncated_replay(h5_fixture):
    compute = _compute(h5_fixture, _leaking, warmup=1)
    full = compute(None)
    # Shape checks pass: a well-formed daily matrix on the requested axes.
    assert full.index.equals(pd.date_range(START, END, freq="D", name="date"))
    assert full.notna().any().any()
    # Truncated replay detects the forward shift: at the cutoff the full
    # prefix holds finite values where the truncated replay must have NaN.
    with pytest.raises(AssertionError):
        check_cutoff(compute, [DECISION_DAY])


def test_cutoff_truncates_market_and_membership_knowledge_together(h5_fixture):
    truncated = DataProvider(h5_fixture, as_of="2024-01-05")
    full = DataProvider(h5_fixture)
    # Market knowledge: no rows after the cutoff are visible.
    assert truncated.get_time_range()[1] == pd.Timestamp("2024-01-05")
    assert full.get_time_range()[1] == END
    # Membership knowledge: LUSDT's decision (2024-01-06) is unknown, so the
    # future entrant does not exist at all for the truncated provider.
    assert "LUSDT" not in truncated.symbols
    assert "LUSDT" in full.symbols
    # The future membership entrant produces output only from its effective
    # date, never before; the transition-day prefix is entirely NaN.
    compute = _compute(h5_fixture, _causal, warmup=3)
    full_values = compute(None)
    entrant = full_values["GUSDT"]
    assert entrant.loc[:DECISION_DAY].isna().all()
    assert entrant.loc[EFFECTIVE_DAY:].notna().all()
    assert entrant.loc[:DECISION_DAY].index[-1] == DECISION_DAY


def test_external_dataframe_input_is_not_verified():
    external = pd.DataFrame(
        np.ones((3, 1)),
        index=pd.date_range("2024-01-01", periods=3, name="date"),
        columns=["AUSDT"],
    )
    report = check_cutoff(external, CUTOFFS)
    assert report["status"] == "not_verified"
    assert report["cutoffs"] == []
