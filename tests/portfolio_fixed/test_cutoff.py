import pandas as pd
import pytest

from portfolio.fixed_audit import audit_fixed, compare_matrix
from portfolio.fixed_config import FixedConfig


def test_real_provider_cutoffs_recompute_pipeline(config_dict):
    audit = audit_fixed(FixedConfig.from_dict(config_dict), ["2024-01-23", "2024-01-24"])
    assert audit["status"] == "verified"
    assert len(audit["cutoffs"]) == 2
    assert all(item["max_abs_diff"] <= 1e-10 for item in audit["cutoffs"])


def test_cutoff_receipt_preserves_incomplete_tail_and_known_segment(config_dict):
    audit = audit_fixed(FixedConfig.from_dict(config_dict), ["2024-01-24"])
    receipt = audit["cutoffs"][0]
    assert receipt["incomplete_tail_expected"] is True
    assert all(receipt["metrics_full_none"].values())
    assert all(status == "incomplete" for status in receipt["scenario_status"].values())
    assert receipt["known_segment_end"] <= receipt["cutoff"]
    assert "omitted" in receipt["input_reads_comparison"]


def test_missing_nonzero_target_column_is_not_ignored():
    index = pd.date_range("2024-01-01", periods=1, name="date")
    full = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=index)
    cut = full[["A"]]
    with pytest.raises(AssertionError):
        compare_matrix(full, cut, index[0], absent_zero=True)


def test_nan_mask_change_is_not_ignored():
    index = pd.date_range("2024-01-01", periods=1, name="date")
    full = pd.DataFrame({"A": [float("nan")]}, index=index)
    cut = pd.DataFrame({"A": [0.0]}, index=index)
    with pytest.raises(AssertionError):
        compare_matrix(full, cut, index[0])


def test_empty_cutoff_list_cannot_claim_verified(config_dict):
    with pytest.raises(ValueError, match="cutoff"):
        audit_fixed(FixedConfig.from_dict(config_dict), [])
