import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/crypto-research/users/wesleywu")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gp.minute_gp_system.engines.unified_v2 import evolution as evo


def _row(
    A="returns",
    B="return_abs",
    window=120,
    slice=None,
    ts_comp_op="none",
    formula="formula",
    composite=0.0,
):
    return {
        "formula": formula,
        "raw_formula": formula,
        "composite": composite,
        "decoded": {
            "mode": 4,
            "mode4_op": "group_rank_mean",
            "A": A,
            "B": B,
            "window": window,
            "slice": slice,
            "mask_field": "open",
            "mask_rule": "none",
            "B_shift_lag": 0,
            "ts_comp_op": ts_comp_op,
            "ts_comp_window": 20,
            "cs_comp_op": "none",
        }
    }


def test_signature_buckets_fields_by_family_and_window_hour():
    returns_sig = evo._signature_of(_row(A="returns", B="return_abs", window=120))
    return_abs_sig = evo._signature_of(_row(A="return_abs", B="returns", window=150))
    flow_sig = evo._signature_of(_row(A="money_flow", B="return_abs", window=120))

    assert returns_sig == return_abs_sig
    assert flow_sig != returns_sig
    assert evo._family_of_field("returns") == "return_family"
    assert evo._family_of_field("money_flow") == "flow_family"
    assert evo._family_of_field("unknown_field") == "other_family"


def test_signature_handles_missing_decoded_without_error():
    assert evo._signature_of({"formula": "x"}) is None
    assert evo._signature_of(None) is None


def test_dedup_by_signature_keeps_first_composite_order():
    rows = [
        _row(A="returns", window=120, formula="best", composite=3.0),
        _row(A="return_abs", window=150, formula="duplicate_family_window", composite=2.0),
        _row(A="money_flow", window=120, formula="different_family", composite=1.0),
    ]

    kept, removed = evo._dedup_by_signature(rows)

    assert removed == 1
    assert [row["formula"] for row in kept] == ["best", "different_family"]
    assert kept[0]["final_signature_dedup_gate"]["passed"] is True
    assert rows[1]["final_signature_dedup_gate"] == {
        "passed": False,
        "reason": "duplicate_signature",
    }


def test_signature_blacklist_removes_external_signature_match(tmp_path):
    blacklist_path = tmp_path / "blacklist.json"
    blacklist_path.write_text(
        json.dumps([_row(A="returns", window=120, formula="prior")])
    )
    blacklist = evo._load_signature_blacklist(str(blacklist_path))
    rows = [
        _row(A="return_abs", window=150, formula="external_duplicate"),
        _row(A="money_flow", window=120, formula="different_family"),
    ]

    kept, removed = evo._dedup_by_signature_blacklist(rows, blacklist)

    assert removed == 1
    assert [row["formula"] for row in kept] == ["different_family"]
    assert rows[0]["final_signature_blacklist_gate"] == {
        "passed": False,
        "reason": "external_signature_match",
    }


def test_external_behavior_corr_logs_dimension_mismatch_once(capsys):
    if hasattr(evo._compute_external_behavior_corr, "_warned_dim_mismatch"):
        delattr(evo._compute_external_behavior_corr, "_warned_dim_mismatch")
    try:
        behavior = np.ones((2, 3), dtype=np.float32)
        archive = np.ones((1, 4), dtype=np.float32)
        valid = np.asarray([True, True])
        corr1 = evo._compute_external_behavior_corr(behavior, valid, archive)
        corr2 = evo._compute_external_behavior_corr(behavior, valid, archive)
        out = capsys.readouterr().out
        assert np.isnan(corr1).all()
        assert np.isnan(corr2).all()
        assert out.count("[EXTERNAL_ARCHIVE] dimension mismatch") == 1
        assert "behavior_signatures.shape[1]=3" in out
        assert "archive.shape[1]=4" in out
    finally:
        if hasattr(evo._compute_external_behavior_corr, "_warned_dim_mismatch"):
            delattr(evo._compute_external_behavior_corr, "_warned_dim_mismatch")


def test_external_behavior_corr_handles_non_2d_inputs(capsys):
    if hasattr(evo._compute_external_behavior_corr, "_warned_dim_mismatch"):
        delattr(evo._compute_external_behavior_corr, "_warned_dim_mismatch")
    try:
        corr = evo._compute_external_behavior_corr(
            np.ones(3, dtype=np.float32),
            np.asarray([True, True]),
            np.ones((1, 3), dtype=np.float32),
        )
        assert np.isnan(corr).all()
        assert "[EXTERNAL_ARCHIVE] dimension mismatch" in capsys.readouterr().out
    finally:
        if hasattr(evo._compute_external_behavior_corr, "_warned_dim_mismatch"):
            delattr(evo._compute_external_behavior_corr, "_warned_dim_mismatch")

def test_final_diversity_caps_pair_a_family(monkeypatch):
    monkeypatch.setattr(evo, "FINAL_PAIR_A_FAMILY_GATE_ENABLED", True)
    monkeypatch.setattr(
        evo,
        "FINAL_PAIR_A_FAMILY_MAX_SHARE",
        {"return_family": 0.30},
    )
    monkeypatch.setattr(evo, "FINAL_PAIR_A_FAMILY_DEFAULT_CAP", 0.50)
    monkeypatch.setattr(evo, "FINAL_TS_COMP_OP_MAX_SHARE", 1.0)
    monkeypatch.setattr(evo, "FINAL_DIVERSITY_MAX_PER_OPERATOR", 20)

    rows = [
        _row(A="returns", formula="r1"),
        _row(A="return_abs", formula="r2"),
        _row(A="price_range_pct", formula="r3"),
        _row(A="money_flow", formula="f1"),
        _row(A="turnover", formula="f2"),
        _row(A="beta_btc_60d", formula="c1"),
        _row(A="funding", formula="o1"),
        _row(A="open", formula="p1"),
    ]

    kept, removed = evo._apply_final_diversity_constraints(rows)
    assert [row["formula"] for row in kept if evo._family_of_field(row["decoded"]["A"]) == "return_family"] == [
        "r1",
        "r2",
        "r3",
    ]
    assert removed.get("diversity_pair_a_family", 0) == 0

    rows.append(_row(A="returns", formula="r4"))
    kept, removed = evo._apply_final_diversity_constraints(rows)
    return_family_rows = [
        row for row in kept if evo._family_of_field(row["decoded"]["A"]) == "return_family"
    ]
    assert [row["formula"] for row in return_family_rows] == ["r1", "r2", "r3"]
    assert removed["diversity_pair_a_family"] == 1
    assert rows[-1]["final_diversity_gate"]["reason"] == "diversity_pair_a_family"


def test_final_diversity_caps_ts_comp_op(monkeypatch):
    monkeypatch.setattr(evo, "FINAL_PAIR_A_FAMILY_GATE_ENABLED", False)
    monkeypatch.setattr(evo, "FINAL_TS_COMP_OP_MAX_SHARE", 0.50)
    monkeypatch.setattr(evo, "FINAL_DIVERSITY_MAX_PER_OPERATOR", 20)
    rows = [
        _row(A="returns", ts_comp_op="ts_decay", formula="a"),
        _row(A="money_flow", ts_comp_op="ts_decay", formula="b"),
        _row(A="turnover", ts_comp_op="ts_decay", formula="c"),
        _row(A="funding", ts_comp_op="none", formula="d"),
    ]
    kept, removed = evo._apply_final_diversity_constraints(rows)
    assert [row["formula"] for row in kept] == ["a", "b", "d"]
    assert removed["diversity_ts_comp_op"] == 1
    assert rows[2]["final_diversity_gate"]["reason"] == "diversity_ts_comp_op"
