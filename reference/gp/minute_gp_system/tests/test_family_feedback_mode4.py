from gp.minute_gp_system.family_feedback import build_candidate_feedback_payload


def test_mode4_operator_family_is_not_other():
    payload = build_candidate_feedback_payload(
        {
            "mode": 4,
            "A": "money_flow",
            "B": "rv_20d",
            "mode4_op": "group_dispersion",
            "window": 180,
            "mask_rule": "high_0.8",
            "mask_field": "turnover",
            "ts_comp_op": "ts_decay",
            "cs_comp_op": "none",
        }
    )

    assert payload["operator_name"] == "group_dispersion"
    assert payload["operator_family"] != "other"
