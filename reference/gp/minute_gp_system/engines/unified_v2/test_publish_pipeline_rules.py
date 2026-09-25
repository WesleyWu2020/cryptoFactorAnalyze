import sys
import types

factor_platform = types.ModuleType("factor_platform")
factor_platform_config = types.ModuleType("factor_platform.config")
factor_platform_config.DEFAULT_GP_REVIEW_QUEUE_PATH = "/tmp/gp_review_queue.json"
sys.modules.setdefault("factor_platform", factor_platform)
sys.modules.setdefault("factor_platform.config", factor_platform_config)

from gp.minute_gp_system.engines.unified_v2 import publish_pipeline as publish


def _passing_yearly():
    return {
        "summary": {
            "min_ls_netret": 0.01,
            "min_ls_net_sharpe": 0.6,
            "max_turnover": 0.5,
            "min_coverage": 0.8,
        }
    }


def test_publish_gate_matches_submission_thresholds():
    reasons = publish._gate_reasons(
        {"ls_netret": 0.01, "ls_net_sharpe": 0.8},
        {
            "ls_netret": 0.079,
            "ls_net_sharpe": 0.79,
            "ls_turnover": 0.5,
            "ls_seg_positive_ratio": 0.5,
            "ls_seg_min": -0.11,
        },
        _passing_yearly(),
    )

    assert "oos_netret" in reasons
    assert "oos_sharpe" in reasons
    assert "oos_segment_min" in reasons


def test_publish_yearly_gate_matches_submission_thresholds():
    reasons = publish._gate_reasons(
        {"ls_netret": 0.01, "ls_net_sharpe": 0.8},
        {
            "ls_netret": 0.08,
            "ls_net_sharpe": 0.8,
            "ls_turnover": 0.5,
            "ls_seg_positive_ratio": 0.5,
            "ls_seg_min": -0.1,
        },
        {
            "summary": {
                "min_ls_netret": 0.01,
                "min_ls_net_sharpe": 0.49,
                "max_turnover": 0.5,
                "min_coverage": 0.69,
            }
        },
    )

    assert "yearly_sharpe" in reasons
    assert "yearly_coverage" in reasons
