from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from daily_pipeline.artifacts import publish_signal, write_failure, write_run
from daily_pipeline.config import load_strategy_config
from daily_pipeline.pipeline import _long, is_rebalance_signal_day


CONFIG = Path(__file__).resolve().parents[2] / "daily_pipeline" / "configs" / "portfolio_v1.json"


def test_frozen_strategy_loads_and_matches_source_report():
    config = load_strategy_config(CONFIG)
    assert config.strategy_id == "fixed_portfolio_756193dec1b9"
    assert len(config.factors) == 10
    assert sum(item.allocation for item in config.factors) == 1.0


def test_rebalance_signal_day_uses_next_day_execution_schedule():
    config = load_strategy_config(CONFIG)
    assert is_rebalance_signal_day(config, "2026-08-22")
    assert not is_rebalance_signal_day(config, "2026-08-23")


def test_long_form_keeps_missing_instruments():
    frame = pd.DataFrame(
        [[1.0, float("nan")]],
        index=pd.DatetimeIndex(["2026-08-22"], name="date"),
        columns=["AUSDT", "BUSDT"],
    )
    result = _long(frame, "factor_value", "2026-08-22")
    assert result["instrument"].tolist() == ["AUSDT", "BUSDT"]
    assert pd.isna(result.loc[1, "factor_value"])


def test_artifact_and_publication_are_idempotent(tmp_path):
    day = pd.Timestamp("2026-08-22")
    table = pd.DataFrame({"signal_date": [day], "instrument": ["AUSDT"], "factor_value": [1.0]})
    result = {
        "signal": {"schema_version": 1, "signal_id": "abc", "signal_date": "2026-08-22",
                   "generated_at": "2026-08-23T00:01:00+00:00", "action": "hold", "targets": []},
        "readiness": {"ready": True},
        "factor_values": table,
        "factor_groups": table.rename(columns={"factor_value": "group"}),
        "member_targets": table.rename(columns={"factor_value": "member_weight"}),
        "combined_targets": table.rename(columns={"factor_value": "target_weight"}),
        "risk": pd.DataFrame({"gross": [0.0]}, index=[day]),
        "member_diagnostics": {},
        "manifest": {"status": "complete"},
    }
    first = write_run(result, tmp_path)
    published = publish_signal(first, tmp_path)
    second = write_run(result, tmp_path)
    assert publish_signal(second, tmp_path) == published
    assert json.loads((tmp_path / "published" / "latest.json").read_text())["signal_id"] == "abc"


def test_failure_receipt_does_not_publish(tmp_path):
    receipt = write_failure(tmp_path, "2026-08-22", ValueError("not ready"))
    manifest = json.loads((receipt / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "ValueError"
    assert not (tmp_path / "published" / "latest.json").exists()
