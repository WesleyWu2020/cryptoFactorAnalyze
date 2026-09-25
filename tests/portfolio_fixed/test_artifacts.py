import json
import pytest

import pandas as pd

from portfolio.fixed_artifacts import write_result
from portfolio.fixed_config import FixedConfig
from portfolio.fixed_pipeline import run_fixed
import portfolio.fixed_artifacts as artifacts


def test_complete_artifacts_are_readable_and_not_overwritten(config_dict, tmp_path):
    result = run_fixed(FixedConfig.from_dict(config_dict))
    first = write_result(result, tmp_path / "out")
    second = write_result(result, tmp_path / "out")
    assert first != second
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["artifact_complete"] is True
    assert (first / "artifact_complete.json").exists()
    pd.testing.assert_frame_equal(
        pd.read_parquet(first / "targets.parquet"), result["targets"], check_freq=False
    )
    assert not list((tmp_path / "out").glob(".partial_*"))
    report = (first / "report.html").read_text()
    assert "研究回测" in report
    assert "365" in report
    assert "保证金" in report
    assert "echarts.min.js" in report
    assert "组合净值" in report
    assert "累计成本分解" in report
    assert "风险约束实际使用情况" in report


def test_incomplete_full_metrics_remain_null(config_dict, tmp_path):
    result = run_fixed(FixedConfig.from_dict(config_dict), as_of="2024-01-24")
    assert result["status"] == "incomplete"
    output = write_result(result, tmp_path / "out")
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["all_costs"]["full"] is None
    assert metrics["all_costs"]["known_segment"] is not None
    assert json.loads((output / "artifact_complete.json").read_text())["written"] is True


def test_html_escapes_configuration(config_dict, tmp_path):
    config_dict["market_mode"] = "<script>alert(1)</script>"
    # Config validation rejects arbitrary modes; use a valid result and inject
    # a hostile display-only value to exercise escaping without changing schema.
    config_dict["market_mode"] = "perp_long_short"
    result = run_fixed(FixedConfig.from_dict(config_dict))
    result["configuration"]["display_note"] = "<script>alert(1)</script>"
    output = write_result(result, tmp_path / "out")
    html = (output / "report.html").read_text()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html


def test_existing_final_directory_is_never_replaced(config_dict, tmp_path, monkeypatch):
    result = run_fixed(FixedConfig.from_dict(config_dict))
    root = tmp_path / "out"
    root.mkdir()
    run_id = "fixed_collision"
    staged = root / (".partial_" + run_id)
    final = root / run_id
    final.mkdir()
    sentinel = final / "sentinel"
    sentinel.write_text("keep")
    monkeypatch.setattr(artifacts, "_new_run_id", lambda _: (run_id, staged, final))
    with pytest.raises(FileExistsError):
        artifacts.write_result(result, root)
    assert sentinel.read_text() == "keep"


def test_path_component_keys_reject_traversal(config_dict, tmp_path):
    result = run_fixed(FixedConfig.from_dict(config_dict))
    values = result["values"]
    key, frame = next(iter(values.items()))
    values["../escape"] = frame
    with pytest.raises(ValueError, match="member name"):
        write_result(result, tmp_path / "out")
    assert not (tmp_path / "escape").exists()
