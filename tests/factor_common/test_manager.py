"""Tests for FactorManager orchestration and read APIs.

All tests run against temp artifact stores and the shared H5 fixture; an
autouse fixture hard-fails any attempted network use.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_common.manager import FactorManager
from factor_common.storage import ConcurrentSourceChangeError

FACTOR_SOURCE = '''
import numpy as np

TYPE = "regular"

META = {{"factor_name": "{name}", "author": "test", "level": "daily",
        "category": "momentum", "description": "1-day momentum"}}

SETTING = {{"data_needed": ["close"], "universe": "historical_top50",
          "warmup_bars": 1, "preprocessing": "none",
          "params": {{"window": 1}}, "factor_direction": {direction}}}


def calc_factor(data_ctx):
    close = data_ctx["close"]
    window = SETTING["params"]["window"]
    {formula}
'''

LEAKY_FORMULA = "return close.shift(-1) / close - 1.0"
CAUSAL_FORMULA = "return close / close.shift(window) - 1.0"

# The shared fixture has an intentional EUSDT kline gap on 2024-01-05; signals
# from 2024-01-06 on never hold it (its factor value is NaN there), so the
# default window below supports a fully complete accounting.
BASE_PARAMS = {
    "start": "2024-01-06",
    "end": "2024-01-08",
    "n_groups": 3,
    "include_funding": False,
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access is not allowed in manager tests")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


def _write_factor(tmp_path: Path, name: str = "mom1", *, direction: int = 1,
                  formula: str = CAUSAL_FORMULA) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(
        FACTOR_SOURCE.format(name=name, direction=direction, formula=formula),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def factor_file(tmp_path):
    return _write_factor(tmp_path)


@pytest.fixture
def manager(tmp_path, h5_fixture):
    return FactorManager(
        h5_path=h5_fixture,
        base_dir=tmp_path / "factor_results",
        reports_dir=tmp_path / "reports",
        persist_evaluations=True,
    )


def _external_matrix() -> pd.DataFrame:
    dates = pd.date_range("2024-01-06", "2024-01-08", freq="D", name="date")
    columns = ["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT", "FUSDT"]
    values = np.arange(len(dates) * len(columns), dtype="float64").reshape(
        len(dates), len(columns)
    )
    return pd.DataFrame(values + 1.0, index=dates, columns=columns)


def test_evaluate_module_source_returns_standard_result(manager, factor_file):
    result = manager.evaluate(str(factor_file), params=dict(BASE_PARAMS), plot=True)

    assert result["status"] == "complete"
    for key in (
        "status", "factor_value", "factor_performance", "factor_result",
        "diagnostics", "metadata", "paths", "group_returns",
        "run_id", "evaluation_id",
    ):
        assert key in result

    values = result["factor_value"]
    assert values.index.equals(pd.date_range("2024-01-06", "2024-01-08", freq="D"))
    assert values.notna().any().any()

    performance = result["factor_performance"]
    assert performance["profile_id"] == "perp_1d"
    assert set(performance["samples"]) == {"full", "in_sample", "out_of_sample"}
    assert set(performance["scenarios"]) == {"gross", "trading_net", "all_costs"}
    assert performance["samples"]["full"]["ic"]["n_dates"] > 0

    accounting = result["factor_result"]
    assert accounting["status"] == "complete"
    for name in ("gross", "trading_net", "all_costs"):
        assert accounting["scenarios"][name]["status"] == "complete"
        assert not accounting["scenarios"][name]["ledger"].empty

    group_returns = result["group_returns"]
    assert list(group_returns.columns) == ["group_1", "group_2", "group_3"]
    assert group_returns.notna().any().any()

    metadata = result["metadata"]
    assert metadata["factor_name"] == "mom1"
    assert metadata["profile_id"] == "perp_1d"
    assert metadata["factor_direction"] == 1
    assert metadata["n_groups"] == 3

    paths = result["paths"]
    assert paths["run_id"] == result["run_id"]
    assert Path(paths["factor_path"]).is_file()
    assert Path(paths["report_path"]).is_file()

    coverage = result["diagnostics"]["coverage"]
    assert set(coverage) == {"signal", "label", "funding"}


def test_evaluate_dataframe_source_requires_explicit_identity(manager):
    matrix = _external_matrix()
    with pytest.raises(ValueError, match="factor_name"):
        manager.evaluate(matrix, params=dict(BASE_PARAMS), plot=False)

    result = manager.evaluate(
        matrix, factor_name="external_mom", params=dict(BASE_PARAMS), plot=False
    )
    assert result["status"] == "complete"
    assert result["metadata"]["factor_name"] == "external_mom"
    assert result["metadata"]["source_type"] == "dataframe"
    validation = result["diagnostics"]["validation"]
    assert validation["cutoff"]["status"] == "not_verified"
    pd.testing.assert_frame_equal(
        manager.get_value("external_mom"), result["factor_value"]
    )


def test_evaluate_does_not_persist_evaluation_by_default(tmp_path, h5_fixture, factor_file):
    manager = FactorManager(
        h5_path=h5_fixture,
        base_dir=tmp_path / "factor_results",
        reports_dir=tmp_path / "reports",
    )

    result = manager.evaluate(
        str(factor_file), params=dict(BASE_PARAMS), plot=False
    )

    assert result["evaluation_id"] is None
    assert result["paths"]["evaluation_dir"] is None
    assert list((tmp_path / "factor_results").glob("mom1.evaluation*")) == []


def test_evaluate_dataframe_long_table(manager):
    matrix = _external_matrix()
    long = matrix.reset_index().melt(
        id_vars="date", var_name="instrument", value_name="factor"
    )
    result = manager.evaluate(
        long, factor_name="external_long", params=dict(BASE_PARAMS), plot=False
    )
    assert result["status"] == "complete"
    pd.testing.assert_frame_equal(
        result["factor_value"],
        matrix.reindex(pd.date_range("2024-01-06", "2024-01-08", freq="D", name="date")),
        check_names=False,
    )


def test_evaluate_dataframe_rejects_duplicate_keys(manager):
    matrix = _external_matrix()
    long = matrix.reset_index().melt(
        id_vars="date", var_name="instrument", value_name="factor"
    )
    duplicated = pd.concat([long, long.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        manager.evaluate(
            duplicated, factor_name="dup_factor", params=dict(BASE_PARAMS), plot=False
        )


def test_evaluate_dataframe_rejects_unrecognized_axes(manager):
    bogus = pd.DataFrame({"alpha": [1.0, 2.0], "beta": [3.0, 4.0]})
    with pytest.raises(ValueError, match="axes"):
        manager.evaluate(
            bogus, factor_name="bogus_factor", params=dict(BASE_PARAMS), plot=False
        )


def test_factor_run_reused_when_only_fees_change(manager, factor_file):
    first = manager.evaluate(
        str(factor_file), params={**BASE_PARAMS, "fee_rate": 0.0003}, plot=False
    )
    second = manager.evaluate(
        str(factor_file), params={**BASE_PARAMS, "fee_rate": 0.001}, plot=False
    )
    assert first["run_id"] == second["run_id"]
    assert first["paths"]["factor_path"] == second["paths"]["factor_path"]
    assert first["evaluation_id"] != second["evaluation_id"]
    pd.testing.assert_frame_equal(first["factor_value"], second["factor_value"])

    first_equity = first["factor_result"]["scenarios"]["trading_net"]["ledger"]["equity"]
    second_equity = second["factor_result"]["scenarios"]["trading_net"]["ledger"]["equity"]
    assert not first_equity.equals(second_equity)

    with pytest.raises(FileNotFoundError, match="unknown evaluation"):
        manager.get_performance("mom1", evaluation_id=first["evaluation_id"])
    assert manager.get_performance("mom1")["evaluation_id"] == second["evaluation_id"]


def test_incomplete_evaluation_accessible_by_id(manager, factor_file):
    result = manager.evaluate(
        str(factor_file),
        params={**BASE_PARAMS, "include_funding": True},
        plot=False,
    )
    assert result["status"] == "incomplete"
    all_costs = result["factor_result"]["scenarios"]["all_costs"]
    assert all_costs["status"] == "incomplete"
    assert all_costs["diagnostics"]["halt_reason"] == "unresolved_funding_coverage"
    assert result["factor_performance"]["scenarios"]["all_costs"]["full"] is None

    with pytest.raises(FileNotFoundError, match="no complete evaluation"):
        manager.get_performance("mom1")

    loaded = manager.get_performance("mom1", evaluation_id=result["evaluation_id"])
    assert loaded["status"] == "incomplete"
    assert loaded["factor_performance"]["scenarios"]["all_costs"]["full"] is None
    assert loaded["factor_value"].notna().any().any()


def test_insufficient_data_status_is_structured(manager, factor_file):
    result = manager.evaluate(
        str(factor_file), params={**BASE_PARAMS, "n_groups": 10}, plot=False
    )
    assert result["status"] == "insufficient_data"
    assert result["factor_result"]["diagnostics"]["no_usable_signals"] is True

    with pytest.raises(FileNotFoundError, match="no complete evaluation"):
        manager.get_performance("mom1")
    loaded = manager.get_performance("mom1", evaluation_id=result["evaluation_id"])
    assert loaded["status"] == "insufficient_data"


def test_default_direction_from_definition_and_override(manager, tmp_path):
    factor = _write_factor(tmp_path, name="rev_mom", direction=-1)
    result = manager.evaluate(str(factor), params=dict(BASE_PARAMS), plot=False)
    assert result["metadata"]["factor_direction"] == -1

    override = manager.evaluate(
        str(factor), params={**BASE_PARAMS, "factor_direction": 1}, plot=False
    )
    assert override["metadata"]["factor_direction"] == 1
    assert override["run_id"] == result["run_id"]
    assert override["evaluation_id"] != result["evaluation_id"]


def test_unknown_params_keys_rejected(manager, factor_file):
    with pytest.raises(ValueError, match="Unknown params key"):
        manager.evaluate(str(factor_file), params={"cost": 0.001}, plot=False)
    with pytest.raises(ValueError, match="Unknown params key"):
        manager.evaluate(str(factor_file), params={"bogus": 1}, plot=False)


def test_future_leak_pattern_rejected(manager, tmp_path):
    factor = _write_factor(tmp_path, name="leaky_mom", formula=LEAKY_FORMULA)
    with pytest.raises(ValueError, match="future-leak"):
        manager.evaluate(str(factor), params=dict(BASE_PARAMS), plot=False)


def test_cutoff_replay_verified_for_module_source(manager, factor_file):
    result = manager.evaluate(str(factor_file), params=dict(BASE_PARAMS), plot=False)
    cutoff = result["diagnostics"]["validation"]["cutoff"]
    assert cutoff["status"] == "verified"
    assert cutoff["cutoffs"]
    assert all(entry["max_abs_diff"] == 0.0 for entry in cutoff["cutoffs"])


def test_snapshot_precedes_value_phase_reads(manager, factor_file, monkeypatch):
    """An H5 mtime bump during the value phase must abort the run.

    Store reads are read-only, so the pre-read snapshot taken by the manager
    stays valid; the simulated mid-eval change is detected by save_value's
    before/after stat verification.
    """
    from factor_common.data_provider import DataProvider

    original = DataProvider.get_single_data
    bumped = []

    def wrapped(self, field, *, start, end):
        if not bumped:
            bumped.append(True)
            os.utime(manager.h5_path, ns=(1_600_000_000_000_000_000,) * 2)
        return original(self, field, start=start, end=end)

    monkeypatch.setattr(DataProvider, "get_single_data", wrapped)
    with pytest.raises(ConcurrentSourceChangeError, match="changed"):
        manager.evaluate(str(factor_file), params=dict(BASE_PARAMS), plot=False)
    assert bumped  # the tamper happened inside the value phase


def test_pure_reads_do_not_disturb_snapshot(manager, factor_file):
    """Control: repeated evaluations over pure reads keep a stable stat."""
    first = manager.evaluate(str(factor_file), params=dict(BASE_PARAMS), plot=False)
    second = manager.evaluate(str(factor_file), params=dict(BASE_PARAMS), plot=False)
    assert first["run_id"] == second["run_id"]


def test_short_window_cutoff_replay_uses_pre_end_cutoff(manager, factor_file):
    """Single-day evaluations must replay against a cutoff strictly before end."""
    params = {**BASE_PARAMS, "start": "2024-01-06", "end": "2024-01-06"}
    result = manager.evaluate(str(factor_file), params=params, plot=False)
    assert result["status"] == "complete"
    cutoff = result["diagnostics"]["validation"]["cutoff"]
    assert cutoff["status"] == "verified"
    assert len(cutoff["cutoffs"]) == 1
    entry = cutoff["cutoffs"][0]
    assert pd.Timestamp(entry["cutoff"]) < pd.Timestamp("2024-01-06")


def test_short_window_cutoff_replay_catches_future_leak(manager, tmp_path):
    """A leak evading the static scan must be caught by the short-window replay.

    The reversal trick computes value[t] = close[t+1]/close[t] - 1 without any
    banned AST pattern; a tautological cutoff at ``end`` would pass it.
    """
    factor = _write_factor(
        tmp_path,
        name="sneaky_leak",
        formula=(
            "future_close = close.iloc[::-1].shift(1).iloc[::-1]\n"
            "    return future_close / close - 1.0"
        ),
    )
    params = {**BASE_PARAMS, "start": "2024-01-06", "end": "2024-01-06"}
    with pytest.raises(ValueError, match="future leak"):
        manager.evaluate(str(factor), params=params, plot=False)


def test_get_value_round_trip(manager, factor_file):
    result = manager.evaluate(str(factor_file), params=dict(BASE_PARAMS), plot=False)
    stored = manager.get_value("mom1")
    pd.testing.assert_frame_equal(stored, result["factor_value"])
    by_run = manager.get_value("mom1", run_id=result["run_id"])
    pd.testing.assert_frame_equal(by_run, result["factor_value"])


def test_plot_result_renders_saved_group_sections(manager, factor_file, tmp_path):
    result = manager.evaluate(str(factor_file), params=dict(BASE_PARAMS), plot=False)
    loaded = manager.get_performance("mom1")
    assert "group_returns" in loaded
    assert "factor_value" in loaded

    output = tmp_path / "loaded_report.html"
    manager.plot_result(loaded, output_path=output)
    html = output.read_text(encoding="utf-8")
    assert "Group cumulative return (simple sum, from saved group returns)" in html
    assert "the saved result contains no group return table" not in html
    assert "Group membership on" in html
    assert "the saved result contains no factor value matrix" not in html


def test_create_template_never_overwrites(manager, tmp_path):
    target_dir = tmp_path / "factors"
    target_dir.mkdir()
    path = manager.create_template("brand_new_factor", output_dir=target_dir)
    assert path.is_file()
    assert path.name == "brand_new_factor.py"
    with pytest.raises(FileExistsError, match="overwrite"):
        manager.create_template("brand_new_factor", output_dir=target_dir)
