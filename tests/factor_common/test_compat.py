"""Characterization tests for the legacy factor_miner compatibility adapter.

These pin the public contract of the old ``factor_analyse.factor_analyse_custom.factor_miner``
class as now served by ``factor_common.compat.LegacyFactorMiner``: constructor
positional argument order, custom factor-column rename to the standard
``factor``, no mutation of the caller's DataFrame, and representative public
return shapes (``IC()`` 9-tuple order, ``performance()`` table columns).
Chart element IDs are deliberately never snapshotted.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_common.compat import LegacyFactorMiner

REPO_ROOT = Path(__file__).parents[2]
EXAMPLE_MODULE = REPO_ROOT / "factor_analyse" / "factor_mining" / "example_momentum.py"

DATES = pd.date_range("2024-01-06", "2024-01-08", freq="D")
INSTRUMENTS = ["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT", "FUSDT"]
FACTOR_COL = "MyFactor"

PERFORMANCE_COLUMNS = [
    "group",
    "return",
    "turnover",
    "annual_return",
    "sharp",
    "IC",
    "volatility",
    "annual_volatility",
    "max_drawback",
    "md_period_days",
    "recovery_period_days",
    "win_percent",
    "profit-loss ratio",
]


def _legacy_frame() -> pd.DataFrame:
    rows = []
    for day_index, date in enumerate(DATES):
        for symbol_index, instrument in enumerate(INSTRUMENTS):
            factor = (symbol_index + 1) * 0.1 + day_index * 0.01
            future_ret = factor * 0.3 + (symbol_index % 2) * 0.05 - 0.02 * day_index
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "instrument": instrument,
                FACTOR_COL: factor,
                "future_ret": future_ret,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def miner(tmp_path, h5_fixture):
    frame = _legacy_frame()
    return LegacyFactorMiner(
        frame,
        FACTOR_COL,
        1,
        n_groups=3,
        rebalance_period=1,
        h5_path=h5_fixture,
        base_dir=tmp_path / "factor_results",
        reports_dir=tmp_path / "reports",
        include_funding=False,
    )


def test_constructor_positional_arguments_preserved(tmp_path, h5_fixture):
    frame = _legacy_frame()
    miner = LegacyFactorMiner(
        frame, FACTOR_COL, -1, "report.html", "key", "secret", 5, 10, 90,
        h5_path=h5_fixture, base_dir=tmp_path / "fr", reports_dir=tmp_path / "rep",
    )
    assert miner.factor_name == FACTOR_COL
    assert miner.factor_direction == -1
    assert miner.render_path == "report.html"
    assert miner.api_key == "key"
    assert miner.api_secret == "secret"
    assert miner.n_groups == 5
    assert miner.rebalance_period == 10
    assert miner.out_of_sample_days == 90
    # The Binance client is gone from the active evaluation path.
    assert miner.client is None


def test_custom_factor_column_renamed_to_standard_factor(miner):
    assert "factor" in miner.factor_data.columns
    assert FACTOR_COL not in miner.factor_data.columns
    assert {"date", "instrument", "factor"}.issubset(miner.factor_data.columns)


def test_input_dataframe_is_never_mutated(tmp_path, h5_fixture):
    frame = _legacy_frame()
    snapshot = frame.copy(deep=True)
    miner = LegacyFactorMiner(
        frame, FACTOR_COL, 1, n_groups=3,
        h5_path=h5_fixture, base_dir=tmp_path / "fr", reports_dir=tmp_path / "rep",
        include_funding=False,
    )
    miner.IC()
    miner.performance(2)
    miner.calculate_hedged_returns_with_fees()
    miner.calculate_rank_ic_decay()
    miner.calculate_rank_ic_autocorr()
    pd.testing.assert_frame_equal(frame, snapshot)


def test_ic_tuple_order_and_external_future_ret_labels(miner):
    frame = _legacy_frame()
    result = miner.IC()
    assert isinstance(result, tuple) and len(result) == 9
    ic, acc_ic, ir, ic_ir, t_stat, p_value, spearman_corr, rank_ic, daily_ic = result

    # Provided future_ret is the external-label IC input: exact daily Pearson/
    # Spearman means over the supplied column, not the H5 open labels.
    expected_ic = frame.groupby("date").apply(
        lambda group: group[FACTOR_COL].corr(group["future_ret"])
    ).mean()
    expected_rank_ic = frame.groupby("date").apply(
        lambda group: group[FACTOR_COL].corr(group["future_ret"], method="spearman")
    ).mean()
    assert ic == pytest.approx(expected_ic)
    assert rank_ic == pytest.approx(expected_rank_ic)

    assert list(acc_ic.columns) == ["date", "acc_ic"]
    assert list(daily_ic.columns) == ["date", "ic", "rank_ic", "acc_ic"]
    assert len(daily_ic) == len(DATES)
    assert acc_ic["acc_ic"].iloc[-1] == pytest.approx(daily_ic["ic"].sum())
    assert daily_ic["acc_ic"].equals(acc_ic["acc_ic"])
    for value in (ir, ic_ir, t_stat, p_value, spearman_corr):
        assert value is None or np.isfinite(value)


def test_ic_with_sample_split_tuple(miner):
    result = miner.IC_with_sample_split(sample_type="样本内")
    assert isinstance(result, tuple) and len(result) == 9
    result_out = miner.IC_with_sample_split(sample_type="样本外")
    assert isinstance(result_out, tuple) and len(result_out) == 9
    with pytest.raises(ValueError, match="data"):
        miner.IC_with_sample_split(data=pd.DataFrame(), sample_type="样本内")


def test_performance_table_columns_and_group_labels(miner):
    table = miner.performance(2)
    assert list(table.columns) == PERFORMANCE_COLUMNS
    assert table["group"].iloc[0] == "long"
    short_table = miner.performance(0)
    assert short_table["group"].iloc[0] == "short"

    split_table = miner.performance_with_sample_split(2, sample_type="样本外")
    assert list(split_table.columns) == PERFORMANCE_COLUMNS


def test_performance_respects_factor_direction(tmp_path, h5_fixture):
    frame = _legacy_frame()
    miner = LegacyFactorMiner(
        frame, FACTOR_COL, -1, n_groups=3,
        h5_path=h5_fixture, base_dir=tmp_path / "fr", reports_dir=tmp_path / "rep",
        include_funding=False,
    )
    assert miner.performance(0)["group"].iloc[0] == "long"
    assert miner.performance(2)["group"].iloc[0] == "short"


def test_hedged_returns_fee_rate_keyword_preserved(miner):
    result, stats_no_fee, stats_with_fee = miner.calculate_hedged_returns_with_fees(
        fee_rate=0.001
    )
    assert {"annualized_return", "sharpe_ratio", "max_drawdown", "win_rate"} == set(
        stats_no_fee
    )
    assert set(stats_with_fee) == set(stats_no_fee)
    assert "cum_return_no_fee" in result.columns
    assert "cum_return_with_fee" in result.columns
    assert result["cum_return_with_fee"].iloc[-1] <= result["cum_return_no_fee"].iloc[-1]

    split_result, _, _ = miner.calculate_hedged_returns_with_fees_sample_split(
        fee_rate=0.001, sample_type="样本外"
    )
    assert isinstance(split_result, pd.DataFrame)
    with pytest.raises(ValueError, match="data"):
        miner.calculate_hedged_returns_with_fees_sample_split(
            data=pd.DataFrame(), fee_rate=0.001
        )


def test_decay_autocorr_and_halflife_shapes(miner):
    decay = miner.calculate_rank_ic_decay(max_lag=10)
    assert list(decay.columns) == ["lag", "rank_ic"]
    assert decay["lag"].tolist() == list(range(1, 11))

    autocorr = miner.calculate_rank_ic_autocorr(max_lag=20)
    assert list(autocorr.columns) == ["lag", "autocorr"]
    assert autocorr["lag"].tolist() == list(range(1, 21))

    halflife = miner.calc_rankic_halflife(decay)
    assert halflife is None or isinstance(halflife, (int, float))


def test_get_latest_group_symbols(miner):
    latest_date, group_data = miner.get_latest_group_symbols()
    assert pd.Timestamp(latest_date) == DATES[-1]
    assert set(group_data) == {0, 1, 2}
    assert len(group_data[2]) == 2  # top group holds 2 of 6 instruments
    instrument, value = group_data[2][0]
    assert instrument == "FUSDT"  # highest factor value on the latest date
    assert isinstance(value, float)


def test_render_writes_html_report(miner, tmp_path):
    render_path = tmp_path / "legacy_report.html"
    miner.render_path = str(render_path)
    info = miner.render()
    assert render_path.is_file()
    html = render_path.read_text(encoding="utf-8")
    assert FACTOR_COL in html
    assert info["output_path"] == str(render_path)


def test_show_plot_removed_with_migration_message(miner):
    with pytest.raises(NotImplementedError, match="render"):
        miner.show_plot()


def test_legacy_module_exports_adapter():
    path = REPO_ROOT / "factor_analyse" / "factor_analyse_custom.py"
    spec = importlib.util.spec_from_file_location("factor_analyse_custom_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.factor_miner is LegacyFactorMiner
    text = path.read_text(encoding="utf-8").lower()
    assert "from binance" not in text
    assert "import binance" not in text


def test_factor_config_registers_example_with_module_path():
    sys.path.insert(0, str(REPO_ROOT / "factor_analyse"))
    try:
        import factor_config
    finally:
        sys.path.remove(str(REPO_ROOT / "factor_analyse"))
    config = factor_config.get_factor_config("example_momentum")
    for key in (
        "file_prefix", "factor_name", "factor_direction", "factor_desc",
        "rebalance_period", "module_path",
    ):
        assert key in config
    assert config["factor_name"] == "example_momentum"
    assert (REPO_ROOT / config["module_path"]).is_file()


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "factor_analyse/main.py", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )


def test_cli_list_shows_existing_names_and_migration_state():
    proc = _run_cli("--list")
    assert proc.returncode == 0, proc.stderr
    assert "price_momentum_60d" in proc.stdout
    assert "example_momentum" in proc.stdout
    # Migration state is explicit in both directions.
    migrated_line = next(
        line for line in proc.stdout.splitlines() if "example_momentum" in line
    )
    assert "migrated" in migrated_line
    legacy_line = next(
        line for line in proc.stdout.splitlines() if "price_momentum_60d" in line
    )
    assert "not migrated" in legacy_line


def test_cli_unmigrated_factor_reports_status_and_exits_nonzero():
    proc = _run_cli("price_momentum_60d")
    assert proc.returncode != 0
    output = proc.stdout + proc.stderr
    assert "not migrated" in output
