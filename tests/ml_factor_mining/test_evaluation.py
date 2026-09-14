from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ML_factor_mining.config import Config
from ML_factor_mining.evaluation import evaluate_oos, manager_params, quarterly_metrics


def _ledger(start="2025-01-01", periods=95):
    dates = pd.date_range(start, periods=periods, freq="D", name="date")
    returns = pd.Series(np.resize([0.01, -0.002, 0.003], periods), index=dates)
    equity = (1.0 + returns).cumprod()
    return pd.DataFrame(
        {
            "equity": equity,
            "return": returns,
            "price_pnl": returns,
            "funding_cashflow": 0.0,
            "fee": 0.0,
            "slippage": 0.0,
            "trade_notional": 0.0,
        },
        index=dates,
    )


def test_quarterly_metrics_slices_continuous_ledger_and_reconciles():
    ledger = _ledger(periods=95)
    result = quarterly_metrics(ledger)

    assert list(result["quarter"]) == ["2025Q1", "2025Q2"]
    assert result["total_return"].sum() != ledger["return"].sum()
    compounded = float((1.0 + result["total_return"]).prod() - 1.0)
    np.testing.assert_approx_equal(
        compounded, float((1.0 + ledger["return"]).prod() - 1.0), significant=12
    )
    assert result["reconciled"].all()
    assert result.loc[0, "starting_equity"] == 1.0
    assert result.loc[1, "starting_equity"] == ledger.loc["2025-04-01", "equity"]


def test_quarterly_metrics_keeps_prediction_quarters_after_accounting_halt():
    ledger = _ledger(start="2025-01-01", periods=2)
    prediction_dates = pd.to_datetime(["2025-01-01", "2025-07-01"])
    predictions = pd.DataFrame(
        {"date": prediction_dates, "instrument": "A", "factor": 1.0}
    )

    result = quarterly_metrics(
        ledger,
        predictions=predictions,
        evidence_end="2025-07-03",
        status="incomplete",
    )

    assert list(result["quarter"]) == ["2025Q1", "2025Q2", "2025Q3"]
    assert result.loc[result["quarter"].eq("2025Q2"), "total_return"].isna().all()
    assert result.loc[result["quarter"].eq("2025Q2"), "missing_tail"].item()
    assert result.loc[result["quarter"].eq("2025Q3"), "missing_tail"].item()


def test_quarter_flags_are_local_when_global_accounting_is_incomplete():
    ledger = _ledger(start="2025-01-01", periods=91)  # complete Q1 plus Apr 1
    predictions = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-01"]), "instrument": "A", "factor": 1.0}
    )

    result = quarterly_metrics(
        ledger,
        predictions=predictions,
        evidence_end="2025-04-02",
        status="incomplete",
    ).set_index("quarter")

    assert result.loc["2025Q1", "complete_calendar_quarter"]
    assert not result.loc["2025Q1", "missing_tail"]
    assert result.loc["2025Q2", "missing_tail"]
    assert result.loc["2025Q2", "accounting_tail_only"]


def test_manager_params_allowlist_and_iso_dates():
    config = Config(
        holding_days=5,
        anchor_date="2024-01-01",
        fee_rate=0.002,
        slippage=0.003,
    )
    assert manager_params(config, start=pd.Timestamp("2025-01-01"), end="2025-03-31") == {
        "start": "2025-01-01",
        "end": "2025-03-31",
        "rebalance_days": 5,
        "anchor_date": "2024-01-01",
        "n_groups": 5,
        "fee_rate": 0.002,
        "slippage": 0.003,
    }


def test_manager_params_supports_plan_style_positional_bounds():
    config = Config()
    params = manager_params(config, "2025-01-01", "2025-03-31")
    start, end, overrides = __import__("factor_common.manager", fromlist=["FactorManager"]).FactorManager._split_params(params)
    from factor_common.profiles import resolve_profile
    from ML_factor_mining.scoring import backtest_profile

    actual = resolve_profile("perp_1d", overrides)
    expected = backtest_profile(config)
    assert start == pd.Timestamp("2025-01-01")
    assert end == pd.Timestamp("2025-03-31")
    assert actual.rebalance_days == expected.rebalance_days
    assert actual.fee_rate == expected.fee_rate
    assert actual.slippage == expected.slippage
    assert actual.split_date == "2024-12-31"


def test_evaluate_oos_writes_original_tables_and_honest_status(tmp_path):
    dates = pd.date_range("2025-01-01", periods=3, name="date")
    predictions = pd.DataFrame(
        {
            "date": dates.repeat(3),
            "instrument": ["A", "B", "C"] * 3,
            "factor": np.arange(9, dtype=float),
        }
    )
    ledger = _ledger(periods=3)
    empty = pd.DataFrame()

    class FakeManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def evaluate(self, source, **kwargs):
            assert list(source.columns) == ["date", "instrument", "factor"]
            assert kwargs["params"] == {
                "start": "2025-01-01",
                "end": "2025-01-03",
                "split_date": "2024-12-31",
                "n_groups": 5,
            }
            return {
                "status": "incomplete",
                "metadata": {"factor_name": kwargs["factor_name"], "profile_id": "perp_1d"},
                "factor_value": source,
                "factor_performance": {
                    "samples": {
                        "full": {"ic": {"daily": [{"date": "2025-01-01", "ic": 0.1, "rank_ic": 0.2}]}},
                    }
                },
                "factor_result": {
                    "status": "incomplete",
                    "scenarios": {
                        "gross": {"status": "complete", "ledger": ledger, "orders": empty, "positions": empty, "funding": empty, "diagnostics": {}},
                        "trading_net": {"status": "complete", "ledger": ledger, "orders": empty, "positions": empty, "funding": empty, "diagnostics": {}},
                        "all_costs": {"status": "incomplete", "ledger": ledger.iloc[:2], "orders": empty, "positions": empty, "funding": empty, "funding_coverage": empty, "diagnostics": {"missing_tail": True}},
                    },
                },
                "group_returns": pd.DataFrame({"group_1": [0.1, 0.2, 0.3]}, index=dates),
                "diagnostics": {"grouping": {"insufficient_dates": []}},
                "paths": {"report_path": str(tmp_path / "standard.html")},
            }

    result = evaluate_oos(
        predictions,
        output_dir=tmp_path,
        factor_name="ml_demo",
        manager_cls=FakeManager,
        h5_path=tmp_path / "input.h5",
        coverage=pd.DataFrame({"date": dates, "coverage": [0.7, 0.8, 0.9]}),
    )

    assert result["status"] == "incomplete"
    assert (tmp_path / "factor.parquet").is_file()
    assert (tmp_path / "factor_oos.parquet").is_file()
    assert (tmp_path / "daily_ic.parquet").is_file()
    assert (tmp_path / "quarterly_metrics.parquet").is_file()
    assert (tmp_path / "all_costs__ledger.parquet").is_file()
    assert (tmp_path / "orders.parquet").is_file()
    assert (tmp_path / "positions.parquet").is_file()
    assert (tmp_path / "funding.parquet").is_file()
    assert (tmp_path / "funding_coverage.parquet").is_file()
    assert (tmp_path / "daily_ledger.parquet").is_file()
    assert (tmp_path / "group_forward_return_diagnostics.parquet").is_file()
    assert (tmp_path / "quarterly.html").is_file()
    quarterly_html = (tmp_path / "quarterly.html").read_text()
    assert "quarterly OOS evaluation" in quarterly_html
    assert "continuous all-costs ledger" in quarterly_html
    assert "forward-label diagnostics" in quarterly_html
    assert "Continuous NAV / standard evaluation report" in quarterly_html
    payload = json.loads((tmp_path / "evaluation.json").read_text())
    assert payload["status"] == "incomplete"
    assert payload["all_costs_status"] == "incomplete"
    assert payload["prediction_start"] == "2025-01-01"
    assert payload["prediction_end"] == "2025-01-03"
    assert payload["quarterly_metrics"][0]["coverage_mean"] == 0.8
    assert payload["scenarios"]["all_costs"]["diagnostics"]["missing_tail"] is True
    assert "forward-label diagnostics" in payload["group_diagnostics_scope"]
    assert "NaN" not in (tmp_path / "evaluation.json").read_text()


def test_evaluate_oos_mapping_config_bounds_and_as_of(tmp_path):
    dates = pd.date_range("2025-01-01", periods=6, name="date")
    predictions = pd.DataFrame(
        {"date": dates.repeat(3), "instrument": ["A", "B", "C"] * 6,
         "factor": np.arange(18, dtype=float)}
    )
    ledger = _ledger(periods=2)
    captured = {}

    class FakeManager:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def evaluate(self, source, **kwargs):
            captured["source_end"] = source["date"].max()
            captured["params"] = kwargs["params"]
            return {
                "status": "incomplete", "metadata": {},
                "factor_performance": {},
                "factor_result": {"scenarios": {"all_costs": {
                    "status": "incomplete", "ledger": ledger,
                    "orders": pd.DataFrame(), "positions": pd.DataFrame(),
                    "funding": pd.DataFrame(), "diagnostics": {},
                }}},
                "diagnostics": {},
            }

    evaluate_oos(
        predictions, tmp_path, config={"holding_days": 2},
        evidence_end="2025-01-06", manager_cls=FakeManager,
    )
    assert captured["init"]["as_of"] == pd.Timestamp("2025-01-06")
    assert captured["params"]["start"] == "2025-01-01"
    assert captured["params"]["end"] == "2025-01-03"
    assert captured["params"]["rebalance_days"] == 2
    assert captured["source_end"] == pd.Timestamp("2025-01-03")


def test_evaluate_oos_rejects_ineligible_external_prediction(tmp_path):
    dates = pd.date_range("2025-01-01", periods=2, name="date")
    predictions = pd.DataFrame(
        {"date": dates.repeat(2), "instrument": ["A", "B"] * 2,
         "factor": np.arange(4, dtype=float)}
    )

    class Provider:
        def get_universe(self, *, start, end):
            return pd.DataFrame(
                {"A": [True, True], "B": [False, False]},
                index=pd.date_range(start, end, freq="D", name="date"),
            )

    class FakeManager:
        dp = Provider()

        def evaluate(self, *args, **kwargs):
            raise AssertionError("membership failure must happen before evaluation")

    with pytest.raises(ValueError, match="ineligible historical instruments"):
        evaluate_oos(predictions, tmp_path, manager=FakeManager())
