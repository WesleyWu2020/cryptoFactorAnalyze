from __future__ import annotations

import json

import numpy as np
import pandas as pd

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
        "fee_rate": 0.002,
        "slippage": 0.003,
    }


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
                        "all_costs": {"status": "incomplete", "ledger": ledger.iloc[:2], "orders": empty, "positions": empty, "funding": empty, "diagnostics": {"missing_tail": True}},
                    },
                },
                "group_returns": pd.DataFrame({"group_1": [0.1, 0.2, 0.3]}, index=dates),
                "diagnostics": {"grouping": {"insufficient_dates": []}},
                "paths": {},
            }

    result = evaluate_oos(
        predictions,
        output_dir=tmp_path,
        factor_name="ml_demo",
        manager_cls=FakeManager,
        h5_path=tmp_path / "input.h5",
    )

    assert result["status"] == "incomplete"
    assert (tmp_path / "factor.parquet").is_file()
    assert (tmp_path / "daily_ic.parquet").is_file()
    assert (tmp_path / "quarterly_metrics.parquet").is_file()
    assert (tmp_path / "all_costs__ledger.parquet").is_file()
    payload = json.loads((tmp_path / "evaluation.json").read_text())
    assert payload["status"] == "incomplete"
    assert payload["scenarios"]["all_costs"]["diagnostics"]["missing_tail"] is True
    assert "NaN" not in (tmp_path / "evaluation.json").read_text()
