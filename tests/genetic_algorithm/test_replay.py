"""Tests for bounded stage replay of exported GP candidates via FactorManager."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from data.crypto_quant.panel import build_research_panel
from data.crypto_quant.store import CryptoQuantStore
from factor_common.loader import load_factor
from Genetic_Algorithm.config import Stage
from Genetic_Algorithm.evolution import Candidate
from Genetic_Algorithm.expression import (
    Node,
    canonical_tree,
    expression_hash,
    node_count,
)
from Genetic_Algorithm.replay import ReplayEvaluationError, replay

SYMBOLS = [f"{letter}USDT" for letter in "ABCDEFGHIJKL"]
CALENDAR = pd.date_range("2024-01-01", "2024-02-19", freq="D")
STAGE = Stage("validation", "2024-01-20", "2024-01-24")
TREE = Node("rank", (Node("rolling_mean", (Node("return_1d"),), window=2),))

# One boundary (00:00) and one intraday (08:00) funding event for every symbol,
# both inside the replay window, with an exhaustive schedule elsewhere.
FUNDING_DAYS = {
    pd.Timestamp("2024-01-22"): ["2024-01-22 00:00:00"],
    pd.Timestamp("2024-01-23"): ["2024-01-23 08:00:00"],
}


def _candidate(tree=TREE):
    canonical = canonical_tree(tree)
    return Candidate(
        canonical, expression_hash(canonical), (0.1, 0.05, -node_count(canonical))
    )


def _universe() -> pd.DataFrame:
    rows = []
    for cmc_id, symbol in enumerate(SYMBOLS, start=1):
        rows.append({
            "decision_date": "2024-01-01",
            "effective_date": "2024-01-02",
            "effective_end_date": pd.NaT,
            "cmc_id": cmc_id,
            "cmc_symbol": symbol[:-4],
            "binance_symbol": symbol,
            "market_cap_rank": cmc_id,
            "cmc_weight": 1.0 / len(SYMBOLS),
        })
    return pd.DataFrame(rows)


def _klines() -> pd.DataFrame:
    rows = []
    for day in CALENDAR:
        day_offset = (day - CALENDAR[0]).days
        for index, symbol in enumerate(SYMBOLS, start=1):
            close = 100.0 + index * 10.0 + day_offset * (0.4 + 0.05 * index)
            rows.append({
                "date": day,
                "symbol": symbol,
                "close_time": day + pd.Timedelta(hours=23, minutes=59),
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 2.0,
                "close": close,
                "volume": 1_000.0,
                "quote_volume": close * 1_000.0,
                "trade_count": 10,
                "taker_buy_base_volume": 500.0,
                "taker_buy_quote_volume": close * 500.0,
            })
    return pd.DataFrame(rows)


def _funding_events() -> pd.DataFrame:
    rows = []
    for day, times in FUNDING_DAYS.items():
        for time in times:
            for index, symbol in enumerate(SYMBOLS, start=1):
                rows.append({
                    "funding_time": time,
                    "symbol": symbol,
                    "funding_rate": 0.001,
                    "mark_price": 100.0 + index * 10.0,
                    "rate_type": "Regular",
                })
    return pd.DataFrame(rows)


def _funding_schedule() -> pd.DataFrame:
    rows = []
    for day in CALENDAR:
        for symbol in SYMBOLS:
            rows.append({
                "date": day,
                "symbol": symbol,
                "expected_times": FUNDING_DAYS.get(day, []),
            })
    return pd.DataFrame(rows)


def _write_store(path, *, funding_schedule):
    store = CryptoQuantStore(path)
    universe = _universe()
    klines = _klines()
    funding = _funding_events()
    panel = build_research_panel(
        universe, klines, funding, CALENDAR[-1], funding_schedule=funding_schedule
    )
    panel = panel.merge(
        universe[["binance_symbol", "decision_date", "market_cap_rank"]],
        on="binance_symbol",
        how="left",
    )
    store.replace("universe_monthly", universe)
    store.replace("klines_daily", klines)
    store.replace("funding_events", funding)
    store.replace("research_panel_daily", panel)
    return path


@pytest.fixture
def complete_h5(tmp_path):
    return _write_store(
        tmp_path / "replay.h5", funding_schedule=_funding_schedule()
    )


@pytest.fixture
def unknown_h5(tmp_path):
    return _write_store(tmp_path / "replay.h5", funding_schedule=None)


def test_replay_stage_is_bounded_and_complete(complete_h5, tmp_path):
    run_dir = tmp_path / "run"
    outcome = replay(_candidate(), STAGE, complete_h5, run_dir, direction=1)

    result = outcome["result"]
    assert result["status"] == "complete"
    assert outcome["stage"] is STAGE
    assert outcome["identifier"] == f"GP_{expression_hash(TREE)[:16]}"

    # The exported wrapper lives under the run directory, not factor_mining.
    assert str(outcome["export"].path).startswith(str(run_dir))
    exported_spec = load_factor(outcome["export"].path)
    assert exported_spec.setting["factor_direction"] == 1
    assert exported_spec.setting["context_eligible"] is True

    # Signals stop at signal_end; nothing beyond the stage end is read.
    assert result["factor_value"].index.max() == STAGE.signal_end

    # Resolved profile pins the replay contract.
    profile = outcome["profile"]
    assert profile.rebalance_days == 1
    assert profile.n_groups == 10
    assert profile.factor_direction == 1
    assert profile.fee_rate == 0.0005
    assert profile.slippage == 0.001
    assert profile.include_funding is True
    assert profile.funding_price_mode == "strict"
    assert profile.gross_exposure == 1.0

    all_costs = result["factor_result"]["scenarios"]["all_costs"]
    assert all_costs["status"] == "complete"
    ledger = all_costs["ledger"]

    # Entry and exit costs are charged at both boundaries of the holding loop.
    assert ledger["fee"].iloc[0] > 0.0
    assert ledger["fee"].iloc[-1] > 0.0
    assert ledger["slippage"].iloc[0] > 0.0
    assert ledger["slippage"].iloc[-1] > 0.0

    # Boundary funding (00:00, pre-trade) and intraday funding (08:00,
    # post-trade) both settle inside the stage.
    assert ledger.loc[pd.Timestamp("2024-01-22"), "funding_cashflow"] != 0.0
    assert ledger.loc[pd.Timestamp("2024-01-23"), "funding_cashflow"] != 0.0

    # The framework's 50/50 long-short weights are preserved: each trading
    # day is dollar-neutral with unit gross exposure.
    filled = all_costs["orders"]
    filled = filled[filled["status"] == "filled"]
    traded = filled.groupby("date")["target_weight"]
    first_day = filled["date"].min()
    first_day_weights = traded.get_group(first_day)
    assert first_day_weights.sum() == pytest.approx(0.0)
    assert first_day_weights.abs().sum() == pytest.approx(1.0)

    # Final liquidation lands at most at the stage end and flattens the book.
    diagnostics = all_costs["diagnostics"]
    liquidation = pd.Timestamp(diagnostics["liquidation_date"])
    assert liquidation <= STAGE.end
    assert liquidation == STAGE.end  # signal_end plus the two-day tail
    assert diagnostics["final_quantities"] == {}
    positions = all_costs["positions"]
    assert (positions.iloc[-1] == 0.0).all()

    # Stage-wide all_costs metrics come from the full certified ledger via the
    # framework's summary helpers — never the internal IS/OOS split.
    metrics = outcome["metrics"]
    assert metrics is not None
    assert metrics["n_periods"] == len(ledger)
    full_block = result["factor_performance"]["scenarios"]["all_costs"]["full"]
    assert metrics == full_block

    # The report renders under reports/ with stage and run identifiers.
    report_path = outcome["report_path"]
    assert STAGE.name in report_path
    assert outcome["run_id"] in report_path
    assert "reports" in report_path
    text = open(report_path, encoding="utf-8").read()
    assert "Status: complete" in text
    assert outcome["identifier"] in text

    # The replay artifact stores the resolved profile, stage bounds, stage
    # metrics, and discloses the framework's internal split.
    artifact = json.loads(open(outcome["artifact_path"], encoding="utf-8").read())
    assert artifact["stage"]["name"] == STAGE.name
    assert artifact["stage"]["end"] == "2024-01-24"
    assert artifact["stage"]["signal_end"] == "2024-01-22"
    assert artifact["profile"]["fee_rate"] == 0.0005
    assert artifact["profile"]["gross_exposure"] == 1.0
    assert artifact["profile"]["funding_price_mode"] == "strict"
    assert artifact["all_costs_metrics"]["n_periods"] == len(ledger)
    assert artifact["liquidation_date"] == "2024-01-24"
    assert artifact["framework_split"]["split_date"]
    assert "not" in artifact["framework_split_disclosure"]
    assert artifact["expression_hash"] == expression_hash(TREE)


def test_replay_non_complete_all_costs_is_a_failure_not_zero_return(
    unknown_h5, tmp_path
):
    run_dir = tmp_path / "run"
    with pytest.raises(ReplayEvaluationError, match="all_costs"):
        replay(_candidate(), STAGE, unknown_h5, run_dir, direction=1)
    # No replay artifact is published for a failed evaluation.
    assert not (run_dir / f"replay_{STAGE.name}.json").exists()


def test_replay_beyond_available_data_fails_instead_of_zero_return(
    complete_h5, tmp_path
):
    # The stage end lies past the last stored day: the execution tail is
    # unpriceable, so the evaluation must fail loudly.
    stage = Stage("test", "2024-02-15", "2024-02-25")
    with pytest.raises(ReplayEvaluationError):
        replay(_candidate(), stage, complete_h5, tmp_path / "run", direction=1)


def test_replay_uses_the_fixed_training_direction(complete_h5, tmp_path):
    outcome = replay(
        _candidate(), STAGE, complete_h5, tmp_path / "run", direction=-1
    )
    assert outcome["result"]["metadata"]["factor_direction"] == -1
    exported_spec = load_factor(outcome["export"].path)
    assert exported_spec.setting["factor_direction"] == -1

    # Direction flips the long/short legs of the 50/50 portfolio.
    base = replay(_candidate(), STAGE, complete_h5, tmp_path / "run_long", direction=1)

    def first_day_weights(res):
        orders = res["factor_result"]["scenarios"]["gross"]["orders"]
        filled = orders[orders["status"] == "filled"]
        first = filled["date"].min()
        return (
            filled[filled["date"] == first]
            .set_index("instrument")["target_weight"]
            .sort_index()
        )

    pd.testing.assert_series_equal(
        first_day_weights(outcome["result"]), -1 * first_day_weights(base["result"])
    )
