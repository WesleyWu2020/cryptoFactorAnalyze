"""Tests for the migrated example factor and its CLI path.

Uses a dedicated 50-day synthetic H5 store (the shared 12-day fixture is
shorter than the example's 20-day momentum window). Verifies the module
contract, side-effect-free import, end-to-end evaluation with cutoff replay,
and the migrated CLI run.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from data.crypto_quant.panel import build_research_panel
from data.crypto_quant.store import CryptoQuantStore
from factor_common.loader import load_factor
from factor_common.manager import FactorManager

REPO_ROOT = Path(__file__).parents[2]
EXAMPLE_PATH = REPO_ROOT / "factor_analyse" / "factor_mining" / "example_momentum.py"

SYMBOLS = [f"{letter}USDT" for letter in "ABCDEFGHIJKL"]
CALENDAR = pd.date_range("2024-01-01", "2024-02-19", freq="D")

EVAL_PARAMS = {
    "start": "2024-01-25",
    "end": "2024-02-15",
    "rebalance_days": 1,
    "n_groups": 3,
    "include_funding": False,
}


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
    return pd.DataFrame([
        {
            "funding_time": "2024-02-01 00:00:00",
            "symbol": "AUSDT",
            "funding_rate": 0.001,
            "mark_price": 120.0,
            "rate_type": "Regular",
        },
    ])


def _funding_schedule() -> pd.DataFrame:
    # Independent settlement coverage for every date/symbol: an explicit empty
    # list proves no settlement was applicable, so coverage is fully known and
    # the all-costs scenario can complete.
    rows = []
    for day in CALENDAR:
        for symbol in SYMBOLS:
            expected = []
            if symbol == "AUSDT" and day == pd.Timestamp("2024-02-01"):
                expected = ["2024-02-01 00:00:00"]
            rows.append({"date": day, "symbol": symbol, "expected_times": expected})
    return pd.DataFrame(rows)


@pytest.fixture
def example_h5(tmp_path: Path) -> Path:
    path = tmp_path / "crypto_quant_example.h5"
    store = CryptoQuantStore(path)
    universe = _universe()
    klines = _klines()
    funding = _funding_events()
    panel = build_research_panel(
        universe, klines, funding, CALENDAR[-1], funding_schedule=_funding_schedule()
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
def example_manager(tmp_path, example_h5):
    return FactorManager(
        h5_path=example_h5,
        base_dir=tmp_path / "factor_results",
        reports_dir=tmp_path / "reports",
    )


def test_example_module_contract():
    spec = load_factor(EXAMPLE_PATH)
    assert spec.factor_id == "example_momentum"
    assert spec.meta["factor_name"] == "example_momentum"
    assert spec.meta["level"] == "daily"
    assert list(spec.setting["data_needed"]) == ["close"]
    assert spec.setting["params"]["window"] == 20
    assert spec.setting["warmup_bars"] == 20
    assert spec.setting["factor_direction"] == 1
    assert callable(spec.calc_factor)


def test_example_import_is_side_effect_free(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise AssertionError("FactorManager.evaluate ran at import time")

    monkeypatch.setattr(FactorManager, "evaluate", _boom)
    module_name = "example_momentum_import_test"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TYPE == "regular"
    assert module.SETTING["params"]["window"] == 20
    # Importing must not persist any artifacts anywhere.
    assert not (tmp_path / "factor_results").exists()


def test_example_evaluation_complete_with_verified_cutoff(example_manager):
    result = example_manager.evaluate(
        str(EXAMPLE_PATH), params=dict(EVAL_PARAMS), plot=False
    )
    assert result["status"] == "complete"
    assert result["metadata"]["factor_name"] == "example_momentum"

    values = result["factor_value"]
    assert values.index.equals(pd.date_range("2024-01-25", "2024-02-15", freq="D"))
    assert values.notna().all().all()

    cutoff = result["diagnostics"]["validation"]["cutoff"]
    assert cutoff["status"] == "verified"
    assert cutoff["cutoffs"]
    assert all(entry["max_abs_diff"] <= 1e-12 for entry in cutoff["cutoffs"])

    scan = result["diagnostics"]["validation"]["static_scan"]
    assert scan["status"] == "clean"


def test_example_values_track_raw_log_momentum(example_manager, example_h5):
    result = example_manager.evaluate(
        str(EXAMPLE_PATH), params=dict(EVAL_PARAMS), plot=False
    )
    values = result["factor_value"]

    store = CryptoQuantStore(example_h5)
    klines = store.read("klines_daily")
    close = klines.pivot(index="date", columns="symbol", values="close").sort_index()
    raw = np.log(close / close.shift(20))
    raw = raw.reindex(index=values.index, columns=values.columns)

    # mad_rank preprocessing is a per-date monotonic transform, so the saved
    # factor must rank instruments like the raw 20-day log momentum.
    for date in values.index:
        corr, _ = stats.spearmanr(values.loc[date], raw.loc[date])
        assert corr > 0.99, f"rank mismatch on {date.date()}"


def test_cli_runs_migrated_example(example_h5, tmp_path):
    proc = subprocess.run(
        [
            sys.executable, "factor_analyse/main.py", "example_momentum",
            "--h5-path", str(example_h5),
            "--start", EVAL_PARAMS["start"], "--end", EVAL_PARAMS["end"],
            "--no-plot", "--output-dir", str(tmp_path),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert "example_momentum" in proc.stdout
    assert "status=complete" in proc.stdout
    # All run artifacts stay under the requested output directory.
    assert (tmp_path / "factor_results").is_dir()
