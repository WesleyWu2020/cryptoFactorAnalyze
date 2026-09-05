"""Tests for result-only HTML rendering (factor_common.reporting).

The renderer consumes saved result tables only: it must never call data
providers, factor computation, backtests, or metric evaluation. All tests
build synthetic results matching the documented schema in
``factor_common/reporting.py``.
"""

from __future__ import annotations

import copy
import json
import pickle
import re

import pandas as pd
import pytest

from factor_common.reporting import render_result

DATES = pd.date_range("2024-01-01", periods=30, freq="D", name="date")
INSTRUMENTS = ["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT", "FUSDT"]
RETURN_KEYS = (
    "total_return",
    "annual_return",
    "volatility",
    "annual_volatility",
    "sharpe",
    "max_drawdown",
    "win_rate",
    "profit_loss_ratio",
)


def _ledger(dates, equity, *, fee_rate=0.0003, funding_cash=0.0):
    equity = pd.Series(equity, index=dates, dtype="float64")
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] - 1.0  # initial equity 1.0
    trade_notional = pd.Series(
        [0.2 if i % 5 == 0 else 0.0 for i in range(len(dates))], index=dates
    )
    return pd.DataFrame(
        {
            "equity": equity,
            "return": returns,
            "price_pnl": 0.001,
            "funding_cashflow": funding_cash,
            "fee": trade_notional * fee_rate,
            "trade_notional": trade_notional,
        },
        index=dates,
    )


def _orders(dates):
    rows = []
    for day in dates[::5]:
        rows.append(
            {
                "date": day,
                "instrument": "AUSDT",
                "side": "buy",
                "target_weight": 0.5,
                "target_quantity": 0.01,
                "order_quantity": 0.01,
                "price": 100.0,
                "notional": 1.0,
                "fee": 0.0003,
                "status": "filled",
                "reason": None,
            }
        )
    return pd.DataFrame(rows)


def _positions(dates):
    data = {
        "AUSDT": [0.01 if i % 2 == 0 else 0.0 for i in range(len(dates))],
        "BUSDT": [-0.02 if i % 2 == 0 else 0.0 for i in range(len(dates))],
        "CUSDT": [0.03 for _ in dates],
        "DUSDT": [0.0 for _ in dates],
        "EUSDT": [0.0 for _ in dates],
        "FUSDT": [0.0 for _ in dates],
    }
    frame = pd.DataFrame(data, index=dates)
    frame.index.name = "date"
    return frame


def _scenario_diagnostics(**overrides):
    diagnostics = {
        "status": "complete",
        "halt_reason": None,
        "halt_date": None,
        "halt_detail": None,
        "filled_orders": 6,
        "failed_orders": 0,
        "first_order_date": "2024-01-01",
        "last_order_date": "2024-01-26",
        "liquidation_date": "2024-01-30",
        "liquidation_reached": True,
        "missing_tail": False,
        "tail_evaluated_through": "2024-01-30",
        "final_quantities": {},
        "blocked_orders": [],
        "retrospective_nonexecution": [],
        "funding_total": 0.0,
    }
    diagnostics.update(overrides)
    return diagnostics


def _scenario(dates, equity, *, status="complete", diagnostics=None, funding=None, coverage=None):
    return {
        "status": status,
        "ledger": _ledger(dates, equity, funding_cash=0.0001 if funding is not None else 0.0),
        "orders": _orders(dates),
        "positions": _positions(dates),
        "valuation_prices": _positions(dates),
        "funding": funding
        if funding is not None
        else pd.DataFrame(columns=["funding_time", "instrument", "quantity", "funding_rate",
                                   "mark_price", "settlement_price", "price_approximated",
                                   "resolved", "unresolved_reason", "cashflow"]),
        "funding_coverage": coverage
        if coverage is not None
        else pd.DataFrame(columns=["date", "instrument", "observed_status", "status",
                                   "partial_day", "accepted"]),
        "diagnostics": diagnostics or _scenario_diagnostics(),
    }


def _summary(n_periods, *, filled=True):
    summary = {key: None for key in RETURN_KEYS}
    summary["n_periods"] = n_periods
    summary["turnover"] = None
    if filled and n_periods:
        summary.update(
            {
                "total_return": 0.12,
                "annual_return": 1.5,
                "volatility": 0.01,
                "annual_volatility": 0.19,
                "sharpe": 1.2,
                "max_drawdown": 0.05,
                "win_rate": 0.55,
                "profit_loss_ratio": 1.3,
                "turnover": 0.04,
            }
        )
    return summary


def _ic_block(dates, *, empty=False):
    block = {
        "ic_mean": None,
        "rank_ic_mean": None,
        "icir": None,
        "annualized_icir": None,
        "rank_icir": None,
        "annualized_rank_icir": None,
        "t_stat": None,
        "p_value": None,
        "n_dates": 0,
        "daily": [],
    }
    if empty:
        return block
    block.update(
        {
            "ic_mean": 0.04,
            "rank_ic_mean": 0.05,
            "icir": 0.5,
            "annualized_icir": 9.5,
            "rank_icir": 0.6,
            "annualized_rank_icir": 11.4,
            "t_stat": 2.5,
            "p_value": 0.02,
            "n_dates": len(dates),
            "daily": [
                {"date": day.date().isoformat(), "ic": 0.04, "rank_ic": 0.05}
                for day in dates
            ],
        }
    )
    return block


def _sample(dates, *, empty=False):
    if empty:
        return {
            "ic": _ic_block(dates, empty=True),
            "ic_decay": [{"horizon": h, "rank_ic": None} for h in range(1, 11)],
            "rank_ic_autocorr": [{"lag": lag, "autocorr": None} for lag in range(1, 21)],
            "rank_ic_half_life": None,
            "coverage": {"mean_label_coverage": None, "n_dates": 0},
        }
    return {
        "ic": _ic_block(dates),
        "ic_decay": [
            {"horizon": h, "rank_ic": 0.05 / h if h <= 5 else None} for h in range(1, 11)
        ],
        "rank_ic_autocorr": [
            {"lag": lag, "autocorr": 0.8 - 0.04 * lag if lag <= 15 else None}
            for lag in range(1, 21)
        ],
        "rank_ic_half_life": 6,
        "coverage": {"mean_label_coverage": 0.9, "n_dates": len(dates)},
    }


def _performance(*, empty_out=False, incomplete_all_costs=False):
    split = {
        "split_date": "2024-01-20",
        "out_of_sample_days": 180,
        "hold_days": 1,
        "in_sample_signal_dates": 19,
        "out_of_sample_signal_dates": 10,
        "purged_signal_dates": ["2024-01-20"],
    }
    in_dates = DATES[:20]
    out_dates = DATES[20:]
    samples = {
        "full": _sample(DATES),
        "in_sample": _sample(in_dates),
        "out_of_sample": _sample(out_dates, empty=empty_out),
    }
    scenarios = {
        "gross": {
            "status": "complete",
            "full": _summary(30),
            "in_sample": _summary(20),
            "out_of_sample": _summary(10, filled=not empty_out),
        },
        "trading_net": {
            "status": "complete",
            "full": _summary(30),
            "in_sample": _summary(20),
            "out_of_sample": _summary(10, filled=not empty_out),
        },
    }
    if incomplete_all_costs:
        scenarios["all_costs"] = {
            "status": "incomplete",
            "full": None,
            "in_sample": None,
            "out_of_sample": None,
            "known_segment": {**_summary(10), "through": "2024-01-10"},
        }
    else:
        scenarios["all_costs"] = {
            "status": "complete",
            "full": _summary(30),
            "in_sample": _summary(20),
            "out_of_sample": _summary(10, filled=not empty_out),
        }
    return {"profile_id": "perp_1d", "split": split, "samples": samples, "scenarios": scenarios}


def _factor_result(*, incomplete_all_costs=False):
    equity = [1.0 + 0.004 * i for i in range(len(DATES))]
    gross = _scenario(DATES, equity)
    trading_net = _scenario(DATES, [e * 0.999 for e in equity])
    if incomplete_all_costs:
        known_dates = DATES[:10]
        funding = pd.DataFrame(
            [
                {
                    "funding_time": "2024-01-05 08:00:00",
                    "instrument": "AUSDT",
                    "quantity": 0.01,
                    "funding_rate": 0.001,
                    "mark_price": None,
                    "settlement_price": None,
                    "price_approximated": False,
                    "resolved": False,
                    "unresolved_reason": "invalid_mark_price",
                    "cashflow": None,
                }
            ]
        )
        coverage = pd.DataFrame(
            [
                {
                    "date": "2024-01-05",
                    "instrument": "AUSDT",
                    "observed_status": "unknown",
                    "status": "unknown",
                    "partial_day": False,
                    "accepted": False,
                }
            ]
        )
        all_costs = _scenario(
            known_dates,
            equity[:10],
            status="incomplete",
            diagnostics=_scenario_diagnostics(
                status="incomplete",
                halt_reason="unresolved_funding",
                halt_date="2024-01-05",
                halt_detail="invalid_mark_price",
                liquidation_reached=False,
                funding_total=0.0001,
            ),
            funding=funding,
            coverage=coverage,
        )
        status = "incomplete"
    else:
        coverage = pd.DataFrame(
            [
                {
                    "date": "2024-01-03",
                    "instrument": "AUSDT",
                    "observed_status": "complete",
                    "status": "complete",
                    "partial_day": False,
                    "accepted": True,
                }
            ]
        )
        all_costs = _scenario(DATES, [e * 0.998 for e in equity], coverage=coverage)
        status = "complete"
    return {
        "status": status,
        "portfolio": "long_short",
        "profile_id": "perp_1d",
        "scenarios": {"gross": gross, "trading_net": trading_net, "all_costs": all_costs},
        "diagnostics": {
            "anchor_date": "2024-01-01",
            "rebalance_days": 1,
            "signal_delay_days": 1,
            "signal_start": "2024-01-01",
            "signal_end": "2024-01-30",
            "scheduled_dates": [d.date().isoformat() for d in DATES],
            "usable_execution_dates": [d.date().isoformat() for d in DATES[:-1]],
            "executed_dates": [d.date().isoformat() for d in DATES[:-1]],
            "no_usable_signals": False,
        },
    }


def _factor_value():
    data = {
        instrument: [0.01 * i + 0.1 * j for i in range(len(DATES))]
        for j, instrument in enumerate(INSTRUMENTS)
    }
    return pd.DataFrame(data, index=DATES)


def _group_returns():
    data = {
        "group_1": [-0.002 + 0.0001 * i for i in range(len(DATES))],
        "group_2": [0.0005 for _ in DATES],
        "group_3": [0.002 - 0.0001 * i for i in range(len(DATES))],
    }
    frame = pd.DataFrame(data, index=DATES)
    frame.index.name = "date"
    return frame


def _benchmark():
    return pd.DataFrame({"cmc100": [100.0 + i for i in range(len(DATES))]}, index=DATES)


def build_result(
    *,
    factor_name="example_momentum",
    incomplete_all_costs=False,
    empty_out=False,
    with_benchmark=True,
    with_group_returns=True,
    with_factor_value=True,
):
    result = {
        "status": "incomplete" if incomplete_all_costs else "complete",
        "metadata": {
            "factor_name": factor_name,
            "profile_id": "perp_1d",
            "factor_direction": 1,
            "n_groups": 3,
            "generated_at_utc": "2024-02-01T00:00:00+00:00",
        },
        "factor_performance": _performance(
            empty_out=empty_out, incomplete_all_costs=incomplete_all_costs
        ),
        "factor_result": _factor_result(incomplete_all_costs=incomplete_all_costs),
    }
    if with_factor_value:
        result["factor_value"] = _factor_value()
    if with_group_returns:
        result["group_returns"] = _group_returns()
    if with_benchmark:
        result["benchmark"] = _benchmark()
    return result


def _option(html_text: str, chart_id: str) -> dict:
    match = re.search(
        r"var option_" + re.escape(chart_id) + r" = (\{.*?\});\n", html_text, re.DOTALL
    )
    assert match, f"chart option {chart_id} not found in HTML"
    return json.loads(match.group(1))


def _series_y(series: dict) -> list:
    """pyecharts emits [x, y] pairs on category axes; unwrap to y values."""
    return [
        point[1] if isinstance(point, list) else point for point in series["data"]
    ]


@pytest.fixture
def complete_result():
    return build_result()


@pytest.fixture
def incomplete_result():
    return build_result(incomplete_all_costs=True)


def test_render_rejects_non_dict_result(tmp_path):
    with pytest.raises(TypeError):
        render_result("not a result", tmp_path / "report.html")


def test_render_rejects_missing_keys(tmp_path):
    result = build_result()
    del result["factor_result"]
    with pytest.raises(ValueError, match="factor_result"):
        render_result(result, tmp_path / "report.html")


def test_render_rejects_inverted_window(tmp_path, complete_result):
    with pytest.raises(ValueError, match="start"):
        render_result(
            complete_result, tmp_path / "report.html", start="2024-01-20", end="2024-01-10"
        )


def test_render_complete_report(tmp_path, complete_result):
    out = tmp_path / "report.html"
    info = render_result(complete_result, out)
    assert out.exists()
    html_text = out.read_text(encoding="utf-8")
    assert info["output_path"] == str(out)
    assert info["warnings"] == []
    assert info["start"] == "2024-01-01"
    assert info["end"] == "2024-01-30"
    assert "example_momentum" in html_text
    assert "perp_1d" in html_text
    assert "UTC" in html_text
    assert "2024-01-20" in html_text  # split date
    for heading in (
        "Scenario performance",
        "IC summary",
        "Net asset value",
        "Cumulative RankIC",
        "Turnover",
        "Positions",
        "RankIC decay",
        "RankIC autocorrelation",
        "Latest groups",
        "Diagnostics",
    ):
        assert heading in html_text, heading
    assert "NaN" not in html_text
    assert "Infinity" not in html_text
    # Cost scenarios are explicitly named.
    for label in ("gross", "trading_net", "all_costs"):
        assert label in html_text


def test_html_escaping_of_labels(tmp_path):
    evil = 'bad"><script>alert("x")</script>&amp;'
    result = build_result(factor_name=evil)
    out = tmp_path / "report.html"
    render_result(result, out)
    html_text = out.read_text(encoding="utf-8")
    assert 'bad"><script>alert' not in html_text
    assert "bad&quot;&gt;" in html_text
    assert "<script>alert" not in html_text


def test_incomplete_state_is_visible(tmp_path, incomplete_result):
    out = tmp_path / "report.html"
    render_result(incomplete_result, out)
    html_text = out.read_text(encoding="utf-8")
    assert "incomplete" in html_text
    assert "unresolved_funding" in html_text
    assert "Known segment" in html_text
    # Incomplete scenario metrics render as explicit nulls, not numbers/zeros.
    assert '<span class="null">null</span>' in html_text
    # The certified NAV prefix is plotted but clearly labeled as incomplete.
    nav = _option(html_text, "chart_nav")
    names = [series["name"] for series in nav["series"]]
    assert any("all_costs" in name and "incomplete" in name for name in names)
    assert any(name == "gross" for name in names)


def test_empty_sample_renders_unavailable(tmp_path):
    result = build_result(empty_out=True)
    out = tmp_path / "report.html"
    render_result(result, out)
    html_text = out.read_text(encoding="utf-8")
    assert "NaN" not in html_text
    assert '<span class="null">null</span>' in html_text
    assert "no observations" in html_text


def test_no_provider_or_backtest_calls_during_render(tmp_path, complete_result, monkeypatch):
    import factor_common.backtest as backtest_mod
    import factor_common.data_provider as provider_mod
    import factor_common.metrics as metrics_mod
    import factor_common.value_engine as engine_mod

    def boom(*args, **kwargs):
        raise AssertionError("recomputation during rendering")

    monkeypatch.setattr(provider_mod.DataProvider, "__init__", boom)
    monkeypatch.setattr(backtest_mod, "run_backtest", boom)
    monkeypatch.setattr(metrics_mod, "evaluate_metrics", boom)
    monkeypatch.setattr(engine_mod, "compute_factor", boom)
    out = tmp_path / "report.html"
    render_result(complete_result, out)
    assert out.exists()


def test_render_does_not_mutate_result(tmp_path, complete_result):
    before = pickle.dumps(complete_result)
    render_result(complete_result, tmp_path / "report.html")
    assert pickle.dumps(complete_result) == before


def test_benchmark_is_named_cmc100_and_rebased(tmp_path, complete_result):
    out = tmp_path / "report.html"
    render_result(complete_result, out)
    html_text = out.read_text(encoding="utf-8")
    assert "CMC100" in html_text
    nav = _option(html_text, "chart_nav")
    benchmark = [s for s in nav["series"] if s["name"] == "CMC100"]
    assert len(benchmark) == 1
    data = _series_y(benchmark[0])
    assert data[0] == pytest.approx(1.0)
    assert data[1] == pytest.approx(1.01)


def test_missing_benchmark_is_warning_not_failure(tmp_path):
    result = build_result(with_benchmark=False)
    out = tmp_path / "report.html"
    info = render_result(result, out)
    assert out.exists()
    assert any("CMC100" in warning for warning in info["warnings"])
    html_text = out.read_text(encoding="utf-8")
    assert "CMC100" in html_text
    assert "unavailable" in html_text
    nav = _option(html_text, "chart_nav")
    assert all(series["name"] != "CMC100" for series in nav["series"])


def test_display_window_slices_series(tmp_path, complete_result):
    out = tmp_path / "report.html"
    info = render_result(
        complete_result, out, start="2024-01-10", end="2024-01-25"
    )
    assert info["start"] == "2024-01-10"
    assert info["end"] == "2024-01-25"
    html_text = out.read_text(encoding="utf-8")
    nav = _option(html_text, "chart_nav")
    x_data = nav["xAxis"][0]["data"]
    assert x_data[0] == "2024-01-10"
    assert x_data[-1] == "2024-01-25"
    # Every NAV series is rebased to the display start.
    for series in nav["series"]:
        assert _series_y(series)[0] == pytest.approx(1.0)


def test_missing_group_returns_explains_unavailability(tmp_path):
    result = build_result(with_group_returns=False)
    out = tmp_path / "report.html"
    render_result(result, out)
    html_text = out.read_text(encoding="utf-8")
    assert "unavailable" in html_text
    assert "group return" in html_text.lower()
    assert "chart_group_nav" not in html_text


def test_missing_factor_value_explains_latest_groups(tmp_path):
    result = build_result(with_factor_value=False)
    out = tmp_path / "report.html"
    render_result(result, out)
    html_text = out.read_text(encoding="utf-8")
    assert "Latest groups" in html_text
    assert "unavailable" in html_text


def test_latest_groups_table_lists_instruments(tmp_path, complete_result):
    out = tmp_path / "report.html"
    render_result(complete_result, out)
    html_text = out.read_text(encoding="utf-8")
    assert "AUSDT" in html_text  # lowest factor value -> group_1 on the latest date
    assert "FUSDT" in html_text  # highest factor value -> group_3


def test_deterministic_output(tmp_path, complete_result):
    result = copy.deepcopy(complete_result)
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"
    render_result(result, first)
    render_result(result, second)
    assert first.read_bytes() == second.read_bytes()
