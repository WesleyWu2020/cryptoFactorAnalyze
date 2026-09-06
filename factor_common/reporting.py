"""Result-only HTML rendering for standard factor evaluation results.

``render_result(result, output_path, *, start=None, end=None)`` renders a
saved evaluation result to a self-contained HTML report. Rendering is a pure
display transform: it never reads market data, never recomputes factor
values, and never reruns portfolios or metric evaluation. Every table and
chart is derived from the tables already present in ``result``.

Expected result schema (Task 12's ``FactorManager`` is the producer)::

    {
        # Required
        "status": "complete" | "incomplete" | "insufficient_data",
        "metadata": {
            "factor_name": str,            # required; HTML-escaped for display
            "profile_id": str,             # optional; falls back to the
                                           # performance profile_id
            "factor_direction": 1 | -1,    # optional display hint
            "n_groups": int,               # optional; enables the
                                           # latest-groups table together
                                           # with "factor_value"
            "generated_at_utc": str,       # optional ISO timestamp
        },
        "factor_performance": dict,        # metrics.evaluate_metrics() result
        "factor_result": dict,             # backtest.run_backtest() result
        # Optional saved tables
        "factor_value": DataFrame,         # date x instrument factor matrix;
                                           # only used to derive the latest
                                           # group membership via the same
                                           # deterministic grouping rules
        "group_returns": DataFrame,        # date x "group_1".."group_N" mean
                                           # daily group returns saved at
                                           # evaluation time
        "benchmark": DataFrame | Series,   # optional CMC100 closes on a daily
                                           # UTC date index
    }

Extra keys (for example ``diagnostics`` or ``paths`` from the manager) are
ignored. ``start``/``end`` select an inclusive UTC display window; every time
series is sliced to it and NAV-like curves (including the optional CMC100
benchmark) are rebased to 1.0 at the display start. The benchmark is always
named "CMC100"; a missing benchmark is a report warning, never an evaluation
failure, and it is never renamed to the legacy equal-weight index.

Unavailable data is stated explicitly: charts whose source table is missing
or entirely null render a reason instead of a zero-filled curve, and missing
metrics render as an explicit ``null`` marker — never as NaN. All labels are
HTML-escaped and chart options are JSON-encoded with ``<``, ``>`` and ``&``
unicode-escaped for safe embedding in script contexts.
"""

from __future__ import annotations

import html
import math
import string
from pathlib import Path

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar, Line

from .grouping import assign_groups

_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "report.html"

_REQUIRED_KEYS = ("status", "metadata", "factor_performance", "factor_result")
SCENARIOS = ("gross", "trading_net", "all_costs")
SAMPLES = ("full", "in_sample", "out_of_sample")

_SCENARIO_ROWS = (
    ("Total return", "total_return"),
    ("Annual return", "annual_return"),
    ("Volatility (daily)", "volatility"),
    ("Annual volatility", "annual_volatility"),
    ("Sharpe", "sharpe"),
    ("Max drawdown", "max_drawdown"),
    ("Win rate", "win_rate"),
    ("Profit/loss ratio", "profit_loss_ratio"),
    ("Turnover (mean daily)", "turnover"),
    ("Periods", "n_periods"),
)

_IC_ROWS = (
    ("IC mean", "ic_mean"),
    ("RankIC mean", "rank_ic_mean"),
    ("ICIR", "icir"),
    ("Annualized ICIR", "annualized_icir"),
    ("RankICIR", "rank_icir"),
    ("Annualized RankICIR", "annualized_rank_icir"),
    ("t-stat", "t_stat"),
    ("p-value", "p_value"),
    ("IC dates", "n_dates"),
)

_SCENARIO_DIAGNOSTIC_ROWS = (
    ("Status", "status"),
    ("Halt reason", "halt_reason"),
    ("Halt date", "halt_date"),
    ("Halt detail", "halt_detail"),
    ("Filled orders", "filled_orders"),
    ("Failed orders", "failed_orders"),
    ("First order date", "first_order_date"),
    ("Last order date", "last_order_date"),
    ("Liquidation date", "liquidation_date"),
    ("Liquidation reached", "liquidation_reached"),
    ("Missing tail", "missing_tail"),
    ("Tail evaluated through", "tail_evaluated_through"),
    ("Funding total", "funding_total"),
)


# ---------------------------------------------------------------------------
# formatting / safety helpers
# ---------------------------------------------------------------------------


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def _fmt(value) -> str:
    """Format a metric cell; missing values are explicit nulls, never NaN."""
    if value is None:
        return '<span class="null">null</span>'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return '<span class="null">null</span>'
        return f"{value:.4f}"
    return _esc(value)


def _table(headers, rows) -> str:
    """Build an HTML table; header/label text is escaped, cells preformatted."""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _unavailable(title: str, reason: str) -> str:
    return (
        f'<div class="unavailable"><strong>{_esc(title)}</strong> '
        f"unavailable: {_esc(reason)}</div>"
    )


def _script_safe(option_json: str) -> str:
    """Make a JSON string safe to embed inside a <script> element."""
    safe = (
        option_json.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    if "NaN" in safe or "Infinity" in safe:
        raise RuntimeError("chart option contains non-finite numbers")
    return safe


def _clean(values) -> list:
    """Convert values to JSON-safe floats; non-finite entries become None."""
    out = []
    for value in values:
        if value is None:
            out.append(None)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(round(number, 6) if math.isfinite(number) else None)
    return out


# ---------------------------------------------------------------------------
# window resolution and slicing
# ---------------------------------------------------------------------------


def _as_day(value, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid date, got {value!r}")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def _resolve_window(result: dict, start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    for scenario in result["factor_result"]["scenarios"].values():
        ledger = scenario.get("ledger")
        if isinstance(ledger, pd.DataFrame) and not ledger.empty:
            dates.extend(pd.Timestamp(day) for day in ledger.index)
    if not dates:
        samples = result["factor_performance"].get("samples", {})
        for sample in samples.values():
            for entry in sample.get("ic", {}).get("daily", []):
                dates.append(pd.Timestamp(entry["date"]))
    if not dates and (start is None or end is None):
        raise ValueError(
            "result contains no dated series; pass explicit start and end"
        )
    start_day = _as_day(start, "start") if start is not None else min(dates)
    end_day = _as_day(end, "end") if end is not None else max(dates)
    if start_day > end_day:
        raise ValueError("start must be on or before end")
    return start_day, end_day


def _slice(frame, start: pd.Timestamp, end: pd.Timestamp):
    if frame is None or len(frame) == 0:
        return frame
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def _aligned(series_map: dict) -> tuple[list, dict]:
    """Align named date-indexed series on their union calendar."""
    all_dates = sorted({day for series in series_map.values() for day in series.index})
    x_labels = [day.date().isoformat() for day in all_dates]
    aligned = {
        name: _clean(series.reindex(all_dates).tolist())
        for name, series in series_map.items()
    }
    return x_labels, aligned


# ---------------------------------------------------------------------------
# chart builders
# ---------------------------------------------------------------------------


def _chart_block(chart_id: str, chart) -> str:
    option = _script_safe(chart.dump_options())
    return (
        f'<div id="{chart_id}" class="chart"></div>\n'
        f"<script>\n"
        f'var chart_{chart_id} = echarts.init(document.getElementById("{chart_id}"));\n'
        f"var option_{chart_id} = {option};\n"
        f"chart_{chart_id}.setOption(option_{chart_id});\n"
        f"</script>"
    )


def _line_chart(title: str, x_labels, series_map: dict, *, y_name: str, mark_zero=False):
    chart = Line(init_opts=opts.InitOpts(width="100%", height="420px"))
    chart.add_xaxis([str(label) for label in x_labels])
    for name, values in series_map.items():
        kwargs = {}
        if mark_zero:
            kwargs["markline_opts"] = opts.MarkLineOpts(
                data=[opts.MarkLineItem(y=0)]
            )
        chart.add_yaxis(name, values, is_symbol_show=False, **kwargs)
    chart.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="6%"),
        xaxis_opts=opts.AxisOpts(type_="category", name="UTC date"),
        yaxis_opts=opts.AxisOpts(type_="value", name=y_name),
        datazoom_opts=[
            opts.DataZoomOpts(range_start=0, range_end=100),
            opts.DataZoomOpts(type_="inside"),
        ],
    )
    return chart


def _bar_chart(title: str, x_labels, values, *, x_name: str, y_name: str):
    chart = Bar(init_opts=opts.InitOpts(width="100%", height="420px"))
    chart.add_xaxis([str(label) for label in x_labels])
    chart.add_yaxis(y_name, values)
    chart.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="6%"),
        xaxis_opts=opts.AxisOpts(type_="category", name=x_name),
        yaxis_opts=opts.AxisOpts(type_="value", name=y_name),
    )
    return chart


def _nav_section(result, start, end, warnings) -> str:
    series_map = {}
    notes = []
    for name in SCENARIOS:
        scenario = result["factor_result"]["scenarios"][name]
        ledger = _slice(scenario.get("ledger"), start, end)
        if ledger is None or ledger.empty:
            notes.append(f"{name}: no certified ledger rows in the display window")
            continue
        nav = ledger["equity"] / float(ledger["equity"].iloc[0])
        label = name
        if scenario.get("status") != "complete":
            label = f"{name} (incomplete — certified segment only)"
        series_map[label] = nav

    benchmark = _benchmark_series(result.get("benchmark"), start, end, warnings)
    if benchmark is not None:
        series_map["CMC100"] = benchmark

    if not series_map:
        return _unavailable(
            "Net asset value", "no scenario ledger has rows in the display window"
        )
    x_labels, aligned = _aligned(series_map)
    chart = _line_chart(
        "NAV (rebased to display start = 1.0)", x_labels, aligned, y_name="NAV"
    )
    note = ""
    if notes:
        note = f'<p class="note">{_esc("; ".join(notes))}.</p>'
    return _chart_block("chart_nav", chart) + note


def _benchmark_series(benchmark, start, end, warnings):
    if benchmark is None:
        warnings.append(
            "Optional CMC100 benchmark was not included in the result; "
            "the NAV chart omits it."
        )
        return None
    if isinstance(benchmark, pd.Series):
        series = benchmark
    elif isinstance(benchmark, pd.DataFrame):
        if benchmark.shape[1] == 0:
            warnings.append("CMC100 benchmark table has no columns; it is omitted.")
            return None
        column = "cmc100" if "cmc100" in benchmark.columns else benchmark.columns[0]
        series = benchmark[column]
    else:
        warnings.append(
            f"CMC100 benchmark has unsupported type {type(benchmark).__name__}; "
            "it is omitted."
        )
        return None
    series = series.dropna()
    series = series[(series.index >= start) & (series.index <= end)]
    if series.empty:
        warnings.append(
            "CMC100 benchmark has no data within the display window; it is omitted."
        )
        return None
    return series / float(series.iloc[0])


def _cumic_section(performance, start, end) -> str:
    series_map = {}
    for sample in SAMPLES:
        daily = performance["samples"][sample]["ic"].get("daily", [])
        points = [
            (pd.Timestamp(entry["date"]), entry.get("rank_ic"))
            for entry in daily
            if start <= pd.Timestamp(entry["date"]) <= end
            and entry.get("rank_ic") is not None
        ]
        if not points:
            continue
        cumulative = pd.Series(
            [float(value) for _, value in points],
            index=[day for day, _ in points],
        ).cumsum()
        series_map[sample] = cumulative
    if not series_map:
        return _unavailable(
            "Cumulative RankIC",
            "no daily RankIC observations in the display window",
        )
    x_labels, aligned = _aligned(series_map)
    chart = _line_chart("Cumulative RankIC", x_labels, aligned, y_name="cumulative RankIC")
    return _chart_block("chart_cumic", chart)


def _turnover_section(result, start, end) -> str:
    series_map = {}
    for name in SCENARIOS:
        ledger = _slice(result["factor_result"]["scenarios"][name].get("ledger"), start, end)
        if ledger is None or ledger.empty:
            continue
        pretrade = ledger["equity"] + ledger["fee"] + ledger.get("slippage", 0.0)
        turnover = ledger["trade_notional"].where(pretrade > 0.0) / pretrade.where(
            pretrade > 0.0
        )
        series_map[name] = turnover
    if not series_map:
        return _unavailable("Turnover", "no scenario ledger rows in the display window")
    x_labels, aligned = _aligned(series_map)
    chart = _line_chart(
        "Daily turnover (traded notional / pretrade equity)",
        x_labels,
        aligned,
        y_name="turnover",
    )
    return _chart_block("chart_turnover", chart)


def _positions_section(result, start, end) -> str:
    for name in SCENARIOS:
        scenario = result["factor_result"]["scenarios"][name]
        positions = _slice(scenario.get("positions"), start, end)
        if positions is None or positions.empty:
            continue
        series_map = {
            "long count": (positions > 0).sum(axis=1).astype(float),
            "short count": (positions < 0).sum(axis=1).astype(float),
        }
        x_labels, aligned = _aligned(series_map)
        chart = _line_chart(
            f"Open positions per day ({name} scenario)",
            x_labels,
            aligned,
            y_name="count",
        )
        return _chart_block("chart_positions", chart)
    return _unavailable(
        "Positions", "no saved positions table has rows in the display window"
    )


def _group_sections(result, start, end) -> tuple[str, str]:
    group_returns = result.get("group_returns")
    if group_returns is None:
        reason = "the saved result contains no group return table"
        return (
            _unavailable("Group cumulative return", reason),
            _unavailable("Demeaned group returns", reason),
        )
    if not isinstance(group_returns, pd.DataFrame):
        raise TypeError("result['group_returns'] must be a DataFrame when present")
    sliced = _slice(group_returns, start, end)
    if sliced is None or sliced.empty:
        reason = "no group return rows fall inside the display window"
        return (
            _unavailable("Group cumulative return", reason),
            _unavailable("Demeaned group returns", reason),
        )
    sliced = sliced.dropna(how="all")
    # Simple-sum (non-compounded) accumulation: each day's cross-sectional
    # mean label return is added, not multiplied, so the curve shows the
    # arithmetic PnL of a fixed-notional daily rebalanced group sleeve.
    cumulative = sliced.cumsum()
    x_labels, aligned_nav = _aligned({str(c): cumulative[c] for c in cumulative.columns})
    nav_chart = _line_chart(
        "Group cumulative return (simple sum, from saved group returns)",
        x_labels,
        aligned_nav,
        y_name="cumulative return",
    )
    demeaned = sliced.sub(sliced.mean(axis=1), axis=0)
    demeaned_map = {str(c): demeaned[c] for c in demeaned.columns}
    if len(demeaned.columns) >= 2:
        demeaned_map["top minus bottom"] = (
            demeaned.iloc[:, -1] - demeaned.iloc[:, 0]
        )
    x_labels, aligned_demeaned = _aligned(demeaned_map)
    demeaned_chart = _line_chart(
        "Demeaned group daily returns",
        x_labels,
        aligned_demeaned,
        y_name="demeaned return",
        mark_zero=True,
    )
    return (
        _chart_block("chart_group_nav", nav_chart),
        _chart_block("chart_group_demeaned", demeaned_chart),
    )


def _decay_section(performance) -> str:
    decay = performance["samples"]["full"]["ic_decay"]
    if not decay or all(entry.get("rank_ic") is None for entry in decay):
        return _unavailable(
            "RankIC decay", "every horizon is null for the full sample"
        )
    chart = _bar_chart(
        "RankIC decay (full sample)",
        [entry["horizon"] for entry in decay],
        _clean([entry.get("rank_ic") for entry in decay]),
        x_name="horizon (signal days)",
        y_name="RankIC",
    )
    return _chart_block("chart_decay", chart)


def _autocorr_section(performance) -> str:
    autocorr = performance["samples"]["full"]["rank_ic_autocorr"]
    if not autocorr or all(entry.get("autocorr") is None for entry in autocorr):
        return _unavailable(
            "RankIC autocorrelation", "every lag is null for the full sample"
        )
    chart = _line_chart(
        "RankIC autocorrelation (full sample)",
        [str(entry["lag"]) for entry in autocorr],
        {
            "autocorr": _clean([entry.get("autocorr") for entry in autocorr]),
        },
        y_name="autocorrelation",
        mark_zero=True,
    )
    return _chart_block("chart_autocorr", chart)


# ---------------------------------------------------------------------------
# table sections
# ---------------------------------------------------------------------------


def _summary_section(result, start, end) -> str:
    metadata = result["metadata"]
    performance = result["factor_performance"]
    split = performance.get("split", {})
    purged = split.get("purged_signal_dates", [])
    profile_id = metadata.get("profile_id") or performance.get("profile_id")
    rows = [
        ["Factor", _esc(metadata.get("factor_name", "unknown"))],
        ["Profile", _fmt(profile_id)],
        ["Factor direction", _fmt(metadata.get("factor_direction"))],
        ["Evaluation status", _esc(result.get("status", "unknown"))],
        ["Split date", _fmt(split.get("split_date"))],
        ["Hold days", _fmt(split.get("hold_days"))],
        ["In-sample signal dates", _fmt(split.get("in_sample_signal_dates"))],
        ["Out-of-sample signal dates", _fmt(split.get("out_of_sample_signal_dates"))],
        [
            "Purged signal dates",
            _esc(f"{len(purged)} ({', '.join(str(day) for day in purged)})"),
        ],
        [
            "Display window (UTC)",
            _esc(f"{start.date().isoformat()} to {end.date().isoformat()}"),
        ],
        ["Generated at (UTC)", _fmt(metadata.get("generated_at_utc"))],
    ]
    return _table(["Item", "Value"], rows)


def _status_banner(result) -> str:
    accounting = result["factor_result"]
    incomplete_parts = []
    for name in SCENARIOS:
        scenario = accounting["scenarios"][name]
        if scenario.get("status") == "complete":
            continue
        diagnostics = scenario.get("diagnostics", {})
        reason = diagnostics.get("halt_reason") or "unknown"
        date = diagnostics.get("halt_date") or "unknown date"
        incomplete_parts.append(f"{name}: {reason} on {date}")
    if result.get("status") == "complete" and not incomplete_parts:
        return '<div class="banner complete">Status: complete — all cost scenarios certified.</div>'
    detail = "; ".join(incomplete_parts) or "see diagnostics"
    return (
        f'<div class="banner incomplete">Status: {_esc(result.get("status", "unknown"))}'
        f" — incomplete segments: {_esc(detail)}. Metrics for incomplete scenarios are"
        " null; only separately labeled certified segments are shown.</div>"
    )


def _scenario_metrics_section(performance) -> str:
    blocks = []
    for name in SCENARIOS:
        scenario = performance["scenarios"][name]
        blocks.append(f"<h3>{_esc(name)}</h3>")
        if scenario.get("status") != "complete":
            blocks.append(
                "<p>This scenario's accounting is incomplete, so full/in/out-of-sample"
                " metrics are unavailable (null). The known segment below summarizes"
                " only the certified ledger prefix and is not comparable to"
                " full-window metrics.</p>"
            )
            rows = [
                [_esc(label), '<span class="null">null</span>',
                 '<span class="null">null</span>', '<span class="null">null</span>']
                for label, _ in _SCENARIO_ROWS
            ]
            blocks.append(_table(["Metric", "Full", "In-sample", "Out-of-sample"], rows))
            known = scenario.get("known_segment")
            if known:
                through = known.get("through", "unknown")
                blocks.append(f"<h3>Known segment (through {_esc(through)})</h3>")
                rows = [
                    [_esc(label), _fmt(known.get(key))] for label, key in _SCENARIO_ROWS
                ]
                blocks.append(_table(["Metric", "Known segment"], rows))
            continue
        rows = []
        for label, key in _SCENARIO_ROWS:
            rows.append(
                [_esc(label)]
                + [_fmt((scenario.get(sample) or {}).get(key)) for sample in SAMPLES]
            )
        blocks.append(_table(["Metric", "Full", "In-sample", "Out-of-sample"], rows))
        empty = [
            sample
            for sample in SAMPLES
            if (scenario.get(sample) or {}).get("n_periods") == 0
        ]
        if empty:
            blocks.append(
                f'<p class="note">{_esc(", ".join(empty))}: no observations.</p>'
            )
    return "".join(blocks)


def _ic_metrics_section(performance) -> str:
    samples = performance["samples"]
    rows = []
    for label, key in _IC_ROWS:
        rows.append(
            [_esc(label)]
            + [_fmt(samples[sample]["ic"].get(key)) for sample in SAMPLES]
        )
    rows.append(
        [_esc("Mean label coverage")]
        + [
            _fmt(samples[sample]["coverage"].get("mean_label_coverage"))
            for sample in SAMPLES
        ]
    )
    rows.append(
        [_esc("Coverage dates")]
        + [_fmt(samples[sample]["coverage"].get("n_dates")) for sample in SAMPLES]
    )
    rows.append(
        [_esc("RankIC half-life")]
        + [_fmt(samples[sample].get("rank_ic_half_life")) for sample in SAMPLES]
    )
    table = _table(["Metric", "Full", "In-sample", "Out-of-sample"], rows)
    empty = [
        sample for sample in SAMPLES if samples[sample]["ic"].get("n_dates") == 0
    ]
    if empty:
        table += f'<p class="note">{_esc(", ".join(empty))}: no observations.</p>'
    return table


def _latest_groups_section(result) -> str:
    factor_value = result.get("factor_value")
    n_groups = result["metadata"].get("n_groups")
    if factor_value is None:
        return _unavailable(
            "Latest groups", "the saved result contains no factor value matrix"
        )
    if not isinstance(factor_value, pd.DataFrame):
        raise TypeError("result['factor_value'] must be a DataFrame when present")
    if not isinstance(n_groups, int) or isinstance(n_groups, bool) or n_groups < 2:
        return _unavailable(
            "Latest groups", "metadata does not record a valid n_groups"
        )
    groups, _ = assign_groups(factor_value, n_groups)
    usable = groups.dropna(how="all")
    if usable.empty:
        return _unavailable(
            "Latest groups",
            "no date has enough valid names for the configured grouping",
        )
    latest = usable.index[-1]
    rows = []
    for group_id in range(1, n_groups + 1):
        members = groups.loc[latest]
        members = members[members == group_id].index
        entries = []
        for instrument in sorted(members, key=str):
            value = factor_value.loc[latest, instrument]
            entries.append(f"{instrument} ({float(value):.4f})")
        rows.append([_esc(f"group_{group_id}"), _esc(", ".join(entries))])
    heading = (
        f'<p class="note">Group membership on {_esc(latest.date().isoformat())}'
        " (UTC), derived from the saved factor value matrix.</p>"
    )
    return heading + _table(["Group", "Instruments (factor value)"], rows)


def _diagnostics_section(result) -> str:
    accounting = result["factor_result"]
    blocks = ["<h3>Backtest calendar</h3>"]
    diagnostics = accounting.get("diagnostics", {})
    rows = [
        [_esc("Anchor date"), _fmt(diagnostics.get("anchor_date"))],
        [_esc("Rebalance days"), _fmt(diagnostics.get("rebalance_days"))],
        [_esc("Signal delay days"), _fmt(diagnostics.get("signal_delay_days"))],
        [_esc("Signal start"), _fmt(diagnostics.get("signal_start"))],
        [_esc("Signal end"), _fmt(diagnostics.get("signal_end"))],
        [
            _esc("Scheduled execution dates"),
            _fmt(len(diagnostics.get("scheduled_dates", []))),
        ],
        [
            _esc("Usable execution dates"),
            _fmt(len(diagnostics.get("usable_execution_dates", []))),
        ],
        [_esc("Executed dates"), _fmt(len(diagnostics.get("executed_dates", [])))],
        [_esc("No usable signals"), _fmt(diagnostics.get("no_usable_signals"))],
    ]
    blocks.append(_table(["Item", "Value"], rows))

    for name in SCENARIOS:
        scenario = accounting["scenarios"][name]
        scenario_diag = scenario.get("diagnostics", {})
        blocks.append(f"<h3>{_esc(name)}</h3>")
        rows = [
            [_esc(label), _fmt(scenario_diag.get(key))]
            for label, key in _SCENARIO_DIAGNOSTIC_ROWS
        ]
        ledger = scenario.get("ledger")
        if isinstance(ledger, pd.DataFrame) and not ledger.empty:
            total_fees = float(ledger["fee"].sum())
            total_slippage = (
                float(ledger["slippage"].sum()) if "slippage" in ledger.columns else 0.0
            )
        else:
            total_fees = None
            total_slippage = None
        rows.append([_esc("Total fees (from ledger)"), _fmt(total_fees)])
        rows.append([_esc("Total slippage (from ledger)"), _fmt(total_slippage)])
        rows.append(
            [_esc("Blocked orders"), _fmt(len(scenario_diag.get("blocked_orders", [])))]
        )
        rows.append(
            [
                _esc("Retrospective nonexecution events"),
                _fmt(len(scenario_diag.get("retrospective_nonexecution", []))),
            ]
        )
        blocks.append(_table(["Item", "Value"], rows))

    blocks.append("<h3>Funding coverage (all_costs)</h3>")
    coverage = accounting["scenarios"]["all_costs"].get("funding_coverage")
    if isinstance(coverage, pd.DataFrame) and not coverage.empty:
        counts = coverage["status"].value_counts()
        rows = [[_esc(str(status)), _fmt(int(count))] for status, count in counts.items()]
        blocks.append(_table(["Coverage status", "Rows"], rows))
    else:
        blocks.append('<p class="note">No funding coverage rows saved.</p>')
    return "".join(blocks)


# ---------------------------------------------------------------------------
# schema validation and entry point
# ---------------------------------------------------------------------------


def _validate_result(result) -> None:
    if not isinstance(result, dict):
        raise TypeError(
            f"result must be a dict assembled from saved evaluation tables, "
            f"got {type(result).__name__}"
        )
    missing = [key for key in _REQUIRED_KEYS if key not in result]
    if missing:
        raise ValueError(f"result is missing required keys: {missing}")
    metadata = result["metadata"]
    if not isinstance(metadata, dict) or not isinstance(metadata.get("factor_name"), str):
        raise ValueError("result['metadata'] must be a dict with a string 'factor_name'")
    performance = result["factor_performance"]
    if not isinstance(performance, dict) or not all(
        key in performance for key in ("samples", "scenarios")
    ):
        raise ValueError(
            "result['factor_performance'] must be an evaluate_metrics result "
            "with 'samples' and 'scenarios'"
        )
    accounting = result["factor_result"]
    if not isinstance(accounting, dict) or not isinstance(accounting.get("scenarios"), dict):
        raise ValueError(
            "result['factor_result'] must be a run_backtest result with 'scenarios'"
        )
    missing_scenarios = [name for name in SCENARIOS if name not in accounting["scenarios"]]
    if missing_scenarios:
        raise ValueError(f"factor_result missing scenarios: {missing_scenarios}")


def render_result(result, output_path, *, start=None, end=None) -> dict:
    """Render a saved evaluation result to an HTML report.

    Parameters
    ----------
    result:
        Dict assembled from saved result tables; see the module docstring for
        the expected schema.
    output_path:
        Destination HTML file path (parent directories are created).
    start, end:
        Optional inclusive UTC display window; all time series are sliced to
        it and NAV-like curves are rebased to 1.0 at the display start.

    Returns a dict with ``output_path``, the resolved ``start``/``end``
    (ISO dates) and any non-fatal ``warnings`` (for example a missing
    optional CMC100 benchmark).
    """
    _validate_result(result)
    start_day, end_day = _resolve_window(result, start, end)

    metadata = result["metadata"]
    performance = result["factor_performance"]
    warnings: list[str] = []

    nav_section = _nav_section(result, start_day, end_day, warnings)
    group_nav_section, group_demeaned_section = _group_sections(
        result, start_day, end_day
    )

    if warnings:
        items = "".join(f"<li>{_esc(warning)}</li>" for warning in warnings)
        warnings_section = f'<ul class="warnings">{items}</ul>'
    else:
        warnings_section = ""

    factor_name = metadata.get("factor_name", "unknown")
    page_title = _esc(f"Factor report: {factor_name} (UTC)")

    template = string.Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    html_text = template.substitute(
        page_title=page_title,
        status_banner=_status_banner(result),
        summary_section=_summary_section(result, start_day, end_day),
        warnings_section=warnings_section,
        scenario_metrics_section=_scenario_metrics_section(performance),
        ic_metrics_section=_ic_metrics_section(performance),
        nav_section=nav_section,
        group_nav_section=group_nav_section,
        group_demeaned_section=group_demeaned_section,
        cumic_section=_cumic_section(performance, start_day, end_day),
        turnover_section=_turnover_section(result, start_day, end_day),
        positions_section=_positions_section(result, start_day, end_day),
        decay_section=_decay_section(performance),
        autocorr_section=_autocorr_section(performance),
        groups_section=_latest_groups_section(result),
        diagnostics_section=_diagnostics_section(result),
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return {
        "output_path": str(path),
        "start": start_day.date().isoformat(),
        "end": end_day.date().isoformat(),
        "warnings": warnings,
    }


__all__ = ["render_result"]
