"""HTML report renderer for portfolio backtest results."""
from __future__ import annotations

import base64
import io
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from portfolio.backtester.metrics import (
    annual_return, annual_volatility, sharpe, max_drawdown, calmar,
)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; }}
    h1 {{ color: #333; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .chart {{ margin: 20px 0; }}
    .better {{ color: #27ae60; font-weight: bold; }}
    .worse  {{ color: #e74c3c; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <h2>Performance Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Portfolio</th>{benchmark_headers}</tr>
    <tr><td>Annual Return</td><td>{annual_return:.2%}</td>{benchmark_annual_return}</tr>
    <tr><td>Annual Volatility</td><td>{annual_volatility:.2%}</td>{benchmark_annual_vol}</tr>
    <tr><td>Sharpe Ratio</td><td>{sharpe:.3f}</td>{benchmark_sharpe}</tr>
    <tr><td>Max Drawdown</td><td>{max_drawdown:.2%}</td>{benchmark_max_dd}</tr>
    <tr><td>Calmar Ratio</td><td>{calmar:.3f}</td>{benchmark_calmar}</tr>
  </table>
  <h2>Equity Curve</h2>
  <div class="chart">
    <img src="data:image/png;base64,{chart_b64}" style="max-width:100%;" />
  </div>
</body>
</html>"""


def _make_equity_chart(returns: pd.Series, benchmarks: dict | None = None) -> str:
    """Render equity curve with optional benchmark series, return base64-encoded PNG."""
    cum = (1 + returns).cumprod()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cum.index, cum.values, linewidth=1.5, label="Portfolio", color="#1f77b4")

    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    if benchmarks:
        for (name, bret), color in zip(benchmarks.items(), colors):
            bcum = (1 + bret.reindex(cum.index).fillna(0)).cumprod()
            ax.plot(bcum.index, bcum.values, linewidth=1.2, linestyle="--", label=name, color=color)
        ax.legend(loc="upper left", fontsize=9)

    ax.set_title("Equity Curve")
    ax.set_ylabel("Cumulative Return (base=1)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_html(
    returns: pd.Series,
    output_path: str | pathlib.Path,
    title: str = "Portfolio Backtest",
    benchmarks: dict | None = None,
) -> pathlib.Path:
    """Render HTML backtest report with metrics table and equity curve.

    Args:
        returns: Daily portfolio return series indexed by date.
        output_path: File path for the output HTML.
        title: Report title displayed in the HTML header.
        benchmarks: Optional dict of {name: return_series} to overlay on chart
                    and include in metrics table.

    Returns:
        pathlib.Path of the written HTML file.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chart_b64 = _make_equity_chart(returns, benchmarks)

    # Build benchmark table columns
    benchmark_headers = ""
    benchmark_annual_return = ""
    benchmark_annual_vol = ""
    benchmark_sharpe = ""
    benchmark_max_dd = ""
    benchmark_calmar_str = ""

    if benchmarks:
        port_ann = annual_return(returns)
        port_sharpe = sharpe(returns)
        for name, bret in benchmarks.items():
            benchmark_headers += f"<th>{name}</th>"
            b_ann = annual_return(bret)
            b_vol = annual_volatility(bret)
            b_sh = sharpe(bret)
            b_dd = max_drawdown(bret)
            b_cal = calmar(bret)
            cls_ann = "better" if port_ann >= b_ann else "worse"
            cls_sh  = "better" if port_sharpe >= b_sh else "worse"
            benchmark_annual_return += f'<td><span class="{cls_ann}">{b_ann:.2%}</span></td>'
            benchmark_annual_vol    += f"<td>{b_vol:.2%}</td>"
            benchmark_sharpe        += f'<td><span class="{cls_sh}">{b_sh:.3f}</span></td>'
            benchmark_max_dd        += f"<td>{b_dd:.2%}</td>"
            benchmark_calmar_str    += f"<td>{b_cal:.3f}</td>"

    html = _HTML_TEMPLATE.format(
        title=title,
        annual_return=annual_return(returns),
        annual_volatility=annual_volatility(returns),
        sharpe=sharpe(returns),
        max_drawdown=max_drawdown(returns),
        calmar=calmar(returns),
        chart_b64=chart_b64,
        benchmark_headers=benchmark_headers,
        benchmark_annual_return=benchmark_annual_return,
        benchmark_annual_vol=benchmark_annual_vol,
        benchmark_sharpe=benchmark_sharpe,
        benchmark_max_dd=benchmark_max_dd,
        benchmark_calmar=benchmark_calmar_str,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
