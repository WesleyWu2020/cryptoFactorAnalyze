"""Immutable, auditable artifacts for fixed portfolio research runs."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import string
import uuid
from typing import Any, Mapping
from contextlib import contextmanager

import numpy as np
import pandas as pd

from factor_common.reporting import (
    _aligned,
    _chart_block,
    _esc,
    _fmt,
    _line_chart,
    _table,
    _unavailable,
)
from factor_common.storage import _json_safe
from portfolio.fixed_pipeline import summarize_accounting

_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "fixed_report.html"

TABLES = (
    "ledger", "orders", "positions", "valuation_prices", "funding",
    "funding_coverage",
)
LIMITATION = (
    "研究回测：日频 UTC，365 天年化，信号在下一日 open 基准执行。"
    "滑点按成交名义金额扣款；不是实际成交回放。未建模最小订单、数量精度、"
    "盘口冲击、保证金、强平、交易所故障或数据历史修订。"
    "基准各按单位总敞口运行，与受风险约束的组合暴露不同。"
    "历史分组成绩、冻结时间和 cutoff 一致性均不证明未来盈利。"
)


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2,
                   sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def observed_exposure(account: Mapping[str, Any]) -> pd.DataFrame:
    positions = account["positions"]
    marked = positions.mul(account["valuation_prices"])
    marked = marked.where(positions != 0.0, 0.0)
    weight = marked.div(account["ledger"]["equity"], axis=0)
    return pd.DataFrame({
        "gross": weight.abs().sum(axis=1),
        "long": weight.clip(lower=0).sum(axis=1),
        "short": -weight.clip(upper=0).sum(axis=1),
        "net": weight.sum(axis=1),
        "single": weight.abs().max(axis=1),
    }, index=weight.index)


def comparison_tables(result: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns: dict[str, pd.Series] = {}
    for name, baseline in result["baselines"].items():
        account = baseline["scenarios"]["all_costs"]
        if account["status"] == "complete":
            returns[name] = account["ledger"]["return"]
    names = list(result["baselines"])
    panel = pd.DataFrame(returns).reindex(columns=names)
    correlation = panel.corr(min_periods=20)
    valid = panel.notna().astype("int64")
    counts = valid.T.dot(valid)
    overlap = pd.DataFrame(np.nan, index=names, columns=names)
    for first in names:
        left = result["member_targets"][first]
        for second in names:
            right = result["member_targets"][second]
            active = (left.abs().sum(axis=1) > 0) & (right.abs().sum(axis=1) > 0)
            shared = np.minimum(left.abs(), right.abs()).where(left * right > 0, 0.0).sum(axis=1)
            denominator = 0.5 * (left.abs().sum(axis=1) + right.abs().sum(axis=1))
            series = shared.div(denominator.where(denominator > 0))
            overlap.loc[first, second] = series[active].mean()
    return correlation, counts, overlap


def report_html(result: Mapping[str, Any], correlation: pd.DataFrame,
                counts: pd.DataFrame, overlap: pd.DataFrame) -> str:
    """Render the fixed portfolio through the common factor-report visual system."""
    config = result["configuration"]
    scenarios = result["accounting"]["scenarios"]
    metrics = result["metrics"]
    status = escape(str(result["status"]))
    banner_class = "complete" if result["status"] == "complete" else "incomplete"
    status_banner = (
        f'<div class="banner {banner_class}"><strong>研究状态：{status}</strong> '
        "本报告仅由已保存的组合回测结果渲染，未重读市场数据或重新优化。</div>"
    )
    summary = _table(
        ["项目", "值"],
        [
            ["信号区间", _esc(f'{config["signal_start"]} 至 {config["signal_end"]}')],
            ["知识截止日", _esc(str(result["knowledge_cutoff"]))],
            ["调仓周期", _esc(f'{config["rebalance_days"]} 天')],
            ["成员因子数", _esc(str(len(result["baselines"])))],
            ["总/多/空/单币/净敞口上限", _esc(
                f'{config["gross_limit"]:.0%} / {config["long_limit"]:.0%} / '
                f'{config["short_limit"]:.0%} / {config["single_limit"]:.0%} / {config["net_limit"]:.0%}'
            )],
        ],
    )
    summary += f'<p class="note">{_esc(LIMITATION)}</p>'

    metric_rows = []
    for name in ("gross", "trading_net", "all_costs"):
        item = metrics[name]
        full = item["full"] or {}
        metric_rows.append([
            _esc(name), _esc(item["status"]), _percent(full.get("total_return")),
            _percent(full.get("annual_return")), _fmt(full.get("sharpe")),
            _percent(full.get("max_drawdown")), _fmt(full.get("n_periods")),
            _fmt(item.get("fee_total")), _fmt(item.get("slippage_total")),
            _fmt(item.get("funding_total")),
        ])
    scenario_metrics = _table(
        ["情景", "状态", "总收益", "年化收益", "Sharpe", "最大回撤", "期数", "手续费", "滑点", "资金费"],
        metric_rows,
    )

    nav_series = {}
    for name in ("gross", "trading_net", "all_costs"):
        ledger = scenarios[name]["ledger"]
        if not ledger.empty:
            nav_series[name] = ledger["equity"] / float(ledger["equity"].iloc[0])
    nav_section = _line_section("chart_portfolio_nav", "组合净值（起点 = 1.0）", nav_series, "NAV")

    all_costs = scenarios["all_costs"]["ledger"]
    cost_series = {}
    if not all_costs.empty:
        cost_series = {
            "累计手续费": all_costs["fee"].cumsum(),
            "累计滑点": all_costs["slippage"].cumsum(),
            "累计资金费现金流": all_costs["funding_cashflow"].cumsum(),
        }
    cost_section = _line_section("chart_costs", "累计成本分解（all_costs）", cost_series, "账户单位")

    turnover_series = {}
    for name in ("gross", "trading_net", "all_costs"):
        ledger = scenarios[name]["ledger"]
        if not ledger.empty:
            pretrade = ledger["equity"] + ledger["fee"] + ledger["slippage"]
            turnover_series[name] = ledger["trade_notional"].div(pretrade.where(pretrade > 0))
    turnover_section = _line_section("chart_turnover", "日换手率", turnover_series, "成交名义金额 / 调仓前权益")

    exposure = observed_exposure(scenarios["all_costs"])
    risk_series = {
        "实际总敞口": exposure["gross"], "实际多头": exposure["long"],
        "实际空头": exposure["short"], "实际绝对净敞口": exposure["net"].abs(),
        "实际单币最大敞口": exposure["single"],
        "总敞口上限": pd.Series(config["gross_limit"], index=exposure.index),
        "多头上限": pd.Series(config["long_limit"], index=exposure.index),
        "空头上限": pd.Series(config["short_limit"], index=exposure.index),
        "绝对净敞口上限": pd.Series(config["net_limit"], index=exposure.index),
        "单币敞口上限": pd.Series(config["single_limit"], index=exposure.index),
    }
    risk_section = _line_section("chart_exposure", "风险约束实际使用情况（all_costs）", risk_series, "权益权重")

    rows = []
    baselines = []
    for name, baseline in result["baselines"].items():
        item = summarize_accounting(baseline)["all_costs"]
        full = item["full"] or {}
        baselines.append([
            _esc(name), _esc(item["status"]), _percent(full.get("annual_return")),
            _fmt(full.get("sharpe")), _percent(full.get("max_drawdown")),
        ])
    baseline_section = _table(
        ["成员因子", "状态", "年化收益", "Sharpe", "最大回撤"], baselines
    )
    settlements = []
    for scenario, account in scenarios.items():
        for event in account["diagnostics"].get("contract_settlements", []):
            settlements.append([
                _esc(scenario), _esc(event.get("date")), _esc(event.get("instrument")),
                _fmt(event.get("price")),
            ])
    settlement_section = _table(["情景", "结算日", "合约", "结算价"], settlements) if settlements else _unavailable(
        "合约结算", "本次组合未持有需要结算的终止合约"
    )
    diagnostics = {name: account["diagnostics"] for name, account in result["accounting"]["scenarios"].items()}
    config_json = escape(json.dumps(_json_safe(config), ensure_ascii=False, indent=2))
    diag = escape(json.dumps(_json_safe(diagnostics), ensure_ascii=False, indent=2))
    template = string.Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        page_title="固定组合研究报告（UTC）", status_banner=status_banner,
        summary_section=summary, scenario_metrics_section=scenario_metrics,
        nav_section=nav_section, cost_section=cost_section, turnover_section=turnover_section,
        risk_section=risk_section, baseline_section=baseline_section,
        correlation_section=correlation.to_html(escape=True), counts_section=counts.to_html(escape=True),
        overlap_section=overlap.to_html(escape=True), settlement_section=settlement_section,
        scaling_section=result["risk"].describe(include="all").to_html(escape=True),
        exposure_summary=exposure.describe().to_html(escape=True), config_section=config_json,
        diagnostics_section=diag,
    )


def _percent(value: Any) -> str:
    if value is None or not isinstance(value, (float, int, np.number)) or not np.isfinite(value):
        return '<span class="null">null</span>'
    return f"{float(value):.2%}"


def _line_section(chart_id: str, title: str, series: Mapping[str, pd.Series], y_name: str) -> str:
    usable = {name: values for name, values in series.items() if isinstance(values, pd.Series) and not values.empty}
    if not usable:
        return _unavailable(title, "保存的账本没有可显示的数据")
    labels, aligned = _aligned(usable)
    return _chart_block(chart_id, _line_chart(title, labels, aligned, y_name=y_name))


def _new_run_id(root: Path) -> tuple[str, Path, Path]:
    while True:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
        staged, final = root / (".partial_" + run_id), root / run_id
        if not staged.exists() and not final.exists():
            return run_id, staged, final


def _safe_component(value: Any, label: str) -> str:
    """Validate a user/result-controlled value before using it in a path."""
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"invalid {label}: {value!r}")
    if "/" in value or "\\" in value or Path(value).name != value:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


@contextmanager
def _publication_lock(root: Path):
    """Serialize publishers and make the final existence check exclusive."""
    lock = root / ".publish.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"another artifact publication is in progress: {root}") from exc
    try:
        yield
    finally:
        lock.rmdir()


def write_result(result: Mapping[str, Any], output_root: str | Path) -> Path:
    """Write one complete result directory and publish it atomically.

    A failed write intentionally leaves its ``.partial_*`` directory for
    diagnosis. Existing runs are never replaced.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for name in result.get("values", {}):
        _safe_component(name, "member name")
    for name in result.get("member_targets", {}):
        _safe_component(name, "member target name")
    for name in result.get("accounting", {}).get("scenarios", {}):
        _safe_component(name, "scenario name")
    for name in result.get("baselines", {}):
        _safe_component(name, "baseline name")

    with _publication_lock(root):
        run_id, staged, final = _new_run_id(root)
        staged.mkdir(exist_ok=False)
        scalar_keys = ("status", "configuration", "knowledge_cutoff", "environment",
                       "source_stat", "value_pipeline_fingerprint")
        manifest = {key: result[key] for key in scalar_keys if key in result}
        manifest.update({"run_id": run_id, "research_only": True,
                         "strategy_certified": False, "artifact_complete": True,
                         "cutoff_audit_attached": False, "limitations": LIMITATION})
        write_json(staged / "manifest.json", manifest)
        write_json(staged / "metrics.json", result["metrics"])
        write_json(staged / "config.json", result["configuration"])
        write_json(staged / "provenance.json", {"environment": result.get("environment"),
                                                 "value_pipeline_fingerprint": result.get("value_pipeline_fingerprint")})
        write_json(staged / "source_stat.json", result.get("source_stat", {}))
        write_json(staged / "input_reads.json", result["input_reads"])
        result["targets"].to_parquet(staged / "targets.parquet")
        result["risk"].to_parquet(staged / "risk.parquet")
        for name, values in result["values"].items():
            path = staged / "members" / name
            path.mkdir(parents=True)
            values.to_parquet(path / "values.parquet")
            result["member_targets"][name].to_parquet(path / "targets.parquet")
            result["member_diagnostics"][name]["daily"].to_parquet(path / "diagnostics.parquet")
            write_json(path / "metadata.json", {key: value for key, value in result["member_diagnostics"][name].items() if key != "daily"})
        for name, account in result["accounting"]["scenarios"].items():
            path = staged / "scenarios" / name
            path.mkdir(parents=True)
            for table in TABLES:
                account[table].to_parquet(path / f"{table}.parquet")
            observed_exposure(account).to_parquet(path / "observed_exposure.parquet")
            write_json(path / "diagnostics.json", account["diagnostics"])
        for factor, baseline in result["baselines"].items():
            path = staged / "baselines" / factor
            path.mkdir(parents=True)
            write_json(path / "metrics.json", summarize_accounting(baseline))
            for scenario, account in baseline["scenarios"].items():
                _safe_component(scenario, "baseline scenario name")
                child = path / scenario
                child.mkdir()
                account["ledger"].to_parquet(child / "ledger.parquet")
                write_json(child / "diagnostics.json", account["diagnostics"])
        correlation, counts, overlap = comparison_tables(result)
        correlation.to_parquet(staged / "baseline_return_correlation.parquet")
        counts.to_parquet(staged / "baseline_return_pair_counts.parquet")
        overlap.to_parquet(staged / "target_overlap.parquet")
        (staged / "report.html").write_text(report_html(result, correlation, counts, overlap), encoding="utf-8")
        write_json(staged / "artifact_complete.json", {"run_id": run_id, "written": True})
        if final.exists():
            raise FileExistsError(f"artifact run already exists: {final}")
        os.rename(staged, final)
        return final


__all__ = ["write_result", "write_json", "observed_exposure", "comparison_tables", "report_html"]
