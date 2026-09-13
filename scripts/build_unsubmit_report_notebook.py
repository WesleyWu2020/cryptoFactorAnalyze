"""Build and execute the unsubmit-factor performance summary notebook.

Reads outputs/unsubmit_results.json (produced by
scripts/batch_evaluate_unsubmit.py) and writes an executed notebook to
factor_analyse/unsubmit_performance_report.ipynb.

Usage:
    ./.venv/bin/python scripts/build_unsubmit_report_notebook.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "outputs" / "unsubmit_results.json"
NOTEBOOK_PATH = PROJECT_ROOT / "factor_analyse" / "unsubmit_performance_report.ipynb"


def md(source: str):
    return nbformat.v4.new_markdown_cell(source)


def code(source: str):
    return nbformat.v4.new_code_cell(source)


def build_notebook(params: dict) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3"}

    nb.cells = [
        md(
            "# unsubmit 因子批量回测绩效统计\n"
            "\n"
            f"- 回测区间：`{params['start']}` ~ `{params['end']}`（信号窗口）\n"
            f"- 调仓周期：{params['rebalance_days']} 天；分组数：{params['n_groups']}\n"
            f"- 费率：{params['fee_rate']}；滑点：{params['slippage']}；含资金费：{params['include_funding']}\n"
            "- 因子由分钟频框架批量转换为 `factor_common` 日频契约（`TYPE`/`META`/`SETTING`/`calc_factor`），"
            "截面去极值与秩归一化由框架 `mad_rank` 完成\n"
            "- 字段映射：`dollar_volume`/`turnover`/`xbinance_quote_volume` → `quote_volume`，"
            "`xbinance_taker_buy_quote_volume` → `taker_buy_quote_volume`，"
            "`xbinance_trade_count` → `trade_count`，`funding` → 日频平均资金费率\n"
            "- 数据源：`data/crypto_quant.h5`（point-in-time top50 宇宙）\n"
            "- 每个因子的完整 HTML 报告见 `reports/<factor>.html`\n"
            "\n"
            "> 注意：受 H5 资金费覆盖限制，`all_costs` 场景在真实数据上通常无法完整认证"
            "（status=incomplete），此时以 `trading_net`（含手续费+滑点）作为净绩效参考。"
            "分钟级滚动窗口已按公式意图改写为日频等价形式，语义判断记录在各因子模块 docstring。"
        ),
        code(
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "RESULTS_PATH = Path('../outputs/unsubmit_results.json')\n"
            "payload = json.loads(RESULTS_PATH.read_text())\n"
            "rows = payload['results']\n"
            "df = pd.DataFrame(rows)\n"
            "print(f\"factors: {len(df)}, params: {payload['params']}\")\n"
            "df['status'].value_counts().to_frame('count')"
        ),
        md(
            "## 1. 总览：IC 指标（全样本 full）\n"
            "\n"
            "按 `rank_ic_mean` 降序。`icir` 未年化；`t_stat`/`p_value` 为 IC 序列显著性。\n"
            "`static_scan`/`cutoff` 为未来函数检查结论（静态扫描 + 截断重放一致性）。"
        ),
        code(
            "ic_cols = ['factor', 'status', 'full.rank_ic_mean', 'full.ic_mean', 'full.icir',\n"
            "           'full.t_stat', 'full.p_value', 'static_scan', 'cutoff']\n"
            "ic_table = (df[ic_cols]\n"
            "            .sort_values('full.rank_ic_mean', ascending=False, na_position='last')\n"
            "            .reset_index(drop=True))\n"
            "ic_table.index += 1\n"
            "ic_table.style.format({c: '{:.4f}' for c in ic_cols if c.startswith('full.')},\n"
            "                      na_rep='-')"
        ),
        md("## 2. RankIC 分布图"),
        code(
            "plot_df = df.dropna(subset=['full.rank_ic_mean']).sort_values('full.rank_ic_mean')\n"
            "fig, ax = plt.subplots(figsize=(10, max(8, 0.28 * len(plot_df))))\n"
            "colors = ['#d62728' if v < 0 else '#2ca02c' for v in plot_df['full.rank_ic_mean']]\n"
            "ax.barh(plot_df['factor'], plot_df['full.rank_ic_mean'], color=colors)\n"
            "ax.axvline(0, color='black', linewidth=0.8)\n"
            "ax.set_title('unsubmit factors - full-sample mean RankIC (2024-01-01 ~ 2026-09-01)')\n"
            "ax.set_xlabel('rank_ic_mean')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## 3. 样本内 / 样本外 IC 稳定性\n"
            "\n"
            "样本外 = 最后 180 天。`rank_ic_oos` 与全样本符号一致且幅度未大幅衰减的因子更稳健。"
        ),
        code(
            "split_cols = ['factor', 'full.rank_ic_mean', 'in_sample.rank_ic_mean',\n"
            "              'out_of_sample.rank_ic_mean', 'in_sample.t_stat', 'out_of_sample.t_stat']\n"
            "split_table = (df[split_cols]\n"
            "               .sort_values('full.rank_ic_mean', ascending=False, na_position='last')\n"
            "               .reset_index(drop=True))\n"
            "split_table.index += 1\n"
            "split_table.style.format({c: '{:.4f}' for c in split_cols if c != 'factor'}, na_rep='-')"
        ),
        md(
            "## 4. 多空组合绩效（trading_net = 含手续费+滑点）\n"
            "\n"
            "`gross` 为无成本对照。收益为复利累计；`max_drawdown` 为复合 NAV 上的最大回撤（小数）。"
        ),
        code(
            "perf_cols = ['factor', 'trading_net.sharpe', 'trading_net.total_return',\n"
            "             'trading_net.annual_return', 'trading_net.max_drawdown',\n"
            "             'trading_net.win_rate', 'gross.sharpe', 'gross.total_return']\n"
            "perf_table = (df[perf_cols]\n"
            "              .sort_values('trading_net.sharpe', ascending=False, na_position='last')\n"
            "              .reset_index(drop=True))\n"
            "perf_table.index += 1\n"
            "perf_table.style.format({c: '{:.3f}' for c in perf_cols if c != 'factor'}, na_rep='-')"
        ),
        md("## 5. 净 Sharpe 分布图"),
        code(
            "plot_df = df.dropna(subset=['trading_net.sharpe']).sort_values('trading_net.sharpe')\n"
            "fig, ax = plt.subplots(figsize=(10, max(8, 0.28 * len(plot_df))))\n"
            "colors = ['#d62728' if v < 0 else '#2ca02c' for v in plot_df['trading_net.sharpe']]\n"
            "ax.barh(plot_df['factor'], plot_df['trading_net.sharpe'], color=colors)\n"
            "ax.axvline(0, color='black', linewidth=0.8)\n"
            "ax.set_title('unsubmit factors - trading_net Sharpe (long-short, fees+slippage)')\n"
            "ax.set_xlabel('sharpe')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## 6. 筛选：可进一步研究的因子\n"
            "\n"
            "标准：全样本 `|rank_ic_mean| >= 0.02` 且 `|t_stat| >= 2`，且样本外 rank_ic 与全样本同号。\n"
            "按 `|rank_ic_mean|` 降序。"
        ),
        code(
            "mask = ((df['full.rank_ic_mean'].abs() >= 0.02)\n"
            "        & (df['full.t_stat'].abs() >= 2)\n"
            "        & (df['out_of_sample.rank_ic_mean'] * df['full.rank_ic_mean'] > 0))\n"
            "selected = (df.loc[mask, ['factor', 'full.rank_ic_mean', 'full.t_stat',\n"
            "                          'in_sample.rank_ic_mean', 'out_of_sample.rank_ic_mean',\n"
            "                          'trading_net.sharpe', 'trading_net.total_return',\n"
            "                          'trading_net.max_drawdown']]\n"
            "            .assign(abs_rank_ic=lambda x: x['full.rank_ic_mean'].abs())\n"
            "            .sort_values('abs_rank_ic', ascending=False)\n"
            "            .drop(columns='abs_rank_ic')\n"
            "            .reset_index(drop=True))\n"
            "selected.index += 1\n"
            "print(f'selected: {len(selected)} / {len(df)}')\n"
            "selected.style.format({c: '{:.4f}' for c in selected.columns if c != 'factor'}, na_rep='-')"
        ),
        md(
            "## 7. 跳过 / 失败因子\n"
            "\n"
            "`skipped` = 依赖日频 H5 未提供的数据（open_interest、mark/index_close、"
            "bybit standards、跨交易所份额），未做转换；其余为评估异常。"
        ),
        code(
            "bad = df[~df['status'].isin(['complete', 'incomplete'])]\n"
            "if bad.empty:\n"
            "    print('无跳过/失败因子')\n"
            "else:\n"
            "    display(bad[['factor', 'status', 'error']].reset_index(drop=True))"
        ),
        md(
            "## 8. all_costs 总绩效（含手续费+滑点+资金费）\n"
            "\n"
            "按 `all_costs.annual_return` 降序。与第 4 节的 `trading_net` 相比，"
            "`all_costs` 额外计入了资金费现金流；`cost_drag` = gross − all_costs 年收益差。"
        ),
        code(
            "ac_cols = ['factor', 'all_costs.annual_return', 'all_costs.total_return',\n"
            "         'all_costs.sharpe', 'all_costs.max_drawdown', 'all_costs.win_rate',\n"
            "         'gross.annual_return']\n"
            "ac_table = (df.dropna(subset=['all_costs.annual_return'])[ac_cols]\n"
            "            .assign(cost_drag=lambda x: x['gross.annual_return'] - x['all_costs.annual_return'])\n"
            "            .sort_values('all_costs.annual_return', ascending=False)\n"
            "            .reset_index(drop=True))\n"
            "ac_table.index += 1\n"
            "ac_table.style.format({c: '{:.3f}' for c in ac_table.columns if c != 'factor'}, na_rep='-')"
        ),
        md(
            "## 9. all_costs 分年度绩效\n"
            "\n"
            "从 `reports/<factor>.html` 内嵌的 all_costs NAV 序列复合出每年收益"
            "（2026 年为年初至今）。先给全库分年度透视表（按全样本 all_costs 年收益排序），"
            "再给按年份的横截面汇总。"
        ),
        code(
            "import re\n"
            "\n"
            "def _matched_json_array(text, start_at):\n"
            "    start = text.find('[', start_at)\n"
            "    if start < 0:\n"
            "        raise ValueError('NAV data array not found')\n"
            "    depth, quoted, escaped = 0, False, False\n"
            "    for index in range(start, len(text)):\n"
            "        char = text[index]\n"
            "        if quoted:\n"
            "            if escaped:\n"
            "                escaped = False\n"
            "            elif char == '\\\\':\n"
            "                escaped = True\n"
            "            elif char == '\"':\n"
            "                quoted = False\n"
            "        else:\n"
            "            if char == '\"':\n"
            "                quoted = True\n"
            "            elif char == '[':\n"
            "                depth += 1\n"
            "            elif char == ']':\n"
            "                depth -= 1\n"
            "                if depth == 0:\n"
            "                    return text[start:index + 1]\n"
            "    raise ValueError('NAV data array is not closed')\n"
            "\n"
            "def _scenario_nav(text, scenario):\n"
            "    nav_section = text[text.index('var option_chart_nav'):text.index('chart_chart_nav.setOption')]\n"
            "    match = re.search(r'\"name\":\\s*\"' + re.escape(scenario) + r'\"', nav_section)\n"
            "    if match is None:\n"
            "        raise ValueError(f'{scenario} NAV series not found')\n"
            "    data_match = re.search(r'\"data\":\\s*', nav_section[match.end():])\n"
            "    values = json.loads(_matched_json_array(nav_section, match.end() + data_match.start()))\n"
            "    return pd.DataFrame(values, columns=['date', 'nav']).assign(\n"
            "        date=lambda frame: pd.to_datetime(frame['date']),\n"
            "        nav=lambda frame: pd.to_numeric(frame['nav']),\n"
            "    )\n"
            "\n"
            "def _annual_return(nav):\n"
            "    frame = nav.sort_values('date').copy()\n"
            "    frame['year'] = frame['date'].dt.year\n"
            "    year_end = frame.groupby('year', sort=True)['nav'].last()\n"
            "    opening_nav = year_end.shift(1)\n"
            "    opening_nav.iloc[0] = frame.groupby('year', sort=True)['nav'].first().iloc[0]\n"
            "    return year_end.div(opening_nav).sub(1).rename('return')\n"
            "\n"
            "REPORT_DIR = Path('../reports')\n"
            "records, nav_errors = [], []\n"
            "for factor in df.loc[df['status'].isin(['complete', 'incomplete']), 'factor']:\n"
            "    report_file = REPORT_DIR / f'{factor}.html'\n"
            "    try:\n"
            "        html = report_file.read_text(encoding='utf-8')\n"
            "        gross = _annual_return(_scenario_nav(html, 'gross'))\n"
            "        all_costs = _annual_return(_scenario_nav(html, 'all_costs'))\n"
            "        for year in gross.index:\n"
            "            records.append({'factor': factor, 'year': int(year),\n"
            "                            'gross': gross.loc[year], 'all_costs': all_costs.loc[year],\n"
            "                            'cost_drag': gross.loc[year] - all_costs.loc[year]})\n"
            "    except Exception as error:\n"
            "        nav_errors.append((factor, str(error)))\n"
            "yearly = pd.DataFrame(records).sort_values(['factor', 'year']).reset_index(drop=True)\n"
            "print(f'parsed {yearly.factor.nunique()} reports, {len(yearly)} factor-year rows, '\n"
            "      f'years {yearly.year.min()}-{yearly.year.max()}, errors: {len(nav_errors)}')\n"
            "if nav_errors:\n"
            "    print(nav_errors)"
        ),
        code(
            "annual_pivot = yearly.pivot(index='factor', columns='year', values='all_costs')\n"
            "order = (df.set_index('factor')['all_costs.annual_return']\n"
            "         .sort_values(ascending=False).index)\n"
            "annual_pivot = annual_pivot.reindex(order).dropna(how='all')\n"
            "annual_pivot['mean'] = annual_pivot.mean(axis=1)\n"
            "annual_pivot.style.format('{:.1%}', na_rep='-') \\\n"
            "    .background_gradient(cmap='RdYlGn', vmin=-0.6, vmax=0.6)"
        ),
        code(
            "year_summary = yearly.groupby('year').agg(\n"
            "    factors=('factor', 'size'),\n"
            "    mean_gross=('gross', 'mean'),\n"
            "    mean_all_costs=('all_costs', 'mean'),\n"
            "    median_all_costs=('all_costs', 'median'),\n"
            "    mean_cost_drag=('cost_drag', 'mean'),\n"
            "    positive_all_costs=('all_costs', lambda values: int((values > 0).sum())),\n"
            ")\n"
            "print('各年份全库横截面汇总（mean/median 为因子间统计；positive_all_costs 为当年 all_costs 为正的因子数）')\n"
            "year_summary.style.format({c: '{:.1%}' for c in\n"
            "                           ['mean_gross', 'mean_all_costs', 'median_all_costs', 'mean_cost_drag']})"
        ),
    ]
    return nb


def main() -> int:
    if not RESULTS_PATH.is_file():
        print(f"error: {RESULTS_PATH} not found; run batch_evaluate_unsubmit.py first")
        return 1
    payload = json.loads(RESULTS_PATH.read_text())

    nb = build_notebook(payload["params"])
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
    nbformat.write(nb, NOTEBOOK_PATH)
    print(f"notebook written: {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
