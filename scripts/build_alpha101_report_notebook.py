"""Build and execute the Alpha101 performance summary notebook.

Reads outputs/alpha101_results.json (produced by
scripts/batch_evaluate_alpha101.py) and writes an executed notebook to
factor_analyse/alpha101_performance_report.ipynb.

Usage:
    ./.venv/bin/python scripts/build_alpha101_report_notebook.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "outputs" / "alpha101_results.json"
NOTEBOOK_PATH = PROJECT_ROOT / "factor_analyse" / "alpha101_performance_report.ipynb"


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
            "# Alpha101 因子批量回测绩效统计\n"
            "\n"
            f"- 回测区间：`{params['start']}` ~ `{params['end']}`（信号窗口）\n"
            f"- 调仓周期：{params['rebalance_days']} 天；分组数：{params['n_groups']}\n"
            f"- 费率：{params['fee_rate']}；滑点：{params['slippage']}；含资金费：{params['include_funding']}\n"
            "- 因子全部转换为 `factor_common` 契约格式（`TYPE`/`META`/`SETTING`/`calc_factor`），"
            "截面去极值与秩归一化由框架 `mad_rank` 完成\n"
            "- 数据源：`data/crypto_quant.h5`（point-in-time top50 宇宙）\n"
            "- 每个因子的完整 HTML 报告见 `reports/<factor>.html`\n"
            "\n"
            "> 注意：受 H5 资金费覆盖限制，`all_costs` 场景在真实数据上通常无法完整认证"
            "（status=incomplete），此时以 `trading_net`（含手续费+滑点）作为净绩效参考。"
        ),
        code(
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "RESULTS_PATH = Path('../outputs/alpha101_results.json')\n"
            "payload = json.loads(RESULTS_PATH.read_text())\n"
            "rows = payload['results']\n"
            "df = pd.DataFrame(rows)\n"
            "print(f\"factors: {len(df)}, params: {payload['params']}\")\n"
            "df[['factor', 'status', 'static_scan', 'cutoff']].value_counts(['status']).to_frame('count')"
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
            "ax.set_title('Alpha101 factors - full-sample mean RankIC (2024-01-01 ~ 2026-09-01)')\n"
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
            "ax.set_title('Alpha101 factors - trading_net Sharpe (long-short, fees+slippage)')\n"
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
            "## 7. 失败 / 异常因子\n"
            "\n"
            "评估未到达 complete/incomplete 状态的因子及错误信息。"
        ),
        code(
            "bad = df[~df['status'].isin(['complete', 'incomplete'])]\n"
            "if bad.empty:\n"
            "    print('无失败因子')\n"
            "else:\n"
            "    display(bad[['factor', 'status', 'error']].reset_index(drop=True))"
        ),
    ]
    return nb


def main() -> int:
    if not RESULTS_PATH.is_file():
        print(f"error: {RESULTS_PATH} not found; run batch_evaluate_alpha101.py first")
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
