"""因子分析 CLI —— common 框架适配器。

用法:
    python factor_analyse/main.py --list
    python factor_analyse/main.py <factor_type> [rebalance_days] [--start ...]
        [--end ...] [--h5-path ...] [--output-dir ...] [--no-plot]
        [--funding-price-mode strict|daily_open_approx]

位置参数 rebalance_days 显式覆盖 factor_config 中的注册默认值。
未迁移因子（无 module_path）报告迁移状态并以非零码退出，不再静默搜索
已删除的 CSV 因子目录。
"""

import argparse
import shutil
import sys
from pathlib import Path

_FACTOR_ANALYSE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FACTOR_ANALYSE_DIR.parent
for _path in (str(_PROJECT_ROOT), str(_FACTOR_ANALYSE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factor_config import FACTOR_CONFIG, get_factor_config, list_all_factors  # noqa: E402


def _migration_state(factor_id):
    """返回 (module_path 或 None, 迁移状态描述)。"""
    config = get_factor_config(factor_id) or {}
    module_path = config.get("module_path")
    if not module_path:
        return None, "not migrated (无 module_path; 旧 CSV 管线已移除)"
    path = _PROJECT_ROOT / module_path
    if not path.is_file():
        return None, f"not runnable (module_path 文件不存在: {module_path})"
    return path, "migrated (common framework)"


def _cmd_list():
    print(f"{'factor_type':<45} {'rebalance':>9}  migration")
    for factor_id in list_all_factors():
        config = FACTOR_CONFIG[factor_id]
        _, state = _migration_state(factor_id)
        rebalance = config.get("rebalance_period", "-")
        print(f"{factor_id:<45} {rebalance:>9}  {state}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="因子分析入口（common 框架适配器）",
    )
    parser.add_argument("factor_type", nargs="?", help="factor_config 中的因子类型")
    parser.add_argument(
        "rebalance_days", nargs="?", type=int, default=None,
        help="调仓周期（天），显式指定时覆盖注册默认值",
    )
    parser.add_argument("--list", "-l", action="store_true", help="列出所有因子及迁移状态")
    parser.add_argument("--start", default=None, help="信号开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="信号结束日期 YYYY-MM-DD")
    parser.add_argument("--h5-path", default=None, help="CryptoQuant H5 数据文件路径")
    parser.add_argument(
        "--output-dir", default=None,
        help="报告与运行产物输出目录（默认 reports/ 与 data/factor_results/）",
    )
    parser.add_argument("--no-plot", action="store_true", help="跳过 HTML 报告渲染")
    parser.add_argument(
        "--funding-price-mode", default=None,
        choices=["strict", "daily_open_approx"],
        help="资金费定价模式（默认 strict）",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list:
        return _cmd_list()

    if not args.factor_type:
        print("错误: 缺少 factor_type（或使用 --list 查看支持的因子）", file=sys.stderr)
        return 2

    config = get_factor_config(args.factor_type)
    if config is None:
        print(f"错误: 不支持的因子类型 '{args.factor_type}'", file=sys.stderr)
        print(f"支持的因子类型: {', '.join(list_all_factors())}", file=sys.stderr)
        return 2

    module_path, state = _migration_state(args.factor_type)
    if module_path is None:
        print(f"因子 '{args.factor_type}' 迁移状态: {state}", file=sys.stderr)
        print(
            "请先将其迁移为 factor_common 模块（loader 契约），并在 "
            "factor_config.py 中注册 module_path。",
            file=sys.stderr,
        )
        return 3

    from factor_common import FactorManager

    rebalance_days = args.rebalance_days or config.get("rebalance_period") or 1
    output_dir = Path(args.output_dir) if args.output_dir else None
    manager = FactorManager(
        h5_path=args.h5_path,
        base_dir=(output_dir / "factor_results") if output_dir else None,
        reports_dir=output_dir,
    )
    params = {
        "rebalance_days": rebalance_days,
        "n_groups": 10,
        "factor_direction": config["factor_direction"],
        "out_of_sample_days": 180,
    }
    if args.start:
        params["start"] = args.start
    if args.end:
        params["end"] = args.end
    if args.funding_price_mode:
        params["funding_price_mode"] = args.funding_price_mode

    print(
        f"开始分析 {config['factor_desc']} ({args.factor_type}) - "
        f"{rebalance_days}天调仓 [{state}]"
    )
    try:
        result = manager.evaluate(
            str(module_path),
            factor_name=config["factor_name"],
            params=params,
            plot=not args.no_plot,
        )
    except Exception as exc:
        print(f"错误: 评估失败: {exc}", file=sys.stderr)
        return 1

    ic = result["factor_performance"]["samples"]["full"]["ic"]
    print(
        f"status={result['status']} ic_mean={ic['ic_mean']} "
        f"rank_ic_mean={ic['rank_ic_mean']} t_stat={ic['t_stat']}"
    )
    report_path = result["paths"]["report_path"]
    if report_path:
        print(f"✅ 因子分析报告已生成: {report_path}")
        # 保留旧的 web 目录复制行为，仅在该目录已配置存在时生效
        web_dir = _PROJECT_ROOT / "web"
        if web_dir.is_dir():
            target = web_dir / f"{args.factor_type}_rebalance{rebalance_days}d_latest.html"
            shutil.copy2(report_path, target)
            print(f"✅ 报告已复制到web目录: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
