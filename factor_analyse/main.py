"""因子分析 CLI —— common 框架适配器。

用法:
    python factor_analyse/main.py --list
    python factor_analyse/main.py <factor_path> [rebalance_days] [--start ...]
        [--end ...] [--h5-path ...] [--output-dir ...] [--no-plot]
        [--funding-price-mode strict|daily_open_approx]

``factor_path`` 是因子模块文件的完整路径（满足 factor_common loader 契约的
.py 文件），不再使用 factor_config 注册表。因子的方向、预热窗口等元信息
由模块自身的 SETTING 提供。例如:

    python factor_analyse/main.py \
        factor_analyse/factor_mining/Volume_Stability_Factor.py 1
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

_NON_FACTOR_MODULES = {"util_factor", "operator_utils"}


def _cmd_list():
    """列出 factor_mining/ 下所有可加载的因子模块及其 SETTING 摘要。"""
    from factor_common.loader import load_factor

    factor_dir = _PROJECT_ROOT / "factor_analyse" / "factor_mining"
    print(f"{'factor file':<50} {'direction':>9}  description")
    for path in sorted(factor_dir.glob("*.py")):
        if path.stem in _NON_FACTOR_MODULES:
            continue
        try:
            spec = load_factor(path)
        except Exception as exc:
            print(f"{path.name:<50} {'-':>9}  unloadable: {exc}")
            continue
        direction = spec.setting["factor_direction"]
        print(f"{path.name:<50} {direction:>9}  {spec.meta['description']}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="因子分析入口（common 框架适配器，直接传因子文件路径）",
    )
    parser.add_argument(
        "factor_path", nargs="?",
        help="因子模块 .py 文件的完整路径（相对路径基于当前工作目录解析）",
    )
    parser.add_argument(
        "rebalance_days", nargs="?", type=int, default=None,
        help="调仓周期（天），缺省为 1",
    )
    parser.add_argument("--list", "-l", action="store_true", help="列出 factor_mining/ 下的因子模块")
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

    if not args.factor_path:
        print("错误: 缺少 factor_path（或使用 --list 查看可用因子模块）", file=sys.stderr)
        return 2

    factor_path = Path(args.factor_path).expanduser()
    if not factor_path.is_absolute():
        factor_path = (Path.cwd() / factor_path).resolve()
    if not factor_path.is_file():
        print(f"错误: 因子文件不存在: {factor_path}", file=sys.stderr)
        return 2

    from factor_common import FactorManager
    from factor_common.loader import load_factor

    try:
        spec = load_factor(factor_path)
    except Exception as exc:
        print(f"错误: 因子文件不满足 loader 契约: {exc}", file=sys.stderr)
        return 3

    rebalance_days = args.rebalance_days or 1
    output_dir = Path(args.output_dir) if args.output_dir else None
    manager = FactorManager(
        h5_path=args.h5_path,
        base_dir=(output_dir / "factor_results") if output_dir else None,
        reports_dir=output_dir,
    )
    params = {
        "rebalance_days": rebalance_days,
        "n_groups": 10,
        "out_of_sample_days": 180,
    }
    if args.start:
        params["start"] = args.start
    if args.end:
        params["end"] = args.end
    if args.funding_price_mode:
        params["funding_price_mode"] = args.funding_price_mode

    print(
        f"开始分析 {spec.meta['description']} ({spec.factor_id}) - "
        f"{rebalance_days}天调仓 direction={spec.setting['factor_direction']}"
    )
    try:
        result = manager.evaluate(
            str(factor_path),
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
            target = web_dir / f"{spec.factor_id}_rebalance{rebalance_days}d_latest.html"
            shutil.copy2(report_path, target)
            print(f"✅ 报告已复制到web目录: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
