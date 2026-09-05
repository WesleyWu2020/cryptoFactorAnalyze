"""示例动量因子：N 日对数动量（common 框架迁移示例）。

公式复用 Task 3 模板：log(close / close.shift(window))，window 来自
SETTING["params"]。导入本模块无任何副作用；可执行入口位于
``if __name__ == "__main__"`` 下，解析项目根目录并调用 FactorManager。
"""

import numpy as np

TYPE = "regular"

META = {"factor_name": "example_momentum", "author": "local",
        "level": "daily", "category": "momentum", "description": "N-day log momentum"}

SETTING = {"data_needed": ["close"], "universe": "historical_top50",
           "warmup_bars": 20, "preprocessing": "mad_rank",
           "params": {"window": 20}, "factor_direction": 1}


def calc_factor(data_ctx):
    close = data_ctx["close"]
    window = SETTING["params"]["window"]
    return np.log(close / close.shift(window))


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from factor_common import FactorManager

    _parser = argparse.ArgumentParser(
        description="Evaluate the example_momentum factor via FactorManager"
    )
    _parser.add_argument("--start", default=None, help="signal start date (YYYY-MM-DD)")
    _parser.add_argument("--end", default=None, help="signal end date (YYYY-MM-DD)")
    _parser.add_argument("--rebalance-days", type=int, default=1)
    _parser.add_argument("--no-plot", action="store_true", help="skip HTML rendering")
    _args = _parser.parse_args()

    _params = {"rebalance_days": _args.rebalance_days}
    if _args.start:
        _params["start"] = _args.start
    if _args.end:
        _params["end"] = _args.end

    _manager = FactorManager(project_root=_PROJECT_ROOT)
    _result = _manager.evaluate(
        str(Path(__file__).resolve()), params=_params, plot=not _args.no_plot
    )
    _ic = _result["factor_performance"]["samples"]["full"]["ic"]
    print(f"status={_result['status']} run_id={_result['run_id']}")
    print(f"ic_mean={_ic['ic_mean']} rank_ic_mean={_ic['rank_ic_mean']}")
    print(f"report={_result['paths']['report_path']}")
