import argparse
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# util_factor 在 factor_mining 目录下
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FACTOR_ANALYSE_DIR = os.path.dirname(THIS_DIR)
FACTOR_MINING_DIR = os.path.join(FACTOR_ANALYSE_DIR, "factor_mining")
if FACTOR_MINING_DIR not in sys.path:
    sys.path.insert(0, FACTOR_MINING_DIR)

from util_factor import (
    FACTOR_DIR,
    load_available_tokens_from_index_cache,
    load_kline_df,
    print_availability_sample,
    print_factor_summary,
    print_latest_date_inference,
    save_factor_df,
)

from expr import Node
from factory import random_tree
from fitness import build_factor_df_for_expr, evaluate_expression
from ga import GAConfig, run_ga, split_train_valid_dates


def _slugify_expr(expr: str, max_len: int = 80) -> str:
    s = expr.strip().lower()
    # keep letters/numbers, turn everything else into underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "expr"
    return s[:max_len]


def _append_ga_log(
    expr: str,
    out_path: str,
    best_valid,
    args: argparse.Namespace,
) -> None:
    os.makedirs(FACTOR_DIR, exist_ok=True)
    log_path = os.path.join(FACTOR_DIR, "ga_best_expr_log.csv")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "datetime": now,
        "rebalance_period": args.rebalance_period,
        "top_n": args.top_n,
        "group_n": args.group_n,
        "ic_weight": args.ic_weight,
        "ls_weight": args.ls_weight,
        "fitness": best_valid.fitness,
        "ic_ir": best_valid.ic_ir,
        "ls_ir": best_valid.ls_ir,
        "ic_mean": getattr(best_valid, "ic_mean", np.nan),
        "ls_mean": getattr(best_valid, "ls_mean", np.nan),
        "n_rows": getattr(best_valid, "n_rows", np.nan),
        "expr": expr,
        "factor_file": out_path,
    }

    import pandas as pd

    df = pd.DataFrame([row])
    if not os.path.exists(log_path):
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode="a", header=False, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rebalance-period", type=int, default=10)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--cache-file", type=str, default="index_cache_equal_weight_30_50.csv")

    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--population", type=int, default=80)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--max-nodes", type=int, default=35)

    p.add_argument("--group-n", type=int, default=5)
    p.add_argument("--ic-weight", type=float, default=0.6)
    p.add_argument("--ls-weight", type=float, default=0.4)

    # quality gates (only export high-quality candidates)
    p.add_argument("--min-rows", type=int, default=8000, help="最少有效样本行数（过滤低质量候选）")
    p.add_argument("--min-days", type=int, default=60, help="最少有效天数（IC/LS 序列长度；兼容旧参数 --min-ic-days/--min-ls-days）")
    # backward-compatible aliases (older commands)
    p.add_argument("--min-ic-days", type=int, default=None, help="[兼容参数] 最少 IC 有效天数（将与 --min-days 取最大值）")
    p.add_argument("--min-ls-days", type=int, default=None, help="[兼容参数] 最少 LS 有效天数（将与 --min-days 取最大值）")

    p.add_argument("--seed", type=int, default=7)

    # parallelism
    # 0 => auto (use all CPU cores), 1 => single-thread (default), N => N worker threads
    p.add_argument("--n-jobs", type=int, default=1, help="并行评估线程数；0=自动；1=关闭并行")

    # evaluate more candidates on valid each generation
    p.add_argument(
        "--valid-eval-k",
        type=int,
        default=0,
        help="每一代在 valid 上评估的 unique 候选数量；0=自动(保持当前策略)",
    )

    # export multiple candidates
    p.add_argument("--export-topk", type=int, default=20, help="导出 valid 上候选TopK到日志（不一定保存因子文件）")
    p.add_argument(
        "--save-topk",
        type=int,
        default=1,
        help="保存 valid 上TopK因子文件数量（默认只保存第1名；设为0表示不保存任何因子文件）",
    )
    p.add_argument(
        "--overwrite-top-log",
        action="store_true",
        help="覆盖写入 ga_top_expr_log.csv（不追加历史记录），避免多次运行后出现重复观感",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    effective_min_days = int(
        max(
            args.min_days,
            0 if args.min_ic_days is None else args.min_ic_days,
            0 if args.min_ls_days is None else args.min_ls_days,
        )
    )

    print("加载K线数据...")
    df = load_kline_df()
    all_dates = sorted(df["date"].unique())

    print("从指数缓存加载可用token列表...")
    available_tokens_by_date = load_available_tokens_from_index_cache(
        cache_file=args.cache_file,
        top_n=args.top_n,
        all_dates=all_dates,
    )
    if not available_tokens_by_date:
        raise FileNotFoundError(f"无法从指数缓存加载成分股列表: {args.cache_file}")

    print_availability_sample(available_tokens_by_date, n=3)

    train_range, valid_range = split_train_valid_dates(all_dates, train_ratio=0.7)
    print(f"Train range: {train_range[0].date()} ~ {train_range[1].date()}")
    print(f"Valid range: {valid_range[0].date()} ~ {valid_range[1].date()}")

    cfg = GAConfig(
        population=args.population,
        generations=args.generations,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        seed=args.seed,
        group_n=args.group_n,
        ic_weight=args.ic_weight,
        ls_weight=args.ls_weight,
        min_rows=args.min_rows,
        min_days=effective_min_days,
        n_jobs=args.n_jobs,
        valid_eval_k=args.valid_eval_k,
    )

    best_train_node, best_train, best_valid_node, best_valid, valid_ranked = run_ga(
        df=df,
        available_tokens_by_date=available_tokens_by_date,
        rebalance_period=args.rebalance_period,
        cfg=cfg,
        train_range=train_range,
        valid_range=valid_range,
    )

    print("\n===== Best (train) =====")
    print(best_train)
    print("\n===== Best (valid) =====")
    print(best_valid)

    # 只导出高质量候选：evaluate_expression 失败/样本太少会返回 fitness 约 -1e9
    # 并且导出时按 expr 去重（不同代/不同个体可能生成同一表达式）
    unique_valid_ranked = []
    seen_expr = set()
    for node, res in valid_ranked:
        if not np.isfinite(res.fitness):
            continue
        if res.fitness <= -1e8:
            continue
        if res.expr in seen_expr:
            continue
        seen_expr.add(res.expr)
        unique_valid_ranked.append((node, res))

    # 1) 导出 valid 候选 TopK 名单（单独日志，不影响 ga_best_expr_log.csv 的兼容性）
    os.makedirs(FACTOR_DIR, exist_ok=True)
    top_log_path = os.path.join(FACTOR_DIR, "ga_top_expr_log.csv")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    export_topk = max(1, int(args.export_topk))
    if len(unique_valid_ranked) < export_topk:
        print(
            f"[WARN] 高质量候选只有 {len(unique_valid_ranked)} 个 (< export_topk={export_topk}). "
            f"可尝试增大 --population/--generations，或放宽 --min-rows/--min-days。"
        )
    for rank, (node, res) in enumerate(unique_valid_ranked[:export_topk], start=1):
        rows.append(
            {
                "datetime": now,
                "rank": rank,
                "rebalance_period": args.rebalance_period,
                "top_n": args.top_n,
                "group_n": args.group_n,
                "ic_weight": args.ic_weight,
                "ls_weight": args.ls_weight,
                "min_rows": args.min_rows,
                "min_days": effective_min_days,
                "fitness": res.fitness,
                "ic_ir": res.ic_ir,
                "ls_ir": res.ls_ir,
                "ic_mean": getattr(res, "ic_mean", np.nan),
                "ls_mean": getattr(res, "ls_mean", np.nan),
                "n_rows": getattr(res, "n_rows", np.nan),
                "expr": res.expr,
                "expr_slug": _slugify_expr(res.expr),
                "factor_file": "",
            }
        )

    import pandas as pd

    df_rows = pd.DataFrame(rows)
    if args.overwrite_top_log or (not os.path.exists(top_log_path)):
        df_rows.to_csv(top_log_path, index=False)
    else:
        df_rows.to_csv(top_log_path, mode="a", header=False, index=False)

    # 2) 保存 TopK 因子文件（默认仅保存第1名：best_valid；save_topk=0 则不保存）
    save_topk = max(0, int(args.save_topk))
    saved_paths = []
    if save_topk == 0:
        print("\n已设置 --save-topk 0：不保存因子文件，仅记录表达式与指标。")
        _append_ga_log(best_valid.expr, "", best_valid, args)
    else:
        print(f"\n保存 valid Top{save_topk} 因子文件...")
        for rank, (node, res) in enumerate(unique_valid_ranked[:save_topk], start=1):
            fac = build_factor_df_for_expr(
                df=df,
                expr=node,
                available_tokens_by_date=available_tokens_by_date,
                rebalance_period=args.rebalance_period,
            )
            factor_df = fac.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]

            expr_slug = _slugify_expr(res.expr)
            run_tag = datetime.now().strftime("%H%M%S")
            out_path = save_factor_df(
                factor_df,
                file_prefix=f"ga_{expr_slug}_rebalance{args.rebalance_period}d_",
                suffix=f"rank{rank}_run{run_tag}_",
            )
            saved_paths.append(out_path)

            if rank == 1:
                print_factor_summary(factor_df, out_path)
                latest_date = factor_df["date"].max()
                print_latest_date_inference(factor_df, latest_date, groups=args.group_n)

            # 记录“best”日志（保持原文件语义：只记录最佳）
            if rank == 1:
                _append_ga_log(res.expr, out_path, res, args)

    print("\nBest expr (valid):")
    print(best_valid.expr)
    print(f"\n候选表达式日志: {top_log_path}")
    if saved_paths:
        print(f"已保存 {len(saved_paths)} 个因子文件")


if __name__ == "__main__":
    main()
