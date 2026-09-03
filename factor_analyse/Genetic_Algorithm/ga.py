from __future__ import annotations

import random
from dataclasses import dataclass
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from expr import Node
from factory import random_tree
from fitness import FitnessResult, evaluate_expression
from tree_ops import crossover_subtree, random_path, replace_subtree


@dataclass
class GAConfig:
    population: int = 80
    generations: int = 30
    max_depth: int = 5
    max_nodes: int = 35
    tournament_k: int = 5
    crossover_p: float = 0.6
    mutation_p: float = 0.3
    elitism: int = 5
    seed: int = 7

    group_n: int = 5
    ic_weight: float = 0.6
    ls_weight: float = 0.4
    complexity_penalty: float = 0.002
    missing_penalty: float = 1.0

    # quality gates (shared by train/valid evaluation)
    min_rows: int = 8000
    min_days: int = 60

    # parallelism
    # NOTE: we use threads (not processes) because expression trees contain lambdas
    # which are not picklable by default.
    # 0 => auto (use os.cpu_count())
    n_jobs: int = 1

    # how many unique candidates (by expr) to evaluate on valid each generation
    # 0 => auto (keep current behavior)
    valid_eval_k: int = 0


def _auto_n_jobs(n_jobs: int) -> int:
    if n_jobs is None:
        return 1
    if int(n_jobs) == 0:
        return max(1, int(os.cpu_count() or 1))
    return max(1, int(n_jobs))


def _eval_many(
    nodes: Iterable[Node],
    *,
    df: pd.DataFrame,
    available_tokens_by_date: Dict,
    rebalance_period: int,
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
    cfg: GAConfig,
    cache: Dict[str, FitnessResult],
) -> None:
    # Only evaluate unique expr that are not in cache.
    uniq: Dict[str, Node] = {}
    for n in nodes:
        k = n.to_str()
        if k not in cache and k not in uniq:
            uniq[k] = n
    if not uniq:
        return

    def _one(n: Node) -> Tuple[str, FitnessResult]:
        k = n.to_str()
        r = evaluate_expression(
            df=df,
            expr=n,
            available_tokens_by_date=available_tokens_by_date,
            rebalance_period=rebalance_period,
            train_date_range=date_range,
            group_n=cfg.group_n,
            ic_weight=cfg.ic_weight,
            ls_weight=cfg.ls_weight,
            complexity_penalty=cfg.complexity_penalty,
            missing_penalty=cfg.missing_penalty,
            min_rows=cfg.min_rows,
            min_days=cfg.min_days,
        )
        return k, r

    n_jobs = _auto_n_jobs(cfg.n_jobs)
    if n_jobs <= 1:
        for n in uniq.values():
            k, r = _one(n)
            cache[k] = r
        return

    # ThreadPool avoids pickling Node (it may contain lambdas).
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_one, n) for n in uniq.values()]
        for fut in as_completed(futures):
            k, r = fut.result()
            cache[k] = r


def tournament_select(scored: List[Tuple[Node, FitnessResult]], k: int) -> Node:
    cand = random.sample(scored, k)
    cand.sort(key=lambda x: x[1].fitness, reverse=True)
    return cand[0][0]


def _is_valid(ind: Node, max_nodes: int) -> bool:
    return ind.size() <= max_nodes


def mutate(ind: Node, max_depth: int, max_nodes: int) -> Node:
    # 子树变异：随机位置替换为新随机树
    for _ in range(5):
        p = random_path(ind, include_root=True)
        new_sub = random_tree(max_depth=max_depth)
        child = replace_subtree(ind, p, new_sub)
        if _is_valid(child, max_nodes):
            return child
    return ind


def crossover(a: Node, b: Node, max_nodes: int) -> Tuple[Node, Node]:
    for _ in range(5):
        c1, c2 = crossover_subtree(a, b)
        if _is_valid(c1, max_nodes) and _is_valid(c2, max_nodes):
            return c1, c2
    return a, b


def split_train_valid_dates(
    all_dates: List[pd.Timestamp],
    train_ratio: float = 0.7,
) -> Tuple[Tuple[pd.Timestamp, pd.Timestamp], Tuple[pd.Timestamp, pd.Timestamp]]:
    ds = sorted(pd.to_datetime(all_dates))
    if len(ds) < 10:
        raise ValueError("Not enough dates to split train/valid")
    cut = int(len(ds) * train_ratio)
    train = (ds[0], ds[cut - 1])
    valid = (ds[cut], ds[-1])
    return train, valid


def run_ga(
    df: pd.DataFrame,
    available_tokens_by_date: Dict,
    rebalance_period: int,
    cfg: GAConfig,
    train_range: Tuple[pd.Timestamp, pd.Timestamp],
    valid_range: Tuple[pd.Timestamp, pd.Timestamp],
) -> Tuple[Node, FitnessResult, Node, FitnessResult, List[Tuple[Node, FitnessResult]]]:
    random.seed(cfg.seed)

    pop: List[Node] = []
    while len(pop) < cfg.population:
        ind = random_tree(cfg.max_depth)
        if _is_valid(ind, cfg.max_nodes):
            pop.append(ind)

    cache_train: Dict[str, FitnessResult] = {}
    cache_valid: Dict[str, FitnessResult] = {}
    # Keep a representative node for each expr string seen on valid side.
    valid_nodes: Dict[str, Node] = {}

    best_train_node: Optional[Node] = None
    best_valid_node: Optional[Node] = None
    best_train = FitnessResult(-1e9, np.nan, np.nan, np.nan, np.nan, 0, "")
    best_valid = FitnessResult(-1e9, np.nan, np.nan, np.nan, np.nan, 0, "")

    last_valid_ranked: List[Tuple[Node, FitnessResult]] = []

    for gen in range(cfg.generations):
        scored: List[Tuple[Node, FitnessResult]] = []
        _eval_many(
            pop,
            df=df,
            available_tokens_by_date=available_tokens_by_date,
            rebalance_period=rebalance_period,
            date_range=train_range,
            cfg=cfg,
            cache=cache_train,
        )
        for ind in pop:
            scored.append((ind, cache_train[ind.to_str()]))

        scored.sort(key=lambda x: x[1].fitness, reverse=True)
        cur_best_train = scored[0][1]

        # 记录 valid 上最优（用 train 排名前列 candidates 做 valid 评估）
        # 注意：population 里可能存在重复表达式；这里优先确保用于 valid 评估的候选是 unique expr。
        target_k = int(cfg.valid_eval_k) if int(cfg.valid_eval_k) > 0 else max(cfg.elitism * 4, 20)
        topk_unique: List[Node] = []
        seen = set()
        for ind, _ in scored:
            key = ind.to_str()
            if key in seen:
                continue
            seen.add(key)
            topk_unique.append(ind)
            if len(topk_unique) >= target_k:
                break

        _eval_many(
            topk_unique,
            df=df,
            available_tokens_by_date=available_tokens_by_date,
            rebalance_period=rebalance_period,
            date_range=valid_range,
            cfg=cfg,
            cache=cache_valid,
        )
        for ind in topk_unique:
            key = ind.to_str()
            if key not in valid_nodes:
                valid_nodes[key] = ind

        cur_best_valid = max((cache_valid[i.to_str()] for i in topk_unique), key=lambda r: r.fitness)

        # 保存本代 valid 上候选的排序列表（供 runner 导出 topK 因子）
        last_valid_ranked = [(ind, cache_valid[ind.to_str()]) for ind in topk_unique]
        last_valid_ranked.sort(key=lambda x: x[1].fitness, reverse=True)

        if cur_best_train.fitness > best_train.fitness:
            best_train = cur_best_train
            best_train_node = scored[0][0]
        if cur_best_valid.fitness > best_valid.fitness:
            best_valid = cur_best_valid
            # 找到对应的 node
            for ind in topk_unique:
                if ind.to_str() == cur_best_valid.expr:
                    best_valid_node = ind
                    break

        print(
            f"[Gen {gen+1:02d}/{cfg.generations}] "
            f"train_best={cur_best_train.fitness:.4f} (ICIR={cur_best_train.ic_ir:.3f}, LSIR={cur_best_train.ls_ir:.3f}, rows={cur_best_train.n_rows}) "
            f"valid_best={cur_best_valid.fitness:.4f} (ICIR={cur_best_valid.ic_ir:.3f}, LSIR={cur_best_valid.ls_ir:.3f})"
        )
        print(f"  train_expr: {cur_best_train.expr}")
        print(f"  valid_expr: {cur_best_valid.expr}")

        # 下一代：elitism + tournament + crossover/mutation
        next_pop: List[Node] = [scored[i][0] for i in range(min(cfg.elitism, len(scored)))]

        while len(next_pop) < cfg.population:
            p1 = tournament_select(scored, cfg.tournament_k)
            p2 = tournament_select(scored, cfg.tournament_k)

            if random.random() < cfg.crossover_p:
                c1, c2 = crossover(p1, p2, cfg.max_nodes)
            else:
                c1, c2 = p1, p2

            if random.random() < cfg.mutation_p:
                c1 = mutate(c1, cfg.max_depth, cfg.max_nodes)
            if random.random() < cfg.mutation_p:
                c2 = mutate(c2, cfg.max_depth, cfg.max_nodes)

            next_pop.append(c1)
            if len(next_pop) < cfg.population:
                next_pop.append(c2)

        pop = next_pop[: cfg.population]

    if best_train_node is None:
        best_train_node = pop[0]
    if best_valid_node is None:
        best_valid_node = best_train_node

    # Export candidates: use all valid evaluations across all generations.
    all_valid_ranked: List[Tuple[Node, FitnessResult]] = [(valid_nodes[k], cache_valid[k]) for k in cache_valid.keys()]
    all_valid_ranked.sort(key=lambda x: x[1].fitness, reverse=True)
    return best_train_node, best_train, best_valid_node, best_valid, all_valid_ranked
