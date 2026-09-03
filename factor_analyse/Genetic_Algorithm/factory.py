from __future__ import annotations

import random
from typing import List

import numpy as np

from expr import (
    Binary,
    CSRank,
    Const,
    Delay,
    Delta,
    IndNeutralize,
    Node,
    RollingBinary,
    RollingUnary,
    Scale,
    SignedPower,
    Terminal,
    Unary,
    safe_div,
    safe_log,
    safe_sqrt,
)


# 基础 terminals（来自 kline csv header）
TERMINALS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades_count",
    "taker_buy_base",
    "taker_buy_quote",
    # derived features (由 evaluate 时预先在 gp 里计算)
    "ret1",
    "hl",
    "hl_pct",
    "oc_ret",
    "co_pct",
]

ROLL_WINDOWS = [3, 5, 10, 20, 30, 60]
DELAYS = [1, 2, 3, 5, 10]
POWERS = [0.5, 1.0, 2.0, 3.0]
GROUP_COLS = ["base_asset"]


def random_terminal() -> Node:
    return Terminal(random.choice(TERMINALS))


def random_const() -> Node:
    # 小常数有助于生成更稳定的表达式
    return Const(random.choice([0.5, 1.0, 2.0, -1.0, -0.5]))


def make_unary(child: Node) -> Node:
    ops = [
        ("abs", lambda x: x.abs()),
        ("log", safe_log),
        ("sqrt", safe_sqrt),
        ("neg", lambda x: -x),
        ("sign", lambda x: np.sign(x)),
    ]
    name, fn = random.choice(ops)
    return Unary(name, fn, child)


def make_binary(left: Node, right: Node) -> Node:
    ops = [
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
        ("div", safe_div),
        ("max", lambda a, b: np.maximum(a, b)),
        ("min", lambda a, b: np.minimum(a, b)),
    ]
    name, fn = random.choice(ops)
    return Binary(name, fn, left, right)


def make_rolling(child: Node) -> Node:
    op = random.choice([
        "mean",
        "std",
        "zscore",
        "sum",
        "min",
        "max",
        "product",
        "ts_rank",
        "ts_argmax",
        "ts_argmin",
        "decay_linear",
    ])
    w = random.choice(ROLL_WINDOWS)
    return RollingUnary(op, w, child)


def make_delay(child: Node) -> Node:
    return Delay(random.choice(DELAYS), child)


def make_delta(child: Node) -> Node:
    return Delta(random.choice(DELAYS), child)


def make_signedpower(child: Node) -> Node:
    return SignedPower(child, random.choice(POWERS))


def make_cs_rank(child: Node) -> Node:
    return CSRank(child)


def make_scale(child: Node) -> Node:
    return Scale(child, a=1.0)


def make_indneutralize(child: Node) -> Node:
    return IndNeutralize(child, random.choice(GROUP_COLS))


def make_rolling_binary(left: Node, right: Node) -> Node:
    op = random.choice(["correlation", "covariance"])
    w = random.choice(ROLL_WINDOWS)
    return RollingBinary(op, w, left, right)


def random_tree(max_depth: int, p_const: float = 0.15) -> Node:
    if max_depth <= 1:
        return random_const() if random.random() < p_const else random_terminal()

    r = random.random()
    if r < 0.30:
        return random_const() if random.random() < p_const else random_terminal()
    if r < 0.46:
        return make_unary(random_tree(max_depth - 1, p_const=p_const))
    if r < 0.66:
        return make_rolling(random_tree(max_depth - 1, p_const=p_const))
    if r < 0.74:
        return make_delay(random_tree(max_depth - 1, p_const=p_const))

    if r < 0.80:
        return make_delta(random_tree(max_depth - 1, p_const=p_const))

    if r < 0.86:
        return make_signedpower(random_tree(max_depth - 1, p_const=p_const))

    if r < 0.90:
        return make_cs_rank(random_tree(max_depth - 1, p_const=p_const))

    if r < 0.93:
        return make_scale(random_tree(max_depth - 1, p_const=p_const))

    if r < 0.96:
        return make_indneutralize(random_tree(max_depth - 1, p_const=p_const))

    if r < 0.985:
        left = random_tree(max_depth - 1, p_const=p_const)
        right = random_tree(max_depth - 1, p_const=p_const)
        return make_rolling_binary(left, right)

    left = random_tree(max_depth - 1, p_const=p_const)
    right = random_tree(max_depth - 1, p_const=p_const)
    return make_binary(left, right)
