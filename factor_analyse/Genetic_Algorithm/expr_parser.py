"""
解析遗传算法表达式字符串为 Node 对象
支持格式: terminal, unary(expr), op[param](expr), signedpower(expr,num), scale(expr,num), rank(expr), binary(a,b)
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from expr import (
    Binary,
    CSRank,
    Delay,
    Delta,
    Node,
    RollingUnary,
    Scale,
    SignedPower,
    Terminal,
    Unary,
    safe_div,
    safe_log,
    safe_sqrt,
)


def _find_matching_paren(s: str, start: int) -> int:
    """找到与 start 处 '(' 匹配的 ')' 位置"""
    count = 1
    i = start + 1
    while i < len(s) and count > 0:
        if s[i] == "(":
            count += 1
        elif s[i] == ")":
            count -= 1
        i += 1
    return i - 1 if count == 0 else -1


def _split_top_level_args(s: str) -> list[str]:
    """按顶层逗号分割参数，正确处理嵌套括号"""
    parts = []
    start = 0
    depth = 0
    for i, c in enumerate(s):
        if c == "(" or c == "[":
            depth += 1
        elif c == ")" or c == "]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return parts


def parse_expr(s: str) -> Node:
    """
    解析表达式字符串为 Node
    支持: terminal, neg(x), abs(x), min[60](x), delay[1](x), delta[5](x),
         signedpower(x,0.5), scale(x,1), rank(x), sub(a,b) 等
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty expression")

    # Terminal: 无括号、无方括号的标识符
    if "(" not in s and "[" not in s:
        return Terminal(s)

    # op[param](child) - RollingUnary, Delay, Delta
    m = re.match(r"^(\w+)\[(\d+)\]\((.+)\)\s*$", s)
    if m:
        op_name = m.group(1)
        param = int(m.group(2))
        rest = m.group(3)
        # 找到与第一个 ( 匹配的 )，提取 inner
        lp = s.index("](") + 2  # 即 op[param]( 之后的位置
        rp = _find_matching_paren(s, lp - 1)  # lp-1 是 ( 的位置
        inner = s[lp : rp]
        child = parse_expr(inner)

        if op_name == "delay":
            return Delay(param, child)
        if op_name == "delta":
            return Delta(param, child)
        if op_name in {"min", "max", "mean", "std", "sum", "product", "zscore",
                       "ts_rank", "ts_argmax", "ts_argmin", "decay_linear"}:
            return RollingUnary(op_name, param, child)
        raise ValueError(f"Unknown op[{param}]: {op_name}")

    # unary(child) - neg, abs, log, sqrt, sign
    unary_ops = {
        "neg": lambda x: -x,
        "abs": lambda x: x.abs(),
        "log": safe_log,
        "sqrt": safe_sqrt,
        "sign": lambda x: np.sign(x),
    }
    for op_name, op_fn in unary_ops.items():
        if s.startswith(op_name + "(") and s.endswith(")"):
            inner = s[len(op_name) + 1 : -1]
            child = parse_expr(inner)
            return Unary(op_name, op_fn, child)

    # signedpower(child, num)
    if s.startswith("signedpower(") and s.endswith(")"):
        args = _split_top_level_args(s[len("signedpower(") : -1])
        if len(args) == 2:
            child = parse_expr(args[0])
            a_str = args[1].replace("_", ".")  # 0_5 -> 0.5
            a = float(a_str)
            return SignedPower(child, a)

    # scale(child, num)
    if s.startswith("scale(") and s.endswith(")"):
        args = _split_top_level_args(s[len("scale(") : -1])
        if len(args) == 2:
            child = parse_expr(args[0])
            a = float(args[1])
            return Scale(child, a)

    # rank(child)
    if s.startswith("rank(") and s.endswith(")"):
        inner = s[5:-1]
        child = parse_expr(inner)
        return CSRank(child)

    # binary(left, right): sub, add, mul, div, max, min
    binary_ops = {
        "sub": lambda a, b: a - b,
        "add": lambda a, b: a + b,
        "mul": lambda a, b: a * b,
        "div": safe_div,
        "max": lambda a, b: np.maximum(a, b),
        "min": lambda a, b: np.minimum(a, b),
    }
    for op_name, op_fn in binary_ops.items():
        if s.startswith(op_name + "(") and s.endswith(")"):
            args_str = s[len(op_name) + 1 : -1]
            args = _split_top_level_args(args_str)
            if len(args) == 2:
                left = parse_expr(args[0])
                right = parse_expr(args[1])
                return Binary(op_name, op_fn, left, right)

    raise ValueError(f"Cannot parse expression: {s[:80]}...")


def load_slug_to_expr_map(ga_log_path: str) -> dict[str, str]:
    """从 ga_top_expr_log.csv 加载 expr_slug -> expr 映射"""
    import csv
    slug_to_expr = {}
    try:
        with open(ga_log_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return {}
            # expr, expr_slug 为倒数第3、第2列（兼容列数不一致的行）
            for row in reader:
                if len(row) >= 13:
                    expr = str(row[-3]).strip().strip('"')
                    slug = str(row[-2]).strip()
                    if slug and expr and expr != "nan":
                        slug_to_expr[slug] = expr
    except Exception:
        pass
    return slug_to_expr
