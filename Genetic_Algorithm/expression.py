"""Immutable expression trees and their causal metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .features import TERMINAL_DEPENDENCIES, TERMINAL_FIELDS, TERMINAL_HISTORY
from .operators import (
    COMMUTATIVE_OPERATORS, LAG_STYLE_OPERATORS, MIN_OPERATOR_WINDOW,
    MIXED_DIMENSION_OPERATORS, OPERATORS, OPERATOR_ARITY,
    PRESERVE_DIMENSION_OPERATORS, RATIO_DIMENSION_OPERATORS,
    ROLLING_OPERATORS, TERMINAL_DIMENSIONS, WINDOW_OPERATORS,
    combine_dimensions,
)


@dataclass(frozen=True)
class Node:
    op: str
    children: tuple["Node", ...] = ()
    field: str | None = None
    window: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))


def _config_value(config: Any, name: str, default: int) -> int:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _stats(node: Node, depth: int = 0) -> tuple[int, int, int, set[str]]:
    validate_node_attributes(node)
    if node.op not in OPERATOR_ARITY:
        history = TERMINAL_HISTORY.get(node.field or node.op, 0)
        terminal = node.field or node.op
        fields = set(TERMINAL_DEPENDENCIES.get(terminal, {terminal}))
        return depth, 1, history, fields
    child_stats = [_stats(child, depth + 1) for child in node.children]
    max_depth = max((item[0] for item in child_stats), default=depth)
    count = 1 + sum(item[1] for item in child_stats)
    history = max((item[2] for item in child_stats), default=0)
    if node.op in ROLLING_OPERATORS:
        history += node.window - 1
    elif node.op in LAG_STYLE_OPERATORS:
        history += node.window
    fields = set().union(*(item[3] for item in child_stats))
    return max_depth, count, history, fields


def validate_node_attributes(node: Node) -> None:
    if not isinstance(node, Node):
        raise TypeError("tree must contain Node instances")
    if node.op in TERMINAL_FIELDS:
        if node.field is not None and node.field != node.op:
            raise ValueError(f"terminal {node.op} field must match op")
        if node.children or node.window is not None:
            raise ValueError("terminal cannot have children or window")
        return
    if node.op not in OPERATORS or node.op not in OPERATOR_ARITY:
        raise ValueError(f"unknown op: {node.op}")
    if len(node.children) != OPERATOR_ARITY[node.op]:
        raise ValueError(f"operator {node.op} requires arity {OPERATOR_ARITY[node.op]}")
    if node.field is not None:
        raise ValueError("operator cannot have field")
    if node.op in WINDOW_OPERATORS:
        if type(node.window) is not int or node.window <= 0:
            raise ValueError("window must be positive")
        minimum = MIN_OPERATOR_WINDOW.get(node.op)
        if minimum is not None and node.window < minimum:
            raise ValueError(
                f"operator {node.op} requires window >= {minimum}, got {node.window}"
            )
    elif node.window is not None:
        raise ValueError("non-window operator cannot have window")


def expression_dimension(node: Node) -> str:
    """Infer the dimension (量纲) of a tree bottom-up; illegal combos raise."""
    validate_node_attributes(node)
    if node.op not in OPERATOR_ARITY:
        field = node.field or node.op
        if field not in TERMINAL_DIMENSIONS:
            raise ValueError(f"unknown terminal dimension: {field}")
        return TERMINAL_DIMENSIONS[field]
    child_dimensions = [expression_dimension(child) for child in node.children]
    if node.op in PRESERVE_DIMENSION_OPERATORS:
        return child_dimensions[0]
    if node.op in RATIO_DIMENSION_OPERATORS:
        return "ratio"
    if node.op in MIXED_DIMENSION_OPERATORS:
        return "mixed"
    return combine_dimensions(node.op, child_dimensions[0], child_dimensions[1])


def validate_tree(node: Node, config: Any) -> None:
    depth, count, history, fields = _stats(node)
    expression_dimension(node)
    expression_value_domain(node)
    max_depth = _config_value(config, "max_depth", 4)
    max_nodes = _config_value(config, "max_nodes", 15)
    max_history = _config_value(config, "max_history", 180)
    if depth > max_depth:
        raise ValueError("tree exceeds max depth")
    if count > max_nodes:
        raise ValueError("tree exceeds max nodes")
    if history > max_history:
        raise ValueError("tree exceeds max history")
    if any(field not in TERMINAL_FIELDS for field in fields):
        raise ValueError("unknown field")


def expression_value_domain(node: Node) -> str:
    """Conservative sign inference, distinct from physical dimensions."""
    domains = [expression_value_domain(child) for child in node.children]
    if node.op == "rolling_hit_rate":
        if domains[0] in {"positive", "negative", "nonpositive", "zero"}:
            raise ValueError("degenerate rolling_hit_rate: input has a fixed positive/nonpositive sign")
        return "nonnegative"
    if node.op in {"rank", "ts_rank"}:
        return "positive"
    if node.op in {"abs", "rolling_std", "rolling_mad", "rolling_iqr",
                   "rolling_downside_std", "rolling_r2", "efficiency_ratio", "rolling_cv"}:
        return "nonnegative"
    if node.op == "negate":
        return {"positive": "negative", "negative": "positive",
                "nonnegative": "nonpositive", "nonpositive": "nonnegative",
                "zero": "zero"}.get(domains[0], "unknown")
    if node.op in {"lag", "rolling_mean", "rolling_min", "rolling_max", "rolling_median"}:
        return domains[0]
    return "unknown"


def canonical_tree(node: Node) -> Node:
    validate_node_attributes(node)
    children = tuple(canonical_tree(child) for child in node.children)
    if node.op in COMMUTATIVE_OPERATORS:
        children = tuple(sorted(children, key=_canonical_key))
    return Node(node.op, children, node.field, node.window)


def _canonical_key(node: Node) -> str:
    return json.dumps(_canonical_payload(node), sort_keys=True, separators=(",", ":"))


def _canonical_payload(node: Node) -> dict[str, Any]:
    return {
        "op": node.op,
        "field": node.field,
        "window": node.window,
        "children": [_canonical_payload(child) for child in node.children],
    }


def expression_hash(node: Node) -> str:
    payload = _canonical_key(canonical_tree(node)).encode()
    return hashlib.sha256(payload).hexdigest()


def required_fields(node: Node) -> set[str]:
    return _stats(node)[3]


def history_days(node: Node) -> int:
    return _stats(node)[2]


def node_count(node: Node) -> int:
    return _stats(node)[1]


__all__ = [
    "Node", "validate_tree", "canonical_tree", "expression_hash", "required_fields",
    "history_days", "node_count", "expression_dimension",
]
