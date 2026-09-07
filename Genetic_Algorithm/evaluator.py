"""Causal expression evaluation with masked ranks and bounded caching."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from . import operators
from .expression import Node, expression_hash, validate_tree
from .features import evaluate_terminal


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([str(x) for x in frame.index], default=str).encode())
    digest.update(json.dumps([str(x) for x in frame.columns]).encode())
    values = frame.astype("float64").to_numpy(copy=True)
    values[np.isnan(values)] = 0.0
    digest.update(np.ascontiguousarray(values).tobytes())
    digest.update(np.ascontiguousarray(frame.isna().to_numpy(dtype=np.uint8)).tobytes())
    return digest.hexdigest()


def _ctx_fingerprint(data_ctx: dict[str, pd.DataFrame]) -> str:
    supplied = data_ctx.get("fingerprint") if isinstance(data_ctx, dict) else None
    if supplied is not None and not isinstance(supplied, pd.DataFrame):
        return str(supplied)
    digest = hashlib.sha256()
    for name in sorted(key for key, value in data_ctx.items() if isinstance(value, pd.DataFrame)):
        digest.update(name.encode())
        digest.update(_frame_fingerprint(data_ctx[name]).encode())
    return digest.hexdigest()


def _operator_source_hash() -> str:
    return hashlib.sha256(inspect.getsource(operators).encode()).hexdigest()


def _eligible_fingerprint(eligible: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(eligible.shape)).encode())
    digest.update(json.dumps([str(x) for x in eligible.index], default=str).encode())
    digest.update(json.dumps([str(x) for x in eligible.columns], default=str).encode())
    values = eligible.astype(bool).to_numpy(dtype=np.uint8)
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _apply(node: Node, data_ctx: dict[str, pd.DataFrame], eligible: pd.DataFrame) -> pd.DataFrame:
    if node.op not in operators.OPERATORS:
        return evaluate_terminal(node.field or node.op, data_ctx)
    children = [_apply(child, data_ctx, eligible) for child in node.children]
    if node.op == "rank":
        return operators.cross_sectional_rank(children[0], eligible)
    if node.op in {"lag", "delta", "rolling_mean", "rolling_std", "rolling_min", "rolling_max"}:
        return operators.OPERATORS[node.op](children[0], node.window)
    if node.op == "rolling_corr":
        return operators.rolling_correlation(children[0], children[1], node.window)
    return operators.OPERATORS[node.op](*children)


def _cache_put(cache: dict, key: tuple, value: pd.DataFrame, cache_bytes: int) -> None:
    size = int(value.to_numpy(copy=False).nbytes)
    if size > cache_bytes:
        return
    if isinstance(cache, OrderedDict):
        cache[key] = value
        cache.move_to_end(key)
    else:
        cache[key] = value
    total = sum(int(item.to_numpy(copy=False).nbytes) for item in cache.values() if isinstance(item, pd.DataFrame))
    while total > cache_bytes and cache:
        oldest = next(iter(cache))
        removed = cache.pop(oldest)
        if isinstance(removed, pd.DataFrame):
            total -= int(removed.to_numpy(copy=False).nbytes)


def evaluate_tree(
    node: Node,
    data_ctx: dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
    cache: dict | None = None,
    *,
    cache_bytes: int = 268435456,
) -> pd.DataFrame:
    validate_tree(node, {"max_depth": 10_000, "max_nodes": 10_000, "max_history": 10_000_000})
    if cache_bytes < 0:
        raise ValueError("cache_bytes must be non-negative")
    if not isinstance(eligible, pd.DataFrame):
        raise TypeError("eligible must be a DataFrame")
    key = (
        expression_hash(node), _ctx_fingerprint(data_ctx), _operator_source_hash(),
        _eligible_fingerprint(eligible),
    )
    if cache is not None and key in cache:
        value = cache.pop(key)
        cache[key] = value
        return value.copy()
    result = _apply(node, data_ctx, eligible).astype("float64")
    result = result.reindex(index=eligible.index, columns=eligible.columns)
    result = result.where(eligible.astype(bool))
    if cache is not None:
        _cache_put(cache, key, result.copy(), cache_bytes)
    return result


__all__ = ["evaluate_tree"]
