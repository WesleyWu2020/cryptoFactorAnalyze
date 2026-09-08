"""Export GP candidate expressions as standalone regular factor modules.

An export renders the candidate's canonical AST as a literal dictionary into
a ``factor_common`` regular factor module whose ``calc_factor`` reconstructs
and revalidates the tree, then delegates to
``Genetic_Algorithm.evaluator.evaluate_tree`` with the point-in-time
eligibility matrix injected by the loader contract (``context_eligible``).
The module carries no runs-directory dependency and no training labels.

Exports are immutable: an existing file is never overwritten. A name reuse is
accepted only when the embedded full expression hash and the baked-in
direction both match exactly, so prefix collisions between different
expressions are detected against the full hashes and a flipped training
direction can never silently reuse a module. Alongside the module a manifest
records the wrapper hash and the
content hashes of every transitive runtime module (``expression``,
``evaluator``, ``operators``, ``features``), because the wrapper source alone
is not a complete fingerprint of the formula implementation.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import read_verified_manifest, write_artifact
from .expression import (
    Node,
    canonical_tree,
    expression_hash,
    history_days,
    required_fields,
    validate_node_attributes,
)

_IDENTIFIER_PREFIX_CHARS = 16
_HASH_MARKER_RE = re.compile(r'^_EXPRESSION_HASH = "([0-9a-f]{64})"$', re.MULTILINE)
_DIRECTION_MARKER_RE = re.compile(r'"factor_direction": (-?1)\b')
_AST_KEYS = {"op", "field", "window", "children"}
# Transitive runtime closure of the exported calc_factor: evaluator imports
# expression, operators, and features; features imports operators.
_RUNTIME_MODULES = ("expression", "evaluator", "operators", "features")


@dataclass(frozen=True)
class ExportResult:
    """Identity and provenance of one exported factor module."""

    identifier: str
    path: Path
    manifest_path: Path
    expression_hash: str
    runtime_hashes: dict[str, str]


def _identifier(full_hash: str) -> str:
    return f"GP_{full_hash[:_IDENTIFIER_PREFIX_CHARS]}"


def _validate_direction(direction: Any) -> int:
    if isinstance(direction, bool) or not isinstance(direction, int):
        raise TypeError("direction must be an integer in {-1, 1}")
    if direction not in (-1, 1):
        raise ValueError("direction must be an integer in {-1, 1}")
    return direction


def _node_from_payload(payload: Mapping[str, Any]) -> Node:
    if not isinstance(payload, Mapping) or set(payload) != _AST_KEYS:
        raise ValueError("candidate AST payload is malformed")
    op, field, window, children = (
        payload["op"], payload["field"], payload["window"], payload["children"]
    )
    if not isinstance(op, str) or (field is not None and not isinstance(field, str)):
        raise ValueError("candidate AST payload is malformed")
    if window is not None and type(window) is not int:
        raise ValueError("candidate AST payload is malformed")
    if not isinstance(children, (list, tuple)):
        raise ValueError("candidate AST payload is malformed")
    return Node(
        op,
        tuple(_node_from_payload(child) for child in children),
        field,
        window,
    )


def _candidate_tree(candidate: Any) -> Node:
    if isinstance(candidate, Mapping):
        payload = candidate.get("ast")
        if payload is None:
            payload = candidate.get("tree")
        if isinstance(payload, Node):
            return payload
        if payload is not None:
            return _node_from_payload(payload)
        raise TypeError("candidate mapping requires an 'ast' payload")
    tree = getattr(candidate, "tree", None)
    if not isinstance(tree, Node):
        raise TypeError("candidate must provide a Node tree")
    return tree


def _candidate_expression_id(candidate: Any) -> str | None:
    if isinstance(candidate, Mapping):
        value = candidate.get("expression_id")
    else:
        value = getattr(candidate, "expression_id", None)
    return None if value is None else str(value)


def _ast_payload(node: Node) -> dict[str, Any]:
    return {
        "op": node.op,
        "field": node.field,
        "window": node.window,
        "children": [_ast_payload(child) for child in node.children],
    }


def _runtime_module_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    return {
        f"Genetic_Algorithm/{name}.py": hashlib.sha256(
            (package_dir / f"{name}.py").read_bytes()
        ).hexdigest()
        for name in _RUNTIME_MODULES
    }


def _existing_expression_hash(target: Path) -> str | None:
    source = target.read_text(encoding="utf-8")
    match = _HASH_MARKER_RE.search(source)
    return match.group(1) if match else None


def _existing_factor_direction(target: Path) -> int | None:
    source = target.read_text(encoding="utf-8")
    match = _DIRECTION_MARKER_RE.search(source)
    return int(match.group(1)) if match else None


def _render_module(
    identifier: str,
    full_hash: str,
    ast_payload: dict[str, Any],
    data_needed: list[str],
    warmup: int,
    direction: int,
) -> str:
    description = f"GP export {identifier} (expression {full_hash})"
    return f'''"""Exported GP factor {identifier}.

Generated by Genetic_Algorithm.export.export_factor. The literal ``_AST``
dictionary below is the complete formula; evaluation is delegated to
``Genetic_Algorithm.evaluator.evaluate_tree`` with the point-in-time
eligibility context injected by factor_common (``context_eligible``).
"""

from Genetic_Algorithm.evaluator import evaluate_tree
from Genetic_Algorithm.expression import (
    Node,
    expression_hash,
    validate_node_attributes,
)

TYPE = "regular"

META = {{"factor_name": "{identifier}", "author": "genetic_algorithm",
        "level": "daily", "category": "gp",
        "description": {description!r}}}

_EXPRESSION_HASH = "{full_hash}"

_AST = {ast_payload!r}

SETTING = {{"data_needed": {data_needed!r}, "universe": "historical_top50",
           "warmup_bars": {warmup}, "preprocessing": "none",
           "params": {{}}, "factor_direction": {direction},
           "context_eligible": True}}


def _build_node(payload):
    node = Node(
        payload["op"],
        tuple(_build_node(child) for child in payload["children"]),
        field=payload["field"],
        window=payload["window"],
    )
    validate_node_attributes(node)
    return node


def calc_factor(data_ctx):
    tree = _build_node(_AST)
    if expression_hash(tree) != _EXPRESSION_HASH:
        raise ValueError(
            "exported AST does not match the recorded expression hash"
        )
    return evaluate_tree(tree, data_ctx, data_ctx["__eligible__"])
'''


def _write_manifest(
    manifest_path: Path,
    *,
    identifier: str,
    full_hash: str,
    ast_payload: dict[str, Any],
    direction: int,
    data_needed: list[str],
    warmup: int,
    wrapper_sha256: str,
    runtime_hashes: dict[str, str],
) -> None:
    write_artifact(
        manifest_path,
        {
            "export_version": 1,
            "identifier": identifier,
            "expression_hash": full_hash,
            "ast": ast_payload,
            "factor_direction": direction,
            "data_needed": data_needed,
            "warmup_bars": warmup,
            "preprocessing": "none",
            "universe": "historical_top50",
            "context_eligible": True,
            "wrapper_sha256": wrapper_sha256,
            "runtime_module_hashes": runtime_hashes,
        },
        immutable=True,
    )


def export_factor(candidate: Any, destination: str | Path, *, direction: int) -> ExportResult:
    """Export one candidate as a regular factor module under ``destination``.

    ``direction`` is the fixed training direction (1 or -1); it is baked into
    the module unchanged and never re-derived here. Existing exports are
    reused only when the embedded full expression hash and the baked-in
    direction both match; a different expression at the same path raises
    ``FileExistsError`` and a direction mismatch raises ``ValueError``.
    """
    direction = _validate_direction(direction)
    tree = canonical_tree(_candidate_tree(candidate))
    validate_node_attributes(tree)
    full_hash = expression_hash(tree)
    declared = _candidate_expression_id(candidate)
    if declared is not None and declared != full_hash:
        raise ValueError(
            f"candidate expression_id {declared!r} does not match the "
            f"canonical expression hash {full_hash}"
        )
    identifier = _identifier(full_hash)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{identifier}.py"
    manifest_path = destination / f"{identifier}.export_manifest.json"
    runtime_hashes = _runtime_module_hashes()
    ast_payload = _ast_payload(tree)
    data_needed = sorted(required_fields(tree))
    warmup = history_days(tree)

    if os.path.lexists(target):
        existing = _existing_expression_hash(target)
        if existing is None:
            raise FileExistsError(
                f"refusing to overwrite a foreign file: {target}"
            )
        if existing != full_hash:
            raise FileExistsError(
                f"export identifier collision at {target}: {identifier} is "
                f"shared with a different expression {existing}"
            )
        # The expression hash covers only the tree; direction is part of the
        # reuse key, or a caller with the opposite direction would silently
        # reuse a module with the wrong baked-in factor_direction.
        existing_direction = _existing_factor_direction(target)
        if existing_direction is None:
            raise FileExistsError(
                f"refusing to reuse an export without a direction marker: {target}"
            )
        if existing_direction != direction:
            raise ValueError(
                f"export direction mismatch at {target}: existing export "
                f"fixes factor_direction={existing_direction}, requested "
                f"{direction}"
            )
        if manifest_path.exists():
            manifest = read_verified_manifest(manifest_path)
            if manifest.get("expression_hash") != full_hash:
                raise FileExistsError(
                    f"export manifest collision at {manifest_path}: recorded "
                    f"expression {manifest.get('expression_hash')} differs "
                    f"from {full_hash}"
                )
            if manifest.get("factor_direction") != direction:
                raise ValueError(
                    f"export manifest direction mismatch at {manifest_path}: "
                    f"recorded factor_direction="
                    f"{manifest.get('factor_direction')}, requested {direction}"
                )
        else:
            _write_manifest(
                manifest_path,
                identifier=identifier,
                full_hash=full_hash,
                ast_payload=ast_payload,
                direction=direction,
                data_needed=data_needed,
                warmup=warmup,
                wrapper_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                runtime_hashes=runtime_hashes,
            )
        return ExportResult(identifier, target, manifest_path, full_hash, runtime_hashes)

    source = _render_module(
        identifier, full_hash, ast_payload, data_needed, warmup, direction
    )
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(source)
    except FileExistsError:
        # Lost a creation race: re-enter the collision/idempotency path.
        return export_factor(candidate, destination, direction=direction)
    _write_manifest(
        manifest_path,
        identifier=identifier,
        full_hash=full_hash,
        ast_payload=ast_payload,
        direction=direction,
        data_needed=data_needed,
        warmup=warmup,
        wrapper_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        runtime_hashes=runtime_hashes,
    )
    return ExportResult(identifier, target, manifest_path, full_hash, runtime_hashes)


__all__ = ["ExportResult", "export_factor"]
