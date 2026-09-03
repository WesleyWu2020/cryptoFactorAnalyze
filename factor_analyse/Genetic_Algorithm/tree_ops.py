from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from expr import Node


Path = Tuple[int, ...]


def iter_paths(root: Node) -> List[Tuple[Path, Node]]:
    out: List[Tuple[Path, Node]] = []

    def rec(node: Node, path: Path):
        out.append((path, node))
        for idx, ch in enumerate(node.children()):
            rec(ch, path + (idx,))

    rec(root, ())
    return out


def get_subtree(root: Node, path: Path) -> Node:
    node = root
    for idx in path:
        node = node.children()[idx]
    return node


def replace_subtree(root: Node, path: Path, new_subtree: Node) -> Node:
    if path == ():
        return new_subtree

    # rebuild from bottom up
    parents: List[Tuple[Node, int]] = []
    node = root
    for idx in path[:-1]:
        parents.append((node, idx))
        node = node.children()[idx]

    last_idx = path[-1]
    rebuilt = node.with_child(last_idx, new_subtree)
    for parent, idx in reversed(parents):
        rebuilt = parent.with_child(idx, rebuilt)
    return rebuilt


def random_path(root: Node, include_root: bool = True) -> Path:
    paths = iter_paths(root)
    if not include_root:
        paths = [(p, n) for p, n in paths if p != ()]
    if not paths:
        return ()
    return random.choice(paths)[0]


def crossover_subtree(a: Node, b: Node) -> Tuple[Node, Node]:
    pa = random_path(a, include_root=True)
    pb = random_path(b, include_root=True)
    sa = get_subtree(a, pa)
    sb = get_subtree(b, pb)
    return replace_subtree(a, pa, sb), replace_subtree(b, pb, sa)
