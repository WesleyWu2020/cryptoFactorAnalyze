"""Deterministic, bounded genetic search for causal expression trees."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from factor_common.labels import make_labels

from .evaluator import evaluate_tree
from .expression import Node, canonical_tree, expression_hash, node_count, validate_tree
from .features import TERMINAL_FIELDS
from .fitness import score_training
from .operators import OPERATOR_ARITY, WINDOW_OPERATORS


@dataclass(frozen=True)
class Candidate:
    tree: Node
    expression_id: str
    score: tuple[Any, ...]
    eligible: bool = True
    reasons: tuple[str, ...] = ()
    direction: int | None = None


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[Candidate, ...]
    generation_log: tuple[dict[str, Any], ...]
    evaluations: int
    values_by_id: dict[str, Any] = field(default_factory=dict)


class SearchFormationError(RuntimeError):
    """Raised when the structural population cannot be formed in its budget."""


class CandidateInvalidError(ValueError):
    """Raised when one candidate is invalid but search may try another."""


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _score(candidate: Candidate | Sequence[Any]) -> tuple[Any, ...]:
    values = candidate.score if isinstance(candidate, Candidate) else candidate
    return tuple(values)


def _validate_objective_vectors(candidates: Sequence[Candidate], context: str) -> int:
    """Require a nonempty, shared objective width at algorithm boundaries."""
    if not candidates:
        return 0
    width = len(_score(candidates[0]))
    if width == 0:
        raise ValueError(f"{context} objective vector length must be positive")
    for candidate in candidates[1:]:
        candidate_width = len(_score(candidate))
        if candidate_width != width:
            raise ValueError(
                f"{context} objective vector length mismatch: expected {width}, "
                f"got {candidate_width} for {candidate.expression_id}"
            )
    return width


def _objective_value(value: Any) -> float:
    """Map an objective to a comparison value without mutating the score."""
    if value is None:
        return float("-inf")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return numeric if np.isfinite(numeric) else float("-inf")


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    left_values = tuple(_objective_value(value) for value in left)
    right_values = tuple(_objective_value(value) for value in right)
    return all(a >= b for a, b in zip(left_values, right_values)) and any(a > b for a, b in zip(left_values, right_values))


def nondominated_sort(candidates: Iterable[Candidate]) -> tuple[tuple[Candidate, ...], ...]:
    """Return Pareto fronts for maximization objectives in deterministic order."""
    items = sorted(tuple(candidates), key=lambda item: (item.expression_id, node_count(item.tree)))
    _validate_objective_vectors(items, "nondominated sort")
    dominates: dict[int, list[int]] = {i: [] for i in range(len(items))}
    dominated_by = [0] * len(items)
    for i, left in enumerate(items):
        for j, right in enumerate(items):
            if i == j:
                continue
            if _dominates(_score(left), _score(right)):
                dominates[i].append(j)
            elif _dominates(_score(right), _score(left)):
                dominated_by[i] += 1
    current = [i for i, count in enumerate(dominated_by) if count == 0]
    fronts: list[tuple[Candidate, ...]] = []
    while current:
        front = tuple(sorted(
            (items[i] for i in current),
            key=lambda item: (node_count(item.tree), item.expression_id),
        ))
        fronts.append(front)
        next_front: list[int] = []
        for index in current:
            for child in dominates[index]:
                dominated_by[child] -= 1
                if dominated_by[child] == 0:
                    next_front.append(child)
        current = sorted(set(next_front), key=lambda i: items[i].expression_id)
    return tuple(fronts)


def crowding_distance(front: Sequence[Candidate]) -> dict[str, float]:
    """Calculate NSGA-II crowding distance for one front."""
    _validate_objective_vectors(tuple(front), "crowding distance")
    result = {candidate.expression_id: 0.0 for candidate in front}
    if len(front) <= 2:
        return {candidate.expression_id: float("inf") for candidate in front}
    width = len(_score(front[0]))
    for objective in range(width):
        ordered = sorted(front, key=lambda item: (_objective_value(_score(item)[objective]), item.expression_id))
        low = _objective_value(_score(ordered[0])[objective])
        high = _objective_value(_score(ordered[-1])[objective])
        result[ordered[0].expression_id] = float("inf")
        result[ordered[-1].expression_id] = float("inf")
        if high == low:
            continue
        for index in range(1, len(ordered) - 1):
            if result[ordered[index].expression_id] == float("inf"):
                continue
            previous = _objective_value(_score(ordered[index - 1])[objective])
            following = _objective_value(_score(ordered[index + 1])[objective])
            if not all(np.isfinite(value) for value in (previous, following, low, high)):
                continue
            contribution = (following - previous) / (high - low)
            if np.isfinite(contribution):
                result[ordered[index].expression_id] += contribution
    return result


def select_population(candidates: Iterable[Candidate], limit: int) -> tuple[Candidate, ...]:
    """Select at most ``limit`` candidates by fronts, crowding, complexity, hash."""
    if limit <= 0:
        return ()
    unique: dict[str, Candidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.expression_id, candidate)
    chosen: list[Candidate] = []
    for front in nondominated_sort(unique.values()):
        if len(chosen) + len(front) <= limit:
            chosen.extend(front)
            continue
        distances = crowding_distance(front)
        remaining = limit - len(chosen)
        chosen.extend(sorted(
            front,
            key=lambda item: (-distances[item.expression_id], node_count(item.tree), item.expression_id),
        )[:remaining])
        break
    return tuple(chosen)


def _exploratory_key(candidate: Candidate) -> tuple[Any, ...]:
    """Order exploratory candidates reproducibly, including incomplete scores."""
    objective_key = tuple(
        (value is None, -_objective_value(value)) for value in candidate.score
    )
    return (objective_key, node_count(candidate.tree), candidate.expression_id, candidate.reasons)


def select_exploratory(candidates: Iterable[Candidate], limit: int) -> tuple[Candidate, ...]:
    unique = {candidate.expression_id: candidate for candidate in candidates}
    return tuple(sorted(unique.values(), key=_exploratory_key)[:max(0, limit)])


def _tournament(pool: Sequence[Candidate], rng: np.random.Generator, config: Any) -> Candidate:
    """Pick a parent by nondominated rank, crowding, complexity, and hash."""
    fronts = nondominated_sort(pool)
    ranks = {candidate.expression_id: rank for rank, front in enumerate(fronts) for candidate in front}
    distances = {
        expression_id: distance
        for front in fronts
        for expression_id, distance in crowding_distance(front).items()
    }
    size = max(2, int(_value(config, "tournament_size", 2)))
    indices = rng.integers(0, len(pool), size=size)
    contenders = [pool[int(index)] for index in indices]
    return min(
        contenders,
        key=lambda candidate: (
            ranks[candidate.expression_id],
            -distances[candidate.expression_id],
            node_count(candidate.tree),
            candidate.expression_id,
        ),
    )


def _paths(node: Node, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    result = [prefix]
    for index, child in enumerate(node.children):
        result.extend(_paths(child, prefix + (index,)))
    return result


def _at_path(node: Node, path: tuple[int, ...]) -> Node:
    for index in path:
        node = node.children[index]
    return node


def _replace(node: Node, path: tuple[int, ...], replacement: Node) -> Node:
    if not path:
        return replacement
    index = path[0]
    children = list(node.children)
    children[index] = _replace(children[index], path[1:], replacement)
    return Node(node.op, tuple(children), node.field, node.window)


def _random_tree(rng: np.random.Generator, config: Any, depth: int = 0) -> Node:
    terminals = sorted(TERMINAL_FIELDS)
    max_depth = int(_value(config, "max_depth", 4))
    if depth >= max_depth or (depth > 0 and rng.random() < 0.35):
        return Node(str(rng.choice(terminals)))
    operators = sorted(OPERATOR_ARITY)
    op = str(rng.choice(operators))
    children = tuple(_random_tree(rng, config, depth + 1) for _ in range(OPERATOR_ARITY[op]))
    window = None
    if op in WINDOW_OPERATORS:
        values = tuple(_value(config, "windows", (3, 5, 10, 20, 40, 60)))
        if op in {"lag", "delta"}:
            values = tuple(_value(config, "lags", values))
        window = int(rng.choice(values))
    return Node(op, children, window=window)


def _variation(parent_a: Node, parent_b: Node, rng: np.random.Generator, config: Any) -> Node:
    crossover = float(_value(config, "crossover_probability", 0.6))
    mutation = float(_value(config, "mutation_probability", 0.3))
    copy = float(_value(config, "copy_probability", 0.1))
    draw = float(rng.random())
    if draw < crossover:
        parent_a_paths = _paths(parent_a)
        parent_b_paths = _paths(parent_b)
        target = parent_a_paths[int(rng.integers(len(parent_a_paths)))]
        source = parent_b_paths[int(rng.integers(len(parent_b_paths)))]
        return _replace(parent_a, target, _at_path(parent_b, source))
    if draw < crossover + mutation:
        parent_paths = _paths(parent_a)
        target = parent_paths[int(rng.integers(len(parent_paths)))]
        return _replace(parent_a, target, _random_tree(rng, config))
    if draw < crossover + mutation + copy:
        return parent_a
    raise ValueError("variation probabilities must cover the random draw")


def _training_evaluate(
    tree: Node, stage_data: Any, config: Any
) -> tuple[tuple[float, ...], bool, tuple[str, ...], Any]:
    """Evaluate one tree; forward labels are created only within this function."""
    custom = _value(config, "evaluate_candidate")
    opens = _value(stage_data, "opens")
    labels = make_labels(opens, int(_value(config, "hold_days", 1))) if opens is not None else None
    if callable(custom):
        raw = custom(tree, stage_data, labels, config)
        if isinstance(raw, dict):
            values = next(
                (raw[key] for key in ("values", "factor_values", "training_values") if key in raw),
                None,
            )
            return (
                tuple(raw["score"]),
                bool(raw.get("eligible", True)),
                tuple(raw.get("reasons", ())),
                values, raw.get("direction"),
            )
        return tuple(raw), True, (), None, None
    features = _value(stage_data, "features")
    eligible = _value(stage_data, "eligible")
    quality = _value(stage_data, "quality_eligible")
    values = evaluate_tree(tree, features, eligible, cache=_value(config, "cache"), cache_bytes=int(_value(config, "cache_bytes", 268435456)))
    values = values.loc[opens.index]
    score_config = dict(config) if isinstance(config, dict) else config.__dict__.copy()
    score_config["node_count"] = node_count(tree)
    score = score_training(values, labels, quality, score_config)
    return tuple(score.objective_vector), score.eligible, score.reasons, values, score.direction


def search(stage_data: Any, config: Any) -> SearchResult:
    """Run a deterministic GP search using one local random generator."""
    from .config import SearchConfig

    if isinstance(config, dict):
        config = SearchConfig(**config)
    elif not isinstance(config, SearchConfig):
        raise TypeError("search config must be a SearchConfig or mapping")
    rng = np.random.default_rng(int(_value(config, "seed", 42)))
    population_size = int(_value(config, "population", 200))
    generations = int(_value(config, "generations", 20))
    max_attempts = min(config.max_attempts, population_size * 50)
    cache: dict[str, tuple[tuple[Any, ...], bool, tuple[str, ...], Node, Any, int | None]] = {}
    values_by_id: dict[str, Any] = {}
    evaluations = 0
    objective_width: int | None = None
    logs: list[dict[str, Any]] = []
    supplied = list(_value(config, "initial_trees", ()))

    def evaluate_pool(trees: Iterable[Node], generation: int) -> tuple[list[Candidate], list[Candidate], dict[str, Any]]:
        nonlocal evaluations, objective_width
        attempted = 0
        unique = 0
        hits = 0
        reasons: Counter[str] = Counter()
        structural: dict[str, Candidate] = {}
        for tree in trees:
            attempted += 1
            try:
                canonical = canonical_tree(tree)
                validate_tree(canonical, config)
            except Exception as exc:
                reasons[type(exc).__name__ + ": " + str(exc)] += 1
                continue
            identifier = expression_hash(canonical)
            if identifier in structural:
                hits += 1
                continue
            if identifier in cache:
                score, eligible, failure_reasons, cached_tree, values, direction = cache[identifier]
                hits += 1
                canonical = cached_tree
            else:
                unique += 1
                evaluations += 1
                try:
                    score, eligible, failure_reasons, values, direction = _training_evaluate(canonical, stage_data, config)
                except CandidateInvalidError as exc:
                    reasons[type(exc).__name__ + ": " + str(exc)] += 1
                    continue
                cache[identifier] = (score, eligible, tuple(failure_reasons), canonical, values, direction)
            if values is not None:
                values_by_id[identifier] = {"values": values}
            score = tuple(score)
            if not score:
                raise ValueError("evaluator objective vector length must be positive")
            if objective_width is None:
                objective_width = len(score)
            elif len(score) != objective_width:
                raise ValueError(
                    "evaluator objective vector length mismatch: "
                    f"expected {objective_width}, got {len(score)} for {identifier}"
                )
            for reason in failure_reasons:
                reasons[str(reason)] += 1
            if score:
                structural[identifier] = Candidate(canonical, identifier, score, bool(eligible), tuple(failure_reasons), direction)
        valid = [candidate for candidate in structural.values() if candidate.eligible]
        exploratory = list(structural.values())
        return valid, exploratory, {
            "generation": generation,
            "attempted_trees": attempted,
            "unique_evaluations": unique,
            "cache_hits": hits,
            "invalid_reasons": dict(sorted(reasons.items())),
            "eligible_population": len(valid),
        }

    def fill(trees: list[Node], generation: int) -> tuple[list[Candidate], list[Candidate], dict[str, Any]]:
        attempts = 0
        all_valid: dict[str, Candidate] = {}
        all_exploratory: dict[str, Candidate] = {}
        aggregate = {"generation": generation, "attempted_trees": 0, "unique_evaluations": 0, "cache_hits": 0, "invalid_reasons": Counter(), "eligible_population": 0}
        pending = list(trees)
        while attempts < max_attempts and (
            len(all_valid) < population_size
            and (all_valid or len(all_exploratory) < population_size)
        ):
            if not pending:
                pending.append(_random_tree(rng, config))
            valid, exploratory, log = evaluate_pool(pending[:1], generation)
            pending = pending[1:]
            attempts += 1
            for candidate in valid:
                all_valid.setdefault(candidate.expression_id, candidate)
            for candidate in exploratory:
                all_exploratory.setdefault(candidate.expression_id, candidate)
            aggregate["attempted_trees"] += log["attempted_trees"]
            aggregate["unique_evaluations"] += log["unique_evaluations"]
            aggregate["cache_hits"] += log["cache_hits"]
            aggregate["invalid_reasons"].update(log["invalid_reasons"])
        aggregate["eligible_population"] = len(all_valid)
        aggregate["invalid_reasons"] = dict(sorted(aggregate["invalid_reasons"].items()))
        if len(all_valid) < population_size and (all_valid or len(all_exploratory) < population_size):
            raise SearchFormationError(
                f"could not form population: unique exploratory candidates "
                f"{len(all_exploratory)}/{population_size} after {attempts}/{max_attempts} attempts; "
                f"eligible candidates {len(all_valid)}/{population_size}; "
                f"reasons={aggregate['invalid_reasons']}"
            )
        return list(all_valid.values()), list(all_exploratory.values()), aggregate

    valid, exploratory, log = fill(supplied, 0)
    current = select_population(valid, population_size)
    exploratory_current = select_exploratory(exploratory, population_size)
    logs.append(log)
    for generation in range(1, generations):
        parents = current if current else exploratory_current
        offspring: list[Node] = []
        while len(offspring) < population_size:
            parent_a = _tournament(parents, rng, config)
            parent_b = _tournament(parents, rng, config)
            offspring.append(_variation(parent_a.tree, parent_b.tree, rng, config))
        offspring_valid, offspring_exploratory, log = fill(offspring, generation)
        merged_valid = select_population(tuple(current) + tuple(offspring_valid), population_size)
        merged_exploratory = select_exploratory(tuple(exploratory_current) + tuple(offspring_exploratory), population_size)
        current = merged_valid
        exploratory_current = merged_exploratory
        logs.append(log)
    retained_values = {
        candidate.expression_id: values_by_id[candidate.expression_id]
        for candidate in current
        if candidate.expression_id in values_by_id
    }
    return SearchResult(tuple(current), tuple(logs), evaluations, retained_values)


__all__ = [
    "Candidate", "CandidateInvalidError", "SearchResult", "SearchFormationError", "nondominated_sort",
    "crowding_distance", "select_population", "select_exploratory", "search",
]
