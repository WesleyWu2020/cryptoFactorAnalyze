"""Deterministic, bounded genetic search for causal expression trees."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

import numpy as np

from factor_common.labels import make_labels

from .evaluator import evaluate_tree
from .expression import Node, canonical_tree, expression_dimension, expression_hash, node_count, validate_tree
from .features import TERMINAL_FIELDS, FEATURE_FAMILIES, expression_families
from .fitness import score_training
from .operators import MIN_OPERATOR_WINDOW, OPERATOR_ARITY, TERMINAL_DIMENSIONS, WINDOW_OPERATORS


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
    fitness_diagnostics: dict[str, Any] = field(default_factory=dict)
    research_diagnostics: dict[str, Any] = field(default_factory=dict)


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
    if _value(config, "fitness_mode") == "all_costs_sharpe":
        contenders = [pool[int(i)] for i in rng.integers(0, len(pool), size=max(2, config.tournament_size))]
        return min(contenders, key=_cost_key)
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


def _cost_key(candidate):
    return (not candidate.eligible, -_objective_value(candidate.score[0]),
            -_objective_value(candidate.score[1]), node_count(candidate.tree), candidate.expression_id)


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


def _random_tree_raw(rng: np.random.Generator, config: Any, depth: int = 0, family=None) -> Node:
    if family is None and _value(config, "family_diversity", False):
        family = str(rng.choice(sorted(FEATURE_FAMILIES)))
    terminals = sorted(FEATURE_FAMILIES[family] if family else TERMINAL_FIELDS)
    max_depth = int(_value(config, "max_depth", 4))
    if depth >= max_depth or (depth > 0 and rng.random() < 0.35):
        return Node(str(rng.choice(terminals)))
    operators = sorted(OPERATOR_ARITY)
    op = str(rng.choice(operators))
    children = tuple(_random_tree_raw(rng, config, depth + 1, family) for _ in range(OPERATOR_ARITY[op]))
    window = None
    if op in WINDOW_OPERATORS:
        values = tuple(_value(config, "windows", (3, 5, 10, 20, 40, 60)))
        if op in {"lag", "delta"}:
            values = tuple(_value(config, "lags", values))
        minimum = MIN_OPERATOR_WINDOW.get(op)
        if minimum is not None:
            values = tuple(value for value in values if value >= minimum)
        if not values:
            return Node(str(rng.choice(terminals)))
        window = int(rng.choice(values))
    return Node(op, children, window=window)


def _semantic_valid(tree, config):
    validate_tree(tree, config)
    if _value(config, "semantic_generation", False):
        for path in _paths(tree):
            if expression_dimension(_at_path(tree, path)) == "mixed":
                raise ValueError("semantic generation excludes mixed dimensions")


def _random_tree(rng: np.random.Generator, config: Any, depth: int = 0, family=None,
                 required_dimension=None) -> Node:
    if not _value(config, "semantic_generation", False) and required_dimension is None:
        return _random_tree_raw(rng, config, depth, family)
    if family is None and _value(config, "family_diversity", False):
        family = str(rng.choice(sorted(FEATURE_FAMILIES)))
    desired = required_dimension
    if desired is None and rng.random() < _value(config, "dimensionless_probability", 0.75):
        desired = "ratio"
    # Bounded rejection happens before costly factor evaluation. The existing
    # dimension algebra is authoritative; no second unit system is introduced.
    for _ in range(24):
        tree = _random_tree_raw(rng, config, depth, family)
        try:
            _semantic_valid(tree, config)
            if desired is None or expression_dimension(tree) == desired:
                return tree
        except ValueError:
            pass
    terminals = sorted(FEATURE_FAMILIES[family] if family else TERMINAL_FIELDS)
    matches = [name for name in terminals if desired is None or TERMINAL_DIMENSIONS[name] == desired]
    if not matches:
        matches = sorted(name for name in TERMINAL_FIELDS if TERMINAL_DIMENSIONS[name] == desired)
    if not matches:
        raise ValueError(f"no terminal fallback for dimension {desired}")
    return Node(str(rng.choice(matches)))


def _structured_mutation(parent, kind, rng, config):
    """One local edit, with bounded validation and unchanged-parent fallback."""
    paths = _paths(parent)
    for _ in range(24):
        path = paths[int(rng.integers(len(paths)))]
        node = _at_path(parent, path)
        if kind == "window":
            if node.window is None:
                continue
            windows = config.lags if node.op in {"lag", "delta"} else config.windows
            options = sorted(set(w for w in windows if w != node.window and w >= MIN_OPERATOR_WINDOW.get(node.op, 1)))
            # Adjacent configured values: local search rather than a new subtree.
            lower = [w for w in options if w < node.window]
            upper = [w for w in options if w > node.window]
            options = lower[-1:] + upper[:1]
            if not options:
                continue
            replacement = replace(node, window=int(rng.choice(options)))
        elif kind == "field":
            if node.children:
                continue
            source = node.field or node.op
            options = sorted({name for fields in FEATURE_FAMILIES.values() if source in fields
                              for name in fields if name != source
                              and TERMINAL_DIMENSIONS[name] == TERMINAL_DIMENSIONS[source]})
            if not options:
                continue
            replacement = Node(str(rng.choice(options)))
        elif kind == "prune":
            # Removal proposes a simpler hypothesis, not an assertion that an
            # arbitrary unary operator is mathematically redundant.
            if len(node.children) != 1 or expression_dimension(node) != expression_dimension(node.children[0]):
                continue
            replacement = node.children[0]
        elif kind == "subtree":
            try:
                replacement = _random_tree(rng, config, required_dimension=expression_dimension(node))
            except ValueError:
                continue
        else:
            raise ValueError(f"unknown mutation kind: {kind}")
        tree = _replace(parent, path, replacement)
        try:
            _semantic_valid(tree, config)
        except ValueError:
            continue
        if expression_hash(tree) != expression_hash(parent):
            return tree
    return parent


def _select_cost_population(candidates, size, config):
    ordered = sorted(candidates, key=_cost_key)
    if not _value(config, "family_diversity", False):
        return ordered[:size]
    buckets = {}
    for candidate in ordered:
        families = expression_families(candidate.tree)
        bucket = next(iter(families)) if len(families) == 1 else "mixed"
        buckets.setdefault(bucket, []).append(candidate)
    selected = []
    # Reserve exploration across available families; eligibility remains a
    # hard requirement when candidates leave search for validation.
    while len(selected) < size and buckets:
        for bucket in sorted(tuple(buckets)):
            selected.append(buckets[bucket].pop(0))
            if not buckets[bucket]:
                del buckets[bucket]
            if len(selected) == size:
                break
    return selected


def _variation(parent_a: Node, parent_b: Node, rng: np.random.Generator, config: Any,
               statistics=None) -> Node:
    crossover = float(_value(config, "crossover_probability", 0.6))
    mutation = float(_value(config, "mutation_probability", 0.3))
    copy = float(_value(config, "copy_probability", 0.1))
    draw = float(rng.random())
    if draw < crossover:
        parent_a_paths = _paths(parent_a)
        parent_b_paths = _paths(parent_b)
        for _ in range(24 if _value(config, "semantic_generation", False) else 1):
            target = parent_a_paths[int(rng.integers(len(parent_a_paths)))]
            source = parent_b_paths[int(rng.integers(len(parent_b_paths)))]
            replacement = _at_path(parent_b, source)
            if _value(config, "semantic_generation", False) and expression_dimension(_at_path(parent_a, target)) != expression_dimension(replacement):
                continue
            tree = _replace(parent_a, target, replacement)
            if _value(config, "semantic_generation", False):
                try:
                    _semantic_valid(tree, config)
                except ValueError:
                    continue
            return tree
        return parent_a
    if draw < crossover + mutation:
        if _value(config, "structured_mutation", False):
            kind = str(rng.choice(("window", "field", "prune", "subtree"), p=config.mutation_weights))
            tree = _structured_mutation(parent_a, kind, rng, config)
            if statistics is not None:
                statistics[kind + "_attempts"] += 1
                statistics[kind + "_changed"] += int(expression_hash(tree) != expression_hash(parent_a))
            return tree
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
    stage = _value(stage_data, "stage")
    if stage is not None:
        from .config import STAGES
        walk_forward_train = stage.name == "walk_forward_train" and config.fitness_mode == "all_costs_sharpe"
        if stage != STAGES["train"] and not walk_forward_train:
            raise ValueError("search fitness requires the configured training stage")
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
    if config.fitness_mode == "all_costs_sharpe":
        from .cost_fitness import score_all_costs
        score, accepted, reasons, direction, diagnostics = score_all_costs(
            values, labels, stage_data, config, node_count(tree),
        )
        stage_data.fitness_diagnostics[expression_hash(tree)] = diagnostics
        return score, accepted, reasons, values, direction
    score_config = dict(config) if isinstance(config, dict) else config.__dict__.copy()
    score_config["node_count"] = node_count(tree)
    if stage is not None:
        score_config["stage"] = stage
    score = score_training(values, labels, quality, score_config)
    return tuple(score.objective_vector), score.eligible, score.reasons, values, score.direction


def search(stage_data: Any, config: Any, *, progress_callback=None) -> SearchResult:
    """Run a deterministic GP search using one local random generator."""
    from .config import SearchConfig

    if isinstance(config, dict):
        config = SearchConfig(**config)
    elif not isinstance(config, SearchConfig):
        raise TypeError("search config must be a SearchConfig or mapping")
    rng = np.random.default_rng(int(_value(config, "seed", 42)))
    population_size = int(_value(config, "population", 200))
    cost_mode = config.fitness_mode == "all_costs_sharpe"
    if config.behavior_diversity or config.layered_elites or config.dsr_diagnostics:
        from .config import STAGES
        if _value(stage_data, "stage") != STAGES["train"]:
            raise ValueError("research search requires fixed 2024 training stage")
    generations = int(_value(config, "generations", 20))
    max_attempts = min(config.max_attempts, population_size * 50)
    cache: dict[str, tuple[tuple[Any, ...], bool, tuple[str, ...], Node, Any, int | None]] = {}
    values_by_id: dict[str, Any] = {}
    evaluations = 0
    objective_width: int | None = None
    logs: list[dict[str, Any]] = []
    supplied = list(_value(config, "initial_trees", ()))
    robustness_checked: set[str] = set()
    base_scores = {}
    elite_archive = {}
    elite_pool = []
    exploration_pool = []
    diagnostics = _value(stage_data, "fitness_diagnostics", {})
    exploration_slots = min(population_size - 1, max(1, int(np.ceil(population_size * config.exploration_fraction))))

    def diverse_select(items, limit):
        if limit <= 0:
            return []
        if config.behavior_diversity:
            from .research import select_behavior
            return select_behavior(items, limit, diagnostics, _cost_key, config.behavior_cell_capacity)
        return _select_cost_population(items, limit, config)

    def research_parent(pool):
        if config.behavior_diversity:
            from .research import behavior_buckets
            buckets = behavior_buckets(pool, diagnostics, _cost_key)
            cells = sorted(buckets)
            pool = buckets[cells[int(rng.integers(len(cells)))]]
        return _tournament(pool, rng, config)

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
                _semantic_valid(canonical, config)
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
                base_scores[identifier] = (tuple(score), bool(eligible), tuple(failure_reasons))
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
        def needs_more():
            if cost_mode:
                return len(all_exploratory) < population_size
            return len(all_valid) < population_size and (all_valid or len(all_exploratory) < population_size)

        while attempts < max_attempts and needs_more():
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
        if needs_more():
            raise SearchFormationError(
                f"could not form population: unique exploratory candidates "
                f"{len(all_exploratory)}/{population_size} after {attempts}/{max_attempts} attempts; "
                f"eligible candidates {len(all_valid)}/{population_size}; "
                f"reasons={aggregate['invalid_reasons']}"
            )
        return list(all_valid.values()), list(all_exploratory.values()), aggregate

    def apply_training_parameter_stability(
        candidates: Iterable[Candidate], generation_log: dict[str, Any],
    ) -> list[Candidate]:
        items = list(candidates)
        generation_log["parameter_stability_candidates"] = 0
        generation_log["parameter_stability_backtests"] = 0
        if not _value(config, "training_parameter_stability", False):
            return items
        from .cost_fitness import training_parameter_stability

        unchecked = [
            candidate for candidate in sorted(items, key=_cost_key)
            if candidate.eligible and candidate.expression_id not in robustness_checked
            and (config.layered_elites or any(
                _at_path(candidate.tree, path).window is not None
                for path in _paths(candidate.tree)
            ))
        ]
        unchecked = diverse_select(unchecked, config.training_parameter_stability_top_k) if config.behavior_diversity else unchecked[:config.training_parameter_stability_top_k]
        replacements: dict[str, Candidate] = {}
        for candidate in unchecked:
            identifier = candidate.expression_id
            diagnostics = stage_data.fitness_diagnostics.get(identifier)
            if not isinstance(diagnostics, dict):
                raise ValueError(f"missing base fitness diagnostics for {identifier}")
            baseline = diagnostics.get("full_training_period")
            if not isinstance(baseline, dict):
                raise ValueError(f"missing 2024 baseline metrics for {identifier}")
            evidence = training_parameter_stability(
                candidate.tree, candidate.direction, stage_data, config, baseline,
            )
            penalty = (
                config.training_parameter_stability_penalty
                * float(evidence["penalty_units"])
            )
            adjusted_score = (
                _objective_value(candidate.score[0]) - penalty,
                *candidate.score[1:],
            )
            evidence["penalty_weight"] = config.training_parameter_stability_penalty
            evidence["fitness_penalty"] = penalty
            evidence["base_score"] = list(candidate.score)
            evidence["adjusted_score"] = list(adjusted_score)
            diagnostics["training_parameter_stability"] = evidence
            diagnostics["score"] = list(adjusted_score)
            eligible = candidate.eligible
            failure_reasons = candidate.reasons
            if config.layered_elites and _objective_value(adjusted_score[0]) <= 0:
                eligible = False
                failure_reasons += ("stability-adjusted objective is not positive",)
            adjusted = replace(candidate, score=adjusted_score, eligible=eligible, reasons=failure_reasons)
            replacements[identifier] = adjusted
            score, _, _, tree, values, direction = cache[identifier]
            cache[identifier] = (
                adjusted_score, eligible, failure_reasons, tree, values, direction,
            )
            diagnostics["reasons"] = list(failure_reasons)
            robustness_checked.add(identifier)
            generation_log["parameter_stability_candidates"] += 1
            generation_log["parameter_stability_backtests"] += int(
                evidence["backtest_count"]
            )
        return [replacements.get(candidate.expression_id, candidate) for candidate in items]

    def select_layered(items, log):
        nonlocal elite_pool, exploration_pool, elite_archive
        merged = {c.expression_id: c for c in (*elite_archive.values(), *items)}
        # Proposals are compared using base scores before admission. Already
        # checked members recover their final score from cache below.
        proposals = [replace(c, score=base_scores[c.expression_id][0],
                             eligible=base_scores[c.expression_id][1],
                             reasons=base_scores[c.expression_id][2])
                     for c in merged.values()]
        checked = apply_training_parameter_stability(proposals, log)
        final = []
        for candidate in checked:
            identifier = candidate.expression_id
            if identifier in robustness_checked:
                score, eligible, reasons, tree, _, direction = cache[identifier]
                if eligible:
                    final.append(Candidate(tree, identifier, score, eligible, reasons, direction))
        elite_pool = diverse_select(final, population_size - exploration_slots)
        elite_archive = {c.expression_id: c for c in elite_pool}
        # Separate ranking and reserved slots: an unchecked score never
        # displaces a checked elite. Nonelite parents retain base fitness.
        exploration_pool = diverse_select(
            [c for c in proposals if c.expression_id not in elite_archive],
            population_size - len(elite_pool),
        )
        log["elite_population"] = len(elite_pool)
        log["exploration_population"] = len(exploration_pool)
        return elite_pool + exploration_pool

    def log_research(log, current):
        log["cumulative_evaluations"] = evaluations
        log["cumulative_parameter_stability_candidates"] = len(robustness_checked)
        log["cumulative_parameter_stability_backtests"] = sum(
            entry.get("parameter_stability_backtests", 0) for entry in logs
        ) + log.get("parameter_stability_backtests", 0)
        log["selected_eligible_population"] = sum(c.eligible for c in current)
        if config.behavior_diversity:
            from .research import behavior_buckets
            log["behavior_cells"] = len(behavior_buckets(current, diagnostics, _cost_key))

    valid, exploratory, log = fill(supplied, 0)
    if config.layered_elites:
        current = select_layered(exploratory, log)
    else:
        exploratory = apply_training_parameter_stability(exploratory, log)
        valid = [candidate for candidate in exploratory if candidate.eligible]
        current = diverse_select(exploratory, population_size) if cost_mode else select_population(valid, population_size)
    exploratory_current = select_exploratory(exploratory, population_size)
    log_research(log, current)
    logs.append(log)
    if progress_callback is not None:
        progress_callback(dict(log))
    if cost_mode:
        print(f"generation 0: evaluations={evaluations}, eligible={sum(c.eligible for c in current)}/{population_size}", flush=True)
    for generation in range(1, generations):
        parents = current if current else exploratory_current
        offspring: list[Node] = []
        variation_statistics = Counter()
        if config.layered_elites:
            fresh = set()
            for _ in range(max_attempts):
                tree = _random_tree(rng, config)
                try:
                    _semantic_valid(tree, config)
                except ValueError:
                    continue
                identifier = expression_hash(tree)
                if identifier not in cache and identifier not in fresh:
                    fresh.add(identifier)
                    offspring.append(tree)
                if len(offspring) == exploration_slots:
                    break
            if len(offspring) < exploration_slots:
                raise SearchFormationError("could not reserve fresh exploration formula budget")
        while len(offspring) < population_size:
            def choose_parent():
                pool = parents
                if config.layered_elites:
                    pool = exploration_pool if rng.random() < config.exploration_fraction else elite_pool
                    pool = pool or parents
                return research_parent(pool) if config.behavior_diversity else _tournament(pool, rng, config)
            parent_a = choose_parent()
            parent_b = choose_parent()
            if config.structured_mutation:
                offspring.append(_variation(parent_a.tree, parent_b.tree, rng, config, variation_statistics))
            else:
                offspring.append(_variation(parent_a.tree, parent_b.tree, rng, config))
        offspring_valid, offspring_exploratory, log = fill(offspring, generation)
        if config.structured_mutation:
            log["mutation_statistics"] = dict(variation_statistics)
        if config.layered_elites:
            log["random_exploration_proposals"] = exploration_slots
        if cost_mode:
            merged = {c.expression_id: c for c in (*current, *offspring_exploratory)}
            if config.layered_elites:
                merged_valid = tuple(select_layered(merged.values(), log))
            else:
                merged = {
                    candidate.expression_id: candidate
                    for candidate in apply_training_parameter_stability(merged.values(), log)
                }
                merged_valid = tuple(diverse_select(merged.values(), population_size))
        else:
            merged_valid = select_population(tuple(current) + tuple(offspring_valid), population_size)
        merged_exploratory = select_exploratory(tuple(exploratory_current) + tuple(offspring_exploratory), population_size)
        current = merged_valid
        if cost_mode and _value(config, "family_diversity", False):
            log["family_population"] = dict(Counter(
                "+".join(sorted(expression_families(c.tree))) for c in current))
        exploratory_current = merged_exploratory
        log_research(log, current)
        logs.append(log)
        if progress_callback is not None:
            progress_callback(dict(log))
        if cost_mode:
            print(f"generation {generation}: evaluations={evaluations}, eligible={sum(c.eligible for c in current)}/{population_size}", flush=True)
    current = tuple(c for c in (elite_pool if config.layered_elites else current) if c.eligible)
    retained_values = {
        candidate.expression_id: values_by_id[candidate.expression_id]
        for candidate in current
        if candidate.expression_id in values_by_id
    }
    research = {}
    if config.dsr_diagnostics:
        from .research import dsr_report
        research["dsr"] = dsr_report(diagnostics, config.dsr_effective_trials, evaluations)
    if config.behavior_diversity:
        from .research import BEHAVIOR_BINS, behavior_cell
        research["behavior_bins"] = BEHAVIOR_BINS
        research["final_cells"] = {c.expression_id: behavior_cell(diagnostics.get(c.expression_id, {}).get("behavior", {})) for c in current}
    return SearchResult(tuple(current), tuple(logs), evaluations, retained_values,
                        dict(_value(stage_data, "fitness_diagnostics", {})), research)


__all__ = [
    "Candidate", "CandidateInvalidError", "SearchResult", "SearchFormationError", "nondominated_sort",
    "crowding_distance", "select_population", "select_exploratory", "search",
]
