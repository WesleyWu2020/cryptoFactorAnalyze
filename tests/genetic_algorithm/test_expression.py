import pytest

from Genetic_Algorithm.expression import (
    Node,
    canonical_tree,
    expression_hash,
    history_days,
    required_fields,
    validate_tree,
)


def test_node_is_frozen_and_commutative_forms_are_canonical():
    left = Node("add", (Node("close"), Node("open")))
    right = Node("add", (Node("open"), Node("close")))
    assert canonical_tree(left) == canonical_tree(right)
    assert expression_hash(left) == expression_hash(right)
    with pytest.raises((AttributeError, TypeError)):
        left.op = "subtract"


def test_canonicalization_preserves_non_commutative_order():
    left = Node("subtract", (Node("close"), Node("open")))
    right = Node("subtract", (Node("open"), Node("close")))
    assert canonical_tree(left) != canonical_tree(right)


def test_tree_metadata_accumulates_causal_history_and_fields():
    tree = Node("rolling_mean", (Node("lag", (Node("close"),), window=2),), window=3)
    assert history_days(tree) == 4
    assert required_fields(tree) == {"close"}


def test_derived_terminal_reports_raw_dependencies():
    assert required_fields(Node("return_1d")) == {"close"}
    assert required_fields(Node("taker_quote_ratio")) == {"taker_buy_quote_volume", "quote_volume"}


def test_validation_rejects_illegal_lags_unknown_fields_and_budget_overruns():
    config = {"max_depth": 2, "max_nodes": 5, "max_history": 10}
    with pytest.raises(ValueError, match="positive"):
        validate_tree(Node("lag", (Node("close"),), window=-1), config)
    with pytest.raises(ValueError, match="unknown field"):
        validate_tree(Node("not_a_field"), config)
    too_deep = Node("negate", (Node("negate", (Node("negate", (Node("close"),)),)),))
    with pytest.raises(ValueError, match="depth"):
        validate_tree(too_deep, config)
