"""Tests for the dimension (量纲) system enforced in validate_tree."""

import pytest

from Genetic_Algorithm.expression import (
    Node,
    expression_dimension,
    history_days,
    validate_tree,
)
from Genetic_Algorithm.operators import combine_dimensions


def test_terminal_dimensions_cover_known_fields():
    assert expression_dimension(Node("close")) == "price"
    assert expression_dimension(Node("volume")) == "base_volume"
    assert expression_dimension(Node("quote_volume")) == "quote_volume"
    assert expression_dimension(Node("trade_count")) == "count"
    assert expression_dimension(Node("quote_per_trade")) == "money_per_trade"
    assert expression_dimension(Node("return_1d")) == "ratio"


def test_add_and_subtract_require_matching_dimensions():
    with pytest.raises(ValueError, match="matching dimensions"):
        combine_dimensions("add", "price", "base_volume")
    assert combine_dimensions("add", "price", "price") == "price"
    assert combine_dimensions("subtract", "ratio", "ratio") == "ratio"


def test_multiply_and_division_algebra():
    assert combine_dimensions("multiply", "ratio", "price") == "price"
    assert combine_dimensions("multiply", "ratio", "ratio") == "ratio"
    assert combine_dimensions("multiply", "price", "price") == "mixed"
    assert combine_dimensions("safe_div", "price", "price") == "ratio"
    assert combine_dimensions("safe_div", "price", "ratio") == "price"
    assert combine_dimensions("safe_div", "quote_volume", "base_volume") == "price"
    assert combine_dimensions("safe_div", "quote_volume", "count") == "money_per_trade"
    assert combine_dimensions("safe_div", "base_volume", "count") == "base_per_trade"
    assert combine_dimensions("safe_div", "price", "count") == "mixed"


def test_mixed_operands_are_illegal_in_binary_operators():
    for op in ("add", "subtract", "multiply", "safe_div"):
        with pytest.raises(ValueError, match="mixed"):
            combine_dimensions(op, "mixed", "price")


def test_expression_dimension_rejects_price_plus_volume():
    tree = Node("add", (Node("close"), Node("volume")))
    with pytest.raises(ValueError, match="dimensions"):
        expression_dimension(tree)
    with pytest.raises(ValueError, match="dimensions"):
        validate_tree(tree, {})


def test_price_ratio_and_unit_aware_division_are_legal():
    assert expression_dimension(Node("safe_div", (Node("close"), Node("open")))) == "ratio"
    tree = Node("safe_div", (Node("quote_volume"), Node("volume")))
    assert expression_dimension(tree) == "price"
    validate_tree(tree, {})


def test_ratio_times_price_is_legal_and_preserves_price():
    tree = Node("multiply", (Node("return_1d"), Node("close")))
    assert expression_dimension(tree) == "price"
    validate_tree(tree, {})


def test_signed_sqrt_output_cannot_enter_binary_operators():
    tree = Node("add", (Node("signed_sqrt", (Node("close"),)), Node("volume")))
    with pytest.raises(ValueError, match="mixed"):
        expression_dimension(tree)
    with pytest.raises(ValueError, match="mixed"):
        validate_tree(tree, {})


def test_mixed_output_can_feed_preserve_and_ratio_operators():
    assert expression_dimension(Node("rolling_mean", (Node("signed_sqrt", (Node("close"),)),), window=3)) == "mixed"
    assert expression_dimension(Node("ts_rank", (Node("safe_log", (Node("close"),)),), window=3)) == "ratio"


def test_operator_output_dimensions_follow_their_class():
    assert expression_dimension(Node("rolling_std", (Node("close"),), window=3)) == "price"
    assert expression_dimension(Node("rolling_corr", (Node("close"), Node("volume")), window=3)) == "ratio"
    assert expression_dimension(Node("rank", (Node("close"),))) == "ratio"
    assert expression_dimension(Node("cross_sectional_zscore", (Node("close"),))) == "ratio"
    assert expression_dimension(Node("rolling_skew", (Node("close"),), window=20)) == "ratio"
    assert expression_dimension(Node("safe_log", (Node("close"),))) == "mixed"


@pytest.mark.parametrize("op", ["rolling_skew", "rolling_kurt", "rolling_autocorr"])
def test_min_operator_window_is_enforced(op):
    with pytest.raises(ValueError, match="window >= 20"):
        validate_tree(Node(op, (Node("close"),), window=10), {})
    validate_tree(Node(op, (Node("close"),), window=20), {})


def test_efficiency_ratio_history_counts_full_shift_window():
    assert history_days(Node("efficiency_ratio", (Node("close"),), window=5)) == 5
    assert history_days(Node("rolling_cv", (Node("close"),), window=5)) == 4
