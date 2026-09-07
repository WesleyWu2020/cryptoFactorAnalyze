from collections import OrderedDict

import numpy as np
import pandas as pd

from Genetic_Algorithm.evaluator import evaluate_tree
from Genetic_Algorithm.expression import Node, expression_hash


def _ctx():
    dates = pd.date_range("2025-01-01", periods=4, freq="D")
    symbols = ["A", "B", "C"]
    def frame(values):
        return pd.DataFrame(values, index=dates, columns=symbols, dtype="float64")
    return {
        "close": frame([[1, 1, 1], [2, 2, 2], [3, 4, 3], [4, 5, 4]]),
        "open": frame([[1, 1, 1], [1, 2, 1], [2, 3, 2], [3, 4, 3]]),
        "high": frame([[2, 2, 2]] * 4),
        "low": frame([[0, 0, 0]] * 4),
        "volume": frame([[10, 20, 30]] * 4),
        "quote_volume": frame([[100, 200, 300]] * 4),
        "trade_count": frame([[1, 2, 4]] * 4),
        "taker_buy_base_volume": frame([[5, 10, 15]] * 4),
        "taker_buy_quote_volume": frame([[50, 100, 150]] * 4),
    }, dates, symbols


def test_evaluate_tree_applies_eligibility_to_nested_rank_and_final_result():
    ctx, dates, symbols = _ctx()
    eligible = pd.DataFrame(True, index=dates, columns=symbols, dtype=object)
    eligible.loc[dates[1], "C"] = False
    tree = Node("rank", (Node("rank", (Node("close"),)),))
    result = evaluate_tree(tree, ctx, eligible)
    assert result.index.equals(dates)
    assert result.columns.tolist() == symbols
    assert pd.isna(result.loc[dates[1], "C"])
    assert result.loc[dates[0]].tolist() == [2 / 3, 2 / 3, 2 / 3]


def test_evaluate_tree_treats_unknown_eligibility_as_ineligible():
    ctx, dates, symbols = _ctx()
    eligible = pd.DataFrame(True, index=dates, columns=symbols, dtype=object)
    eligible.loc[dates[0], "B"] = np.nan
    tree = Node("rank", (Node("close"),))

    result = evaluate_tree(tree, ctx, eligible)

    assert pd.isna(result.loc[dates[0], "B"])
    assert result.loc[dates[0], ["A", "C"]].tolist() == [0.75, 0.75]


def test_evaluate_tree_has_cumulative_history_and_ignores_future_only_symbols():
    ctx, dates, symbols = _ctx()
    ctx["close"].loc[dates[0], "C"] = np.nan
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    tree = Node("delta", (Node("close"),), window=2)
    result = evaluate_tree(tree, ctx, eligible)
    assert pd.isna(result.loc[dates[0], "A"])
    assert result.loc[dates[2], "A"] == 2.0
    assert pd.isna(result.loc[dates[2], "C"])


def test_evaluator_cache_respects_byte_budget():
    ctx, dates, symbols = _ctx()
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    cache = {}
    evaluate_tree(Node("close"), ctx, eligible, cache=cache, cache_bytes=1)
    assert cache == {}


def test_evaluator_cache_enforces_new_budget_on_existing_entries_and_lru_order():
    ctx, dates, symbols = _ctx()
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    cache = OrderedDict()
    entry_bytes = 4 * 3 * 8
    cache_bytes = entry_bytes * 2

    evaluate_tree(Node("close"), ctx, eligible, cache=cache, cache_bytes=cache_bytes)
    evaluate_tree(Node("open"), ctx, eligible, cache=cache, cache_bytes=cache_bytes)
    evaluate_tree(Node("close"), ctx, eligible, cache=cache, cache_bytes=cache_bytes)
    evaluate_tree(Node("high"), ctx, eligible, cache=cache, cache_bytes=cache_bytes)

    assert len(cache) == 2
    assert [key[0] for key in cache] == [
        expression_hash(Node("close")), expression_hash(Node("high")),
    ]


def test_evaluator_cache_drops_existing_entries_when_budget_becomes_zero_or_entry_is_oversized():
    ctx, dates, symbols = _ctx()
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    cache = {}

    evaluate_tree(Node("close"), ctx, eligible, cache=cache)
    assert cache
    evaluate_tree(Node("close"), ctx, eligible, cache=cache, cache_bytes=0)
    assert cache == {}

    evaluate_tree(Node("close"), ctx, eligible, cache=cache)
    evaluate_tree(Node("open"), ctx, eligible, cache=cache, cache_bytes=1)
    assert cache == {}


def test_evaluator_cache_fingerprint_preserves_axis_label_types():
    labels = [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")]
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=labels, columns=["A", "B"])
    string_frame = frame.copy()
    string_frame.index = pd.Index([str(label) for label in labels], dtype=object)
    cache = {}
    eligible = pd.DataFrame(True, index=frame.index, columns=frame.columns)
    string_eligible = pd.DataFrame(True, index=string_frame.index, columns=string_frame.columns)

    evaluate_tree(Node("close"), {"close": frame}, eligible, cache=cache)
    result = evaluate_tree(Node("close"), {"close": string_frame}, string_eligible, cache=cache)

    assert result.index.tolist() == string_frame.index.tolist()
    assert all(isinstance(label, str) for label in result.index)


def test_evaluator_cache_fingerprint_includes_eligibility_axes():
    ctx, dates, symbols = _ctx()
    cache = {}
    first_eligible = pd.DataFrame(True, index=dates, columns=symbols)
    second_eligible = first_eligible.copy()
    second_eligible = second_eligible.loc[:, ["C", "B", "A"]]

    evaluate_tree(Node("close"), ctx, first_eligible, cache=cache)
    result = evaluate_tree(Node("close"), ctx, second_eligible, cache=cache)

    assert result.index.equals(second_eligible.index)
    assert result.columns.equals(second_eligible.columns)
    expected = ctx["close"].reindex(index=second_eligible.index, columns=second_eligible.columns)
    assert result.equals(expected)


def test_nested_rank_ignores_extreme_future_only_ineligible_symbol():
    dates = pd.date_range("2025-01-01", periods=4, freq="D")
    symbols = ["A", "B", "FUTURE_EXTREME"]
    close = pd.DataFrame(
        [[1.0, 2.0, np.nan], [2.0, 3.0, np.nan], [3.0, 4.0, np.nan], [4.0, 5.0, 1e300]],
        index=dates,
        columns=symbols,
    )
    ctx = {"close": close}
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    eligible["FUTURE_EXTREME"] = False
    tree = Node("rank", (Node("rank", (Node("close"),)),))

    result = evaluate_tree(tree, ctx, eligible)
    baseline = evaluate_tree(
        tree,
        {"close": close.drop(columns="FUTURE_EXTREME")},
        eligible.drop(columns="FUTURE_EXTREME"),
    )

    pd.testing.assert_frame_equal(result.drop(columns="FUTURE_EXTREME"), baseline)


def test_terminal_formulas_are_causal_and_use_safe_division():
    ctx, dates, symbols = _ctx()
    ctx["volume"].iloc[0, 0] = 0.0
    ctx["quote_volume"].iloc[0, 0] = 0.0
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    ratio = evaluate_tree(Node("taker_base_ratio"), ctx, eligible)
    assert pd.isna(ratio.loc[dates[0], "A"])
    ctx["quote_volume"].iloc[0, 0] = 100.0
    quote_trade = evaluate_tree(Node("quote_per_trade"), ctx, eligible)
    assert quote_trade.loc[dates[0], "A"] == 100.0
    assert pd.isna(evaluate_tree(Node("return_1d"), ctx, eligible).loc[dates[0], "A"])
