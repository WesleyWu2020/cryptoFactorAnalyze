from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

import Genetic_Algorithm.evaluator as evaluator_module
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


@pytest.mark.parametrize("index_mutation, message", [
    (lambda index: index[[1, 0, 2, 3]], "monotonic"),
    (lambda index: index[[0, 1, 1, 3]], "duplicate"),
])
def test_evaluate_tree_rejects_invalid_date_indexes_before_row_order_operations(
    index_mutation, message
):
    ctx, dates, symbols = _ctx()
    invalid = ctx["close"].copy()
    invalid.index = index_mutation(dates)
    eligible = pd.DataFrame(True, index=dates, columns=symbols)

    with pytest.raises(ValueError, match=message):
        evaluate_tree(Node("rolling_mean", (Node("close"),), window=2), {"close": invalid}, eligible)


@pytest.mark.parametrize("mutation, message", [
    (lambda frame: frame.rename(columns={"A": "AA"}), "axes"),
    (lambda frame: frame.iloc[:, [0, 0, 2]], "duplicate"),
])
def test_evaluate_tree_rejects_context_axis_mismatches_and_duplicates(mutation, message):
    ctx, dates, symbols = _ctx()
    invalid = mutation(ctx["close"])
    eligible = pd.DataFrame(True, index=dates, columns=symbols)

    with pytest.raises(ValueError, match=message):
        evaluate_tree(Node("close"), {"close": invalid}, eligible)


def test_evaluate_tree_rejects_mismatched_required_panel_axes():
    ctx, dates, symbols = _ctx()
    ctx["open"] = ctx["open"].rename(columns={"A": "AA"})
    eligible = pd.DataFrame(True, index=dates, columns=symbols)

    with pytest.raises(ValueError, match="identical axes"):
        evaluate_tree(Node("body_relative"), ctx, eligible)


@pytest.mark.parametrize("invalid_value", [1, 0, "True", object()])
def test_evaluate_tree_rejects_non_boolean_eligibility_values(invalid_value):
    ctx, dates, symbols = _ctx()
    eligible = pd.DataFrame(True, index=dates, columns=symbols, dtype=object)
    eligible.iloc[0, 0] = invalid_value

    with pytest.raises(ValueError, match="boolean"):
        evaluate_tree(Node("close"), ctx, eligible)


def test_evaluate_tree_accepts_boolean_na_eligibility_and_treats_na_as_false():
    ctx, dates, symbols = _ctx()
    eligible = pd.DataFrame(True, index=dates, columns=symbols, dtype=object)
    eligible.iloc[0, 0] = pd.NA

    result = evaluate_tree(Node("close"), ctx, eligible)

    assert pd.isna(result.iloc[0, 0])


@pytest.mark.parametrize("axis", ["index", "columns"])
def test_evaluate_tree_rejects_duplicate_eligible_axes(axis):
    ctx, dates, symbols = _ctx()
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    if axis == "index":
        eligible = eligible.iloc[[0, 0, 2, 3]]
    else:
        eligible = eligible.iloc[:, [0, 0, 2]]

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_tree(Node("close"), ctx, eligible)


def test_evaluate_tree_cache_fingerprint_includes_feature_and_evaluator_sources(monkeypatch):
    ctx, dates, symbols = _ctx()
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    cache = {}

    evaluate_tree(Node("close"), ctx, eligible, cache=cache)
    monkeypatch.setattr(evaluator_module, "_features_source_hash", lambda: "changed-features")
    monkeypatch.setattr(evaluator_module, "_evaluator_source_hash", lambda: "changed-evaluator")
    result = evaluate_tree(Node("close"), ctx, eligible, cache=cache)

    assert result.equals(ctx["close"])
    assert len(cache) == 2


def test_evaluator_cache_content_hash_rejects_stale_reused_supplied_fingerprint():
    ctx, dates, symbols = _ctx()
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    cache = {}
    first = {"close": ctx["close"], "fingerprint": "reused-by-caller"}
    changed_close = ctx["close"].copy()
    changed_close.iloc[0, 0] = 999.0
    second = {"close": changed_close, "fingerprint": "reused-by-caller"}

    evaluate_tree(Node("close"), first, eligible, cache=cache)
    result = evaluate_tree(Node("close"), second, eligible, cache=cache)

    assert result.iloc[0, 0] == 999.0


def test_evaluate_tree_full_and_cutoff_replays_match_through_cutoff_for_nested_expression():
    ctx, dates, symbols = _ctx()
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    tree = Node(
        "rank",
        (Node("rolling_mean", (Node("delta", (Node("close"),), window=1),), window=2),),
    )
    cutoff = dates[2]
    full = evaluate_tree(tree, ctx, eligible)
    cutoff_ctx = {name: frame.loc[:cutoff] for name, frame in ctx.items()}
    cutoff_eligible = eligible.loc[:cutoff]
    replay = evaluate_tree(tree, cutoff_ctx, cutoff_eligible)
    expected = full.loc[:cutoff]

    pd.testing.assert_frame_equal(expected.isna(), replay.isna())
    np.testing.assert_allclose(
        expected.to_numpy(dtype="float64"), replay.to_numpy(dtype="float64"), equal_nan=True
    )


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


def test_evaluate_tree_rejects_reordered_eligibility_axes():
    ctx, dates, symbols = _ctx()
    cache = {}
    first_eligible = pd.DataFrame(True, index=dates, columns=symbols)
    second_eligible = first_eligible.copy()
    second_eligible = second_eligible.loc[:, ["C", "B", "A"]]

    evaluate_tree(Node("close"), ctx, first_eligible, cache=cache)
    with pytest.raises(ValueError, match="axes"):
        evaluate_tree(Node("close"), ctx, second_eligible, cache=cache)


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
