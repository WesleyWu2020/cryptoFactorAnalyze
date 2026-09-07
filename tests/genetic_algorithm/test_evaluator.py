import numpy as np
import pandas as pd

from Genetic_Algorithm.evaluator import evaluate_tree
from Genetic_Algorithm.expression import Node


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
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    eligible.loc[dates[1], "C"] = False
    tree = Node("rank", (Node("rank", (Node("close"),)),))
    result = evaluate_tree(tree, ctx, eligible)
    assert result.index.equals(dates)
    assert result.columns.tolist() == symbols
    assert pd.isna(result.loc[dates[1], "C"])
    assert result.loc[dates[0]].tolist() == [2 / 3, 2 / 3, 2 / 3]


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
