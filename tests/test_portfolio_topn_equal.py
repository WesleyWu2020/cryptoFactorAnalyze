import unittest
import pathlib
import sys
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.portfolio_builder.topn_equal import build_topn_weights  # noqa: E402


class TestTopNEqual(unittest.TestCase):
    def setUp(self):
        self.scores = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 5),
            "instrument": ["A", "B", "C", "D", "E"],
            "composite_score": [0.9, 0.7, 0.5, 0.3, 0.1],
        })
        self.universe = {
            pd.Timestamp("2024-01-01"): {"A", "B", "C", "D", "E"}
        }

    def test_top3_equal_weight(self):
        weights = build_topn_weights(self.scores, self.universe, top_n=3)
        d = pd.Timestamp("2024-01-01")
        self.assertIn(d, weights)
        w = weights[d]
        self.assertEqual(set(w.keys()), {"A", "B", "C"})
        for v in w.values():
            self.assertAlmostEqual(v, 1/3, places=6)

    def test_universe_filter(self):
        univ = {pd.Timestamp("2024-01-01"): {"B", "C", "D"}}
        weights = build_topn_weights(self.scores, univ, top_n=3)
        w = weights[pd.Timestamp("2024-01-01")]
        self.assertEqual(set(w.keys()), {"B", "C", "D"})

    def test_fewer_than_n_available(self):
        univ = {pd.Timestamp("2024-01-01"): {"A", "B"}}
        weights = build_topn_weights(self.scores, univ, top_n=5)
        w = weights[pd.Timestamp("2024-01-01")]
        self.assertEqual(len(w), 2)
        for v in w.values():
            self.assertAlmostEqual(v, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
