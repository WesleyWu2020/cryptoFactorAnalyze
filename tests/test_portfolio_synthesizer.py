import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.combiner.synthesizer import synthesize  # noqa: E402


class TestSynthesizer(unittest.TestCase):
    def setUp(self):
        self.panel = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 4),
            "instrument": ["A", "B", "C", "D"],
            "f1": [1.0, 0.5, -0.5, -1.0],
            "f2": [0.8, 0.2, -0.2, -0.8],
        })
        self.weights = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "f1": [0.6],
            "f2": [0.4],
        })

    def test_output_schema(self):
        out = synthesize(self.panel, self.weights, factor_cols=["f1", "f2"])
        self.assertListEqual(list(out.columns), ["date", "instrument", "composite_score"])

    def test_weighted_sum_correct(self):
        out = synthesize(self.panel, self.weights, factor_cols=["f1", "f2"])
        row_a = out[out["instrument"] == "A"]["composite_score"].iloc[0]
        expected = 1.0 * 0.6 + 0.8 * 0.4
        self.assertAlmostEqual(row_a, expected, places=6)

    def test_nan_weights_fallback_equal(self):
        weights_nan = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "f1": [np.nan],
            "f2": [np.nan],
        })
        out = synthesize(self.panel, weights_nan, factor_cols=["f1", "f2"])
        row_a = out[out["instrument"] == "A"]["composite_score"].iloc[0]
        expected = 1.0 * 0.5 + 0.8 * 0.5
        self.assertAlmostEqual(row_a, expected, places=6)


if __name__ == "__main__":
    unittest.main()
