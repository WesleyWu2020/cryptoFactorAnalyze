"""Tests for factor cluster selector."""
import unittest
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.factor_pool.cluster_selector import cluster_and_select  # noqa: E402


class TestClusterAndSelect(unittest.TestCase):
    def _make_panel(self, factor_dict: dict) -> pd.DataFrame:
        """Build a flat panel DataFrame from {name: array} dict."""
        df = pd.DataFrame(factor_dict)
        return df

    def test_disabled_when_threshold_zero(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"f1": rng.standard_normal(100), "f2": rng.standard_normal(100)})
        result = cluster_and_select(df, ["f1", "f2"], threshold=0.0)
        self.assertEqual(result, ["f1", "f2"])

    def test_single_factor_unchanged(self):
        df = pd.DataFrame({"f1": np.ones(50)})
        result = cluster_and_select(df, ["f1"], threshold=0.6)
        self.assertEqual(result, ["f1"])

    def test_highly_correlated_pair_merged(self):
        rng = np.random.default_rng(42)
        base = rng.standard_normal(200)
        df = pd.DataFrame({
            "f1": base + rng.standard_normal(200) * 0.05,  # nearly identical
            "f2": base + rng.standard_normal(200) * 0.05,
            "f3": rng.standard_normal(200),               # uncorrelated
        })
        result = cluster_and_select(df, ["f1", "f2", "f3"], threshold=0.6)
        # f1 and f2 should be merged → 2 factors total
        self.assertEqual(len(result), 2)
        self.assertIn("f3", result)
        # One of f1/f2 must be present
        self.assertTrue("f1" in result or "f2" in result)

    def test_uncorrelated_factors_all_kept(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({f"f{i}": rng.standard_normal(200) for i in range(5)})
        result = cluster_and_select(df, [f"f{i}" for i in range(5)], threshold=0.6)
        self.assertEqual(len(result), 5)

    def test_missing_col_ignored(self):
        df = pd.DataFrame({"f1": np.ones(50), "f2": np.zeros(50)})
        result = cluster_and_select(df, ["f1", "f_nonexistent"], threshold=0.6)
        self.assertEqual(result, ["f1"])

    def test_coverage_tiebreak_prefers_more_data(self):
        """Representative should be the factor with more non-NaN values."""
        rng = np.random.default_rng(99)
        base = rng.standard_normal(200)
        f1 = base.copy()
        f2 = base + rng.standard_normal(200) * 0.01
        f2[:50] = np.nan  # f2 has 50 fewer valid rows
        df = pd.DataFrame({"f1": f1, "f2": f2})
        result = cluster_and_select(df, ["f1", "f2"], threshold=0.6)
        self.assertEqual(result, ["f1"])  # f1 has more coverage


if __name__ == "__main__":
    unittest.main()
