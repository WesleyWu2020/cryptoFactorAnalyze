import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.combiner.orthogonalizer import orthogonalize  # noqa: E402


class TestOrthogonalizer(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 30
        f1 = rng.standard_normal(n)
        f2 = f1 * 0.9 + rng.standard_normal(n) * 0.1   # highly correlated with f1
        f3 = rng.standard_normal(n)
        dates = pd.to_datetime(["2024-01-01"] * n)
        instruments = [f"C{i:03d}" for i in range(n)]
        self.panel = pd.DataFrame({
            "date": dates,
            "instrument": instruments,
            "f1": f1, "f2": f2, "f3": f3,
        })

    def test_output_shape_unchanged(self):
        out = orthogonalize(self.panel, factor_cols=["f1", "f2", "f3"])
        self.assertEqual(out.shape, self.panel.shape)

    def test_correlation_reduced(self):
        out = orthogonalize(self.panel, factor_cols=["f1", "f2", "f3"])
        cor_before = abs(self.panel["f1"].corr(self.panel["f2"]))
        cor_after  = abs(out["f1"].corr(out["f2"]))
        self.assertLess(cor_after, cor_before)

    def test_degenerate_fallback_returns_unchanged(self):
        tiny = self.panel.head(2).copy()  # 2 instruments, 3 factors → degenerate
        out = orthogonalize(tiny, factor_cols=["f1", "f2", "f3"])
        pd.testing.assert_frame_equal(out, tiny)


if __name__ == "__main__":
    unittest.main()
