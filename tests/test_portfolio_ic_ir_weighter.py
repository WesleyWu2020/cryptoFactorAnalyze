import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.combiner.ic_ir_weighter import compute_ic_ir_weights  # noqa: E402


class TestICIRWeighter(unittest.TestCase):
    def _make_panel(self, n_dates=20, n_ins=30, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
        instruments = [f"C{i:03d}" for i in range(n_ins)]
        rows = []
        for d in dates:
            for ins in instruments:
                rows.append({
                    "date": d,
                    "instrument": ins,
                    "f1": rng.standard_normal(),
                    "f2": rng.standard_normal(),
                    "future_ret": rng.standard_normal(),
                })
        return pd.DataFrame(rows)

    def test_output_shape_and_columns(self):
        panel = self._make_panel()
        weights = compute_ic_ir_weights(panel, factor_cols=["f1", "f2"],
                                        label_col="future_ret", window=5)
        self.assertIn("date", weights.columns)
        self.assertIn("f1", weights.columns)
        self.assertIn("f2", weights.columns)
        # Each date should have exactly 1 row
        self.assertEqual(weights.groupby("date").size().max(), 1)

    def test_weights_sum_to_one_abs(self):
        panel = self._make_panel()
        weights = compute_ic_ir_weights(panel, factor_cols=["f1", "f2"],
                                        label_col="future_ret", window=5)
        # Drop NaN rows (first window-1 dates won't have weights)
        valid = weights.dropna()
        for _, row in valid.iterrows():
            total = abs(row["f1"]) + abs(row["f2"])
            if total > 1e-12:
                self.assertAlmostEqual(total, 1.0, places=6)

    def test_no_future_leak(self):
        """Weights at date t must not use factor/label data at date t."""
        panel = self._make_panel(n_dates=15, n_ins=20)
        weights_full = compute_ic_ir_weights(panel, factor_cols=["f1"],
                                             label_col="future_ret", window=5)
        cutoff = pd.Timestamp("2024-01-10")
        panel_cut = panel[panel["date"] <= cutoff].copy()
        weights_cut = compute_ic_ir_weights(panel_cut, factor_cols=["f1"],
                                            label_col="future_ret", window=5)
        merged = weights_full[weights_full["date"] <= cutoff].merge(
            weights_cut, on="date", suffixes=("_full", "_cut"))
        diff = (merged["f1_full"] - merged["f1_cut"]).abs().max()
        self.assertLess(diff, 1e-10)


if __name__ == "__main__":
    unittest.main()
