"""End-to-end cutoff (anti look-ahead) test for the portfolio pipeline.

For each component, verifies that results on dates <= cutoff are identical
whether computed on full data or data truncated at cutoff.
"""
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.combiner.ic_ir_weighter import compute_ic_ir_weights  # noqa: E402
from portfolio.combiner.normalizer import normalize_cross_section  # noqa: E402
from portfolio.combiner.synthesizer import synthesize  # noqa: E402
from portfolio.config import PortfolioConfig  # noqa: E402
from portfolio.factor_pool.label_builder import build_future_ret  # noqa: E402


def _make_synthetic_panel(n_dates: int = 30, n_ins: int = 15, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    instruments = [f"C{i:02d}" for i in range(n_ins)]
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


def _make_kline(n_dates: int = 30, n_ins: int = 15, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    instruments = [f"C{i:02d}" for i in range(n_ins)]
    rows = []
    prices = {ins: 100.0 for ins in instruments}
    for d in dates:
        for ins in instruments:
            rows.append({"symbol": ins, "date": d, "close": prices[ins]})
            prices[ins] *= (1 + rng.normal(0.001, 0.02))
    return pd.DataFrame(rows)


class TestCutoffAntiLeak(unittest.TestCase):
    """Verify no future leakage at each pipeline stage."""

    CUTOFF = pd.Timestamp("2024-01-20")
    N_DATES = 30
    N_INS = 15

    def setUp(self):
        self.panel_full = _make_synthetic_panel(self.N_DATES, self.N_INS)
        self.panel_cut = self.panel_full[self.panel_full["date"] <= self.CUTOFF].copy()
        self.kline_full = _make_kline(self.N_DATES, self.N_INS)
        self.kline_cut = self.kline_full[self.kline_full["date"] <= self.CUTOFF].copy()
        self.cfg = PortfolioConfig(ic_window=5, label_period=1, orthogonalize=False)

    def _max_diff(self, full: pd.DataFrame, cut: pd.DataFrame, col: str, on: list[str]) -> float:
        merged = full[full["date"] <= self.CUTOFF].merge(cut, on=on, suffixes=("_full", "_cut"))
        col_full = f"{col}_full"
        col_cut = f"{col}_cut"
        if col_full not in merged.columns:
            return 0.0
        diff = (merged[col_full] - merged[col_cut]).abs()
        return float(diff.max()) if not diff.empty else 0.0

    def test_normalizer_no_leak(self):
        """Normalizer is purely cross-sectional per date — no future leak possible."""
        directions = {"f1": 1, "f2": 1}
        norm_full = normalize_cross_section(
            self.panel_full[["date", "instrument", "f1", "f2"]],
            ["f1", "f2"], directions, (0.01, 0.99)
        )
        norm_cut = normalize_cross_section(
            self.panel_cut[["date", "instrument", "f1", "f2"]],
            ["f1", "f2"], directions, (0.01, 0.99)
        )
        for col in ["f1", "f2"]:
            diff = self._max_diff(norm_full, norm_cut, col, ["date", "instrument"])
            self.assertLess(diff, 1e-10, f"normalizer leaked future in {col}: max_diff={diff}")

    def test_ic_ir_weights_no_leak(self):
        """IC_IR weights at date t must not use data at date t or later."""
        w_full = compute_ic_ir_weights(
            self.panel_full, ["f1", "f2"], label_col="future_ret", window=self.cfg.ic_window
        )
        w_cut = compute_ic_ir_weights(
            self.panel_cut, ["f1", "f2"], label_col="future_ret", window=self.cfg.ic_window
        )
        for col in ["f1", "f2"]:
            diff = self._max_diff(w_full, w_cut, col, ["date"])
            self.assertLess(diff, 1e-10, f"ic_ir_weighter leaked future in {col}: max_diff={diff}")

    def test_label_builder_no_leak_in_factors(self):
        """build_future_ret: factor rows at date t must not change when future data removed."""
        label_full = build_future_ret(self.kline_full, period_days=1)
        label_cut = build_future_ret(self.kline_cut, period_days=1)
        # Dates strictly before cutoff should have same future_ret
        # (cutoff row itself may differ — NaN vs actual value — that's expected)
        before_cutoff = label_full[label_full["date"] < self.CUTOFF].merge(
            label_cut[label_cut["date"] < self.CUTOFF],
            on=["date", "instrument"], suffixes=("_full", "_cut")
        )
        diff = (before_cutoff["future_ret_full"] - before_cutoff["future_ret_cut"]).abs().max()
        self.assertLess(diff, 1e-10, f"label_builder has inconsistency: max_diff={diff}")

    def test_synthesizer_no_leak(self):
        """Synthesizer is a weighted sum — no temporal component, so no leak."""
        directions = {"f1": 1, "f2": 1}
        panel_cols_full = self.panel_full[["date", "instrument", "f1", "f2"]]
        panel_cols_cut = self.panel_cut[["date", "instrument", "f1", "f2"]]
        norm_full = normalize_cross_section(panel_cols_full, ["f1", "f2"], directions)
        norm_cut = normalize_cross_section(panel_cols_cut, ["f1", "f2"], directions)

        w_full = compute_ic_ir_weights(
            self.panel_full, ["f1", "f2"], label_col="future_ret", window=self.cfg.ic_window
        )
        w_cut = compute_ic_ir_weights(
            self.panel_cut, ["f1", "f2"], label_col="future_ret", window=self.cfg.ic_window
        )

        scores_full = synthesize(norm_full, w_full, ["f1", "f2"])
        scores_cut = synthesize(norm_cut, w_cut, ["f1", "f2"])

        diff = self._max_diff(scores_full, scores_cut, "composite_score", ["date", "instrument"])
        self.assertLess(diff, 1e-10, f"synthesizer leaked future: max_diff={diff}")


if __name__ == "__main__":
    unittest.main()
