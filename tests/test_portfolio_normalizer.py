import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.combiner.normalizer import normalize_cross_section  # noqa: E402


class TestNormalizer(unittest.TestCase):
    def setUp(self):
        self.panel = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 5 + ["2024-01-02"] * 5),
            "instrument": ["A", "B", "C", "D", "E"] * 2,
            "f1": [1, 2, 3, 4, 100, 1, 2, 3, 4, 100],
            "f2": [10, 20, 30, 40, 50, 11, 22, 33, 44, 55],
        })

    def test_z_score_zero_mean(self):
        out = normalize_cross_section(self.panel, factor_cols=["f1", "f2"],
                                      directions={"f1": 1, "f2": 1},
                                      winsorize_pct=(0.0, 1.0))
        grp = out.groupby("date")["f1"].mean()
        for v in grp:
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_direction_flip(self):
        out = normalize_cross_section(self.panel, factor_cols=["f1"],
                                      directions={"f1": -1},
                                      winsorize_pct=(0.0, 1.0))
        row_a = out[(out["date"] == pd.Timestamp("2024-01-01")) &
                    (out["instrument"] == "A")]["f1"].iloc[0]
        row_e = out[(out["date"] == pd.Timestamp("2024-01-01")) &
                    (out["instrument"] == "E")]["f1"].iloc[0]
        self.assertGreater(row_a, row_e)

    def test_winsorize_clips_outliers(self):
        out = normalize_cross_section(self.panel, factor_cols=["f1"],
                                      directions={"f1": 1},
                                      winsorize_pct=(0.0, 0.8))
        row_e = out[(out["date"] == pd.Timestamp("2024-01-01")) &
                    (out["instrument"] == "E")]["f1"].iloc[0]
        row_d = out[(out["date"] == pd.Timestamp("2024-01-01")) &
                    (out["instrument"] == "D")]["f1"].iloc[0]
        self.assertAlmostEqual(row_e, row_d, places=6)


if __name__ == "__main__":
    unittest.main()
