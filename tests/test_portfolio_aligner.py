import unittest
import pathlib
import sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.factor_pool.aligner import align_factors  # noqa: E402


def _mk(name, rows):
    return pd.DataFrame(rows, columns=["date", "instrument", name]).assign(
        date=lambda d: pd.to_datetime(d["date"]),
    )


class TestAligner(unittest.TestCase):
    def test_align_two_factors(self):
        f1 = _mk("f1", [["2024-01-01", "BTCUSDT", 0.1],
                        ["2024-01-02", "BTCUSDT", 0.2]])
        f2 = _mk("f2", [["2024-01-01", "BTCUSDT", 1.0],
                        ["2024-01-02", "BTCUSDT", 2.0]])
        panel = align_factors({"f1": f1, "f2": f2})
        self.assertEqual(set(panel.columns), {"date", "instrument", "f1", "f2"})
        self.assertEqual(len(panel), 2)

    def test_align_outer_join_missing_filled_nan(self):
        f1 = _mk("f1", [["2024-01-01", "BTCUSDT", 0.1]])
        f2 = _mk("f2", [["2024-01-01", "ETHUSDT", 9.0]])
        panel = align_factors({"f1": f1, "f2": f2})
        self.assertEqual(len(panel), 2)
        self.assertTrue(panel["f1"].isna().any())
        self.assertTrue(panel["f2"].isna().any())

    def test_universe_filter(self):
        f1 = _mk("f1", [["2024-01-01", "BTCUSDT", 0.1],
                        ["2024-01-01", "DOGEUSDT", 0.3]])
        universe_by_date = {pd.Timestamp("2024-01-01"): {"BTCUSDT"}}
        panel = align_factors({"f1": f1}, universe_by_date=universe_by_date)
        self.assertEqual(set(panel["instrument"]), {"BTCUSDT"})


if __name__ == "__main__":
    unittest.main()
