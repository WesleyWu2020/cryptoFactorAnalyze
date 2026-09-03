import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.factor_pool.label_builder import build_future_ret  # noqa: E402


class TestLabelBuilder(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        self.kline = pd.concat([
            pd.DataFrame({"date": dates, "symbol": "BTCUSDT",
                          "close": [100, 101, 102, 103, 104, 105]}),
            pd.DataFrame({"date": dates, "symbol": "ETHUSDT",
                          "close": [10, 11, 12, 13, 14, 15]}),
        ], ignore_index=True)

    def test_future_ret_N3(self):
        fr = build_future_ret(self.kline, period_days=3)
        cell = fr[(fr["date"] == pd.Timestamp("2024-01-01")) &
                  (fr["instrument"] == "BTCUSDT")]["future_ret"].iloc[0]
        self.assertAlmostEqual(cell, 103 / 100 - 1, places=10)

    def test_future_ret_last_N_rows_are_nan(self):
        fr = build_future_ret(self.kline, period_days=3)
        last_dates = fr[fr["instrument"] == "BTCUSDT"].sort_values("date")["date"].iloc[-3:]
        for d in last_dates:
            v = fr[(fr["date"] == d) & (fr["instrument"] == "BTCUSDT")]["future_ret"].iloc[0]
            self.assertTrue(pd.isna(v))

    def test_output_schema(self):
        fr = build_future_ret(self.kline, period_days=2)
        self.assertEqual(list(fr.columns), ["date", "instrument", "future_ret"])


if __name__ == "__main__":
    unittest.main()
