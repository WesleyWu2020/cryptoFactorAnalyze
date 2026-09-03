import unittest
import pathlib
import sys
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.backtester.engine import run_backtest  # noqa: E402


class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        # 5 dates, 3 instruments
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        instruments = ["A", "B", "C"]
        rows = []
        # Instrument A: +1% daily, B: flat, C: -1% daily
        prices = {"A": 100.0, "B": 100.0, "C": 100.0}
        for d in dates:
            for ins in instruments:
                rows.append({"symbol": ins, "date": d, "close": prices[ins]})
            prices["A"] *= 1.01
            prices["B"] *= 1.00
            prices["C"] *= 0.99
        self.kline = pd.DataFrame(rows)

        # Equal weight across all 3
        self.weights = {
            d: {"A": 1/3, "B": 1/3, "C": 1/3}
            for d in dates[:-1]  # no weight for last date
        }

    def test_output_is_series(self):
        ret = run_backtest(self.kline, self.weights, fee_rate=0.0)
        self.assertIsInstance(ret, pd.Series)

    def test_zero_fee_equal_weight(self):
        ret = run_backtest(self.kline, self.weights, fee_rate=0.0)
        # With A+1%, B+0%, C-1%, equal weight → ~0% return each period
        for r in ret:
            self.assertAlmostEqual(r, 0.0, places=4)

    def test_fee_reduces_return(self):
        ret_no_fee = run_backtest(self.kline, self.weights, fee_rate=0.0)
        ret_with_fee = run_backtest(self.kline, self.weights, fee_rate=0.001)
        self.assertLessEqual(ret_with_fee.sum(), ret_no_fee.sum())


if __name__ == "__main__":
    unittest.main()
