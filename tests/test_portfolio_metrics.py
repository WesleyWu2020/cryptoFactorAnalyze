import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.backtester.metrics import annual_return, annual_volatility, sharpe, max_drawdown, calmar  # noqa: E402


class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.ret = pd.Series(rng.normal(0.001, 0.02, 252))

    def test_annual_return_positive(self):
        ar = annual_return(self.ret)
        self.assertGreater(ar, -1.0)  # valid return

    def test_annual_volatility_positive(self):
        av = annual_volatility(self.ret)
        self.assertGreater(av, 0.0)

    def test_sharpe_reasonable(self):
        s = sharpe(self.ret)
        self.assertTrue(np.isfinite(s))

    def test_max_drawdown_negative(self):
        mdd = max_drawdown(self.ret)
        self.assertLessEqual(mdd, 0.0)

    def test_flat_series_sharpe_zero(self):
        flat = pd.Series([0.0] * 100)
        self.assertAlmostEqual(sharpe(flat), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
